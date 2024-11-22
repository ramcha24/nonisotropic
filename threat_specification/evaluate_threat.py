import os
from pathlib import Path
from datetime import datetime

import pickle
import numpy as np
import pandas as pd
import math
import seaborn as sns

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from textwrap import wrap

from dreamsim import dreamsim
from autoattack_mod.autoattack import AutoAttack

import torch
from torch.utils.data import DataLoader

from platforms.platform import get_platform
from foundations import hparams, paths
import datasets.registry as datareg
from datasets.afc import TwoAFCDataset
from utilities.miscellaneous import timeprint, _cast, _move
from datasets.perturbations import (
    transformations,
    common_corruptions_2d,
    common_corruptions_2d_bar,
    common_corruptions_3d,
    backgrounds,
)
from threat_specification.subset_selection import load_threat_specification
from threat_specification.projected_displacement_old import NIT_alt
from threat_specification.projected_displacement import ProjectedDisplacement

from models.utils import load_model
from models.utils import clean_accuracy as compute_accuracy
from models.robustbench_registry import imagenet_Linf_RB_model_names

from training.adv_train_util import get_attack

color_map = {
    "Unsafe": "r",
    "Blur": "b",
    "Noise": "g",
    "Digital": "y",
    "Weather": "c",
    "Compression": "k",
    "Geometric": "magenta",
    "Occlusion": "purple",
    "Augmentation": "orange",
    "Depth": "brown",
}


def find_unsafe_shuffe(examples, labels):
    n = labels.size(0)
    indices = torch.randperm(examples.shape[0])
    duplicate = True
    count = 0
    while duplicate and count < 10:
        count += 1
        indices = torch.randperm(examples.shape[0])

        for i in range(examples.shape[0]):
            duplicate = False
            if labels[i] == labels[indices[i]]:
                duplicate = True
                break

    # there might be some duplicates, but we can ignore them for now
    return indices


def collect_threat_statistics(
    loader,
    num_batches,
    ds_model,
    threat_model,
    device,
    apply_transform=None,
    perturbation_style=None,
    severity=None,
    float_16=False,
    weighted=False,
    segmented=False,
):
    # info_str = f"\nCollecting threat statistics for {perturbation_style}"
    # info_str += f" with severity {severity}" if severity else ""
    # timeprint(info_str)

    info = {}
    info["L2 stats"] = np.array([], dtype=float)
    info["Linf stats"] = np.array([], dtype=float)
    info["Threat stats"] = np.array([], dtype=float) if threat_model else None
    info["DS stats"] = np.array([], dtype=float) if ds_model else None
    info["perturbation_style"] = perturbation_style
    info["severity"] = severity
    info["weighted"] = weighted
    info["segmented"] = segmented

    batch_counter = 0

    device = torch.device(device)

    for batch_items in loader:
        if batch_counter > num_batches:
            break

        if apply_transform:
            examples, labels = batch_items
            if apply_transform == "Unsafe":
                unsafe_indices = find_unsafe_shuffe(examples, labels)
                perturbed_examples = examples.clone()[unsafe_indices]
                perturbed_labels = labels.clone()[unsafe_indices]
            else:
                perturbed_examples = apply_transform(examples)
                perturbed_labels = labels.clone()
        else:
            examples, labels, perturbed_examples, perturbed_labels = batch_items

        diff = perturbed_examples - examples
        perturbations_flat = torch.flatten(perturbed_examples - examples, start_dim=1)

        batch_L2_norms = torch.linalg.norm(perturbations_flat, dim=1, ord=2).numpy()

        batch_Linf_norms = torch.linalg.norm(
            perturbations_flat, dim=1, ord=float("inf")
        ).numpy()

        batch_DS_stats = (
            ds_model(_move(examples, device), _move(perturbed_examples, device))
            .cpu()
            .numpy()
        )

        batch_threat_stats = threat_model.evaluate(
            examples,
            labels,
            perturbed_examples,
            gray_scale=False,
            weighted=weighted,
            segmented=segmented,
        )

        info["L2 stats"] = np.concatenate((info["L2 stats"], batch_L2_norms))
        info["Linf stats"] = np.concatenate((info["Linf stats"], batch_Linf_norms))
        info["DS stats"] = np.concatenate((info["DS stats"], batch_DS_stats))
        info["Threat stats"] = np.concatenate(
            (info["Threat stats"], batch_threat_stats)
        )
        batch_counter += 1

    return info


def print_statistics(info, previously_computed: bool = False):
    weighted = info["weighted"]
    segmented = info["segmented"]
    perturbation_style = info["perturbation_style"]

    assert not (
        weighted and segmented
    ), "Not evaluating both weighted and segmented together currently"

    avg_threat = np.mean(info["Threat stats"])
    avg_l2 = np.mean(info["L2 stats"])
    avg_linf = np.mean(info["Linf stats"])
    avg_ds = np.mean(info["DS stats"])
    severity = info["severity"] if "severity" in info else None
    starter_text = f"Perturbation style : {perturbation_style}"
    starter_text += f" with severity {str(severity)}" if severity else ""
    starter_text += " (Previously Computed) " if previously_computed else ""
    starter_text += " \n Average PD"
    if weighted:
        starter_text += " (Weighted)"
    elif segmented:
        starter_text += " (Segmented)"
    else:
        starter_text += ""
    timeprint(
        starter_text
        + f" : {avg_threat:.4f}, \t Average Linf : {avg_linf:.4f}, \t Average L2 : {avg_l2:.4f}, \t Average DS : {avg_ds:.4f}"
    )


def perturbation_threat_statistics_all(
    threat_run_path,
    dataset_hparams: hparams.DatasetHparams,
    threat_hparams: hparams.ThreatHparams,
    perturbation_hparams: hparams.PerturbationHparams,
    max_data_size: int = 128,  # 5000,
    verbose=False,
    float_16=True,
    in_memory=True,
):

    batch_size = dataset_hparams.batch_size
    num_batches = 1  # max_data_size // batch_size
    num_devices = torch.cuda.device_count()
    mini_batch_size = batch_size // 4

    prepared = False
    file_name = "threat_statistics_all.pkl"
    file_name_w = "threat_statistics_weighted_all.pkl"
    file_name_s = "threat_statistics_segmented_all.pkl"
    per_label = threat_hparams.per_label

    # Evaluate on unsafe and transformations
    perturbations_list = ["Unsafe"]
    # Load the safe perturbations
    # if perturbation_hparams.v2_transformations:
    #     perturbations_list.extend(list(transformations.keys()))

    test_loader = datareg.get(dataset_hparams, train=False)
    test_loader.shuffle(perturbation_hparams.shuffle_seed)

    experiment_info = {}
    experiment_info_w = {}
    experiment_info_s = {}
    print("-" * 80)

    timeprint(
        f"\nEvaluating threat specification for standard transformations\n",
        condition=verbose,
    )

    for perturbation_style in perturbations_list:
        save_path = paths.threat_evaluation_path(
            threat_run_path, per_label, perturbation_style
        )
        file_path = os.path.join(save_path, file_name)
        file_path_w = os.path.join(save_path, file_name_w)
        file_path_s = os.path.join(save_path, file_name_s)

        all_previously_computed = (
            get_platform().exists(file_path)
            and get_platform().exists(file_path_w)
            and get_platform().exists(file_path_s)
        )

        if (not all_previously_computed) and (not prepared):
            threat_model = ProjectedDisplacement(
                dataset_hparams,
                hparams.ThreatHparams(num_chunks=50),
                weighted=True,
                segmented=True,
            )

            threat_model.prepare(num_devices=8)

            ds_model, preprocess = dreamsim(
                pretrained=True, dreamsim_type="dino_vitb16", device="cuda:7"
            )
            prepared = True  # only prepare once

        apply_transform = (
            transformations[perturbation_style]
            if perturbation_style != "Unsafe"
            else "Unsafe"
        )

        if get_platform().exists(file_path):
            with open(file_path, "rb") as f:
                info = pickle.load(f)
            print_statistics(info, previously_computed=True)
            experiment_info[perturbation_style] = info
        else:
            info = collect_threat_statistics(
                test_loader,
                num_batches,
                ds_model=ds_model,
                threat_model=threat_model,
                device="cuda:7",
                apply_transform=apply_transform,
                perturbation_style=perturbation_style,
                float_16=float_16,
                weighted=False,
                segmented=False,
            )
            print_statistics(info, previously_computed=False)
            with open(file_path, "wb") as f:
                pickle.dump(info, f)

            experiment_info[perturbation_style] = info

        if get_platform().exists(file_path_w):
            with open(file_path_w, "rb") as f:
                info = pickle.load(f)
            print_statistics(info, previously_computed=True)
            experiment_info_w[perturbation_style] = info
        else:
            info = collect_threat_statistics(
                test_loader,
                num_batches,
                ds_model=ds_model,
                threat_model=threat_model,
                device="cuda:7",
                apply_transform=apply_transform,
                perturbation_style=perturbation_style,
                float_16=float_16,
                weighted=True,
                segmented=False,
            )
            print_statistics(info, previously_computed=False)
            with open(file_path_w, "wb") as f:
                pickle.dump(info, f)

            experiment_info_w[perturbation_style] = info

        if get_platform().exists(file_path_s):
            with open(file_path_s, "rb") as f:
                info = pickle.load(f)
            print_statistics(info, previously_computed=True)
            experiment_info_s[perturbation_style] = info
        else:
            info = collect_threat_statistics(
                test_loader,
                num_batches,
                ds_model=ds_model,
                threat_model=threat_model,
                device="cuda:7",
                apply_transform=apply_transform,
                perturbation_style=perturbation_style,
                float_16=float_16,
                weighted=False,
                segmented=True,
            )
            print_statistics(info, previously_computed=False)
            with open(file_path_s, "wb") as f:
                pickle.dump(info, f)

            experiment_info_s[perturbation_style] = info

    print("-" * 80)
    timeprint(
        f"Evaluating threat specification for common corruptions\n", condition=verbose
    )
    perturbations_list = []
    if perturbation_hparams.common_2d:
        perturbations_list.extend(list(common_corruptions_2d.keys()))

    if perturbation_hparams.common_2d_bar:
        perturbations_list.extend(list(common_corruptions_2d_bar.keys()))

    if perturbation_hparams.common_3d:
        perturbations_list.extend(list(common_corruptions_3d.keys()))

    # num_batches = num_batches // 10

    for perturbation_style in perturbations_list:
        experiment_info[perturbation_style] = {}
        experiment_info_w[perturbation_style] = {}
        experiment_info_s[perturbation_style] = {}

        for severity in range(1, 6):
            save_path = paths.threat_evaluation_path(
                threat_run_path, per_label, perturbation_style, severity
            )
            if not get_platform().exists(save_path):
                get_platform().makedirs(save_path)

            file_path = os.path.join(save_path, file_name)
            file_path_w = os.path.join(save_path, file_name_w)
            file_path_s = os.path.join(save_path, file_name_s)

            all_previously_computed = (
                get_platform().exists(file_path)
                and get_platform().exists(file_path_w)
                and get_platform().exists(file_path_s)
            )

            if (not all_previously_computed) and (not prepared):
                threat_model = ProjectedDisplacement(
                    dataset_hparams,
                    hparams.ThreatHparams(num_chunks=50),
                    weighted=True,
                    segmented=True,
                )

                threat_model.prepare(num_devices=8)

                ds_model, preprocess = dreamsim(
                    pretrained=True, dreamsim_type="dino_vitb16", device="cuda:7"
                )
                prepared = True  # only prepare once

            test_loader = datareg.get_perturbed(
                dataset_hparams,
                perturbation_style,
                severity,
                in_memory=in_memory,
            )
            test_loader.shuffle(perturbation_hparams.shuffle_seed)

            if get_platform().exists(file_path):
                with open(file_path, "rb") as f:
                    info = pickle.load(f)
                print_statistics(info, previously_computed=True)
                experiment_info[perturbation_style][severity] = info
            else:
                info = collect_threat_statistics(
                    test_loader,
                    num_batches,
                    ds_model=ds_model,
                    threat_model=threat_model,
                    device="cuda:7",
                    apply_transform=None,
                    perturbation_style=perturbation_style,
                    severity=severity,
                    float_16=float_16,
                    weighted=False,
                    segmented=False,
                )
                print_statistics(info, previously_computed=False)
                with open(file_path, "wb") as f:
                    pickle.dump(info, f)

                experiment_info[perturbation_style][severity] = info

            if get_platform().exists(file_path_w):
                with open(file_path_w, "rb") as f:
                    info = pickle.load(f)
                print_statistics(info, previously_computed=True)
                experiment_info_w[perturbation_style][severity] = info
            else:
                info = collect_threat_statistics(
                    test_loader,
                    num_batches,
                    ds_model=ds_model,
                    threat_model=threat_model,
                    device="cuda:7",
                    apply_transform=None,
                    perturbation_style=perturbation_style,
                    severity=severity,
                    float_16=float_16,
                    weighted=True,
                    segmented=False,
                )
                print_statistics(info, previously_computed=False)

                with open(file_path_w, "wb") as f:
                    pickle.dump(info, f)

                experiment_info_w[perturbation_style][severity] = info

            if get_platform().exists(file_path_s):
                with open(file_path_s, "rb") as f:
                    info = pickle.load(f)
                print_statistics(info, previously_computed=True)
                experiment_info_s[perturbation_style][severity] = info
            else:
                info = collect_threat_statistics(
                    test_loader,
                    num_batches,
                    ds_model=ds_model,
                    threat_model=threat_model,
                    device="cuda:7",
                    apply_transform=None,
                    perturbation_style=perturbation_style,
                    severity=severity,
                    float_16=float_16,
                    weighted=False,
                    segmented=True,
                )
                print_statistics(info, previously_computed=False)

                with open(file_path_s, "wb") as f:
                    pickle.dump(info, f)

                experiment_info_s[perturbation_style][severity] = info

    return experiment_info, experiment_info_w, experiment_info_s


def per_label_threat_statistics(
    threat_run_path,
    dataset_hparams,
    threat_hparams,
    perturbation_hparams,
    max_data_size,
    verbose=False,
    threat_replicate=1,
):
    timeprint(
        "Per Label threat statistics has not been implemented yet.", condition=verbose
    )
    per_label_array = [10, 20, 30, 40]
    for per_label in per_label_array:
        threat_hp = hparams.ThreatHparams(per_label=per_label)

        # check for unsafe perturbation and a candidate transformation.
        perturbations_list = ["Unsafe", "Gaussian Blur"]
        for perturbation_style in perturbations_list:
            if perturbation_style == "Unsafe":
                pass


def replicate_threat_statistics(
    threat_run_path,
    dataset_hparams,
    threat_hparams,
    perturbation_hparams,
    max_data_size,
    verbose=False,
):
    timeprint(
        "Replicate threat statistics has not been implemented yet.", condition=verbose
    )


def convert_threat_evaluation_to_df(
    threat_evaluation_info, dataset_name="imagenet", weighted=False, segmented=False
):
    """
    Threat evaluation info has the structure:
    {
        "Unsafe": {
            "Threat stats": np.array,
            "L2 stats": np.array,
            "Linf stats": np.array
            "DS stats": np.array
            "perturbation_style": "Unsafe"
            "severity": None
            "weighted": False
            "segmented": False
            }
        "Gaussian Blur": {
            "1": {
                "Threat stats": np.array,
                "L2 stats": np.array,
                "Linf stats": np.array
                "DS stats": np.array
                "perturbation_style": "Unsafe"
                "severity": None
                "weighted": False
                "segmented": False
            },
            "2": {
                "Threat stats": np.array,
                "L2 stats": np.array,
                "Linf stats": np.array
                "DS stats": np.array
                "perturbation_style": "Unsafe"
                "severity": None
                "weighted": False
                "segmented": False
            },
        ...
        "Pixelate": {
            "1": {  ... },
    }
    """

    linf_scaling = 4.6  # max linf of images in imagenet
    if dataset_name == "imagenet":
        l2_scaling = math.sqrt(3 * 224 * 224)
    elif dataset_name == "cifar10" or dataset_name == "cifar100":
        l2_scaling = math.sqrt(3 * 32 * 32)

    assert isinstance(threat_evaluation_info, dict)
    assert not (
        segmented and weighted
    ), "Currently only testing either segmented or weighted"

    # Process the threat evaluation info dictionary into a DataFrame
    records = []

    for perturbation_style, perturbation_info in threat_evaluation_info.items():
        # if perturbation info is a dictionary, then it has severity levels
        group = corruption_group(perturbation_style)
        key0 = list(perturbation_info.keys())[
            0
        ]  # Check if perturbation has severity levels
        if isinstance(perturbation_info[key0], dict):
            if perturbation_style in list(common_corruptions_2d.keys()):
                tag = "Common Corruptions 2D"
            elif perturbation_style in list(common_corruptions_2d_bar.keys()):
                tag = "Common Corruptions 2D Bar"
            elif perturbation_style in list(common_corruptions_3d.keys()):
                tag = "Common Corruptions 3D"
                raise ValueError("3D common corruptions not implemented yet")
            else:
                raise ValueError("Perturbation style not recognized")

            for severity, metrics in perturbation_info.items():
                metrics["Linf stats"] = metrics["Linf stats"] / linf_scaling
                metrics["L2 stats"] = metrics["L2 stats"] / (l2_scaling * linf_scaling)

                avg_threat = metrics["Threat stats"].mean()
                std_threat = metrics["Threat stats"].std()
                avg_l2 = metrics["L2 stats"].mean()
                std_l2 = metrics["L2 stats"].std()
                avg_linf = metrics["Linf stats"].mean()
                std_linf = metrics["Linf stats"].std()
                avg_DS = metrics["DS stats"].mean()
                std_DS = metrics["DS stats"].std()

                records.append(
                    [
                        group,
                        tag,
                        perturbation_style,
                        severity,
                        avg_threat,
                        avg_l2,
                        avg_linf,
                        avg_DS,
                        std_threat,
                        std_l2,
                        std_linf,
                        std_DS,
                    ]
                )
        else:  # Handle case for "Unsafe"
            if perturbation_style == "Unsafe":
                tag = "Unsafe"
            else:
                tag = "V2 Transformations"

            perturbation_info["Linf stats"] = (
                perturbation_info["Linf stats"] / linf_scaling
            )
            perturbation_info["L2 stats"] = perturbation_info["L2 stats"] / (
                l2_scaling * linf_scaling
            )

            avg_threat = perturbation_info["Threat stats"].mean()
            avg_l2 = perturbation_info["L2 stats"].mean()
            avg_linf = perturbation_info["Linf stats"].mean()
            avg_DS = perturbation_info["DS stats"].mean()

            std_threat = perturbation_info["Threat stats"].std()
            std_l2 = perturbation_info["L2 stats"].std()
            std_linf = perturbation_info["Linf stats"].std()
            std_DS = perturbation_info["DS stats"].std()

            records.append(
                [
                    group,
                    tag,
                    perturbation_style,
                    None,
                    avg_threat,
                    avg_l2,
                    avg_linf,
                    avg_DS,
                    std_threat,
                    std_l2,
                    std_linf,
                    std_DS,
                ]
            )

    if segmented:
        PD_str = "Average PD (segmented)"
    elif weighted:
        PD_str = "Average PD (weighted)"
    else:
        PD_str = "Average PD"

    # Create the DataFrame
    df = pd.DataFrame(
        records,
        columns=[
            "Perturbation Group",
            "Perturbation Tag",
            "Perturbation Style",
            "Severity",
            PD_str,
            r"Average $\ell_2$ (normalized)",
            r"Average $\ell_{\infty}$",
            "Average DS",
            "Std Threat",
            r"Std $\ell_2$ (normalized)",
            r"Std $\ell_{\infty}$",
            "Std DS",
        ],
    )
    # print(df)

    # print("\n\n\n")
    # # Scale by dividing by the max value of the column
    # norm_scaling = 1.0
    # dataset_scaling = 1.0
    # if dataset_name == "imagenet":
    #     dataset_scaling = math.sqrt(3 * 224 * 224)
    #     norm_scaling = 4.75790724023

    # df[r"Average $\ell_2$ (normalized)"] = df["Average L2"] / (
    #     norm_scaling * dataset_scaling
    # )
    # df[r"Average $\ell_{\infty}$"] = df["Average Linf"] / (norm_scaling)

    return df


def corruption_group(corruption_name):
    if corruption_name in [
        "Gaussian Blur",
        "Defocus Blur",
        "Glass Blur",
        "Motion Blur",
        "Zoom Blur",
        "Cocentric Sine Waves",
        "Caustic Refraction",
        "XY-Motion Blur",
        "Z-Motion Blur",
    ]:
        return "Blur"
    elif corruption_name in [
        "Shot Noise",
        "Impulse Noise",
        "Gaussian Noise",
        "Speckle Noise",
        "Blue Noise Sample",
        "Brownish Noise",
        "Perlin Noise",
        "Quantization Noise",
        "Plasma Noise",
        "Low-Light Noise",
        "ISO Noise",
    ]:
        return "Noise"
    elif corruption_name in ["Frost", "Snow", "Fog", "Fog 3D"]:
        return "Weather"
    elif corruption_name in [
        "Pixelate",
        "JPEG Compression",
        "Color Quantization",
        "Bit Error",
        "CRF Compress",
        "ABR Compress",
    ]:
        return "Compression"
    elif corruption_name in [
        "Brightness",
        "Contrast",
        "Elastic Transform",
        "Gray",
        "Jitter",
        "Invert",
        "Posterize",
        "Solarize",
        "Adjust Sharpness",
        "Auto Contrast",
        "Equalize",
        "Saturate",
        "Hue",
        "Single Frequency Greyscale",
    ]:
        return "Digital"
    elif corruption_name in [
        "Perspective",
        "Affine",
        "Rotation",
        "Elastic",
        "Rotate",
        "Translation",
        "Scale",
        "Shear",
        "Horizontal Flip",
        "Vertical Flip",
    ]:
        return "Geometric"
    elif corruption_name in [
        "Flash",
        "Spatter",
        "Checkerboard Cutout",
        "Sparkles",
        "Inverse Sparkles",
    ]:
        return "Occlusion"
    elif corruption_name in [
        "RandAugment",
        "AugMix",
        "Cutout",
        "CutMix",
        "Mixup",
        "AutoAugment",
    ]:
        return "Augmentation"
    elif corruption_name in ["Near Focus", "Far Focus"]:
        return "Depth"
    elif corruption_name == "Unsafe":
        return "Unsafe"
    else:
        raise ValueError("Corruption name not recognized")


# change this to account for corruption groupings
def plot_threat_statistics(
    df,
    per_label,
    threat_run_path,
    dataset_name="imagenet",
    default_severity=None,
    threat_type="PD",
):
    assert threat_type in ["PD", "PD-S", "PD-W"]

    dir_path = paths.threat_evaluation_path(threat_run_path, per_label)

    if threat_type == "PD-S":
        PD_str = "Average PD (segmented)"
    elif threat_type == "PD-W":
        PD_str = "Average PD (weighted)"
    else:
        PD_str = "Average PD"

    if default_severity:
        filter_values = [None, default_severity]
    else:
        filter_values = [None, 1, 2, 3, 4, 5]

    subset_df = df[df["Severity"].isin(filter_values) | df["Severity"].isna()]
    subset_df = subset_df[subset_df["Perturbation Tag"] != "V2 Transformations"]
    assert subset_df.loc[0, "Perturbation Group"] == "Unsafe"
    subset_df.loc[0, "Severity"] = 5.0
    print(f"Inside plot_threat_statistics with {threat_type} threat")
    # print(subset_df)
    print(subset_df[subset_df["Perturbation Group"] == "Unsafe"].iloc[0])

    plt.figure()
    plt.gcf().set_size_inches(5, 5)

    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.family"] = "STIXGeneral"
    plt.rcParams["axes.edgecolor"] = "lightgray"

    LARGE_SIZE = 18

    plt.rc("font", size=LARGE_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=LARGE_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=LARGE_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=LARGE_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=LARGE_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=15)  # legend fontsize
    plt.rc("figure", titlesize=LARGE_SIZE)  # fontsize of the figure title

    # plt.subplot(1, 1, 1)
    plt.gca().set_xticks([0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0])
    plt.gca().set_xticklabels([0, "", 0.25, "", 0.5, "", 0.75, "", 1.0])
    plt.gca().set_yticks([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    plt.gca().set_yticklabels(["", 1.0, "", 2.0, "", 3.0, "", ""])

    plt.gca().grid("on")

    cp = sns.color_palette()
    ax = sns.scatterplot(
        x=r"Average $\ell_{\infty}$",
        y=PD_str,
        data=subset_df,
        hue="Perturbation Group",
        style="Perturbation Group",
        size="Severity",
        sizes={1.0: 10.0, 2.0: 20.0, 3.0: 40.0, 4.0: 80.0, 5.0: 160.0},
        palette=[cp[3], cp[2], cp[1], cp[0], cp[4], cp[5], cp[6]],
    )

    # plt.legend(title="Perturbation Group")
    plt.xlabel(r"Average $\ell_\infty$ Threat")
    plt.ylabel(PD_str)
    # plt.xlim([0, 1.0])
    # plt.ylim([0.0, 3.5])
    y_max = subset_df[PD_str].max()
    x_max = subset_df[r"Average $\ell_{\infty}$"].max()
    ax.set_xlim(0, 1.1 * x_max)
    ax.set_ylim(0, 1.1 * y_max)

    ax.axvline(0.5, color="gray", lw=2)
    ax.axhline(1.0, color="gray", lw=2)
    plt.text(0.9, 0.8, "(I)")
    plt.text(0.9, 0.8 * y_max, "(II)")
    plt.text(0.01, 0.8 * y_max, "(III)")
    plt.text(0.01, 0.8, "(IV)")

    plt.legend(bbox_to_anchor=(2.01, 1.02), loc="upper right")
    plt.tight_layout()
    plt.margins(x=0.01, y=0.01)

    severity_string = str(default_severity) if default_severity else "all"
    file_name_linf = threat_type + "_vs_Linf_severity_" + severity_string + ".pdf"
    plt.savefig(
        os.path.join(dir_path, file_name_linf), format="pdf", bbox_inches="tight"
    )  # You can change format to 'jpg', 'pdf', etc.
    plt.close()

    # PD vs DS

    plt.figure()
    plt.gcf().set_size_inches(5, 5)

    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.family"] = "STIXGeneral"
    plt.rcParams["axes.edgecolor"] = "lightgray"

    LARGE_SIZE = 18

    plt.rc("font", size=LARGE_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=LARGE_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=LARGE_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=LARGE_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=LARGE_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=15)  # legend fontsize
    plt.rc("figure", titlesize=LARGE_SIZE)  # fontsize of the figure title

    plt.subplot(1, 1, 1)
    plt.gca().set_xticks([0, 0.125, 0.25, 0.375, 0.5])
    plt.gca().set_xticklabels([0, "", 0.25, "", 0.5])
    plt.gca().set_yticks([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    plt.gca().set_yticklabels(["", 1.0, "", 2.0, "", 3.0, "", ""])

    plt.gca().grid("on")

    cp = sns.color_palette()
    ax = sns.scatterplot(
        x="Average DS",
        y=PD_str,
        data=subset_df,
        hue="Perturbation Group",
        style="Perturbation Group",
        size="Severity",
        sizes={1.0: 10.0, 2.0: 20.0, 3.0: 40.0, 4.0: 80.0, 5.0: 160.0},
        palette=[cp[3], cp[2], cp[1], cp[0], cp[4], cp[5], cp[6]],
    )

    plt.xlabel("Average DS Threat")
    plt.ylabel(PD_str)
    y_max = subset_df[PD_str].max()
    x_max = subset_df["Average DS"].max()
    ax.set_xlim(0, 1.1 * x_max)
    ax.set_ylim(0, 1.1 * y_max)

    ax.axvline(0.25, color="gray", lw=2)
    ax.axhline(1.0, color="gray", lw=2)
    plt.text(0.53, 0.8, "(I)")
    plt.text(0.53, 0.8 * y_max, "(II)")
    plt.text(0.01, 0.8 * y_max, "(III)")
    plt.text(0.01, 0.8, "(IV)")
    # plt.xlim([0, 0.6])
    # plt.ylim([0.0, 2.75])

    plt.legend(bbox_to_anchor=(2.01, 1.02), loc="upper right")
    plt.tight_layout()
    plt.margins(x=0.01, y=0.01)
    # Save the plot to a file
    severity_string = str(default_severity) if default_severity else "all"
    file_name_DS = threat_type + "_vs_DS_severity_" + severity_string + ".pdf"
    plt.savefig(
        os.path.join(dir_path, file_name_DS), format="pdf", bbox_inches="tight"
    )  # You can change format to 'jpg', 'pdf', etc.
    # Close the figure
    plt.close()


def gradient_bar(ax, theta, radii, width, bottom=0.0, cmap=plt.cm.viridis, **kwargs):
    """
    Creates a radial bar plot with a gradient fill.
    """
    N = len(theta)
    for i in range(N):
        color = cmap(radii[i] / np.max(radii))  # Normalize radii for color mapping
        ax.bar(theta[i], radii[i], width=width, bottom=bottom, color=color, **kwargs)


def gradientbars(bars):
    grad = np.atleast_2d(np.linspace(0, 1, 256)).T
    ax = bars[0].axes
    lim = ax.get_xlim() + ax.get_ylim()
    for bar in bars:
        bar.set_zorder(1)
        bar.set_facecolor("none")
        x, y = bar.get_xy()
        w, h = bar.get_width(), bar.get_height()
        ax.imshow(grad, extent=[x, x + w, y, y + h], aspect="auto", zorder=0)
    ax.axis(lim)


# Gradient function
def add_gradient_bar(ax, theta, radius, width, cmap="viridis", n_grades=100, **kwargs):
    norm = mpl.colors.Normalize(vmin=0, vmax=radius)
    colors = plt.cm.get_cmap(cmap)(np.linspace(0, 1, n_grades))

    # Loop to create gradient
    for i in range(n_grades):
        ax.bar(
            theta,
            radius / n_grades,
            width=width,
            bottom=i * (radius / n_grades),
            color=colors[i],
            edgecolor="none",
            linewidth=0,
            **kwargs,
        )


def heatmap(
    data, row_labels, col_labels, ax=None, cbar_kw=None, cbarlabel="", **kwargs
):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current Axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(
        im, ax=ax, orientation="horizontal", location="bottom", shrink=0.4, **cbar_kw
    )
    # cbar.ax.set_ylabel(cbarlabel)  # rotation=-90, #va="bottom"
    cbar.set_label(
        cbarlabel,
        size=12,
        labelpad=-40,
    )
    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(
    im,
    data=None,
    valfmt="{x:.2f}",
    textcolors=("black", "white"),
    threshold=None,
    **textkw,
):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.0

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = mpl.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def severity_heat_map(
    df, per_label, threat_run_path, dataset_name="imagenet", threat_type="PD"
):
    assert threat_type in ["PD", "PD-S", "PD-W", "DS"]
    if threat_type == "PD":
        PD_str = "Average PD"
    elif threat_type == "PD-S":
        PD_str = "Average PD (segmented)"
    elif threat_type == "PD-W":
        PD_str = "Average PD (weighted)"
    else:
        PD_str = "Average DS"

    # timeprint("Plotting threat statistics has not been implemented yet.")
    dir_path = paths.threat_evaluation_path(threat_run_path, per_label)

    plt.rcParams["legend.fontsize"] = 8

    # plt.rcParams.update({'font.size': 12})

    filter_values = [1, 2, 3, 4, 5]

    subset_df = df[df["Severity"].isin(filter_values)]
    # if threat_type == "DS":
    #     subset_df = subset_df[subset_df["Scaling Factor"] == 1.0]

    # create nparray of perturbation groups across average threat for each severity level
    perturbation_groups = subset_df["Perturbation Group"].unique()
    severity_levels = subset_df["Severity"].unique()
    threat_values = np.zeros((len(perturbation_groups), len(severity_levels)))
    for i, group in enumerate(perturbation_groups):
        for j, severity in enumerate(severity_levels):
            threat_values[i, j] = subset_df[
                (subset_df["Perturbation Group"] == group)
                & (subset_df["Severity"] == severity)
            ][PD_str].values[0]

    fig, ax = plt.subplots()
    im, cbar = heatmap(
        threat_values,
        perturbation_groups,
        severity_levels,
        ax=ax,
        cmap="YlOrRd",
        cbarlabel="Average " + threat_type + " threat",
    )

    texts = annotate_heatmap(im, valfmt="{x:.2f}")
    fig.tight_layout()
    plt.savefig(
        os.path.join(dir_path, threat_type + "_severity_heatmap.pdf"), format="pdf"
    )
    plt.close()


def ridgeplot_threat_specification(
    df,
    per_label,
    threat_run_path,
    dataset_name="imagenet",
    default_severity=3,
    threat_type="PD",
):
    assert threat_type in ["PD", "DS"]
    # timeprint("Plotting threat statistics has not been implemented yet.")
    dir_path = paths.threat_evaluation_path(threat_run_path, per_label)

    plt.rcParams["legend.fontsize"] = 8

    if default_severity:
        filter_values = [default_severity, None]
    else:
        filter_values = [None, 1, 2, 3, 4, 5]

    subset_df = df[df["Severity"].isin(filter_values) | df["Severity"].isna()]
    # subset_df = df[df["Severity"].isin(filter_values)]
    subset_df = subset_df[subset_df["Perturbation Tag"] != "V2 Transformations"]
    # subset_df = subset_df[subset_df["Perturbation Tag"] != "Unsafe"]
    if threat_type == "DS":
        subset_df = subset_df[subset_df["Scaling Factor"] == 1.0]

    # group_averages = subset_df.groupby("Perturbation Group")["Threat"].mean()
    # subset_df["Group Averages"] = subset_df["Perturbation Group"].map(group_averages)

    subset_df.head()

    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    pal = sns.color_palette(palette="coolwarm", n_colors=12)

    g = sns.FacetGrid(
        subset_df,
        row="Perturbation Group",
        hue="Perturbation Group",
        # hue="Group Averages",
        aspect=5,
        height=1.0,
        palette=color_map,
        sharey=True,
    )
    print(subset_df["Threat"].groupby(subset_df["Perturbation Group"]).mean())
    # g.map(
    #     lambda x, color, **kwargs: sns.kdeplot(
    #         x.to_numpy(),
    #         bw_adjust=1,
    #         clip_on=False,
    #         fill=True,
    #         alpha=1,
    #         color=color,
    #         linewidth=1.5,
    #     ),
    #     "Threat",
    # )
    g.map(
        sns.kdeplot,
        "Threat",
        bw_adjust=1,
        clip_on=True,
        fill=True,
        alpha=0.4,
        linewidth=1.5,
        common_norm=False,
    )
    # sns.kdeplot(subset_df[subset_df["Perturbation Tag"] != "Unsafe"])
    # g.map(
    #     lambda x, **kwargs: sns.kdeplot(
    #         np.asarray(x),
    #         bw_adjust=1,
    #         clip_on=False,
    #         lw=2,
    #         color="w",
    #     ),
    #     "Threat",
    # )
    # g.map(sns.kdeplot, "Threat", bw_adjust=1, clip_on=False, color="w", lw=2)
    g.map(plt.axhline, y=0, lw=2, clip_on=False)
    plt.xlim(left=0.0, right=3.0)

    perturbation_groups = subset_df["Perturbation Group"].unique()
    print(perturbation_groups)
    for i, ax in enumerate(g.axes.flat):
        ax.set_ylabel(
            perturbation_groups[i],
            rotation=0,
            fontweight="bold",
            fontsize=8,
            labelpad=-20,
        )
        ax.yaxis.set_label_position("right")
        # ax.text(
        #     -15,
        #     0.02,
        #     perturbation_groups[i],
        #     fontweight="bold",
        #     fontsize=15,
        #     color=ax.lines[-1].get_color(),
        # )

    g.fig.subplots_adjust(hspace=-0.3)

    g.set_titles("")
    g.set(yticks=[])
    g.despine(bottom=True, left=True)
    # plt.setp(ax.get_xticklabels(), fontsize=15, fontweight="bold")
    plt.xlabel(
        str(threat_type) + " threat per corruption group",
        ha="center",
        fontsize=12,
        fontweight="bold",  # 20,
    )

    severity_string = str(default_severity) if default_severity else "all"
    file_name_linf = threat_type + "_histograms_severity_" + severity_string + ".pdf"
    plt.savefig(
        os.path.join(dir_path, file_name_linf),
        format="pdf",
    )
    plt.close()


def radar_threat_specification(
    df,
    per_label,
    threat_run_path,
    fixed_threat=0.05,
    dataset_name="imagenet",
    default_severity=1,
    norm_type="Linf",
):
    # timeprint("Plotting threat statistics has not been implemented yet.")
    dir_path = paths.threat_evaluation_path(threat_run_path, per_label)
    csv_path = os.path.join(dir_path, "info.csv")
    df.to_csv(csv_path, index=False)

    plt.rcParams["legend.fontsize"] = 8

    assert norm_type in ["Linf", "L2"]
    assert default_severity in [1, 2, 3, 4, 5]
    subset_df = df[df["Perturbation Tag"] != "V2 Transformations"]
    filter_values = [None, default_severity]
    subset_df = subset_df[
        subset_df["Severity"].isin(filter_values) | subset_df["Severity"].isna()
    ]
    # exclude v2 transformations
    # subset_df = df[df["Perturbation Tag"] == perturbation_tag]
    # subset_df = subset_df[subset_df["Severity"] == default_severity]
    # Define a color map
    # Values for the x axis

    # Values for the y axis
    perturbations_group = subset_df["Perturbation Group"].unique()

    angles = np.linspace(
        0.05, 2 * np.pi - 0.05, len(perturbations_group), endpoint=False
    )
    norms = []
    threats = []
    for perturbation_group in perturbations_group:
        sub_df = subset_df[subset_df["Perturbation Group"] == perturbation_group]
        if norm_type == "Linf":
            key = r"Average $\ell_{\infty}$"
        else:
            key = r"Average L2"

        norms.append(sub_df[key].mean())
        threats.append(sub_df["Average Threat"].mean())

    # if norm_type == "Linf":
    #     norms = (
    #         subset_df.groupby("Perturbation Group")[r"Average $\ell_{\infty}$"]
    #         .mean()
    #         .values
    #     )
    # elif norm_type == "L2":
    #     norms = subset_df.groupby("Perturbation Group")[r"Average L2"].mean().values
    # else:
    #     raise ValueError("Norm type not recognized")
    print("\n\n\n")
    print("perturbation group: ", perturbations_group)
    print("Norms: ", norms)

    unsafe_df = subset_df[subset_df["Perturbation Tag"] == "Unsafe"]
    unsafe_threat = unsafe_df["Average Threat"].mean()
    print("Unsafe threat: ", unsafe_threat)

    # threats = subset_df.groupby("Perturbation Group")["Average Threat"].mean().values
    print("Threats: ", threats)
    # subset_df["Average PD Threat"].values
    unsafe_threat = df.loc[
        df["Perturbation Tag"] == "Unsafe", r"Average Threat"
    ].values[0]

    # norm_scaling = fixed_threat / unsafe_threat
    for _index in range(len(norms)):
        threat_scaling = fixed_threat / threats[_index]
        norms[_index] = norms[_index] * threat_scaling
    print("Norms after scaling: ", norms)
    # hard_norm_cap = (
    #     min(np.max(norms), 160 / 255) if norm_type == "Linf" else None
    # )  # None for now
    hard_cap = 160  # 256  # np.max(norms)  # 256
    norms = np.clip(
        norms, 0, hard_cap
    )  # norms = np.clip(norms, 0, 160 / 255) if norm_type == "Linf" else None

    if norm_type == "Linf":
        yvals = (16 * np.arange(1, hard_cap // 16 + 1)).tolist()
        # np.linspace(
        #     0, 256, 32, dtype=int
        # ).tolist()  # (8 * np.arange(1, hard_cap // 8, 8)).tolist()  # np.arange(1,5)
        # # [8, 16, 24, 32]
        ymax = (
            float(hard_cap / 255) % 1.0
        )  # hard_cap  # float(np.max(norms))  # 32.0 / 255
        ymin = 0.0
        ylabels = [
            str(item) + "/255" for i, item in enumerate(yvals) if (i + 1) % 2 == 0
        ]
        ymarkers = [
            np.round(item / 255, 2) for i, item in enumerate(yvals) if (i + 1) % 2 == 0
        ]
    elif norm_type == "L2":
        yvals = np.round(np.linspace(0, np.max(norms), 2))
        ymax = yvals[-1]
        ymin = 0.0
        ylabels = [str(item) for item in yvals]
        ymarkers = yvals

    # norms = norms * norm_scaling

    GREY12 = "#1f1f1f"

    # Set default font color to GREY12
    plt.rcParams["text.color"] = GREY12

    # This disables it, and uses a hyphen
    plt.rc("axes", unicode_minus=False)

    # Colors
    color_list = ["#6C5B7B", "#C06C84", "#F67280", "#F8B195"]

    # Colormap
    cmap = mpl.colors.LinearSegmentedColormap.from_list("my color", color_list, N=256)

    # Normalized colors. Each number of tracks is mapped to a color in the
    # color scale 'cmap'
    color_list = cmap(mpl.colors.Normalize(vmin=norms.min(), vmax=norms.max())(norms))

    # Some layout stuff ----------------------------------------------
    # Initialize layout in polar coordinates
    fig, ax = plt.subplots(figsize=(9, 12.6), subplot_kw={"projection": "polar"})

    # Set background color to white, both axis and figure.
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.set_theta_offset(1.2 * np.pi / 2)
    ax.set_ylim(ymin, ymax)

    # Add geometries to the plot -------------------------------------
    # See the zorder to manipulate which geometries are on top

    # Add bars to represent the cumulative track lengths
    # bar = ax.bar(angles, norms, color=color_list, alpha=0.9, width=0.25, zorder=10)

    # gradient_bar(
    #     ax,
    #     angles,
    #     norms,
    #     width=0.25,
    #     bottom=0.0,
    #     cmap=plt.cm.viridis,
    #     alpha=0.9,
    #     zorder=10,
    # )
    ax.set_theta_direction(-1)
    # Plot each bar with gradient
    for t, r in zip(angles, norms):
        add_gradient_bar(ax, t, r, width=0.25, alpha=0.9, zorder=10)

    # Add dashed vertical lines. These are just references
    ax.vlines(angles, ymin, ymax, color=GREY12, ls=(0, (4, 4)), zorder=11)

    # Add dots to represent the mean gain
    # ax.scatter(angles, MEAN_GAIN, s=60, color=GREY12, zorder=11)

    # Add labels for the regions -------------------------------------
    # Note the 'wrap()' function.
    # The '5' means we want at most 5 consecutive letters in a word,
    # but the 'break_long_words' means we don't want to break words
    # longer than 5 characters.
    print(perturbations_group)
    print(angles)
    print(norms)
    perturbations_list = [
        "\n".join(wrap(r, 5, break_long_words=False)) for r in perturbations_group
    ]

    # Set the labels
    ax.set_xticks(angles)
    ax.set_xticklabels(perturbations_list, size=12)

    # Remove lines for polar axis (x)
    ax.xaxis.grid(False)

    # Remove spines
    ax.spines["start"].set_color("none")
    ax.spines["polar"].set_color("none")

    # Adjust padding of the x axis labels ----------------------------
    # This is going to add extra space around the labels for the
    # ticks of the x axis.
    PAD = 20
    XTICKS = ax.xaxis.get_major_ticks()
    for tick in XTICKS:
        tick.set_pad(PAD)

    # Put grid lines for radial axis (y) at 0, 1000, 2000, and 3000
    ax.set_yticks(ymarkers)
    ax.set_yticklabels(ylabels, size=8)

    YTICKS = ax.yaxis.get_major_ticks()
    for tick in YTICKS:
        tick.set_pad(PAD)

    ax.set_rlabel_position(195)
    # ax.set_rgrids([5,10], angle=22)
    # Add legend -----------------------------------------------------

    # First, make some room for the legend and the caption in the bottom.
    fig.subplots_adjust(bottom=0.05)

    # Create an inset axes.
    # Width and height are given by the (0.35 and 0.01) in the
    # bbox_to_anchor
    cbaxes = inset_axes(
        ax,
        width="100%",
        height="100%",
        loc="center",
        bbox_to_anchor=(0.325, 0.1, 0.35, 0.01),
        bbox_transform=fig.transFigure,  # Note it uses the figure.
    )

    v = np.linspace(0, fixed_threat, 10, endpoint=True)
    v = [np.round(item, 2) for item in v.tolist()]
    # n1 = mpl.colors.BoundaryNorm(v, cmap.N)
    n1 = mpl.colors.Normalize(vmin=0, vmax=fixed_threat)
    # print(n1)
    # Create the colorbar
    # plt.colorbar(ticks=v, orientation="horizontal", cax=cbaxes, norm=n1)
    cb = fig.colorbar(
        ScalarMappable(norm=n1, cmap="viridis"),  # cmap
        cax=cbaxes,  # Use the inset_axes created above
        orientation="horizontal",
        # ticks=ymarkers,  # ylabels,
    )
    # Remove the outline of the colorbar
    cb.outline.set_visible(False)

    # Remove tick marks
    cb.ax.xaxis.set_tick_params(size=0)

    # Set legend label and move it to the top (instead of default bottom)
    cb.set_label(
        r"PD threat",
        size=12,
        labelpad=-40,
    )

    # print(ylabels)
    cb.ax.set_yticks(ymarkers)
    cb.ax.set_yticklabels(ylabels, size=12)
    counter = 0
    yticklabels = cb.ax.get_yticklabels()
    for t in yticklabels:
        # print("\n")
        # print(t)
        t.set_text(ylabels[counter])
        # print(t)
        counter += 1
    # cb.ax.set_yticklabels(ylabels)
    cb.ax.set_yticklabels(yticklabels, size=12)
    file_name = (
        "radar_"
        # + perturbation_tag.replace(" ", "_")
        + "severity_"
        + str(default_severity)
        + "_"
        + norm_type
        + ".pdf"
    )

    plt.savefig(
        os.path.join(dir_path, file_name),
        format="pdf",
    )
    # Add annotations ------------------------------------------------

    # # Make some room for the title and subtitle above.
    # fig.subplots_adjust(top=0.8)

    # # Define title, subtitle, and caption
    # title = "\n Anisotropy of PD-Threat specification"

    # # And finally, add them to the plot.
    # fig.text(0.5, 0.83, title, fontsize=25, weight="bold", ha="center")


def score_nights_dataset(
    threat_run_path,
    dataset_hparams: hparams.DatasetHparams,
    threat_hparams: hparams.ThreatHparams,
    perturbation_hparams: hparams.PerturbationHparams,
    max_data_size: int = 5000,
    verbose=False,
    float_16=False,
):
    max_data_size = 5000  # sanity check.
    batch_size = dataset_hparams.batch_size
    num_batches = max_data_size // batch_size
    num_devices = torch.cuda.device_count()
    mini_batch_size = batch_size // 16
    per_label = threat_hparams.per_label

    save_path = paths.threat_evaluation_path(threat_run_path, per_label)
    file_name = "nights_threat_statistics.pkl"
    file_path = os.path.join(save_path, file_name)
    if get_platform().exists(file_path):
        timeprint(f"Skipping nights evaluation as it already exists.")

    DS_model, preprocess = dreamsim(
        pretrained=True, dreamsim_type="dino_vitb16", device="cuda"
    )
    nights_root = os.path.join(get_platform().dataset_root, "nights")
    nights_dataset = TwoAFCDataset(root_dir=nights_root, split="test_imagenet")
    nights_loader = DataLoader(
        nights_dataset, batch_size=batch_size, num_workers=16, shuffle=False
    )

    PD_model = ProjectedDisplacement(dataset_hparams, threat_hparams)
    PD_model.prepare(num_devices=torch.cuda.device_count())

    timeprint("Evaluating NIGHTS dataset.")
    total_threat_stats_left = np.array([], dtype=float)
    total_threat_stats_right = np.array([], dtype=float)

    batch_counter = 0
    device = "cuda"
    for (
        reference_examples,
        left_examples,
        right_examples,
        labels,
        targets,
        idx,
    ) in nights_loader:
        if batch_counter > num_batches:
            break

        labels = torch.tensor(labels)  # convert to tensor
        reference_examples, left_examples, right_examples, labels = (
            reference_examples.to(device),
            left_examples.to(device),
            right_examples.to(device),
            labels.to(device),
        )

        batch_left_threat_stats = (
            PD_model.evaluate(
                reference_examples, labels, left_examples, gray_scale=False
            )
            .cpu()
            .numpy()
        )

        batch_right_threat_stats = (
            PD_model.evaluate(
                reference_examples, labels, right_examples, gray_scale=False
            )
            .cpu()
            .numpy()
        )

        total_threat_stats_left = np.concatenate(
            (total_threat_stats_left, batch_left_threat_stats)
        )
        total_threat_stats_right = np.concatenate(
            (total_threat_stats_right, batch_right_threat_stats)
        )
        batch_counter += 1

    if not get_platform().exists(save_path):
        get_platform().makedirs(save_path)

    threat_statistics = {}
    threat_statistics["left_threat_stats"] = total_threat_stats_left
    threat_statistics["right_threat_stats"] = total_threat_stats_right

    predictions = (total_threat_stats_left <= total_threat_stats_right).astype(int)
    # predictions[index] = True if the left image had lesser threat. corresponds to target 1.
    accuracy = np.mean(predictions == targets.cpu().numpy().astype(int))

    with open(file_path, "wb") as f:
        pickle.dump(threat_statistics, f)

        timeprint(f"2AFC score on nights dataset : {accuracy:.4f}")
        timeprint(f"Average threat stats left : {total_threat_stats_left.mean()}")
        timeprint(f"Average threat stats right : {total_threat_stats_right.mean()}")
        timeprint(f"Saved threat statistics to {file_path}")
        timeprint("Done evaluating NIGHTS dataset.")
        # print("Target : ", targets)
        # print("Predictions : ", predictions)


def adversarial_threat_statistics(
    threat_run_path,
    dataset_hparams,
    threat_hparams,
    perturbation_hparams,
    max_data_size,
    verbose,
    float_16,
    in_memory=False,
    num_models=10,
    use_autoattack=False,
    threat_type="PD",
):
    batch_size = dataset_hparams.batch_size
    per_label = threat_hparams.per_label
    prepared = False

    test_loader = datareg.get(dataset_hparams, train=False)
    test_loader.shuffle(perturbation_hparams.shuffle_seed)

    experiment_info = {}
    print("-" * 80)
    timeprint(
        f"\n Evaluating Adversarial Threat Statistics for RobustBench Models\n",
        condition=verbose,
    )

    # attack_epsilons = [8 / 255, 16 / 255, 24 / 255, 32 / 255, 48 / 255]
    attack_epsilons = [16 / 255]

    model_names = imagenet_Linf_RB_model_names[:num_models]
    model_dir = os.path.join(
        get_platform().runner_root,
        "imagenet",
        "pretrained",
        "robustbenchmark",
        "Linf",
    )

    save_path = paths.threat_evaluation_path(
        threat_run_path,
        per_label,
    )
    save_path = os.path.join(save_path, "RB_evaluation")

    RB_gpu = "cuda:0"
    DS_gpu = "cuda:1"

    if not get_platform().exists(save_path):
        get_platform().makedirs(save_path)

    cast_type = torch.float16 if float_16 else torch.float32
    if threat_type == "PD":
        threat_str = ""
    elif threat_type == "PD_W" or "PD_S":
        threat_str = f"_{threat_type}"

    for model_name in model_names:
        experiment_info[model_name] = {}

        RB_model = None

        for epsilon in attack_epsilons:
            file_name = (
                f"Model_{model_name}_epsilon_{int(255*epsilon)}by255"
                + threat_str
                + ".pkl"
            )
            file_path = os.path.join(save_path, file_name)

            if get_platform().exists(file_path):
                with open(file_path, "rb") as f:
                    info = pickle.load(f)

                    # store clean accuracy, robust accuracy, PD threat of adversarial examples, DS threat of adversarial examples.
                    # PD threat of projected adversarial examples, DS threat of projected adversarial examples.
                    # PD robust accuracy after projecction,
                    clean_accuracy = info["clean_accuracy"]
                    robust_accuracy = info["robust_accuracy"]
                    proj_robust_accuracy = info["projected_robust_accuracy"]

                    avg_PD = np.mean(info["PD_total_stats"])
                    avg_l2 = np.mean(info["L2_total_stats"])
                    avg_linf = np.mean(info["Linf_total_stats"])
                    avg_DS = np.mean(info["DS_total_stats"])

                    avg_proj_PD = np.mean(info["PD_proj_total_stats"])
                    avg_proj_l2 = np.mean(info["L2_proj_total_stats"])
                    avg_proj_linf = np.mean(info["Linf_proj_total_stats"])
                    avg_proj_DS = np.mean(info["DS_proj_total_stats"])

                    timeprint(
                        f"\n\n Statistcs for Model : {model_name}, Epsilon : {epsilon}, {threat_type} (Previously Computed)"
                    )

                    timeprint(
                        f"Clean Accuracy : {clean_accuracy:.4f}, \t Robust Accuracy : {robust_accuracy:.4f} and \t Projeted Robust Accuracy : {proj_robust_accuracy:.4f}"
                    )

                    timeprint(
                        f"Averages BEFORE projection  :: PD threat : {avg_PD:.4f}, \t Linf : {avg_linf:.4f}, \t L2 : {avg_l2:.4f}, \t DS : {avg_DS:.4f}"
                    )
                    timeprint(
                        f"Averages AFTER projection  :: PD threat : {avg_proj_PD:.4f}, \t Linf : {avg_proj_linf:.4f}, \t L2 : {avg_proj_l2:.4f}, \t DS : {avg_proj_DS:.4f}"
                    )

                    experiment_info[model_name][epsilon] = info
                continue

            timeprint(
                f"\n\n Evaluating Model : {model_name}, Epsilon : {int(epsilon*255)}/255, saving stats at {file_path}"
            )

            if not prepared:
                # Load PD threat model and DreamSim threat model once.
                DS_model, preprocess = dreamsim(
                    pretrained=True, dreamsim_type="dino_vitb16", device=DS_gpu
                )
                # if float_16:
                #    DS_model = DS_model.half()  # .to(torch.device(DS_gpu))
                weighted = False
                segmented = False
                if threat_type == "PD_W":
                    weighted = True
                elif threat_type == "PD_S":
                    segmented = True
                else:
                    pass

                PD_model = ProjectedDisplacement(
                    dataset_hparams,
                    threat_hparams,
                    weighted=weighted,
                    segmented=segmented,
                )
                PD_model.prepare(num_devices=torch.cuda.device_count())

                prepared = True

            if RB_model is None:
                RB_model = load_model(
                    model_name=model_name,
                    dataset="imagenet",
                    model_dir=model_dir,
                    threat_model="Linf",
                )
                if float_16:
                    RB_model = RB_model.half()

                RB_model = RB_model.to(torch.device(RB_gpu))

            if use_autoattack:
                adversary = AutoAttack(
                    RB_model,
                    norm="Linf",
                    eps=epsilon,
                    version="custom",
                    attacks_to_run=["apgd-ce"],  # "apgd-dlr"
                    verbose=False,
                    device=torch.device(RB_gpu),
                )
                adversary.apgd.n_restarts = 4
            else:
                attack_fn, attack_power, attack_step, attack_iters = get_attack(
                    hparams.TrainingHparams(
                        adv_train_attack_norm="Linf",
                        adv_train_attack_power_Linf=epsilon,
                    )  # 1.5
                )

            PD_total_stats = np.array([], dtype=float)
            DS_total_stats = np.array([], dtype=float)
            Linf_total_stats = np.array([], dtype=float)
            L2_total_stats = np.array([], dtype=float)
            PD_proj_total_stats = np.array([], dtype=float)
            DS_proj_total_stats = np.array([], dtype=float)
            Linf_proj_total_stats = np.array([], dtype=float)
            L2_proj_total_stats = np.array([], dtype=float)
            clean_accuracy = 0.0
            robust_accuracy = 0.0
            proj_robust_accuracy = 0.0

            example_counter = 0
            for examples, labels in test_loader:
                if example_counter > max_data_size:
                    break

                # Find Linf adversarial examples.
                examples = _move(_cast(examples, cast_type), RB_gpu)
                labels = _move(labels, RB_gpu)
                # examples, labels = examples.to(RB_gpu), labels.to(RB_gpu)
                x_min = examples.min()
                x_max = examples.max()
                x_num = len(examples)
                example_counter += x_num
                # Normalizing to [0,1] for AutoAttack
                if use_autoattack:
                    scaling = x_max - x_min
                    x_min = 0.0
                    x_max = 1.0
                else:
                    scaling = 1.0

                x = (examples - x_min) / (x_max - x_min)
                x = _move(_cast(x, cast_type), RB_gpu)
                clean_acc = compute_accuracy(
                    RB_model, x, labels, device=torch.device(RB_gpu)
                )
                clean_accuracy += clean_acc * x_num

                with torch.autocast(device_type="cuda", dtype=cast_type):
                    if use_autoattack:
                        x_adv, y_adv = adversary.run_standard_evaluation(
                            x, labels, bs=64, return_labels=True
                        )
                    else:
                        timeprint("Running APGD attack")
                        x_adv = x + attack_fn(
                            RB_model,
                            x,
                            labels,
                            attack_power * scaling,
                            attack_step * scaling,
                            attack_iters,
                        )

                x_adv = _move(_cast(x_adv, cast_type), RB_gpu)
                robust_acc = compute_accuracy(
                    RB_model,
                    x_adv,
                    labels,
                    device=torch.device(RB_gpu),
                )
                robust_accuracy += robust_acc * x_num

                # Reverse [0,1] normalization for Linf, L2, DreamSim and PD
                examples_adv = x_adv * (x_max - x_min) + x_min
                diff_adv_flat = torch.flatten(
                    examples_adv - examples, start_dim=1
                ).cpu()
                l2_norms = torch.linalg.norm(diff_adv_flat, dim=1, ord=2).numpy()
                L2_total_stats = np.concatenate((L2_total_stats, l2_norms))

                linf_norms = torch.linalg.norm(
                    diff_adv_flat, dim=1, ord=float("inf")
                ).numpy()
                Linf_total_stats = np.concatenate((Linf_total_stats, linf_norms))

                DS_stats = (
                    DS_model(
                        examples.float().to(DS_gpu), examples_adv.float().to(DS_gpu)
                    )
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                DS_total_stats = np.concatenate((DS_total_stats, DS_stats))

                # Projection to PD threat model
                examples_adv_proj, PD_stats = PD_model.project(
                    examples,
                    labels,
                    examples_adv,
                    threshold=1.0 * epsilon,
                    return_threats=True,
                )

                PD_stats = PD_stats.cpu().numpy()
                PD_total_stats = np.concatenate((PD_total_stats, PD_stats))

                PD_proj_stats = np.clip(PD_stats, 0, 1.0 * epsilon)
                PD_proj_total_stats = np.concatenate(
                    (PD_proj_total_stats, PD_proj_stats)
                )

                diff_adv_proj_flat = torch.flatten(
                    examples_adv_proj - examples, start_dim=1
                ).cpu()
                l2_proj_norms = torch.linalg.norm(
                    diff_adv_proj_flat, dim=1, ord=2
                ).numpy()
                L2_proj_total_stats = np.concatenate(
                    (L2_proj_total_stats, l2_proj_norms)
                )

                linf_proj_norms = torch.linalg.norm(
                    diff_adv_proj_flat, dim=1, ord=float("inf")
                ).numpy()
                Linf_proj_total_stats = np.concatenate(
                    (Linf_proj_total_stats, linf_proj_norms)
                )

                DS_proj_stats = (
                    DS_model(
                        examples.float().to(DS_gpu),
                        examples_adv_proj.float().to(DS_gpu),
                    )
                    .cpu()
                    .numpy()
                )
                DS_proj_total_stats = np.concatenate(
                    (DS_proj_total_stats, DS_proj_stats)
                )

                # Normalize again to [0,1] for projected Robust accuracy
                x_adv_proj = (examples_adv_proj - x_min) / (x_max - x_min)
                x_adv_proj = _move(_cast(x_adv_proj, cast_type), RB_gpu)

                proj_robust_acc = compute_accuracy(
                    RB_model,
                    x_adv_proj,
                    labels,
                    device=torch.device(RB_gpu),
                )
                proj_robust_accuracy += proj_robust_acc * x_num

                avg_PD = np.mean(PD_stats)
                avg_l2 = np.mean(l2_norms)
                avg_linf = np.mean(linf_norms)
                avg_DS = np.mean(DS_stats)

                avg_proj_l2 = np.mean(l2_proj_norms)
                avg_proj_linf = np.mean(linf_proj_norms)
                avg_proj_DS = np.mean(DS_proj_stats)
                avg_proj_PD = np.mean(PD_proj_stats)

                timeprint(f"Results for threat strategy {threat_type}")

                timeprint(
                    f"\n\n Batch Statistcs : {example_counter}/{max_data_size},\t Model : {model_name}, Epsilon : {epsilon}"
                )

                timeprint(
                    f"Clean Accuracy : {clean_acc:.4f}, \t Robust Accuracy : {robust_acc:.4f} and \t Projected Robust Accuracy : {proj_robust_acc:.4f}"
                )

                timeprint(
                    f"Averages BEFORE projection  :: PD threat : {avg_PD:.4f}, \t Linf : {avg_linf:.4f}, \t L2 : {avg_l2:.4f}, \t DS : {avg_DS:.4f}"
                )
                timeprint(
                    f"Averages AFTER projection  :: PD threat : {avg_proj_PD:.4f}, \t Linf : {avg_proj_linf:.4f}, \t L2 : {avg_proj_l2:.4f}, \t DS : {avg_proj_DS:.4f}"
                )

            clean_accuracy /= example_counter
            robust_accuracy /= example_counter
            proj_robust_accuracy /= example_counter

            threat_statistics = {}
            threat_statistics["clean_accuracy"] = clean_accuracy
            threat_statistics["robust_accuracy"] = robust_accuracy
            threat_statistics["projected_robust_accuracy"] = proj_robust_accuracy

            threat_statistics["PD_total_stats"] = PD_total_stats
            threat_statistics["L2_total_stats"] = L2_total_stats
            threat_statistics["Linf_total_stats"] = Linf_total_stats
            threat_statistics["DS_total_stats"] = DS_total_stats

            threat_statistics["PD_proj_total_stats"] = PD_proj_total_stats
            threat_statistics["L2_proj_total_stats"] = L2_proj_total_stats
            threat_statistics["Linf_proj_total_stats"] = Linf_proj_total_stats
            threat_statistics["DS_proj_total_stats"] = DS_proj_total_stats

            avg_PD = np.mean(PD_total_stats)
            avg_l2 = np.mean(L2_total_stats)
            avg_linf = np.mean(Linf_total_stats)
            avg_DS = np.mean(DS_total_stats)

            avg_proj_PD = np.mean(PD_proj_total_stats)
            avg_proj_l2 = np.mean(L2_proj_total_stats)
            avg_proj_linf = np.mean(Linf_proj_total_stats)
            avg_proj_DS = np.mean(DS_proj_total_stats)

            with open(file_path, "wb") as f:
                pickle.dump(threat_statistics, f)

            timeprint(
                f"\n\n Cumulative Batch Statistcs for {max_data_size} points, \t Model : {model_name}, Epsilon : {epsilon}"
            )

            timeprint(
                f"Clean Accuracy : {clean_accuracy:.4f}, \t Robust Accuracy : {robust_accuracy:.4f} and \t Projected Robust Accuracy : {proj_robust_accuracy:.4f}"
            )

            timeprint(
                f"Averages BEFORE projection  :: PD threat : {avg_PD:.4f}, \t Linf : {avg_linf:.4f}, \t L2 : {avg_l2:.4f}, \t DS : {avg_DS:.4f}"
            )

            timeprint(
                f"Averages AFTER projection  :: PD threat : {avg_proj_PD:.4f}, \t Linf : {avg_proj_linf:.4f}, \t L2 : {avg_proj_l2:.4f}, \t DS : {avg_proj_DS:.4f}"
            )

            experiment_info[model_name][epsilon] = threat_statistics
            print("-" * 80)

        print("-" * 80)

    return experiment_info


def weighted_threat_statistics(
    threat_run_path,
    dataset_hparams,
    threat_hparams,
    perturbation_hparams,
    max_data_size=1000,
    float_16=True,
    verbose=True,
    in_memory=False,
):
    batch_size = dataset_hparams.batch_size
    per_label = threat_hparams.per_label
    prepared = False

    test_loader = datareg.get(dataset_hparams, train=False)
    test_loader.shuffle(perturbation_hparams.shuffle_seed)

    experiment_info = {}
    print("-" * 80)
    timeprint(
        f"\n Evaluating Weighted PD Threat\n",
        condition=verbose,
    )

    save_path = paths.threat_evaluation_path(
        threat_run_path,
        per_label,
    )
    save_path = os.path.join(save_path, "Unsafe-Weighted")

    DS_gpu = "cuda:0"

    if not get_platform().exists(save_path):
        # raise ValueError(f"Path {save_path} should already exist.")
        get_platform().makedirs(save_path)

    file_name = f"CD_weighted_threat_statistics.pkl"
    file_path = os.path.join(save_path, file_name)

    if get_platform().exists(file_path):
        with open(file_path, "rb") as f:
            experiment_info = pickle.load(f)

        timeprint("Found saved file.")

        CD_total_stats = experiment_info["CD_total_stats"]
        PD_total_stats = experiment_info["PD_total_stats"]
        L2_total_stats = experiment_info["L2_total_stats"]
        Linf_total_stats = experiment_info["Linf_total_stats"]
        DS_total_stats = experiment_info["DS_total_stats"]
        weighted_PD_total_stats = experiment_info["weighted_PD_total_stats"]
        avg_PD = np.mean(experiment_info["PD_total_stats"])
        avg_l2 = np.mean(experiment_info["L2_total_stats"])
        avg_linf = np.mean(experiment_info["Linf_total_stats"])
        avg_DS = np.mean(experiment_info["DS_total_stats"])
        avg_weighted_PD = np.mean(experiment_info["weighted_PD_total_stats"])
        avg_CD = np.mean(experiment_info["CD_total_stats"])

        max_cd = CD_total_stats.max()
        max_pd = PD_total_stats.max()
        max_l2 = L2_total_stats.max()
        max_linf = Linf_total_stats.max()
        max_ds = DS_total_stats.max()
        max_wpd = weighted_PD_total_stats.max()

        # Define the number of bins and generate evenly spaced bin edges
        num_bins = 10  # Example number of bins
        bin_edges = np.linspace(
            CD_total_stats.min(), CD_total_stats.max(), num_bins + 1
        )

        # Calculate bin centers for plotting
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        num_groups = 3
        bar_width = np.diff(bin_edges)[0] / (num_groups + 1)
        offsets = np.linspace(
            -bar_width * (num_groups - 1) / 2,
            bar_width * (num_groups - 1) / 2,
            num_groups,
        )

        # Digitize B into bins
        bin_indices = np.digitize(CD_total_stats, bins=bin_edges)
        # Calculate the average of A values for each bin
        bin_avg_WPD = [
            weighted_PD_total_stats[bin_indices == i].mean()
            for i in range(1, len(bin_edges))
        ]
        bin_avg_PD = [
            PD_total_stats[bin_indices == i].mean() for i in range(1, len(bin_edges))
        ]
        bin_avg_l2 = [
            L2_total_stats[bin_indices == i].mean() for i in range(1, len(bin_edges))
        ]
        bin_avg_linf = [
            Linf_total_stats[bin_indices == i].mean() for i in range(1, len(bin_edges))
        ]
        bin_avg_DS = [
            DS_total_stats[bin_indices == i].mean() for i in range(1, len(bin_edges))
        ]

        print("-" * 80)

        timeprint(
            f"\n\n Cumulative Batch Statistcs for {max_data_size} points, \n Weighted PD threat : {avg_weighted_PD:.4f},\t PD threat : {avg_PD:.4f}, \t Linf : {avg_linf:.4f}, \t L2 : {avg_l2:.4f}, \t DS : {avg_DS:.4f}, \t CD : {avg_CD:.4f}"
        )
        timeprint(f"Bin values of relative class distances : {bin_edges}")
        timeprint(f"Bin Average Weighted PD threat : {bin_avg_WPD}")
        timeprint(f"Bin Average PD threat : {bin_avg_PD}")
        timeprint(f"Bin Average L2 : {bin_avg_l2}")
        timeprint(f"Bin Average Linf : {bin_avg_linf}")
        timeprint(f"Bin Average DS : {bin_avg_DS}")
        print("-" * 80)

        # Plotting
        plt.figure()
        # Create bar plot
        plt.bar(
            bin_centers + offsets[0],
            bin_avg_WPD / max_wpd,
            width=bar_width,
            align="center",
            edgecolor="k",
            alpha=0.7,
            label="Weighted PD",
        )
        plt.bar(
            bin_centers + offsets[1],
            bin_avg_PD / max_pd,
            width=bar_width,
            align="center",
            edgecolor="k",
            alpha=0.7,
            label="PD",
        )
        plt.bar(
            bin_centers + offsets[2],
            bin_avg_DS / max_ds,
            width=bar_width,
            align="center",
            edgecolor="k",
            alpha=0.7,
            label="DreamSim",
        )
        # plt.bar(
        #     bin_centers + offsets[3],
        #     bin_avg_l2 / max_l2,
        #     width=bar_width,
        #     align="center",
        #     edgecolor="k",
        #     alpha=0.7,
        #     label=r"$\ell_2$",
        # )
        # plt.bar(
        #     bin_centers + offsets[4],
        #     bin_avg_linf / max_linf,
        #     width=bar_width,
        #     align="center",
        #     edgecolor="k",
        #     alpha=0.7,
        #     label=r"$\ell_{infty}$",
        # )

        # Format x-ticks
        plt.xticks(bin_centers, [f"{x:.2f}" for x in bin_centers], fontsize=8)

        # plt.xticks(bin_edges)  # Ensure x-ticks match the unique values in B
        # plt.gca().set_xticklabels([f"{x:.2f}" for x in bin_edges])
        plt.xlabel("Relative distance of class labels")
        plt.ylabel("Relative threat")
        plt.legend()
        plt.title("Average relative threat vs relative class distances")
        plt.savefig(os.path.join(save_path, "CD_vs_all.pdf"), format="pdf")
        plt.close()

        return experiment_info

    cast_type = torch.float16 if float_16 else torch.float32

    if not prepared:
        # Load PD threat model and DreamSim threat model once.
        DS_model, preprocess = dreamsim(
            pretrained=True, dreamsim_type="dino_vitb16", device=DS_gpu
        )
        # if float_16:
        #    DS_model = DS_model.half()  # .to(torch.device(DS_gpu))
        weighted_PD_model = ProjectedDisplacement(
            dataset_hparams, threat_hparams, weighted=True
        )
        weighted_PD_model.prepare(num_devices=torch.cuda.device_count())

        prepared = True

    np.set_printoptions(precision=2)
    PD_total_stats = np.array([], dtype=float)
    DS_total_stats = np.array([], dtype=float)
    Linf_total_stats = np.array([], dtype=float)
    L2_total_stats = np.array([], dtype=float)
    weighted_PD_total_stats = np.array([], dtype=float)
    CD_total_stats = np.array([], dtype=int)

    example_counter = 0
    for examples, labels in test_loader:
        if example_counter > max_data_size:
            break

        example_counter += len(examples)
        examples = _move(examples, DS_gpu)
        labels = _move(labels, DS_gpu)

        unsafe_indices = find_unsafe_shuffe(examples, labels)
        perturbed_labels = labels.clone()[unsafe_indices]
        perturbed_examples = _move(examples.clone()[unsafe_indices], DS_gpu)
        # print(labels)
        # print(perturbed_labels)

        c_dist = np.zeros(len(examples), dtype=float)
        for i in range(len(examples)):
            c_dist[i] = weighted_PD_model.raw_weights[
                labels[i].item(), perturbed_labels[i].item()
            ]
            # print(f"CD at {i}: ", c_dist[i])
            # assert (
            #     c_dist[i] > 0
            # ), f"c_dist : {c_dist[i]}, failed at finding an unsafe perturbation, {labels[i].item()} -> {perturbed_labels[i].item()}"

        diff_flat = torch.flatten(perturbed_examples - examples, start_dim=1)
        l2_norms = torch.linalg.norm(diff_flat, dim=1, ord=2).cpu().numpy()
        linf_norms = torch.linalg.norm(diff_flat, dim=1, ord=float("inf")).cpu().numpy()
        DS_stats = DS_model(examples, perturbed_examples).cpu().numpy()

        examples = _cast(examples, cast_type)
        perturbed_examples = _cast(perturbed_examples, cast_type)

        with torch.autocast(device_type="cuda", dtype=cast_type):
            weighted_PD_stats = (
                weighted_PD_model.evaluate(
                    examples, labels, perturbed_examples, weighted=True
                )
                .cpu()
                .numpy()
            )

            PD_stats = (
                weighted_PD_model.evaluate(examples, labels, perturbed_examples)
                .cpu()
                .numpy()
            )

        CD_total_stats = np.concatenate((CD_total_stats, c_dist))
        L2_total_stats = np.concatenate((L2_total_stats, l2_norms))
        Linf_total_stats = np.concatenate((Linf_total_stats, linf_norms))
        DS_total_stats = np.concatenate((DS_total_stats, DS_stats))
        PD_total_stats = np.concatenate((PD_total_stats, PD_stats))
        weighted_PD_total_stats = np.concatenate(
            (weighted_PD_total_stats, weighted_PD_stats)
        )
        avg_PD = np.mean(PD_total_stats)
        avg_l2 = np.mean(L2_total_stats)
        avg_linf = np.mean(Linf_total_stats)
        avg_DS = np.mean(DS_total_stats)
        avg_weighted_PD = np.mean(weighted_PD_total_stats)
        avg_CD = np.mean(CD_total_stats)

        # Define the number of bins and generate evenly spaced bin edges
        # print(CD_total_stats)
        num_bins = 8  # Example number of bins
        bin_edges = np.linspace(
            CD_total_stats.min(), CD_total_stats.max(), num_bins + 1
        )

        # Digitize B into bins
        bin_indices = np.digitize(CD_total_stats, bins=bin_edges)
        # Calculate the average of A values for each bin
        bin_avg_WPD = [
            weighted_PD_total_stats[bin_indices == i].mean()
            for i in range(1, len(bin_edges))
        ]
        bin_avg_PD = [
            PD_total_stats[bin_indices == i].mean() for i in range(1, len(bin_edges))
        ]
        bin_avg_l2 = [
            L2_total_stats[bin_indices == i].mean() for i in range(1, len(bin_edges))
        ]
        bin_avg_linf = [
            Linf_total_stats[bin_indices == i].mean() for i in range(1, len(bin_edges))
        ]
        bin_avg_DS = [
            DS_total_stats[bin_indices == i].mean() for i in range(1, len(bin_edges))
        ]

        # # Unique values in B and their indices
        # unique_CD = np.sort(np.unique(CD_total_stats))

        # # Compute the average of A for each unique value in B
        # unique_avg_weighted_PD = [
        #     weighted_PD_total_stats[CD_total_stats == x].mean() for x in unique_CD
        # ]
        # unique_avg_PD = [PD_total_stats[CD_total_stats == x].mean() for x in unique_CD]
        # unique_avg_l2 = [L2_total_stats[CD_total_stats == x].mean() for x in unique_CD]
        # unique_avg_linf = [
        #     Linf_total_stats[CD_total_stats == x].mean() for x in unique_CD
        # ]
        # unique_avg_DS = [DS_total_stats[CD_total_stats == x].mean() for x in unique_CD]

        timeprint(
            f"\n\n Batch Statistcs : {example_counter}/{max_data_size},\t Weighted PD threat : {avg_weighted_PD:.4f},\t PD threat : {avg_PD:.4f}, \t Linf : {avg_linf:.4f}, \t L2 : {avg_l2:.4f}, \t DS : {avg_DS:.4f}, \t CD : {avg_CD:.4f}"
        )
        # timeprint(f"Unique CD values : {unique_CD}")
        timeprint(f"Bin values of relative class distances : {bin_edges}")
        timeprint(f"Bin Average Weighted PD threat : {bin_avg_WPD}")
        timeprint(f"Bin Average PD threat : {bin_avg_PD}")
        timeprint(f"Bin Average L2 : {bin_avg_l2}")
        timeprint(f"Bin Average Linf : {bin_avg_linf}")
        timeprint(f"Bin Average DS : {bin_avg_DS}")
        print("-" * 80)

    experiment_info["CD_total_stats"] = CD_total_stats
    experiment_info["PD_total_stats"] = PD_total_stats
    experiment_info["L2_total_stats"] = L2_total_stats / (4.6 * math.sqrt(3) * 224)
    experiment_info["Linf_total_stats"] = Linf_total_stats / 4.6
    experiment_info["DS_total_stats"] = DS_total_stats
    experiment_info["weighted_PD_total_stats"] = weighted_PD_total_stats

    with open(file_path, "wb") as f:
        pickle.dump(experiment_info, f)

    avg_PD = np.mean(experiment_info["PD_total_stats"])
    avg_l2 = np.mean(experiment_info["L2_total_stats"])
    avg_linf = np.mean(experiment_info["Linf_total_stats"])
    avg_DS = np.mean(experiment_info["DS_total_stats"])
    avg_weighted_PD = np.mean(experiment_info["weighted_PD_total_stats"])
    avg_CD = np.mean(experiment_info["CD_total_stats"])

    # Define the number of bins and generate evenly spaced bin edges
    num_bins = 20  # Example number of bins
    bin_edges = np.linspace(CD_total_stats.min(), CD_total_stats.max(), num_bins + 1)

    # Calculate bin centers for plotting
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Digitize B into bins
    bin_indices = np.digitize(CD_total_stats, bins=bin_edges)
    # Calculate the average of A values for each bin
    bin_avg_WPD = [
        weighted_PD_total_stats[bin_indices == i].mean()
        for i in range(1, len(bin_edges))
    ]
    bin_avg_PD = [
        PD_total_stats[bin_indices == i].mean() for i in range(1, len(bin_edges))
    ]
    bin_avg_l2 = [
        L2_total_stats[bin_indices == i].mean() for i in range(1, len(bin_edges))
    ]
    bin_avg_linf = [
        Linf_total_stats[bin_indices == i].mean() for i in range(1, len(bin_edges))
    ]
    bin_avg_DS = [
        DS_total_stats[bin_indices == i].mean() for i in range(1, len(bin_edges))
    ]

    print("-" * 80)

    timeprint(
        f"\n\n Cumulative Batch Statistcs for {max_data_size} points, \n Weighted PD threat : {avg_weighted_PD:.4f},\t PD threat : {avg_PD:.4f}, \t Linf : {avg_linf:.4f}, \t L2 : {avg_l2:.4f}, \t DS : {avg_DS:.4f}, \t CD : {avg_CD:.4f}"
    )
    timeprint(f"Bin values of relative class distances : {bin_edges}")
    timeprint(f"Bin Average Weighted PD threat : {bin_avg_WPD}")
    timeprint(f"Bin Average PD threat : {bin_avg_PD}")
    timeprint(f"Bin Average L2 : {bin_avg_l2}")
    timeprint(f"Bin Average Linf : {bin_avg_linf}")
    timeprint(f"Bin Average DS : {bin_avg_DS}")
    print("-" * 80)

    # Plotting
    plt.figure()
    # Create bar plot
    plt.bar(
        bin_centers,
        bin_avg_WPD,
        width=np.diff(bin_edges),
        align="center",
        edgecolor="k",
        alpha=0.7,
    )
    plt.xticks(bin_edges)  # Ensure x-ticks match the unique values in B
    plt.gca().set_xticklabels([f"{x:.2f}" for x in bin_edges])
    plt.xlabel("Relative distance of class labels")
    plt.ylabel("Average Weighted PD threat")
    plt.savefig(os.path.join(save_path, "CD_vs_WPD.pdf"), format="pdf")
    plt.close()

    # Plotting
    plt.figure()
    # Create bar plot
    plt.bar(
        bin_centers,
        bin_avg_PD,
        width=np.diff(bin_edges),
        align="center",
        edgecolor="k",
        alpha=0.7,
    )
    plt.xticks(bin_edges)  # Ensure x-ticks match the unique values in B
    plt.gca().set_xticklabels([f"{x:.2f}" for x in bin_edges])
    plt.xlabel("Relative distance of class labels")
    plt.ylabel("Average PD threat")
    plt.savefig(os.path.join(save_path, "CD_vs_PD.pdf"), format="pdf")
    plt.close()

    # Plotting
    plt.figure()
    # Create bar plot
    plt.bar(
        bin_centers,
        bin_avg_l2,
        width=np.diff(bin_edges),
        align="center",
        edgecolor="k",
        alpha=0.7,
    )
    plt.xticks(bin_edges)  # Ensure x-ticks match the unique values in B
    plt.gca().set_xticklabels([f"{x:.2f}" for x in bin_edges])
    plt.xlabel("Relative distance of class labels")
    plt.ylabel(r"Average $\ell_2$ threat")
    plt.savefig(os.path.join(save_path, "CD_vs_L2.pdf"), format="pdf")
    plt.close()

    plt.figure()
    # Create bar plot
    plt.bar(
        bin_centers,
        bin_avg_linf,
        width=np.diff(bin_edges),
        align="center",
        edgecolor="k",
        alpha=0.7,
    )
    plt.xticks(bin_edges)  # Ensure x-ticks match the unique values in B
    plt.gca().set_xticklabels([f"{x:.2f}" for x in bin_edges])
    plt.xlabel("Relative distance of class labels")
    plt.ylabel(r"Average $\ell_{\infty}$ threat")
    plt.savefig(os.path.join(save_path, "CD_vs_Linf.pdf"), format="pdf")
    plt.close()

    plt.figure()
    # Create bar plot
    plt.bar(
        bin_centers,
        bin_avg_DS,
        width=np.diff(bin_edges),
        align="center",
        edgecolor="k",
        alpha=0.7,
    )
    plt.xticks(bin_edges)  # Ensure x-ticks match the unique values in B
    plt.gca().set_xticklabels([f"{x:.2f}" for x in bin_edges])
    plt.xlabel("Relative distance of class labels")
    plt.ylabel(r"Average DS threat")
    plt.savefig(os.path.join(save_path, "CD_vs_DS.pdf"), format="pdf")
    plt.close()

    return experiment_info


def evaluate_threat_specification(
    threat_run_path,
    dataset_hparams: hparams.DatasetHparams,
    threat_hparams: hparams.ThreatHparams,
    perturbation_hparams: hparams.PerturbationHparams,
    max_data_size: int = 1000,  # 5000
    verbose=False,
    float_16=False,
    in_memory=False,
):
    if not os.path.exists(threat_run_path):
        os.makedirs(threat_run_path)

    (
        experiment_info,
        experiment_info_w,
        experiment_info_s,
    ) = perturbation_threat_statistics_all(
        threat_run_path,
        dataset_hparams,
        threat_hparams,
        perturbation_hparams,
        verbose=verbose,
        float_16=float_16,
        in_memory=in_memory,
    )

    experiment_df = convert_threat_evaluation_to_df(
        experiment_info, dataset_name="imagenet", weighted=False, segmented=False
    )

    experiment_df_w = convert_threat_evaluation_to_df(
        experiment_info_w, dataset_name="imagenet", weighted=True, segmented=False
    )

    experiment_df_s = convert_threat_evaluation_to_df(
        experiment_info_s, dataset_name="imagenet", weighted=False, segmented=True
    )

    dir_path = paths.threat_evaluation_path(threat_run_path, threat_hparams.per_label)

    experiment_df.to_csv(os.path.join(dir_path, "PD_info.csv"), index=False)
    experiment_df_w.to_csv(os.path.join(dir_path, "Weighted_PD_info.csv"), index=False)
    experiment_df_s.to_csv(os.path.join(dir_path, "Segmented_PD_info.csv"), index=False)

    severity_heat_map(
        experiment_df,
        threat_hparams.per_label,
        threat_run_path,
        threat_type="PD",
    )

    severity_heat_map(
        experiment_df,
        threat_hparams.per_label,
        threat_run_path,
        threat_type="DS",
    )

    severity_heat_map(
        experiment_df_s,
        threat_hparams.per_label,
        threat_run_path,
        threat_type="PD-S",
    )

    severity_heat_map(
        experiment_df_w,
        threat_hparams.per_label,
        threat_run_path,
        threat_type="PD-W",
    )

    plot_threat_statistics(
        experiment_df,
        threat_hparams.per_label,
        threat_run_path,
        threat_type="PD",
    )

    plot_threat_statistics(
        experiment_df_w,
        threat_hparams.per_label,
        threat_run_path,
        threat_type="PD-W",
    )

    plot_threat_statistics(
        experiment_df_s,
        threat_hparams.per_label,
        threat_run_path,
        threat_type="PD-S",
    )

    skip = True  # True
    if skip:
        return  # skipping the rest of the code for now.

    for severity in range(1, 6):
        plot_threat_statistics(
            experiment_df,
            threat_hparams.per_label,
            threat_run_path,
            default_severity=severity,
            threat_type="PD",
        )
        plot_threat_statistics(
            experiment_df_s,
            threat_hparams.per_label,
            threat_run_path,
            default_severity=severity,
            threat_type="PD-S",
        )
        plot_threat_statistics(
            experiment_df_w,
            DS_threat_evaluation_df,
            threat_hparams.per_label,
            threat_run_path,
            default_severity=severity,
            threat_type="PD-W",
        )

    # for perturbtion_tag in [
    #     "Common Corruptions 2D",
    #     "Common Corruptions 2D Bar",
    # ]:
    #     for norm_type in ["Linf"]:
    #         for severity in range(1, 6):
    #             radar_threat_specification(
    #                 PD_threat_evaluation_df,
    #                 threat_hparams.per_label,
    #                 threat_run_path,
    #                 fixed_threat=0.5,
    #                 dataset_name="imagenet",
    #                 default_severity=severity,
    #                 perturbation_tag=perturbtion_tag,
    #                 norm_type=norm_type,
    #             )
    for norm_type in ["Linf"]:
        for severity in range(1, 6):
            radar_threat_specification(
                PD_threat_evaluation_df,
                threat_hparams.per_label,
                threat_run_path,
                fixed_threat=0.5,
                dataset_name="imagenet",
                default_severity=severity,
                norm_type=norm_type,
            )

    # ridgeplot_threat_specification(
    #     PD_threat_evaluation_df_full,
    #     threat_hparams.per_label,
    #     threat_run_path,
    #     threat_type="PD",
    # )

    # weighted_threat_evaluation_info = weighted_threat_statistics(
    #     threat_run_path,
    #     dataset_hparams,
    #     threat_hparams,
    #     perturbation_hparams,
    #     max_data_size,
    #     verbose,
    #     in_memory=in_memory,
    # )

    # Adv_threat_evaluation_info = adversarial_threat_statistics(
    #     threat_run_path,
    #     dataset_hparams,
    #     threat_hparams,
    #     perturbation_hparams,
    #     max_data_size,
    #     verbose,
    #     float_16,
    #     in_memory=in_memory,
    #     threat_type="PD",
    # )

    # Adv_threat_evaluation_info = adversarial_threat_statistics(
    #     threat_run_path,
    #     dataset_hparams,
    #     threat_hparams,
    #     perturbation_hparams,
    #     max_data_size,
    #     verbose,
    #     float_16,
    #     in_memory=in_memory,
    #     threat_type="PD_W",
    # )

    # Adv_threat_evaluation_info = adversarial_threat_statistics(
    #     threat_run_path,
    #     dataset_hparams,
    #     threat_hparams,
    #     perturbation_hparams,
    #     max_data_size,
    #     verbose,
    #     float_16,
    #     in_memory=in_memory,
    #     threat_type="PD_S",
    # )

    # score_nights_dataset(
    #     threat_run_path,
    #     dataset_hparams,
    #     threat_hparams,
    #     perturbation_hparams,
    #     float_16=float_16,
    #     max_data_size=1000,
    # )

    # per_label_threat_statistics(
    #     threat_run_path,
    #     dataset_hparams,
    #     threat_hparams,
    #     perturbation_hparams,
    #     max_data_size,
    #     verbose,
    # )

    # replicate_threat_statistics(
    #     threat_run_path,
    #     dataset_hparams,
    #     threat_hparams,
    #     perturbation_hparams,
    #     max_data_size,
    # )
