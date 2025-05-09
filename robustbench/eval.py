import warnings
from argparse import Namespace
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import random
from autoattack import AutoAttack
from autoattack.state import EvaluationState
from torch import nn
from tqdm import tqdm

from robustbench.data import (
    CORRUPTIONS_DICT,
    get_preprocessing,
    load_clean_dataset,
    CORRUPTION_DATASET_LOADERS,
)
from models.model_zoo.enums import BenchmarkDataset, ThreatModel
from robustbench.utils import clean_accuracy, load_model, parse_args, update_json
from robustbench.model_zoo import model_dicts as all_models


def benchmark(
    model: Union[nn.Module, Sequence[nn.Module]],
    n_examples: int = 10000,
    dataset: Union[str, BenchmarkDataset] = BenchmarkDataset.cifar_10,
    threat_model: Union[str, ThreatModel] = ThreatModel.Linf,
    to_disk: bool = False,
    model_name: Optional[str] = None,
    data_dir: str = "./data",
    corruptions_data_dir: Optional[str] = None,
    device: Optional[Union[torch.device, Sequence[torch.device]]] = None,
    batch_size: int = 32,
    eps: Optional[float] = None,
    log_path: Optional[str] = None,
    preprocessing: Optional[Union[str, Callable]] = None,
    aa_state_path: Optional[Path] = None,
) -> Tuple[float, float]:
    """Benchmarks the given model(s).

    It is possible to benchmark on 3 different threat models, and to save the results on disk. In
    the future benchmarking multiple models in parallel is going to be possible.

    :param model: The model to benchmark.
    :param n_examples: The number of examples to use to benchmark the model.
    :param dataset: The dataset to use to benchmark. Must be one of {cifar10, cifar100}
    :param threat_model: The threat model to use to benchmark, must be one of {L2, Linf
    corruptions}
    :param to_disk: Whether the results must be saved on disk as .json.
    :param model_name: The name of the model to use to save the results. Must be specified if
    to_json is True.
    :param data_dir: The directory where the dataset is or where the dataset must be downloaded.
    :param device: The device to run the computations.
    :param batch_size: The batch size to run the computations. The larger, the faster the
    evaluation.
    :param eps: The epsilon to use for L2 and Linf threat models. Must not be specified for
    corruptions threat model.
    :param preprocessing: The preprocessing that should be used for ImageNet benchmarking. Should be
    specified if `dataset` is `imageget`.
    :param aa_state_path: The path where the AA state will be saved and from where should be
    loaded if it already exists. If `None` no state will be used.

    :return: A Tuple with the clean accuracy and the accuracy in the given threat model.
    """
    if isinstance(model, Sequence) or isinstance(device, Sequence):
        # Multiple models evaluation in parallel not yet implemented
        raise NotImplementedError

    try:
        if model.training:
            warnings.warn(Warning("The given model is *not* in eval mode."))
    except AttributeError:
        warnings.warn(
            Warning("It is not possible to asses if the model is in eval mode")
        )

    dataset_: BenchmarkDataset = BenchmarkDataset(dataset)
    threat_model_: ThreatModel = ThreatModel(threat_model)

    device = device or torch.device("cpu")
    model = model.to(device)

    prepr = get_preprocessing(dataset_, threat_model_, model_name, preprocessing)

    clean_x_test, clean_y_test = load_clean_dataset(
        dataset_, n_examples, data_dir, prepr
    )

    accuracy = clean_accuracy(
        model, clean_x_test, clean_y_test, batch_size=batch_size, device=device
    )
    print(f"Clean accuracy: {accuracy:.2%}")

    extra_metrics = {}  # dict to store corruptions_mce for corruptions threat models
    if threat_model_ in {ThreatModel.Linf, ThreatModel.L2}:
        if eps is None:
            raise ValueError(
                "If the threat model is L2 or Linf, `eps` must be specified."
            )

        adversary = AutoAttack(
            model,
            norm=threat_model_.value,
            eps=eps,
            version="standard",
            device=device,
            log_path=log_path,
        )
        x_adv = adversary.run_standard_evaluation(
            clean_x_test, clean_y_test, bs=batch_size, state_path=aa_state_path
        )
        if aa_state_path is None:
            adv_accuracy = clean_accuracy(
                model, x_adv, clean_y_test, batch_size=batch_size, device=device
            )
        else:
            aa_state = EvaluationState.from_disk(aa_state_path)
            assert aa_state.robust_flags is not None
            adv_accuracy = aa_state.robust_flags.mean().item()

    elif threat_model_ in [ThreatModel.corruptions, ThreatModel.corruptions_3d]:
        corruptions = CORRUPTIONS_DICT[dataset_][threat_model_]
        print(f"Evaluating over {len(corruptions)} corruptions")
        # Exceptionally, for corruptions (2d and 3d) we use only resizing to 224x224
        prepr = get_preprocessing(dataset_, threat_model_, model_name, "Res224")
        # Save into a dict to make a Pandas DF with nested index
        corruptions_data_dir = corruptions_data_dir or data_dir
        adv_accuracy, adv_mce = corruptions_evaluation(
            batch_size,
            corruptions_data_dir,
            dataset_,
            threat_model_,
            device,
            model,
            n_examples,
            to_disk,
            prepr,
            model_name,
        )

        extra_metrics["corruptions_mce"] = adv_mce
    else:
        raise NotImplementedError
    print(f"Adversarial accuracy: {adv_accuracy:.2%}")

    if to_disk:
        if model_name is None:
            raise ValueError("If `to_disk` is True, `model_name` should be specified.")

        update_json(
            dataset_,
            threat_model_,
            model_name,
            accuracy,
            adv_accuracy,
            eps,
            extra_metrics,
        )

    return accuracy, adv_accuracy


def corruptions_evaluation(
    batch_size: int,
    data_dir: str,
    dataset: BenchmarkDataset,
    threat_model: ThreatModel,
    device: torch.device,
    model: nn.Module,
    n_examples: int,
    to_disk: bool,
    prepr: str,
    model_name: Optional[str],
) -> float:
    if to_disk and model_name is None:
        raise ValueError("If `to_disk` is True, `model_name` should be specified.")

    corruptions = CORRUPTIONS_DICT[dataset][threat_model]
    model_results_dict: Dict[Tuple[str, int], float] = {}
    for corruption in tqdm(corruptions):
        for severity in range(1, 6):
            x_corrupt, y_corrupt = CORRUPTION_DATASET_LOADERS[dataset][threat_model](
                n_examples,
                severity,
                data_dir,
                shuffle=False,
                corruptions=[corruption],
                prepr=prepr,
            )

            corruption_severity_accuracy = clean_accuracy(
                model, x_corrupt, y_corrupt, batch_size=batch_size, device=device
            )
            print(
                "corruption={}, severity={}: {:.2%} accuracy".format(
                    corruption, severity, corruption_severity_accuracy
                )
            )

            model_results_dict[(corruption, severity)] = corruption_severity_accuracy

    model_results = pd.DataFrame(model_results_dict, index=[model_name])
    adv_accuracy = model_results.values.mean()

    # Save disaggregated results on disk
    if threat_model == ThreatModel.corruptions_3d:
        unagg_res_file = "unaggregated_results_3d.csv"
    else:
        unagg_res_file = "unaggregated_results.csv"

    existing_results_path = (
        Path("model_info") / dataset.value / "corruptions" / unagg_res_file
    )

    if not existing_results_path.parent.exists():
        existing_results_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        existing_results = pd.read_csv(
            existing_results_path, header=[0, 1], index_col=0
        )
        existing_results.columns = existing_results.columns.set_levels(
            [
                existing_results.columns.levels[0],
                existing_results.columns.levels[1].astype(int),
            ]
        )
        full_results = pd.concat([existing_results, model_results])
    except FileNotFoundError:
        full_results = model_results

    if to_disk:
        full_results.to_csv(existing_results_path)

    adv_mce = 0
    alexnet_accs_corruption = full_results.loc["AlexNet"]
    for corruption in corruptions:
        mce_corruption = (1 - model_results[corruption]) / (
            1 - alexnet_accs_corruption[corruption]
        ).mean()
        adv_mce += mce_corruption.values.mean() / len(corruptions)

    return adv_accuracy, adv_mce


def main(args: Namespace) -> None:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    model = load_model(
        args.model_name,
        model_dir=args.model_dir,
        dataset=args.dataset,
        threat_model=args.threat_model,
    )

    model.eval()

    device = torch.device(args.device)
    benchmark(
        model,
        n_examples=args.n_ex,
        dataset=args.dataset,
        threat_model=args.threat_model,
        to_disk=args.to_disk,
        model_name=args.model_name,
        data_dir=args.data_dir,
        corruptions_data_dir=args.corruptions_data_dir,
        device=device,
        batch_size=args.batch_size,
        eps=args.eps,
    )


if __name__ == "__main__":
    # Example:
    # python -m robustbench.eval --n_ex=5000 --dataset=imagenet --threat_model=Linf \
    #                            --model_name=Salman2020Do_R18 --data_dir=/tmldata1/andriush/imagenet/val \
    #                            --batch_size=128 --eps=0.0156862745
    args_ = parse_args()
    main(args_)
