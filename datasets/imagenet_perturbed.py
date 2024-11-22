import concurrent
import numpy as np
import os
import json
from PIL import Image
import torchvision

from foundations import hparams
from platforms.platform import get_platform
from datasets import base
from datasets.perturbations import (
    transformations,
    common_corruptions_2d,
    common_corruptions_2d_bar,
    common_corruptions_3d,
    backgrounds,
    common_corruptions_2d_bar_severities,
    CC2DTransform,
)

from corruptions.imagenet_c_bar.transform_finder import build_transform
from corruptions.imagenet_c_bar.utils.converters import PilToNumpy, NumpyToTensor
from corruptions.ImageNetC.create_c import make_imagenet_c as make_imagenet_c
from corruptions.ImageNetC.imagenetc.imagenet_c import corrupt
from corruptions.ImageNetC.imagenetc.imagenet_c.corruptions import *


def _get_samples(root, y_name, y_num):
    y_dir = os.path.join(root, y_name)
    if not get_platform().isdir(y_dir):
        return []
    output = [
        (os.path.join(y_dir, f), y_num)
        for f in get_platform().listdir(y_dir)
        if f.lower().endswith("jpeg")
    ]
    return output


class Dataset(base.ImageDataset):
    """Perturbed ImageNet"""

    def __init__(self, loc: str, image_transforms):
        # Load the data.
        classes = sorted(get_platform().listdir(loc))
        samples = []

        if get_platform().num_workers > 0:
            executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=get_platform().num_workers
            )
            futures = [
                executor.submit(_get_samples, loc, y_name, y_num)
                for y_num, y_name in enumerate(classes)
            ]
            for d in concurrent.futures.wait(futures)[0]:
                samples += d.result()
        else:
            for y_num, y_name in enumerate(classes):
                samples += _get_samples(loc, y_name, y_num)

        examples, labels = zip(*samples)
        super(Dataset, self).__init__(
            np.array(examples),
            np.array(labels),
            image_transforms,
            [
                torchvision.transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                )
            ],
        )

    def get_class_labels():
        json_file = os.path.join(
            get_platform().dataset_root, "imagenet", "imagenet_class_index.json"
        )

        # Load the JSON data
        with open(json_file) as f:
            class_info = json.load(f)
        return class_info

    @staticmethod
    def num_train_examples():
        return -1  # 1281167

    @staticmethod
    def num_test_examples():
        return 50000

    @staticmethod
    def num_labels():
        return 1000

    @staticmethod
    def _augment_transforms():
        return [
            torchvision.transforms.RandomResizedCrop(
                224, scale=(0.1, 1.0), ratio=(0.8, 1.25)
            ),
            torchvision.transforms.RandomHorizontalFlip(),
        ]

    @staticmethod
    def _image_transforms(
        perturbation_style: str = None, severity: int = None, in_memory: bool = False
    ):
        if in_memory:
            if perturbation_style in list(common_corruptions_2d.keys()):
                return [
                    torchvision.transforms.Resize(256),
                    torchvision.transforms.CenterCrop(224),
                    CC2DTransform(
                        perturbation_style=perturbation_style,
                        severity=severity,
                        dataset_name="imagenet",
                    ),
                ]

            elif perturbation_style in list(common_corruptions_2d_bar.keys()):
                severity = common_corruptions_2d_bar_severities[perturbation_style][
                    severity - 1
                ]

                return [
                    torchvision.transforms.Resize(256),
                    torchvision.transforms.CenterCrop(224),
                    PilToNumpy(),
                    build_transform(
                        name=common_corruptions_2d_bar[perturbation_style],
                        severity=severity,
                        dataset_type="imagenet",
                    ),
                    # NumpyToTensor(),
                ]
            elif perturbation_style in list(common_corruptions_3d.keys()):
                pass  # 3D corruptions not implemented yet
        else:
            return [
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
            ]

    @staticmethod
    def get_train_set(use_augmentation):
        # transforms = (
        #     Dataset._augment_transforms() if use_augmentation else Dataset._transforms()
        # )
        # return Dataset(
        #     os.path.join(get_platform().dataset_root, "imagenet", "train"), transforms
        # )
        raise NotImplementedError(
            "Perturbed Imagenet has no stored training data, in-memory perturbation has not been implemented yet"
        )

    @staticmethod
    def _dataset_loc(
        perturbation_style: str = None, severity: int = None, in_memory: bool = False
    ):
        if severity not in range(1, 6):
            raise ValueError(f"Invalid severity {severity}")

        dataset_loc = get_platform().dataset_root

        if perturbation_style in common_corruptions_2d:
            dataset_loc = os.path.join(
                dataset_loc,
                "imagenet_c",
                common_corruptions_2d[perturbation_style],
                str(severity),
            )
        elif perturbation_style in common_corruptions_2d_bar:
            dataset_loc = os.path.join(
                dataset_loc,
                "imagenet_c_bar",
                common_corruptions_2d_bar[perturbation_style],
                str(severity),
            )
        elif perturbation_style in common_corruptions_3d:
            dataset_loc = os.path.join(
                dataset_loc,
                "imagenet_3dcc",
                common_corruptions_3d[perturbation_style],
                str(severity),
            )
        else:
            raise ValueError(f"Invalid perturbation style {perturbation_style}")

        if in_memory:
            dataset_loc = os.path.join(get_platform().dataset_root, "imagenet", "val")

        # print("Loading perturbed dataset from ", dataset_loc)

        return dataset_loc

    @staticmethod
    def get_test_set(
        perturbation_style: str = None, severity: int = None, in_memory: bool = False
    ):
        return Dataset(
            Dataset._dataset_loc(perturbation_style, severity, in_memory),
            Dataset._image_transforms(perturbation_style, severity, in_memory),
        )

    @staticmethod
    def example_to_image(example):
        with get_platform().open(example, "rb") as fp:
            return Image.open(fp).convert("RGB")

    @staticmethod
    def default_dataset_hparams() -> "hparams.DatasetHparams":
        return hparams.DatasetHparams(
            dataset_name="imagenet_perturbed",
            batch_size=128,
            num_labels=1000,
            num_channels=3,
            num_spatial_dims=224,
            num_train=-1,
            num_test=50000,
            perturbation_style="Gaussian Noise",
            severity=3,
        )


DataLoader = base.DataLoader
