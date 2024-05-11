# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import os
from PIL import Image
import sys
import torchvision
import math
from datasets import base
from platforms.platform import get_platform
from foundations import hparams


class CIFAR100(torchvision.datasets.CIFAR100):
    """A subclass to suppress an annoying print statement in the torchvision CIFAR-10 library.

    Not strictly necessary - you can just use `torchvision.datasets.CIFAR10 if the print
    message doesn't bother you.
    """

    def download(self):
        if get_platform().is_primary_process:
            with get_platform().open(os.devnull, "w") as fp:
                sys.stdout = fp
                super(CIFAR100, self).download()
                sys.stdout = sys.__stdout__
        get_platform().barrier()


class Dataset(base.ImageDataset):
    """The CIFAR-100 dataset."""

    def __init__(self, examples, labels, image_transforms=None):
        super(Dataset, self).__init__(
            examples,
            labels,
            image_transforms or [],
            Dataset._tensor_transforms,
        )

    @staticmethod
    def num_train_examples():
        return 50000

    @staticmethod
    def num_test_examples():
        return 10000

    @staticmethod
    def num_labels():
        return 100

    @staticmethod
    def dataset_name():
        return "cifar100"

    @staticmethod
    def _tensor_transforms():
        return [
            torchvision.transforms.Normalize(
                mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404],
            )
        ]

    @staticmethod
    def get_train_set(use_augmentation=False):
        # augment = [torchvision.transforms.RandomHorizontalFlip(), torchvision.transforms.RandomCrop(32, 4)]
        train_set = CIFAR100(
            train=True,
            root=os.path.join(get_platform().dataset_root, Dataset.dataset_name()),
            download=True,
            transform=torchvision.transforms.Compose(
                [torchvision.transforms.ToTensor()] + Dataset._tensor_transforms()
            ),
        )
        # return Dataset(train_set.data, np.array(train_set.targets), augment if use_augmentation else [])
        return Dataset(train_set.data, np.array(train_set.targets))

    @staticmethod
    def get_test_set():
        test_set = CIFAR100(
            train=False,
            root=os.path.join(get_platform().dataset_root, Dataset.dataset_name()),
            download=True,
            transform=torchvision.transforms.Compose(
                [torchvision.transforms.ToTensor()] + Dataset._tensor_transforms()
            ),
        )
        return Dataset(test_set.data, np.array(test_set.targets))

    @staticmethod
    def default_dataset_hparams() -> "hparams.DatasetHparams":
        return hparams.DatasetHparams(
            dataset_name="cifar100",
            batch_size=128,
            num_labels=100,
            num_channels=3,
            num_spatial_dims=32,
            num_train=50000,
            num_test=10000,
        )

    def example_to_image(self, example):
        return Image.fromarray(example)


DataLoader = base.DataLoader
