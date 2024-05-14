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


class CIFAR10(torchvision.datasets.CIFAR10):
    """A subclass to suppress an annoying print statement in the torchvision CIFAR-10 library.

    Not strictly necessary - you can just use `torchvision.datasets.CIFAR10 if the print
    message doesn't bother you.
    """

    def download(self):
        if get_platform().is_primary_process:
            with get_platform().open(os.devnull, "w") as fp:
                sys.stdout = fp
                super(CIFAR10, self).download()
                sys.stdout = sys.__stdout__
        get_platform().barrier()


class Dataset(base.ImageDataset):
    """The CIFAR-10 dataset."""

    def __init__(self, examples, labels, image_transforms=None):
        super(Dataset, self).__init__(
            examples,
            labels,
            image_transforms=image_transforms or [],
            tensor_transforms=[
                torchvision.transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2023, 0.1994, 0.2010],
                )
            ],
        )

    @staticmethod
    def num_train_examples():
        return 50000

    @staticmethod
    def num_test_examples():
        return 10000

    @staticmethod
    def num_labels():
        return 10

    @staticmethod
    def dataset_name():
        return "cifar10"

    @staticmethod
    def get_train_set(use_augmentation=False):
        assert use_augmentation is False
        train_set = CIFAR10(
            train=True,
            root=os.path.join(get_platform().dataset_root, Dataset.dataset_name()),
            download=True,
            # transform=torchvision.transforms.Compose(    [torchvision.transforms.ToTensor()] + Dataset._tensor_transforms()),
        )
        # return Dataset(train_set.data, np.array(train_set.targets), augment if use_augmentation else [])
        return Dataset(train_set.data, np.array(train_set.targets), [])

    @staticmethod
    def get_test_set():
        test_set = CIFAR10(
            train=False,
            root=os.path.join(get_platform().dataset_root, Dataset.dataset_name()),
            download=True,
            # transform=torchvision.transforms.Compose(
            #    [torchvision.transforms.ToTensor()] + Dataset._tensor_transforms()
            # ),
        )
        return Dataset(test_set.data, np.array(test_set.targets))

    @staticmethod
    def default_dataset_hparams() -> "hparams.DatasetHparams":
        return hparams.DatasetHparams(
            dataset_name="cifar10",
            batch_size=256,
            num_labels=10,
            num_channels=3,
            num_spatial_dims=32,
            num_train=50000,
            num_test=10000,
        )

    def example_to_image(self, example):
        return Image.fromarray(example)


DataLoader = base.DataLoader
