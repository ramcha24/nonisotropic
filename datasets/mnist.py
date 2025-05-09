import os
from PIL import Image
import torchvision

from datasets import base
from platforms.platform import get_platform
from foundations import hparams

from six.moves import urllib

opener = urllib.request.build_opener()
opener.addheaders = [("User-agent", "Mozilla/5.0")]
urllib.request.install_opener(opener)


class Dataset(base.ImageDataset):
    """The MNIST dataset."""

    @staticmethod
    def num_train_examples():
        return 60000

    @staticmethod
    def num_test_examples():
        return 10000

    @staticmethod
    def num_labels():
        return 10

    @staticmethod
    def get_train_set(use_augmentation):
        """No augmentation for MNIST"""
        train_set = torchvision.datasets.MNIST(
            train=True,
            root=os.path.join(get_platform().dataset_root, "mnist"),
            download=True,
        )
        return Dataset(train_set.data, train_set.targets)

    @staticmethod
    def get_test_set():
        test_set = torchvision.datasets.MNIST(
            train=False,
            root=os.path.join(get_platform().dataset_root, "mnist"),
            download=True,
        )
        return Dataset(test_set.data, test_set.targets)

    @staticmethod
    def default_dataset_hparams() -> "hparams.DatasetHparams":
        return hparams.DatasetHparams(
            dataset_name="mnist",
            batch_size=128,
            num_labels=10,
            num_channels=1,
            num_spatial_dims=28,
            num_train=60000,
            num_test=10000,
        )

    def __init__(self, examples, labels):
        tensor_transforms = [torchvision.transforms.Normalize(mean=[0.0], std=[1.0])]
        # tensor_transforms = [torchvision.transforms.Normalize(mean=[0.1307], std=[0.3081])]
        # tensor_transforms = [torchvision.transforms.Normalize(mean=[0.5], std=[28*0.5])]
        super(Dataset, self).__init__(examples, labels, [], tensor_transforms)

    def example_to_image(self, example):
        return Image.fromarray(example.numpy(), mode="L")


Dataloader = base.DataLoader
