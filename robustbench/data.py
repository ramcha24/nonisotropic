import math
import os
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Set, Tuple, Union

import numpy as np
import timm
import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch import nn

from models.model_zoo import model_dicts as all_models
from models.model_zoo.enums import BenchmarkDataset, ThreatModel
from robustbench.zenodo_download import DownloadError, zenodo_download
from robustbench.loaders import CustomImageFolder

PREPROCESSINGS = {
    "Res256Crop224": transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]
    ),
    "Crop288": transforms.Compose([transforms.CenterCrop(288), transforms.ToTensor()]),
    None: transforms.Compose([transforms.ToTensor()]),
    "Res224": transforms.Compose([transforms.Resize(224), transforms.ToTensor()]),
    "BicubicRes256Crop224": transforms.Compose(
        [
            transforms.Resize(
                256, interpolation=transforms.InterpolationMode("bicubic")
            ),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    ),
}


def get_timm_model_preprocessing(model_name: str) -> Callable:
    model = timm.create_model(model_name)
    if isinstance(model, nn.Sequential):
        # Normalization has been applied, take the inner model to get the other info
        model = model.model
    interpolation = model.default_cfg["interpolation"]
    crop_pct = model.default_cfg["crop_pct"]
    img_size = model.default_cfg["input_size"][1]
    scale_size = int(math.floor(img_size / crop_pct))
    return transforms.Compose(
        [
            transforms.Resize(
                scale_size, interpolation=transforms.InterpolationMode(interpolation)
            ),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
        ]
    )


def get_preprocessing(
    dataset: BenchmarkDataset,
    threat_model: ThreatModel,
    model_name: Optional[str],
    preprocessing: Optional[Union[str, Callable]],
) -> Callable:
    # If preprocessing is already as a function, then return it
    if isinstance(preprocessing, Callable):
        return preprocessing
    # If preprocessing is already specified as a string, then fetch it and return it
    if preprocessing is not None:
        return PREPROCESSINGS[preprocessing]
    # If the dataset is not imagenet, then the only needed preprocessing is ToTensor
    if dataset != BenchmarkDataset.imagenet:
        return PREPROCESSINGS[None]
    # At this point the model name should be specified
    if model_name is None:
        raise Exception(
            "Preprocessing should be specified if the model is not already in the model zoo"
        )
    # See if the model is a timm model, if this is so, then use the custom function
    lower_model_name = model_name.lower().replace("-", "_")
    timm_model_name = (
        f"{lower_model_name}_{dataset.value.lower()}_{threat_model.value.lower()}"
    )
    if timm.is_model(timm_model_name):
        return get_timm_model_preprocessing(timm_model_name)
    # Or directly fetch the preprocessing for the model specified in the dictionary

    # since there is only `corruptions` folder for models in the Model Zoo
    threat_model = ThreatModel(threat_model.value.replace("_3d", ""))
    prepr = all_models[dataset][threat_model][model_name]["preprocessing"]
    return PREPROCESSINGS[prepr]


def _load_dataset(
    dataset: Dataset, n_examples: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = 100
    test_loader = data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    x_test, y_test = [], []
    for i, (x, y) in enumerate(test_loader):
        x_test.append(x)
        y_test.append(y)
        if n_examples is not None and batch_size * i >= n_examples:
            break
    x_test_tensor = torch.cat(x_test)
    y_test_tensor = torch.cat(y_test)

    if n_examples is not None:
        x_test_tensor = x_test_tensor[:n_examples]
        y_test_tensor = y_test_tensor[:n_examples]

    return x_test_tensor, y_test_tensor


def load_cifar10(
    n_examples: Optional[int] = None,
    data_dir: str = "./datasets/cifar10",
    transforms_test: Callable = PREPROCESSINGS[None],
) -> Tuple[torch.Tensor, torch.Tensor]:
    dataset = datasets.CIFAR10(
        root=data_dir, train=False, transform=transforms_test, download=True
    )
    return _load_dataset(dataset, n_examples)


def load_cifar100(
    n_examples: Optional[int] = None,
    data_dir: str = "./datasets/cifar100",
    transforms_test: Callable = PREPROCESSINGS[None],
) -> Tuple[torch.Tensor, torch.Tensor]:
    dataset = datasets.CIFAR100(
        root=data_dir, train=False, transform=transforms_test, download=True
    )
    return _load_dataset(dataset, n_examples)


def load_imagenet(
    n_examples: Optional[int] = 5000,
    data_dir: str = "./datasets/imagenet",
    transforms_test: Callable = PREPROCESSINGS["Res256Crop224"],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if n_examples > 5000:
        raise ValueError("The evaluation is currently possible on at most 5000 points-")

    imagenet = CustomImageFolder(data_dir + "/val", transforms_test)

    test_loader = data.DataLoader(
        imagenet, batch_size=n_examples, shuffle=False, num_workers=4
    )

    x_test, y_test, paths = next(iter(test_loader))

    return x_test, y_test


CleanDatasetLoader = Callable[
    [Optional[int], str, Callable], Tuple[torch.Tensor, torch.Tensor]
]
_clean_dataset_loaders: Dict[BenchmarkDataset, CleanDatasetLoader] = {
    BenchmarkDataset.cifar_10: load_cifar10,
    BenchmarkDataset.cifar_100: load_cifar100,
    BenchmarkDataset.imagenet: load_imagenet,
}


def load_clean_dataset(
    dataset: BenchmarkDataset, n_examples: Optional[int], data_dir: str, prepr: Callable
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _clean_dataset_loaders[dataset](n_examples, data_dir, prepr)


CORRUPTIONS = (
    "shot_noise",
    "motion_blur",
    "snow",
    "pixelate",
    "gaussian_noise",
    "defocus_blur",
    "brightness",
    "fog",
    "zoom_blur",
    "frost",
    "glass_blur",
    "impulse_noise",
    "contrast",
    "jpeg_compression",
    "elastic_transform",
)

CORRUPTIONS_3DCC = (
    "near_focus",
    "far_focus",
    "bit_error",
    "color_quant",
    "flash",
    "fog_3d",
    "h265_abr",
    "h265_crf",
    "iso_noise",
    "low_light",
    "xy_motion_blur",
    "z_motion_blur",
)

CORRUPTIONS_DICT: Dict[BenchmarkDataset, Tuple[str, ...]] = {
    BenchmarkDataset.cifar_10: {ThreatModel.corruptions: CORRUPTIONS},
    BenchmarkDataset.cifar_100: {ThreatModel.corruptions: CORRUPTIONS},
    BenchmarkDataset.imagenet: {
        ThreatModel.corruptions: CORRUPTIONS,
        ThreatModel.corruptions_3d: CORRUPTIONS_3DCC,
    },
}

ZENODO_CORRUPTIONS_LINKS: Dict[BenchmarkDataset, Tuple[str, Set[str]]] = {
    BenchmarkDataset.cifar_10: ("2535967", {"CIFAR-10-C.tar"}),
    BenchmarkDataset.cifar_100: ("3555552", {"CIFAR-100-C.tar"}),
}

CORRUPTIONS_DIR_NAMES: Dict[BenchmarkDataset, str] = {
    BenchmarkDataset.cifar_10: {ThreatModel.corruptions: "CIFAR-10-C"},
    BenchmarkDataset.cifar_100: {ThreatModel.corruptions: "CIFAR-100-C"},
    BenchmarkDataset.imagenet: {
        ThreatModel.corruptions: "ImageNet-C",
        ThreatModel.corruptions_3d: "ImageNet-3DCC",
    },
}


def load_cifar10c(
    n_examples: int,
    severity: int = 5,
    data_dir: str = "./datasets/cifar10c",
    shuffle: bool = False,
    corruptions: Sequence[str] = CORRUPTIONS,
    _: Callable = PREPROCESSINGS[None],
) -> Tuple[torch.Tensor, torch.Tensor]:
    return load_corruptions_cifar(
        BenchmarkDataset.cifar_10, n_examples, severity, data_dir, corruptions, shuffle
    )


def load_cifar100c(
    n_examples: int,
    severity: int = 5,
    data_dir: str = "./datasets/cifar100c",
    shuffle: bool = False,
    corruptions: Sequence[str] = CORRUPTIONS,
    _: Callable = PREPROCESSINGS[None],
) -> Tuple[torch.Tensor, torch.Tensor]:
    return load_corruptions_cifar(
        BenchmarkDataset.cifar_100, n_examples, severity, data_dir, corruptions, shuffle
    )


def load_imagenetc(
    n_examples: Optional[int] = 5000,
    severity: int = 5,
    data_dir: str = "./datasets/imagenetc",
    shuffle: bool = False,
    corruptions: Sequence[str] = CORRUPTIONS,
    prepr: Callable = PREPROCESSINGS[None],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if n_examples > 5000:
        raise ValueError("The evaluation is currently possible on at most 5000 points.")

    assert (
        len(corruptions) == 1
    ), "so far only one corruption is supported (that's how this function is called in eval.py"
    # TODO: generalize this (although this would probably require writing a function similar to `load_corruptions_cifar`
    #  or alternatively creating yet another CustomImageFolder class that fetches images from multiple corruption types
    #  at once -- perhaps this is a cleaner solution)

    data_folder_path = (
        Path(data_dir)
        / CORRUPTIONS_DIR_NAMES[BenchmarkDataset.imagenet][ThreatModel.corruptions]
        / corruptions[0]
        / str(severity)
    )
    imagenet = CustomImageFolder(data_folder_path, prepr)
    test_loader = data.DataLoader(
        imagenet, batch_size=n_examples, shuffle=shuffle, num_workers=2
    )

    x_test, y_test, paths = next(iter(test_loader))

    return x_test, y_test


def load_imagenet3dcc(
    n_examples: Optional[int] = 5000,
    severity: int = 5,
    data_dir: str = "./datasets/imagenet3dcc",
    shuffle_seed: int = 0,
    shuffle: bool = False,
    corruptions: Sequence[str] = CORRUPTIONS_3DCC,
    prepr: Callable = PREPROCESSINGS[None],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if n_examples > 5000:
        raise ValueError("The evaluation is currently possible on at most 5000 points.")

    assert (
        len(corruptions) == 1
    ), "so far only one corruption is supported (that's how this function is called in eval.py"
    # TODO: generalize this (although this would probably require writing a function similar to `load_corruptions_cifar`
    #  or alternatively creating yet another CustomImageFolder class that fetches images from multiple corruption types
    #  at once -- perhaps this is a cleaner solution)

    data_folder_path = (
        Path(data_dir)
        / CORRUPTIONS_DIR_NAMES[BenchmarkDataset.imagenet][ThreatModel.corruptions_3d]
        / corruptions[0]
        / str(severity)
    )
    imagenet = CustomImageFolder(data_folder_path, prepr)
    test_loader = data.DataLoader(
        imagenet, batch_size=n_examples, shuffle=shuffle, num_workers=2
    )
    test_loader.shuffle(shuffle_seed)
    x_test, y_test, paths = next(iter(test_loader))

    return x_test, y_test


CorruptDatasetLoader = Callable[
    [int, int, str, bool, Sequence[str], Callable], Tuple[torch.Tensor, torch.Tensor]
]
CORRUPTION_DATASET_LOADERS: Dict[BenchmarkDataset, CorruptDatasetLoader] = {
    BenchmarkDataset.cifar_10: {ThreatModel.corruptions: load_cifar10c},
    BenchmarkDataset.cifar_100: {ThreatModel.corruptions: load_cifar100c},
    BenchmarkDataset.imagenet: {
        ThreatModel.corruptions: load_imagenetc,
        ThreatModel.corruptions_3d: load_imagenet3dcc,
    },
}


def load_corruptions_cifar(
    dataset: BenchmarkDataset,
    n_examples: int,
    severity: int,
    data_dir: str,
    corruptions: Sequence[str] = CORRUPTIONS,
    shuffle: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert 1 <= severity <= 5
    n_total_cifar = 10000

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    data_dir = Path(data_dir)
    data_root_dir = data_dir / CORRUPTIONS_DIR_NAMES[dataset][ThreatModel.corruptions]

    if not data_root_dir.exists():
        zenodo_download(*ZENODO_CORRUPTIONS_LINKS[dataset], save_dir=data_dir)

    # Download labels
    labels_path = data_root_dir / "labels.npy"
    if not os.path.isfile(labels_path):
        raise DownloadError("Labels are missing, try to re-download them.")
    labels = np.load(labels_path)

    x_test_list, y_test_list = [], []
    n_pert = len(corruptions)
    for corruption in corruptions:
        corruption_file_path = data_root_dir / (corruption + ".npy")
        if not corruption_file_path.is_file():
            raise DownloadError(f"{corruption} file is missing, try to re-download it.")

        images_all = np.load(corruption_file_path)
        images = images_all[(severity - 1) * n_total_cifar : severity * n_total_cifar]
        n_img = int(np.ceil(n_examples / n_pert))
        x_test_list.append(images[:n_img])
        # Duplicate the same labels potentially multiple times
        y_test_list.append(labels[:n_img])

    x_test, y_test = np.concatenate(x_test_list), np.concatenate(y_test_list)
    if shuffle:
        rand_idx = np.random.permutation(np.arange(len(x_test)))
        x_test, y_test = x_test[rand_idx], y_test[rand_idx]

    # Make it in the PyTorch format
    x_test = np.transpose(x_test, (0, 3, 1, 2))
    # Make it compatible with our models
    x_test = x_test.astype(np.float32) / 255
    # Make sure that we get exactly n_examples but not a few samples more
    x_test = torch.tensor(x_test)[:n_examples]
    y_test = torch.tensor(y_test)[:n_examples]

    return x_test, y_test
