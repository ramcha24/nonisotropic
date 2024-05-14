from skimage import color
from skimage import data
from skimage.io import imread_collection

import os
from pathlib import Path

import torch
import torchvision.transforms as T
import numpy as np

from platforms.platform import get_platform

import datasets.registry


def partition_by_loader(
    loader,
    num_labels: int,
    input_shape: list,
    dataset_loc: str = None,
    dataset_type: str = "train",
):
    label_count = torch.zeros(num_labels, dtype=int)
    for (examples, labels) in loader:
        for _label in range(num_labels):
            label_count[labels[_label]] += 1

    for _label in range(num_labels):
        image_partition_folder_path = os.path.join(
            dataset_loc, dataset_type + "_class_partition"
        )
        partition_file_path = "/" + str(_label) + ".pt"
        if os.path.exists(image_partition_folder_path + partition_file_path):
            continue

        per_label = label_count[_label]
        image_partition = torch.zeros([per_label] + input_shape)

        num = 0
        for (examples, labels) in loader:
            num_per_batch = len(examples[labels == _label])
            image_partition[num : num + num_per_batch] = examples[labels == _label]
            num += num_per_batch
        assert num == per_label

        Path(image_partition_folder_path).mkdir(parents=True, exist_ok=True)
        torch.save(
            image_partition,
            image_partition_folder_path + partition_file_path,
        )


def partition_by_folder(dataset_loc: str = None, dataset_type: str = "train"):
    # assuming dataset is imagenet, if there are additional datasets then need to change the transforms.
    transforms = T.Compose(
        [T.ToTensor()]
        + [T.Resize(256), T.CenterCrop(224)]
        + [T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )

    folder_loc = os.path.join(dataset_loc, dataset_type)
    classes = os.listdir(folder_loc)
    for label, label_name in enumerate(classes):
        image_partition_folder_path = os.path.join(
            dataset_loc, dataset_type + "_class_partition"
        )
        partition_file_path = "/" + str(label) + ".pt"
        if os.path.exists(image_partition_folder_path + partition_file_path):
            continue

        label_folder_path = os.path.join(folder_loc, label_name)
        label_selection_string = label_folder_path + "/*.JPEG"
        label_images = imread_collection(label_selection_string)
        list_image_tensors = []
        for image in label_images:
            if len(np.shape(image)) == 3:
                if np.shape(image)[2] == 3:
                    list_image_tensors.append(
                        transforms(img=np.array(image, dtype="f"))
                    )
                elif np.shape(image)[2] == 4:
                    list_image_tensors.append(
                        transforms(img=np.array(color.rgba2rgb(image), dtype="f"))
                    )
                else:
                    raise ValueError(
                        "Image shape is not 3 or 4 but {}".format(np.shape(image)[2])
                    )

        image_partition = torch.stack(list_image_tensors, dim=0)

        image_partition_folder_path = os.path.join(
            dataset_loc, dataset_type + "_class_partition"
        )
        partition_file_path = "/" + str(label) + ".pt"
        Path(image_partition_folder_path).mkdir(parents=True, exist_ok=True)
        torch.save(
            image_partition,
            image_partition_folder_path + partition_file_path,
        )


def save_class_partition(dataset_hparams, dataset_type: str = "train"):
    assert dataset_type in ["train", "val"], f"Invalid dataset type {dataset_type}"
    dataset_loc = os.path.join(get_platform().dataset_root, dataset_name)
    dataset_name = dataset_hparams.dataset_name

    if dataset_name in ["mnist", "cifar10", "cifar100"]:
        train_flag = dataset_type == "train"
        loader = datasets.registry.get(dataset_hparams, train=train_flag)

        num_labels = dataset_hparams.num_labels

        input_shape = [
            dataset_hparams.num_channels,
            dataset_hparams.num_spatial_dims,
            dataset_hparams.num_spatial_dims,
        ]

        partition_by_loader(loader, num_labels, input_shape, dataset_loc, dataset_type)
    elif dataset_name == "imagenet":
        partition_by_folder(dataset_loc, dataset_type)
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")
