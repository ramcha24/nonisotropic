import numpy as np
import torchvision

from datasets import (
    base,
    mnist,
    cifar10,
    cifar100,
    imagenet,
    tinyimagenet,
    imagenet_perturbed,
)
from foundations.hparams import DatasetHparams
from platforms.platform import get_platform

registered_datasets = {
    "mnist": mnist,
    "cifar10": cifar10,
    "cifar100": cifar100,
    "imagenet": imagenet,
    "tinyimagenet": tinyimagenet,
    "imagenet_perturbed": imagenet_perturbed,
}


def get(dataset_hparams: DatasetHparams, train: bool = True):
    """Get the train or test set corresponding to the hyperparameters"""
    seed = dataset_hparams.transformation_seed or 0

    # Get the dataset itself.
    if dataset_hparams.dataset_name in registered_datasets:
        use_augmentation = False  # train and not dataset_hparams.do_not_augment
        if train:
            dataset = registered_datasets[
                dataset_hparams.dataset_name
            ].Dataset.get_train_set(use_augmentation)
        else:
            dataset = registered_datasets[
                dataset_hparams.dataset_name
            ].Dataset.get_test_set()
    else:
        raise ValueError("No such dataset: {}".format(dataset_hparams.dataset_name))

    # Transform the dataset
    if train and dataset_hparams.random_labels_fraction is not None:
        dataset.randomize_labels(
            seed=seed, fraction=dataset_hparams.random_labels_fraction
        )

    if train and dataset_hparams.subsample_fraction is not None:
        dataset.subsample(seed=seed, fraction=dataset_hparams.subsample_fraction)

    if train and dataset_hparams.blur_factor is not None:
        if not isinstance(dataset, base.ImageDataset):
            raise ValueError("Can blur images.")
        else:
            dataset.blur(seed=seed, blur_factor=dataset_hparams.blur_factor)

    if dataset_hparams.unsupervised_labels is not None:
        if dataset_hparams.unsupervised_labels != "rotation":
            raise ValueError(
                "Unknown unsupervised labels: {}".format(
                    dataset_hparams.unsupervised_labels
                )
            )
        elif not isinstance(dataset, base.ImageDataset):
            raise ValueError("Can only do unsupervised rotation to images.")
        else:
            dataset.unsupervised_rotation(seed=seed)

    # create the loader
    # return registered_datasets[dataset_hparams.dataset_name].Dataloader(dataset, batch_size=dataset_hparams.batch_size, num_workers=get_platform().num_workers)
    return base.DataLoader(
        dataset,
        batch_size=dataset_hparams.batch_size,
        num_workers=get_platform().num_workers,
    )


def get_perturbed(
    dataset_hparams: DatasetHparams,
    perturbation_style: str = None,
    severity: int = None,
    in_memory: bool = False,
):
    """Get the perturbed train or test set corresponding to the hyperparameters"""
    seed = dataset_hparams.transformation_seed or 0
    # check that the orignal dataset is clean.
    assert dataset_hparams.perturbation_style is None
    assert dataset_hparams.severity is None
    assert not dataset_hparams.dataset_name.endswith("_perturbed")

    clean_dataset_name = dataset_hparams.dataset_name
    perturbed_dataset_name = clean_dataset_name + "_perturbed"

    if clean_dataset_name in registered_datasets:
        clean_dataset = registered_datasets[clean_dataset_name].Dataset.get_test_set()
    else:
        raise ValueError("No such clean dataset: {}".format(clean_dataset_name))

    if perturbed_dataset_name in registered_datasets:
        perturbed_dataset = registered_datasets[
            perturbed_dataset_name
        ].Dataset.get_test_set(
            perturbation_style=perturbation_style,
            severity=severity,
            in_memory=in_memory,
        )
    else:
        raise ValueError("No such perturbed dataset: {}".format(perturbed_dataset_name))

    twin_dataset = base.TwinDataset(clean_dataset, perturbed_dataset)

    return base.DataLoader(
        twin_dataset,
        batch_size=dataset_hparams.batch_size,
        num_workers=get_platform().num_workers,
    )


def iterations_per_epoch(dataset_hparams: DatasetHparams):
    """Get the number of iterations per training epoch."""

    if dataset_hparams.dataset_name in registered_datasets:
        num_train_examples = registered_datasets[
            dataset_hparams.dataset_name
        ].Dataset.num_train_examples()
    else:
        raise ValueError("No such dataset: {}".format(dataset_hparams.dataset_name))

    if dataset_hparams.subsample_fraction is not None:
        num_train_examples *= dataset_hparams.subsample_fraction

    return np.ceil(num_train_examples / dataset_hparams.batch_size).astype(int)


def num_labels(dataset_hparams: DatasetHparams):
    """Get the number of classes."""

    if dataset_hparams.dataset_name in registered_datasets:
        num_labels = registered_datasets[
            dataset_hparams.dataset_name
        ].Dataset.num_labels()
    else:
        raise ValueError("No such dataset: {}".format(dataset_hparams.dataset_name))

    if dataset_hparams.unsupervised_labels is not None:
        if dataset_hparams.unsupervised_labels != "rotation":
            raise ValueError(
                "Unknown unsupervised labels: {}".format(
                    dataset_hparams.unsupervised_labels
                )
            )
        else:
            return 4

    return num_labels
