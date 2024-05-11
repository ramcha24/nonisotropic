import torch

from platforms.platform import get_platform

from foundations import hparams
from foundations import paths

import datasets.registry

from models.base import Model, DistributedDataParallel

from testing.evaluation_suite import evaluation_suite


def standard_test(
    model: Model,
    train_output_location: str,
    test_output_location: str,
    dataset_hparams: hparams.DatasetHparams,
    testing_hparams: hparams.TestingHparams,
    verbose: bool = True,
    evaluate_batch_only: bool = True,
):
    """Test using the standard callbacks according to the provided hparams."""
    if not get_platform().exists(test_output_location):
        raise ValueError(
            "Test output location does not exists, check if the hyperparameter save is working"
        )

    checkpoint_location = paths.checkpoint(train_output_location)
    if not get_platform().exists(checkpoint_location):
        raise ValueError(
            "The training location does not have a checkpoint, has the model been trained already?"
        )

    train_loader = datasets.registry.get(dataset_hparams, train=True)
    test_loader = datasets.registry.get(dataset_hparams, train=False)

    model.to(get_platform().torch_device)

    checkpoint = get_platform().load_model(
        checkpoint_location, map_location=torch.device("cpu")
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    # Handle parallelism if applicable.
    if get_platform().is_distributed:
        model = DistributedDataParallel(model, device_ids=[get_platform().local_rank])

    # here if the runner is asking for comparing threats, then it would be good to return list(class_wise_loaders)

    evaluations = evaluation_suite(
        dataset_hparams,
        testing_hparams,
        test_output_location,
        train_loader,
        test_loader,
        verbose=verbose,
        evaluate_batch_only=evaluate_batch_only,
    )

    feedback = {}
    for eval_fn in evaluations:
        eval_fn(model, feedback)
