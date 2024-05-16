import torch
import os

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
    model_type: str = None,
    verbose: bool = True,
    evaluate_batch_only: bool = True,
):
    """Test using the standard callbacks according to the provided hparams."""
    if not get_platform().exists(test_output_location):
        raise ValueError(
            "Test output location does not exists, check if the hyperparameter save is working"
        )

    train_loader = datasets.registry.get(dataset_hparams, train=True)
    test_loader = datasets.registry.get(dataset_hparams, train=False)
    model.to(get_platform().torch_device)

    if model_type in [None, "finetuned"]:
        checkpoint_location = paths.checkpoint(train_output_location)

        if not get_platform().exists(checkpoint_location):
            raise ValueError(
                f"The training location does not have a checkpoint at {checkpoint_location},\n has the model been trained already?"
            )

        checkpoint = get_platform().load_model(
            checkpoint_location, map_location=torch.device("cpu")
        )
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        assert model_type == "pretrained", f"Invalid model_type {model_type}"

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

    feedback = {}  # feedback is a dictionary of all the evaluation output.
    for eval_fn in evaluations:
        eval_fn(model, feedback)

    torch.save(feedback, os.path.join(test_output_location, "feedback.pt"))

    return feedback

    # need to store feedback.
    # collect information and do plotting.
    # need a multi-runner to assess all the feedback and generate plots.
    # or perhaps return the feedback to the multi-runner and do the post-processing there.
