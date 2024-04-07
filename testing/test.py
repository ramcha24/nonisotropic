import torch

from platforms.platform import get_platform

from foundations import hparams
from foundations import paths

import datasets.registry

from models.base import Model

from testing import evaluation_suite


def standard_test(
        model: Model,
        train_output_location: str,
        test_output_location: str,
        dataset_hparams: hparams.DatasetHparams,
        testing_hparams: hparams.TestingHparams,
        verbose: bool = True,
        evaluate_batch_only: bool = True):
    """ Test using the standard callbacks according to the provided hparams. """
    model.to(get_platform().torch_device)

    checkpoint_location = paths.checkpoint(train_output_location)
    if not get_platform().exists(checkpoint_location):
        raise ValueError('The training location does not have a checkpoint, has the model been trained already?')

    checkpoint = get_platform().load_model(checkpoint_location, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])

    # create the test output location if it doesn't already exist.
    # if not get_platform().exists(test_output_location) and get_platform().is_primary_process:
    #    get_platform().makedirs(test_output_location)
    if not get_platform().exists(test_output_location):
        raise ValueError('Test output location does not exists, check if the hyperparameter save is working')

    train_loader = datasets.registry.get(dataset_hparams, train=True)
    test_loader = datasets.registry.get(dataset_hparams, train=False)
    name = dataset_hparams.dataset_name

    if model.model_name.startswith('mnist_conv'):
        conv_flag = True

    if name == 'mnist':
        evaluations = evaluation_suite.evaluation_suite_mnist(
            testing_hparams,
            test_output_location,
            train_loader,
            test_loader,
            verbose=verbose,
            evaluate_batch_only=evaluate_batch_only,
            conv_flag=conv_flag
        )
    else:
        raise ValueError('Invalid dataset name : {}'.format(name))

    feedback = {}
    for eval_fn in evaluations:
        eval_fn(model, feedback)
