from dataclasses import dataclass

from cli import arg_utils
from foundations.hparams import Hparams
import models.registry


@dataclass
class JobArgs(Hparams):
    """Arguments shared across jobs"""
    
    replicate: int = 1
    default_hparams: str = None
    quiet: bool = False
    evaluate_only_at_end: bool = False
    evaluate_only_batch_test: bool = False

    _name: str = 'High-level arguments'
    _description: str = 'Arguments that determine how the job is run and where it is stored.'
    _replicate: str = 'The index of this particular replicate. ' \
        'Use a different replicate number to run another copy of the same experiment'
    _default_hparams: str = 'Populate all arguments with the default hyperparameters for this model.'
    _quiet: str = 'Suppress output logging about the training status.'
    _evaluate_only_at_end: str = 'Run the test set only before and after training, otherwise run every epoch'
    _evaluate_only_batch_test: str = 'Run the test runner only on a random batch of the test set'


def maybe_get_default_hparams(runner_name: str = None):
    default_hparams = arg_utils.maybe_get_arg('default_hparams')
    return models.registry.get_default_hparams(default_hparams, runner_name) if default_hparams else None
