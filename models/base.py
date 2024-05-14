import abc
import torch
import typing

from foundations import paths
from foundations.step import Step
from foundations import hparams
import training.desc
from platforms.platform import get_platform


class Model(torch.nn.Module, abc.ABC):
    """The base class used by all models in this codebase."""

    @staticmethod
    @abc.abstractmethod
    def is_valid_model_name(model_name: str, dataset_name: str) -> bool:
        """Is the model name string a valid name for models in this class?"""

        pass

    @staticmethod
    @abc.abstractmethod
    def get_model_from_name(
        model_name: str,
        outputs: int,
        initializer: typing.Callable[[torch.nn.Module], None],
    ) -> "Model":
        """Returns an instance of this class as described by the model_name string."""

        pass

    @staticmethod
    @abc.abstractmethod
    def default_model_hparams() -> "hparams.ModelHparams":
        """The default model hyperparameters for this model."""

        pass

    @staticmethod
    @abc.abstractmethod
    def default_training_hparams() -> "hparams.TrainingHparams":
        """The default training hyperparameters for training this model."""

        pass

    @property
    @abc.abstractmethod
    def loss_criterion(self) -> torch.nn.Module:
        """The loss criterion to use for this model."""

        pass

    def save(self, save_location: str, save_step: Step):
        if not get_platform().is_primary_process:
            return
        if not get_platform().exists(save_location):
            get_platform().makedirs(save_location)
        get_platform().save_model(
            self.state_dict(), paths.model(save_location, save_step)
        )


class DistributedDataParallel(Model, torch.nn.parallel.DistributedDataParallel):
    def __init__(self, module: Model, device_ids):
        super(DistributedDataParallel, self).__init__(
            module=module, device_ids=device_ids
        )

    @property
    def loss_criterion(self):
        return self.module.loss_criterion

    @staticmethod
    def get_model_from_name(model_name, outputs, initializer):
        raise NotImplementedError

    @staticmethod
    def is_valid_model_name(model_name):
        raise NotImplementedError

    @staticmethod
    def default_hparams():
        raise NotImplementedError

    def save(self, save_location: str, save_step: Step):
        self.module.save(save_location, save_step)

    @staticmethod
    def default_model_hparams() -> "hparams.ModelHparams":
        raise NotImplementedError

    @staticmethod
    def default_training_hparams() -> "hparams.TrainingHparams":
        raise NotImplementedError
