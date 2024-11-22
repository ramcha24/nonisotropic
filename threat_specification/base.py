import abc
import torch
import typing

from foundations import paths
from foundations import hparams

from platforms.platform import get_platform


class ThreatModel(abc.ABC):
    """The abstract base class for threat specifications in this codebase."""

    def __init__(
        self,
        dataset_hparams: hparams.DatasetHparams,
        threat_hparams: hparams.ThreatHparams,
    ):
        self.dataset_hparams = dataset_hparams
        self.threat_hparams = threat_hparams

    @abc.abstractmethod
    def prepare(
        self,
        num_devices: int,
    ):
        """Prepare the threat for use on a model and dataset."""
        pass

    @abc.abstractmethod
    def evaluate(
        self,
        examples: torch.Tensor,
        labels: torch.Tensor,
        perturbed_examples: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the threat on a model and dataset."""

        pass

    @abc.abstractmethod
    def project(
        self,
        examples: torch.Tensor,
        labels: torch.Tensor,
        perturbed_examples: torch.Tensor,
        threshold: float,
    ) -> torch.Tensor:
        """Project the perturbed examples onto the permissible set of the threat specification."""

        pass

    def __call__(
        self,
        examples: torch.Tensor,
        labels: torch.Tensor,
        perturbed_examples: torch.Tensor,
    ) -> torch.Tensor:
        return self.evaluate(examples, labels, perturbed_examples)

    @property
    def display(self):
        return "\n".join(
            [
                self.dataset_hparams.display,
                self.threat_hparams.display,
            ]
        )
