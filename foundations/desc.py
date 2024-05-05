import abc
import argparse
from dataclasses import dataclass, fields
import pprint
import hashlib

from foundations import hparams
from foundations import paths
from platforms.platform import get_platform


def printdict(item):
    pp = pprint.PrettyPrinter(indent=4, width=10)
    pp.pprint(item)
    return


@dataclass
class Desc(abc.ABC):
    """The bundle of hyperparameters necessary for a particular kind of job. Contains many hparams objects.
    Each hparams object should be a field of this dataclass.
    """

    @staticmethod
    @abc.abstractmethod
    def name_prefix() -> str:
        """The name to prefix saved runs with."""

        pass

    # @property
    def class_select(self, cls):
        fields_dict = {f.name: getattr(self, f.name) for f in fields(self)}

        field_str = [
            str(fields_dict[k])
            for k in sorted(fields_dict)
            if isinstance(fields_dict[k], cls)
        ]
        return field_str

    def get_dataset_name(self):
        if self.dataset_hparams.dataset_name is None:
            raise ValueError("Dataset name is not set")
        else:
            return self.dataset_hparams.dataset_name

    def get_model_name(self):
        if self.model_hparams.model_name is None:
            raise ValueError("Model name is not set")
        else:
            model_name = ""
            for item in self.model_hparams.model_name.split("_")[1:]:
                model_name += str(item)
            return model_name

    def hashname(self, type_str) -> str:
        hparams_strs = None

        if type_str == "dataset":
            hparams_strs = self.class_select(hparams.DatasetHparams)
        elif type_str == "model":
            hparams_strs = self.class_select(hparams.ModelHparams)
        elif type_str == "train":
            hparams_strs = self.class_select(hparams.TrainingHparams)
        elif type_str == "test":
            hparams_strs = self.class_select(hparams.TestingHparams)
        else:
            raise ValueError("Invalid subclass type of Hparams : {}".format(type_str))

        assert hparams_strs is not None
        hash_str = hashlib.md5(";".join(hparams_strs).encode("utf-8")).hexdigest()[
            :6
        ]  # shortening hash for ease, trade-off with collision factor.
        return hash_str

    @staticmethod
    @abc.abstractmethod
    def add_args(parser: argparse.ArgumentParser, defaults: "Desc" = None) -> None:
        """Add the necessary command-line arguments."""

        pass

    @staticmethod
    @abc.abstractmethod
    def create_from_args(args: argparse.Namespace) -> "Desc":
        """Create from command line arguments."""

        pass

    def save(self, output_location):
        if not get_platform().is_primary_process:
            return
        if not get_platform().exists(output_location):
            get_platform().makedirs(output_location)

        fields_dict = {f.name: getattr(self, f.name) for f in fields(self)}
        hparams_strs = [
            fields_dict[k].display
            for k in sorted(fields_dict)
            if isinstance(fields_dict[k], hparams.Hparams)
        ]
        with get_platform().open(paths.hparams(output_location), "w") as fp:
            fp.write("\n".join(hparams_strs))
