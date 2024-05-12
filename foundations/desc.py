import abc
import argparse
from dataclasses import dataclass, fields
import hashlib

from foundations import hparams
from foundations import paths
from platforms.platform import get_platform


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
            if self.pretraining_hparams is None:
                for item in self.model_hparams.model_name.split("_")[1:]:
                    model_name += str(item)
                return model_name
            else:
                return self.model_hparams.model_name.split("_")[1:]

    def get_hparams_str(self, type_str):
        hparams_strs = None

        if type_str == "data":
            hparams_strs = self.class_select(hparams.DatasetHparams)
        elif type_str == "augment":
            hparams_strs = self.class_select(hparams.AugmentationHparams)
        elif type_str == "model":
            hparams_strs = self.class_select(hparams.ModelHparams)
        elif type_str == "train":
            hparams_strs = self.class_select(hparams.TrainingHparams)
        elif type_str == "test":
            hparams_strs = self.class_select(hparams.TestingHparams)
        else:
            raise ValueError("Invalid subclass type of Hparams : {}".format(type_str))

        assert hparams_strs is not None
        return hparams_strs

    def hashname(self, type_str) -> str:
        hparams_strs = self.get_hparams_str(type_str=type_str)

        hash_str = hashlib.md5(";".join(hparams_strs).encode("utf-8")).hexdigest()[
            :6
        ]  # shortening hash for ease, trade-off with collision factor.
        return hash_str

    def model_hparams_dir(self):
        model_prefix = None
        model_hash = None
        if self.model_hparams is not None:
            model_prefix = "model_"
            model_hash = self.hashname(type_str="model")
        return model_prefix + model_hash

    def train_hparams_dir(self):
        train_prefix = None
        train_hash = None
        if self.training_hparams is not None:
            train_prefix = "train_"
            if not (
                self.training_hparams.adv_train or self.training_hparams.N_adv_train
            ):
                train_prefix += "std_"
            else:
                if self.training_hparams.adv_train:
                    train_prefix += "adv_"
                if self.training_hparams.N_adv_train:
                    train_prefix += "Nadv_"
            train_hash = self.hashname(type_str="train")
        return train_prefix + train_hash

    def augment_hparams_dir(self):
        augment_prefix = None
        augment_hash = None
        if self.augment_hparams is not None:
            augment_prefix = "augment_"
            if not (
                self.augment_hparams.gaussian_augment
                or self.augment_hparams.N_project
                or self.augment_hparams.N_mixup
            ):
                augment_prefix += "std_"
            else:
                if self.augment_hparams.gaussian_augment:
                    augment_prefix += "gaussian_"
                if self.augment_hparams.N_project:
                    augment_prefix += "Nproject_"
                if self.augment_hparams.N_mixup:
                    augment_prefix += "Nmixup_"
            augment_hash = self.hashname(type_str="augment")
        return augment_prefix + augment_hash

    # def test_hparams_dir(self):
    #     test_prefix = "test_"
    #     if self.test_hparams is not None:
    #         test_hash = self.hashname(type_str="test")
    #         return

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
