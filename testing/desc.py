from dataclasses import dataclass, fields
import os
import argparse

from datasets import registry as datasets_registry
from foundations import desc
from foundations import hparams
from foundations import paths
from platforms.platform import get_platform


@dataclass
class TestingDesc(desc.Desc):
    """the hyperparameters necessary to describe a testing run"""

    model_hparams: hparams.ModelHparams
    dataset_hparams: hparams.DatasetHparams
    training_hparams: hparams.TrainingHparams
    testing_hparams: hparams.TestingHparams

    @staticmethod
    def name_prefix():
        return "test"

    @staticmethod
    def add_args(parser: argparse.ArgumentParser, defaults: "TestingDesc" = None):
        hparams.ModelHparams.add_args(
            parser, defaults=defaults.model_hparams if defaults else None
        )
        hparams.DatasetHparams.add_args(
            parser, defaults=defaults.dataset_hparams if defaults else None
        )
        hparams.TrainingHparams.add_args(
            parser, defaults=defaults.training_hparams if defaults else None
        )
        hparams.TestingHparams.add_args(
            parser, defaults=defaults.testing_hparams if defaults else None
        )

    @staticmethod
    def create_from_args(args: argparse.Namespace) -> "TestingDesc":
        model_hparams = hparams.ModelHparams.create_from_args(args)
        dataset_hparams = hparams.DatasetHparams.create_from_args(args)
        training_hparams = hparams.TrainingHparams.create_from_args(args)
        testing_hparams = hparams.TestingHparams.create_from_args(args)

        return TestingDesc(
            model_hparams, dataset_hparams, training_hparams, testing_hparams
        )

    @property
    def test_outputs(self):
        return datasets_registry.num_labels(self.dataset_hparams)

    def run_path(self, replicate, verbose=False):
        root_location = get_platform().runner_root

        dataset_str = self.get_dataset_name()
        model_str = self.get_model_name()
        if model_str.startswith("RB"):
            train_runner_str = "finetuning"
        else:
            train_runner_str = "training"
        # self.name_prefix()

        dataset_prefix = "data_"
        if not (
            self.dataset_hparams.gaussian_augment
            or self.dataset_hparams.N_project
            or self.dataset_hparams.N_mixup
        ):
            dataset_prefix += "std_"
        else:
            if self.dataset_hparams.gaussian_augment:
                dataset_prefix += "gaussian_"
            if self.dataset_hparams.N_project:
                dataset_prefix += "Nproject_"
            if self.dataset_hparams.N_mixup:
                dataset_prefix += "Nmixup_"
        dataset_hash = self.hashname(type_str="data")

        model_prefix = "model_"
        model_hash = self.hashname(type_str="model")

        train_prefix = "train_"
        if not (self.training_hparams.adv_train or self.training_hparams.N_adv_train):
            train_prefix += "std_"
        else:
            if self.training_hparams.adv_train:
                train_prefix += "adv_"
            if self.training_hparams.N_adv_train:
                train_prefix += "Nadv_"
        train_hash = self.hashname(type_str="train")

        replicate_str = f"replicate_{replicate}"

        test_runner_str = self.name_prefix()
        test_prefix = "test_"
        test_hash = self.hashname(type_str="test")

        logger_paths = dict()
        logger_paths["data"] = os.path.join(
            root_location,
            dataset_str,
            model_str,
            train_runner_str,
            dataset_prefix + dataset_hash,
        )

        logger_paths["model"] = os.path.join(
            logger_paths["data"], model_prefix + model_hash
        )

        logger_paths["train"] = os.path.join(
            logger_paths["model"], train_prefix + train_hash
        )

        logger_paths["train_run_path"] = os.path.join(
            logger_paths["train"],
            replicate_str,
        )

        full_run_path = os.path.join(
            logger_paths["train_run_path"], test_runner_str, test_prefix + test_hash
        )

        if get_platform().is_primary_process and verbose:
            print(
                "\nDataset hparams should already be logged at {}".format(
                    logger_paths["data"]
                )
            )
            print(
                "\nModel hparams should already be logged at {}".format(
                    logger_paths["model"]
                )
            )
            print(
                "\nTraining hparams should be logged at {}".format(
                    logger_paths["train"]
                )
            )
            print("\nTesting hparams will be logged at {}".format(full_run_path))

        if not get_platform().exists(logger_paths["train_run_path"]) and verbose:
            raise ValueError("\n No trained models found, Can't run test job~")

        if not get_platform().exists(full_run_path) and verbose:
            print("A job with this configuration has not been run yet.")

        return full_run_path, logger_paths

    @property
    def display(self):
        return "\n".join(
            [
                self.dataset_hparams.display,
                self.model_hparams.display,
                self.training_hparams.display,
                self.testing_hparams.display,
            ]
        )

    def save_param(self, replicate):

        full_run_path, _ = self.run_path(replicate)

        if not get_platform().is_primary_process:
            return
        if not get_platform().exists(full_run_path):
            get_platform().makedirs(full_run_path)
            hparams_strs = self.get_hparams_str(type_str="test")
            with get_platform().open(
                paths.params_loc(full_run_path, "test"), "w"
            ) as fp:
                fp.write("\n".join(hparams_strs))
        else:
            print(
                "A job with this configuration may have already been initiated."  # Stale. Delete the existing job results to run again.
            )
