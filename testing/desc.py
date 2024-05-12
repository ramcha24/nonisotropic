from dataclasses import dataclass, fields
import os
import argparse
from typing import Optional

from datasets import registry as datasets_registry
from foundations import desc
from foundations import hparams
from foundations import paths
from platforms.platform import get_platform


@dataclass
class TestingDesc(desc.Desc):
    """the hyperparameters necessary to describe a testing run"""

    dataset_hparams: hparams.DatasetHparams
    augment_hparams: Optional[hparams.AugmentationHparams] = None
    model_hparams: hparams.ModelHparams
    training_hparams: Optional[hparams.TrainingHparams] = None
    testing_hparams: hparams.TestingHparams

    @staticmethod
    def name_prefix():
        return "test"

    @staticmethod
    def add_args(
        parser: argparse.ArgumentParser, defaults: "TestingDesc" = None, prefix=None
    ):
        hparams.ModelHparams.add_args(
            parser, defaults=defaults.model_hparams if defaults else None, prefix=prefix
        )
        hparams.DatasetHparams.add_args(
            parser,
            defaults=defaults.dataset_hparams if defaults else None,
            prefix=prefix,
        )
        hparams.AugmentationHparams.add_args(
            parser,
            defaults=defaults.augment_hparams if defaults else None,
            prefix=prefix,
        )
        hparams.TrainingHparams.add_args(
            parser,
            defaults=defaults.training_hparams if defaults else None,
            prefix=prefix,
        )
        hparams.TestingHparams.add_args(
            parser,
            defaults=defaults.testing_hparams if defaults else None,
            prefix=prefix,
        )

    @staticmethod
    def create_from_args(args: argparse.Namespace) -> "TestingDesc":
        model_hparams = hparams.ModelHparams.create_from_args(args)
        dataset_hparams = hparams.DatasetHparams.create_from_args(args)
        augment_hparams = hparams.AugmentationHparams.create_from_args(args)
        training_hparams = hparams.TrainingHparams.create_from_args(args)
        testing_hparams = hparams.TestingHparams.create_from_args(args)

        return TestingDesc(
            model_hparams,
            dataset_hparams,
            augment_hparams,
            training_hparams,
            testing_hparams,
        )

    @property
    def test_outputs(self):
        return datasets_registry.num_labels(self.dataset_hparams)

    def run_path(self, verbose=False):
        root_location = get_platform().runner_root

        dataset_dir = self.dataset_hparams.dir_path(identifier_name="dataset_name")
        model_dir = self.dataset_hparams.dir_path(
            identifier_name="model_name",
            include_names=["model_type", "model_source", "threat_model"],
        )

        augment_dir = self.augment_hparams.dir_path() if self.augment_hparams else None
        train_dir = (
            self.training_hparams.dir_path(include_names=["train_replicate"])
            if self.training_hparams
            else None
        )
        test_dir = self.testing_hparams.dir_path(include_names=["test_replicate"])

        logger_paths = dict()
        logger_paths["dataset_path"] = os.path.join(root_location, dataset_dir)
        logger_paths["model_path"] = os.path.join(
            logger_paths["dataset_path"], model_dir
        )

        if augment_dir is not None:
            logger_paths["augment_path"] = os.path.join(
                logger_paths["model_path"], augment_dir
            )
        else:
            logger_paths["augment_path"] = logger_paths["model_path"]

        if train_dir is not None:
            logger_paths["train_path"] = os.path.join(
                logger_paths["augment_path"], train_dir
            )
        else:
            logger_paths["train_path"] = logger_paths["augment_path"]

        logger_paths["test_path"] = os.path.join(logger_paths["train_path"], test_dir)

        if get_platform().is_primary_process and verbose:
            print(
                "\nDataset hparams should already be logged at {}".format(
                    logger_paths["dataset_path"]
                )
            )
            print(
                "\nModel hparams and weights should be found at {}".format(
                    logger_paths["model_path"]
                )
            )
            if augment_dir is not None:
                print(
                    "\nAugmentation hparams should be logged at {}".format(
                        logger_paths["augment_path"]
                    )
                )
            if train_dir is not None:
                print(
                    "\nTraining hparams should be logged at {}".format(
                        logger_paths["train_path"]
                    )
                )
            print(
                "\nTesting hparams and results will be logged at {}".format(
                    logger_paths["test_path"]
                )
            )

        if not get_platform().exists(logger_paths["train_path"]):
            raise ValueError(
                "\n No pretrained/trained/finetuned models found at {},\n Can't run test job~".format(
                    logger_paths["train_path"]
                )
            )

        if not get_platform().exists(logger_paths["test_path"]) and verbose:
            print("A job with this configuration has not been run yet.")

        return logger_paths

    @property
    def display(self):
        return "\n".join(
            [
                self.dataset_hparams.display,
                self.augment_hparams.display
                if self.augment_hparams is not None
                else "",
                self.model_hparams.display,
                self.training_hparams.display
                if self.training_hparams is not None
                else "",
                self.testing_hparams.display,
            ]
        )

    def save_param(self):

        logger_paths = self.run_path()
        full_run_path = logger_paths["test_path"]

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
