import argparse
from dataclasses import dataclass, fields
import os

from datasets import registry as datasets_registry
from foundations import desc
from foundations import hparams
from foundations.step import Step
from foundations import paths
from platforms.platform import get_platform


@dataclass
class TrainingDesc(desc.Desc):
    """The hyperparameters necessary to describe a training run"""

    dataset_hparams: hparams.DatasetHparams
    augment_hparams: hparams.AugmentationHparams
    model_hparams: hparams.ModelHparams
    training_hparams: hparams.TrainingHparams

    @staticmethod
    def name_prefix():
        return "training"

    @staticmethod
    def add_args(parser: argparse.ArgumentParser, defaults: "TrainingDesc" = None):
        hparams.DatasetHparams.add_args(
            parser, defaults=defaults.dataset_hparams if defaults else None
        )
        hparams.AugmentationHparams.add_args(
            parser, defaults=defaults.augment_hparams if defaults else None
        )
        hparams.ModelHparams.add_args(
            parser, defaults=defaults.model_hparams if defaults else None
        )
        hparams.TrainingHparams.add_args(
            parser, defaults=defaults.training_hparams if defaults else None
        )

    @staticmethod
    def create_from_args(args: argparse.Namespace) -> "TrainingDesc":
        dataset_hparams = hparams.DatasetHparams.create_from_args(args)
        augment_hparams = hparams.AugmentationHparams.create_from_args(args)
        model_hparams = hparams.ModelHparams.create_from_args(args)
        training_hparams = hparams.TrainingHparams.create_from_args(args)

        return TrainingDesc(
            dataset_hparams, augment_hparams, model_hparams, training_hparams
        )

    @property
    def end_step(self):
        iterations_per_epoch = datasets_registry.iterations_per_epoch(
            self.dataset_hparams
        )
        return Step.from_str(self.training_hparams.training_steps, iterations_per_epoch)

    @property
    def train_outputs(self):
        return datasets_registry.num_labels(self.dataset_hparams)

    def run_path(self, verbose=False):
        root_location = get_platform().runner_root

        dataset_dir = self.dataset_hparams.dir_path(identifier_name="dataset_name")
        model_dir = self.dataset_hparams.dir_path(
            identifier_name="model_name",
            include_names=["model_type", "model_source", "threat_model"],
        )

        augment_dir = self.augment_hparams.dir_path()
        train_dir = self.training_hparams.dir_path(include_names=["train_replicate"])

        logger_paths = dict()
        logger_paths["dataset_path"] = os.path.join(root_location, dataset_dir)
        logger_paths["model_path"] = os.path.join(
            logger_paths["dataset_path"], model_dir
        )

        logger_paths["augment_path"] = os.path.join(
            logger_paths["model_path"], augment_dir
        )

        logger_paths["train_path"] = os.path.join(
            logger_paths["augment_path"], train_dir
        )

        if get_platform().is_primary_process and verbose:
            print(
                "\nDataset hparams will be logged at {}".format(
                    logger_paths["dataset_path"]
                )
            )
            print(
                "\nModel hparams and weights will be found at {}".format(
                    logger_paths["model_path"]
                )
            )
            print(
                "\nAugmentation hparams will be logged at {}".format(
                    logger_paths["augment_path"]
                )
            )
            print(
                "\nTraining hparams will be logged at {}".format(
                    logger_paths["train_path"]
                )
            )

        if not get_platform().exists(logger_paths["train_path"]):
            raise ValueError(
                "\n A job with this configuration has not been run yet.".format(
                    logger_paths["train_path"]
                )
            )

        return logger_paths

    @property
    def display(self):
        return "\n".join(
            [
                self.dataset_hparams.display,
                self.augment_hparams.display,
                self.model_hparams.display,
                self.training_hparams.display,
            ]
        )

    def save_param(self):

        logger_paths = self.run_path()
        full_run_path = logger_paths["train_path"]
        if not get_platform().is_primary_process:
            return
        if not get_platform().exists(full_run_path):
            get_platform().makedirs(full_run_path)
            for type_str in ["data", "model", "train"]:
                hparams_strs = self.get_hparams_str(type_str=type_str)
                with get_platform().open(
                    paths.params_loc(logger_paths[type_str], type_str), "w"
                ) as fp:
                    fp.write("\n".join(hparams_strs))
        else:
            print(
                "A job with this configuration may have already been initiated."  # Stale. Delete the existing job results to run again.
            )
