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
    model_hparams: hparams.ModelHparams
    augment_hparams: hparams.AugmentationHparams
    training_hparams: hparams.TrainingHparams
    threat_hparams: hparams.ThreatHparams

    @staticmethod
    def name_prefix():
        return "training"

    @staticmethod
    def add_args(
        parser: argparse.ArgumentParser, defaults: "TrainingDesc" = None, prefix=None
    ):
        hparams.DatasetHparams.add_args(
            parser,
            defaults=defaults.dataset_hparams if defaults else None,
            prefix=prefix,
        )
        hparams.ModelHparams.add_args(
            parser, defaults=defaults.model_hparams if defaults else None, prefix=prefix
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
        hparams.ThreatHparams.add_args(
            parser,
            defaults=defaults.threat_hparams if defaults else None,
            prefix=prefix,
        )

    @staticmethod
    def create_from_args(args: argparse.Namespace) -> "TrainingDesc":
        dataset_hparams = hparams.DatasetHparams.create_from_args(args)
        model_hparams = hparams.ModelHparams.create_from_args(args)
        augment_hparams = hparams.AugmentationHparams.create_from_args(args)
        training_hparams = hparams.TrainingHparams.create_from_args(args)
        threat_hparams = hparams.ThreatHparams.create_from_args(args)

        return TrainingDesc(
            dataset_hparams,
            model_hparams,
            augment_hparams,
            training_hparams,
            threat_hparams,
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

    def run_path(self, train_replicate, verbose=False):
        """_summary_

        Args:
            train_replicate (_type_): _description_
            verbose (bool, optional): _description_. Defaults to False.

        Returns:
            logger_paths (dict): Containing paths where hparams and runner output will be stored

        Example train run paths
        nonisotropic/runner_data/cifar10/pretrained/robustbenchmark/Linf/Peng2013robust/augment_std/training_std/train_replicate_1
        nonisotropic/runner_data/cifar10/resnet50/augment_std/training_std/train_replicate_1

        dataset hparams will be stored at nonisotropic/runner_data/cifar10

        model hparams will be stored at
        1) nonisotropic/runner_data/cifar10/resnet50  or
        2) nonisotropic/runner_data/cifar10/pretrained/robustbenchmark/Linf/Peng2013robust

        augment hparams will be stored at
        1) nonisotropic/runner_data/cifar10/resnet50/augment_std_xx or
        2) nonisotropic/runner_data/cifar10/pretrained/robustbenchmark/Linf/Peng2013robust/augment_std_xx

        _xx indicates appropriate hashstring
        Currently not storing hash strings for dataset or model hparams under the assumption these are the same for a given model name dataset_name for all runs

        training hparams will be stored at
        1) nonisotropic/runner_data/cifar10/resnet50/augment_std_xx/training_std_xx or
        2) nonisotropic/runner_data/cifar10/pretrained/robustbenchmark/Linf/Peng2013robust/augment_std_xx/training_std_xx (for finetuned)

        replicate numbers indicate multiple run for the same hyperparameter configurations

        trained model weights for train replicate run 1 will be stored at
        1) nonisotropic/runner_data/cifar10/resnet50/augment_std_xx/training_std_xx/train_replicate_1/x.pt or
        2) nonisotropic/runner_data/cifar10/pretrained/robustbenchmark/Linf/Peng2013robust/augment_std_xx/training_std_xx/train_replicate_1/x.pt (for finetuned) or
        """

        assert (
            train_replicate is not None
        ), "Train replicate must be specified or inferred"

        assert self.model_hparams.model_type in [
            None,
            "pretrained",
        ], f"Invalid model type {self.model_hparams.model_type} for training runner"

        assert (
            self.augment_hparams is not None
        ), "Augment hparams must be specified or inferred from default values"
        assert (
            self.training_hparams is not None
        ), "Training hparams must be specified or inferred from default values"

        logger_paths = dict()
        root_location = get_platform().runner_root  # nonisotropic/runner_data
        dataset_dir = self.dataset_hparams.dataset_name
        logger_paths["dataset_path"] = os.path.join(root_location, dataset_dir)

        if self.model_hparams.model_type in ["pretrained"]:
            if self.model_hparams.model_source == "robustbenchmark":
                if self.model_hparams.threat_model == "Linf":
                    model_dir = os.path.join(
                        "pretrained",
                        "robustbenchmark",
                        "Linf",
                        self.model_hparams.model_name,
                    )
        else:
            model_name = self.model_hparams.model_name
            model_dir = "".join(model_name.split("_")[1:])

        logger_paths["model_path"] = os.path.join(
            logger_paths["dataset_path"], model_dir
        )

        augment_dir = (
            self.augment_hparams.dir_path()
        )  # augment_std_xx / augment_N_aug_xx / augment_mixup_xx
        logger_paths["augment_hparams_path"] = os.path.join(
            logger_paths["model_path"], augment_dir
        )

        train_dir = (
            self.training_hparams.dir_path()
        )  # train_std_xx / train_adv_xx / train_N_adv_xx
        logger_paths["train_hparams_path"] = os.path.join(
            logger_paths["augment_hparams_path"], train_dir
        )

        logger_paths["train_run_path"] = os.path.join(
            logger_paths["train_hparams_path"],
            "train_replicate_" + str(train_replicate),
        )  # train_replicate_1

        if get_platform().is_primary_process and verbose:
            print(
                "\n All runner data for the specified dataset will be found at {}".format(
                    logger_paths["dataset_path"]
                )
            )
            print(
                "\n All runner data for the specified model name will be found at {}".format(
                    logger_paths["model_path"]
                )
            )
            print(
                "\nAugmentation hparams (if needed) will be found at {}".format(
                    logger_paths["augment_hparams_path"]
                )
            )
            print(
                "\nTraining hparams (if needed) will be found at {}".format(
                    logger_paths["train_hparams_path"]
                )
            )
            print(
                "\n Model weights (and training log data if needed) will be found at {}".format(
                    logger_paths["train_run_path"]
                )
            )

        if not get_platform().exists(logger_paths["train_run_path"]) and verbose:
            print("A job with this configuration has not been run yet.")

        return logger_paths

    @property
    def display(self):
        return "\n".join(
            [
                self.dataset_hparams.display,
                self.augment_hparams.display,
                self.model_hparams.display,
                self.training_hparams.display,
                self.threat_hparams.display,
            ]
        )

    def save_param(self, train_replicate):

        logger_paths = self.run_path(train_replicate)
        full_run_path = logger_paths["train_run_path"]
        if not get_platform().is_primary_process:
            return
        if not get_platform().exists(full_run_path):
            get_platform().makedirs(full_run_path)
            for type_str in ["augment", "train"]:
                hparams_strs = self.get_hparams_str(type_str=type_str)
                with get_platform().open(
                    paths.params_loc(
                        logger_paths[type_str + "_hparams_path"], type_str
                    ),
                    "w",
                ) as fp:
                    fp.write("\n".join(hparams_strs))
        else:
            print(
                "A job with this configuration may have already been initiated."  # Stale. Delete the existing job results to run again.
            )
