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
    model_hparams: hparams.ModelHparams
    testing_hparams: hparams.TestingHparams
    augment_hparams: Optional[hparams.AugmentationHparams] = None
    training_hparams: Optional[hparams.TrainingHparams] = None

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
        dataset_hparams = hparams.DatasetHparams.create_from_args(args)
        model_hparams = hparams.ModelHparams.create_from_args(args)
        testing_hparams = hparams.TestingHparams.create_from_args(args)
        augment_hparams = hparams.AugmentationHparams.create_from_args(args)
        training_hparams = hparams.TrainingHparams.create_from_args(args)

        return TestingDesc(
            dataset_hparams,
            model_hparams,
            testing_hparams,
            augment_hparams,
            training_hparams,
        )

    @property
    def test_outputs(self):
        return datasets_registry.num_labels(self.dataset_hparams)

    def run_path(self, train_replicate, test_replicate, verbose=False):
        """_summary_

        Args:
            train_replicate (_type_): _description_
            test_replicate (_type_): _description_
            verbose (bool, optional): _description_. Defaults to False.

        Returns:
            logger_paths (dict): Containing paths where hparams and runner output will be stored

        Example test paths
        nonisotropic/runner_data/cifar10/pretrained/robustbenchmark/Linf/Peng2013robust/test_replicate_1
        nonisotropic/runner_data/cifar10/resnet50/augment_std/training_std/train_replicate_1/test_replicate_1

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

        model weights for train replicate run 1 will be stored at
        1) nonisotropic/runner_data/cifar10/resnet50/augment_std_xx/training_std_xx/train_replicate_1/x.pt or
        2) nonisotropic/runner_data/cifar10/pretrained/robustbenchmark/Linf/Peng2013robust/augment_std_xx/training_std_xx/train_replicate_1/x.pt (for finetuned) or
        3) nonisotropic/runner_data/cifar10/pretrained/robustbenchmark/Linf/Peng2013robust/x.pt (for pretrained)

        test hparams will be stored at
        1) nonisotropic/runner_data/cifar10/resnet50/augment_std_xx/training_std_xx/train_replicate_1/test_std_xx or
        2) nonisotropic/runner_data/cifar10/pretrained/robustbenchmark/Linf/Peng2013robust/augment_std_xx/training_std_xx/train_replicate_1/test_std_xx (for testing finetuned models)
        3) nonisotropic/runner_data/cifar10/pretrained/robustbenchmark/Linf/Peng2013robust/test_std_xx (for testing pretrained models)

        test run data (plots/info) for test replicate run 1 will be stored at
        1) nonisotropic/runner_data/cifar10/resnet50/augment_std_xx/training_std_xx/train_replicate_1/test_std_xx/test_replicate_1 or
        2) nonisotropic/runner_data/cifar10/pretrained/robustbenchmark/Linf/Peng2013robust/augment_std_xx/training_std_xx/train_replicate_1/test_std_xx/test_replicate_1
        3) nonisotropic/runner_data/cifar10/pretrained/robustbenchmark/Linf/Peng2013robust/test_std_xx/test_replicate_1 (for testing pretrained models)

        """

        if train_replicate is None:
            assert (
                self.model_hparams.model_type == "pretrained"
            ), "If model type is not pretrained, then train_replicate must be specified"

        assert (
            test_replicate is not None
        ), "Test replicate must be specified or inferred"

        logger_paths = dict()
        root_location = get_platform().runner_root  # nonisotropic/runner_data
        dataset_dir = self.dataset_hparams.dataset_name
        logger_paths["dataset_path"] = os.path.join(root_location, dataset_dir)

        if self.model_hparams.model_type in ["pretrained", "finetuned"]:
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

        if self.augment_hparams is not None:
            augment_dir = (
                self.augment_hparams.dir_path()
            )  # augment_std_xx / augment_N_aug_xx / augment_mixup_xx
            logger_paths["augment_hparams_path"] = os.path.join(
                logger_paths["model_path"], augment_dir
            )
        else:
            logger_paths["augment_hparams_path"] = logger_paths["model_path"]

        if self.training_hparams is not None:
            train_dir = (
                self.training_hparams.dir_path()
            )  # train_std_xx / train_adv_xx / train_N_adv_xx
            logger_paths["train_hparams_path"] = os.path.join(
                logger_paths["augment_hparams_path"], train_dir
            )
        else:
            logger_paths["train_hparams_path"] = logger_paths["augment_hparams_path"]

        if train_replicate is not None:
            logger_paths["train_run_path"] = os.path.join(
                logger_paths["train_hparams_path"],
                "train_replicate_" + str(train_replicate),
            )  # train_replicate_1
        else:
            logger_paths["train_run_path"] = logger_paths["train_hparams_path"]

        test_dir = self.testing_hparams.dir_path()
        logger_paths["test_hparams_path"] = os.path.join(
            logger_paths["train_run_path"], test_dir
        )  # test_std_xx / test_adv_xx / test_N_adv_xx

        logger_paths["test_run_path"] = os.path.join(
            logger_paths["test_hparams_path"], "test_replicate_" + str(test_replicate)
        )  # test_replicate_1

        if get_platform().is_primary_process and verbose:
            print(
                "\n All runner data for the specified dataset should be found at {}".format(
                    logger_paths["dataset_path"]
                )
            )
            print(
                "\n All runner data for the specified model name should be found at {}".format(
                    logger_paths["model_path"]
                )
            )
            print(
                "\nAugmentation hparams (if needed) should be found at {}".format(
                    logger_paths["augment_hparams_path"]
                )
            )
            print(
                "\nTraining hparams (if needed) should be found at {}".format(
                    logger_paths["train_hparams_path"]
                )
            )
            print(
                "\n Model weights (and training log data if needed) should be found at {}".format(
                    logger_paths["train_run_path"]
                )
            )
            print(
                "\nTesting hparams will be logged at {}".format(
                    logger_paths["test_hparams_path"]
                )
            )
            print(
                "\nTest output (plots/info) will be logged at {}".format(
                    logger_paths["test_run_path"]
                )
            )

        if not get_platform().exists(logger_paths["train_run_path"]) and verbose:
            # raise ValueError(
            print(
                "\n Warning! No pretrained/trained/finetuned models found at {},\n Can't run test job".format(
                    logger_paths["train_run_path"]
                )
            )

        if not get_platform().exists(logger_paths["test_run_path"]) and verbose:
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

    def save_param(self, train_replicate, test_replicate):

        logger_paths = self.run_path(train_replicate, test_replicate)
        full_run_path = logger_paths["test_run_path"]

        if not get_platform().is_primary_process:
            return
        if not get_platform().exists(full_run_path):
            get_platform().makedirs(full_run_path)
            hparams_strs = self.get_hparams_str(type_str="test")
            with get_platform().open(
                paths.params_loc(logger_paths["test_hparams_path"], "test"), "w"
            ) as fp:
                fp.write("\n".join(hparams_strs))
        else:
            print(
                "A job with this configuration may have already been initiated."  # Stale. Delete the existing job results to run again.
            )
