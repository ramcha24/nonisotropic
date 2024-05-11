from dataclasses import dataclass, fields
import os
import argparse

from datasets import registry as datasets_registry
from foundations import desc
from foundations import hparams
from foundations import paths
from platforms.platform import get_platform
from testing.desc import TestingDesc
import json


@dataclass
class MultiTestingDesc(desc.Desc):
    """the hyperparameters necessary to describe a testing run"""

    desc_list: list[TestingDesc]
    dataset: str = None
    threat_model: str = None
    model_type: str = None

    @staticmethod
    def name_prefix():
        return "multi_test"

    @staticmethod
    def add_args(
        parser: argparse.ArgumentParser,
        defaults_list: list[TestingDesc] = None,
    ):
        print("Inside Multi-testing runner")
        print("Default list looks like")
        print(len(defaults_list))
        print(defaults_list[len(defaults_list) // 2])
        print("\n")

        # the bit below should happen on its own via JobArgs
        # dataset_arg_name = "--dataset"
        # parser.add_argument(
        #     dataset_arg_name,
        #     type=str,
        #     default=None,
        #     required=True,
        #     help="(required: str) --dataset",
        # )
        # threat_model_arg_name = "--threat_model"
        # parser.add_argument(
        #     threat_model_arg_name,
        #     type=str,
        #     default=None,
        #     required=True,
        #     help="(required: str) --threat_model",
        # )
        # model_type_arg_name = "--model_type"
        # parser.add_argument(
        #     model_type_arg_name,
        #     type=str,
        #     default=None,
        #     required=True,
        #     help="(required: str) --model_type",
        # )

        for defaults_index, defaults in enumerate(defaults_list):
            TestingDesc.add_args(
                parser,
                defaults,
                prefix=str("multi_test_") + str(defaults_index),
            )
            print(json.dumps(vars(parser.parse_args()), indent=4))

    @staticmethod
    def create_from_args(args: argparse.Namespace) -> "TestingDesc":
        index_list = list(range(16))
        desc_list = []
        for defaults_index in index_list:
            prefix_str = str("multi_test_") + str(defaults_index)
            if hasattr(args, prefix_str + "_model_name"):
                model_hparams = hparams.ModelHparams.create_from_args(
                    args, prefix=prefix_str
                )
                dataset_hparams = hparams.DatasetHparams.create_from_args(
                    args, prefix=prefix_str
                )
                pretraining_hparams = hparams.PretrainingHparams.create_from_args(
                    args, prefix=prefix_str
                )
                testing_hparams = hparams.TestingHparams.create_from_args(
                    args, prefix=prefix_str
                )
                # model_type_arg_name = prefix_str + "_model_type")
                # if not hasattr(args, model_type_arg_name):
                #     raise ValueError(f"Missing argument: {model_type_arg_name}.")
                # model_type = getattr(arg, model_type_arg_name).split("_")[2:]
                if model_hparams.model_type == "finetuned":
                    augment_hparams = hparams.AugmentationHparams.create_from_args(
                        args, prefix=prefix_str
                    )
                    training_hparams = hparams.TrainingHparams.create_from_args(
                        args, prefix=prefix_str
                    )
                    desc_list.append(
                        TestingDesc(
                            model_hparams,
                            dataset_hparams,
                            augment_hparams,
                            pretraining_hparams,
                            training_hparams,
                            testing_hparams,
                        )
                    )
                elif model_hparams.model_type == "pretrained":
                    desc_list.append(
                        TestingDesc(
                            model_hparams,
                            dataset_hparams,
                            pretraining_hparams,
                            testing_hparams,
                        )
                    )
                else:
                    raise ValueError(
                        f"Model type {model_hparams.model_type} is an invalid argument"
                    )

        return MultiTestingDesc(desc_list)

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
