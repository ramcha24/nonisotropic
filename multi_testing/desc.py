from __future__ import annotations
from dataclasses import dataclass, asdict, field, fields
from typing import ClassVar
import os
import argparse
import json
from itertools import product

from datasets import registry as datasets_registry
from foundations.desc import Desc
from foundations.hparams import (
    DatasetHparams,
    ModelHparams,
    TrainingHparams,
    AugmentationHparams,
    TestingHparams,
)
from foundations import paths
from platforms.platform import get_platform
from testing.desc import TestingDesc
from cli.shared_args import MultiTestArgs


@dataclass
class MultiTestingDesc(Desc):
    """the hyperparameters necessary to describe a multi testing run"""

    desc_list: list[TestingDesc] = field(default_factory=list)

    @staticmethod
    def name_prefix():
        return "multi_test"

    @staticmethod
    def add_args(
        parser: argparse.ArgumentParser,
        defaults_list: list[TestingDesc] = None,
    ):
        for defaults_index, defaults in enumerate(defaults_list):
            TestingDesc.add_args(
                parser,
                defaults,
                prefix=str("multi_test_") + str(defaults_index),
            )

    @staticmethod
    def create_from_args(args: argparse.Namespace) -> "TestingDesc":
        index_list = list(
            range(16)
        )  # currently considering less than 16 pretrained models
        augment_toggle_args = AugmentationHparams.get_boolean_field_names()
        training_toggle_args = TrainingHparams.get_boolean_field_names()
        toggle_choices = [True, False]

        selected_augment_toggle_args = [False] * len(augment_toggle_args)
        for (index, arg_name) in enumerate(augment_toggle_args):
            if getattr(args, "toggle_" + arg_name):
                selected_augment_toggle_args[index] = True

        selected_training_toggle_args = [False, False]
        for (index, arg_name) in enumerate(training_toggle_args):
            if getattr(args, "toggle_" + arg_name):
                selected_training_toggle_args[index] = True

        num_augment_selected = sum(selected_augment_toggle_args)
        num_training_selected = sum(selected_training_toggle_args)

        multi_test_args = MultiTestArgs.get_boolean_field_names()
        test_modify_dict = dict()
        for (index, arg_name) in enumerate(multi_test_args):
            if getattr(args, "multi_" + arg_name):
                test_modify_dict[arg_name] = True
            else:
                test_modify_dict[arg_name] = False

        desc_list = []
        for defaults_index in index_list:
            prefix_str = str("multi_test_") + str(defaults_index)
            if hasattr(args, prefix_str + "_model_name"):
                model_hparams = ModelHparams.create_from_args(args, prefix=prefix_str)
                dataset_hparams = DatasetHparams.create_from_args(
                    args, prefix=prefix_str
                )
                default_testing_hparams = TestingHparams.create_from_args(
                    args, prefix=prefix_str
                )
                # modify testing hparams as needed
                testing_hparams = TestingHparams.modified(
                    asdict(default_testing_hparams), test_modify_dict
                )

                model_type = getattr(model_hparams, "model_type")
                if model_type == "pretrained":
                    assert num_augment_selected + num_training_selected == 0
                    desc_list.append(
                        TestingDesc(
                            dataset_hparams,
                            model_hparams,
                            testing_hparams,
                        )
                    )
                elif model_type in ["finetuned", None]:
                    augment_hparams = AugmentationHparams.create_from_args(
                        args, prefix=prefix_str
                    )
                    training_hparams = TrainingHparams.create_from_args(
                        args, prefix=prefix_str
                    )
                    augment_dict = asdict(augment_hparams)
                    training_dict = asdict(training_hparams)

                    for augment_choices in product(
                        toggle_choices, repeat=num_augment_selected
                    ):
                        augment_modify_dict = dict()
                        augment_counter = 0
                        for (index, arg_name) in enumerate(augment_toggle_args):
                            if selected_augment_toggle_args[index]:
                                augment_modify_dict[arg_name] = augment_choices[
                                    augment_counter
                                ]
                                augment_counter += 1
                        assert augment_counter == num_augment_selected
                        modified_augment_hparams = AugmentationHparams.modified(
                            augment_dict, augment_modify_dict
                        )

                        for training_choices in product(
                            toggle_choices, repeat=num_training_selected
                        ):
                            training_modify_dict = dict()
                            training_counter = 0
                            for (index, arg_name) in enumerate(training_toggle_args):
                                if selected_training_toggle_args[index]:
                                    training_modify_dict[arg_name] = training_choices[
                                        training_counter
                                    ]
                                    training_counter += 1
                            assert training_counter == num_training_selected
                            modified_training_hparams = TrainingHparams.modified(
                                training_dict, training_modify_dict
                            )

                            desc_list.append(
                                TestingDesc(
                                    dataset_hparams,
                                    model_hparams,
                                    testing_hparams,
                                    modified_augment_hparams,
                                    modified_training_hparams,
                                )
                            )
                else:
                    raise ValueError(f"Model type {model_type} is an invalid argument")

        multi_desc = MultiTestingDesc(desc_list=desc_list)
        return multi_desc

    @property
    def test_outputs(self):
        return datasets_registry.num_labels(self.desc_list[0].dataset_hparams)

    def run_path(
        self,
        multi_train_replicate: int = 1,
        multi_test_replicate: int = 1,
        verbose: bool = True,
    ):
        # assuming that the following attributes are homogenous across sub runners
        dataset_name = self.desc_list[0].dataset_hparams.dataset_name
        model_type = self.desc_list[0].model_hparams.model_type
        model_source = self.desc_list[0].model_hparams.model_source
        threat_model = self.desc_list[0].model_hparams.threat_model

        if multi_train_replicate is None:
            assert (
                model_type == "pretrained"
            ), "If model type is not pretrained, then multi_train_replicate must be specified"

        assert (
            multi_test_replicate is not None
        ), "Multi_test replicate must be specified or inferred already"

        multi_logger_paths = dict()
        root_location = (
            get_platform().multi_runner_root
        )  # nonisotropic/multi_runner_data
        multi_logger_paths["dataset_path"] = os.path.join(root_location, dataset_name)

        if model_type in ["pretrained", "finetuned"]:
            if model_source == "robustbenchmark":
                if threat_model == "Linf":
                    model_dir = os.path.join(
                        "pretrained",
                        "robustbenchmark",
                        "Linf",
                    )
            if model_type == "finetuned":
                model_dir = os.path.join(
                    model_dir,
                    "finetuned",
                    "multi_train_replicate_" + multi_train_replicate,
                )
        else:
            model_name = self.desc_list[0].model_hparams.model_name
            assert model_name is not None
            model_dir = "".join(model_name.split("_")[1:])
            model_dir = os.path.join(
                model_dir, "multi_train_replicate_" + multi_train_replicate
            )

        multi_logger_paths["feedback_path"] = os.path.join(
            multi_logger_paths["dataset_path"],
            model_dir,
            "multi_test_replicate_" + multi_test_replicate,
        )

        if verbose and get_platform().is_primary_process:
            print(
                f"Evaluation feedback will be stored at {multi_logger_paths['feedback_path']}"
            )

        return multi_logger_paths

    @property
    def display(self):
        raise NotImplementedError(
            "Please call the display of the individual TestingDesc instances"
        )

    def save_param(self):
        raise NotImplementedError(
            "Please call the save param of the individual TestingDesc instances"
        )
