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
        possible_augment_toggle_args = [
            "N_aug",
            "mixup",
        ]
        possible_training_toggle_args = [
            "adv_train",
            "N_adv_train",
        ]

        selected_augment_toggle_args = [False, False]
        for (index, arg_name) in enumerate(possible_augment_toggle_args):
            if getattr(args, "toggle_" + arg_name):
                selected_augment_toggle_args[index] = True

        selected_training_toggle_args = [False, False]
        for (index, arg_name) in enumerate(possible_training_toggle_args):
            if getattr(args, "toggle_" + arg_name):
                selected_training_toggle_args[index] = True

        num_augment_selected = sum(selected_augment_toggle_args)
        num_training_selected = sum(selected_training_toggle_args)

        toggle_choices = [True, False]

        desc_list = []
        for defaults_index in index_list:
            prefix_str = str("multi_test_") + str(defaults_index)
            if hasattr(args, prefix_str + "_model_name"):
                model_hparams = ModelHparams.create_from_args(args, prefix=prefix_str)
                dataset_hparams = DatasetHparams.create_from_args(
                    args, prefix=prefix_str
                )
                testing_hparams = TestingHparams.create_from_args(
                    args, prefix=prefix_str
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
                        for (index, arg_name) in enumerate(
                            possible_augment_toggle_args
                        ):
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
                            for (index, arg_name) in enumerate(
                                possible_training_toggle_args
                            ):
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

    def run_path(self):
        raise NotImplementedError(
            "Please call the run_path of individual TestingDesc instances"
        )

    @property
    def display(self):
        raise NotImplementedError(
            "Please call the display of the individual TestingDesc instances"
        )

    def save_param(self):
        raise NotImplementedError(
            "Please call the save param of the individual TestingDesc instances"
        )
