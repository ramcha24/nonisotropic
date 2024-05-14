from __future__ import annotations
import argparse
from dataclasses import dataclass, asdict, field, fields
import os
from itertools import product
import json

from datasets import registry as datasets_registry
from foundations import desc
from foundations.hparams import (
    DatasetHparams,
    ModelHparams,
    TrainingHparams,
    AugmentationHparams,
)
from foundations.step import Step
from foundations import paths
from platforms.platform import get_platform
from training.desc import TrainingDesc


@dataclass
class MultiTrainingDesc(desc.Desc):
    """The hyperparameters necessary to describe a training run"""

    desc_list: list[TrainingDesc] = field(default_factory=list)

    @staticmethod
    def name_prefix():
        return "multi_train"

    @staticmethod
    def add_args(
        parser: argparse.ArgumentParser, defaults_list: list[TrainingDesc] = None
    ):
        for defaults_index, defaults in enumerate(defaults_list):
            TrainingDesc.add_args(
                parser,
                defaults,
                prefix=str("multi_train_") + str(defaults_index),
            )

    @staticmethod
    def create_from_args(args: argparse.Namespace) -> "TrainingDesc":
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
        # print(json.dumps(vars(args), indent=4))

        desc_list = []
        for defaults_index in index_list:
            prefix_str = str("multi_train_") + str(defaults_index)
            model_name_key = prefix_str + "_model_name"

            if hasattr(args, model_name_key):
                model_hparams = ModelHparams.create_from_args(args, prefix=prefix_str)
                dataset_hparams = DatasetHparams.create_from_args(
                    args, prefix=prefix_str
                )
                model_type = getattr(model_hparams, "model_type")
                assert model_type in [
                    None,
                    "pretrained",
                ], f"Invalid model type {model_type} for multi training runner"

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
                    for (index, arg_name) in enumerate(possible_augment_toggle_args):
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
                            TrainingDesc(
                                dataset_hparams,
                                model_hparams,
                                modified_augment_hparams,
                                modified_training_hparams,
                            )
                        )

        multi_desc = MultiTrainingDesc(desc_list=desc_list)
        return multi_desc

    @property
    def end_step(self):
        raise NotImplementedError(
            "Please call the run_path of individual TrainingDesc instances"
        )

    @property
    def train_outputs(self):
        raise NotImplementedError(
            "Please call the run_path of individual TrainingDesc instances"
        )

    def run_path(self):
        raise NotImplementedError(
            "Please call the run_path of individual TrainingDesc instances"
        )

    @property
    def display(self):
        raise NotImplementedError(
            "Please call the display of the individual TrainingDesc instances"
        )

    def save_param(self):
        raise NotImplementedError(
            "Please call the save param of the individual TrainingDesc instances"
        )
