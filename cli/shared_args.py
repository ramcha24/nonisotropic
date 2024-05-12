from dataclasses import dataclass

from cli import arg_utils
from foundations import hparams

from datasets.registry import registered_datasets

import models.registry
from models.robustbench_registry import rb_registry

from training.desc import TrainingDesc
from testing.desc import TestingDesc


@dataclass
class JobArgs(hparams.Hparams):
    """Arguments shared across jobs"""

    dataset_name: str = None
    model_name: str = None
    quiet: bool = False
    evaluate_only_at_end: bool = False
    evaluate_only_batch_test: bool = False
    # should add dataset, model_type, threat_type

    _name: str = "High-level arguments"
    _description: str = (
        "Arguments that determine how the job is run and where it is stored."
    )
    _dataset_name: str = "Name of the dataset to use for this job"
    _model_name: str = (
        "Populate all arguments with the default hyperparameters for this model."
    )
    _quiet: str = "Suppress output logging about the training status."
    _evaluate_only_at_end: str = (
        "Run the test set only before and after training, otherwise run every epoch"
    )
    _evaluate_only_batch_test: str = (
        "Run the test runner only on a random batch of the test set"
    )


@dataclass
class ToggleArgs(hparams.Hparams):
    toggle_N_aug: bool = False
    toggle_mixup: bool = False
    toggle_adv_train: bool = False
    toggle_N_adv_train: bool = False

    _name: str = "Multi runner toggle hyperparameters"
    _description: str = (
        "Toggle options for multi-runners based on boolean sub attributes"
    )
    _toggle_N_aug: str = "Toggle both non-isotropic augmentations"
    _toggle_mixup: str = "Toggle mixup augmentation"
    _toggle_adv_train: str = "Toggle adversarial training"
    _toggle_N_adv_train: str = "Toggle non-isotropic adversarial training"


def maybe_get_default_hparams(runner_name: str = None):
    model_name = arg_utils.maybe_get_arg("model_name")
    dataset_name = arg_utils.maybe_get_arg("dataset")
    threat_model = arg_utils.maybe_get_arg("threat_model")
    model_type = arg_utils.maybe_get_arg("model_type")

    if dataset_name not in registered_datasets:
        raise ValueError(
            "Cannot provide default runner hparams for invalid dataset : {}".format(
                dataset_name
            )
        )

    dataset_hparams = registered_datasets[dataset_name].default_dataset_hparams()
    augment_hparams = hparams.AugmentationHparams()
    model_hparams = None
    training_hparams = None
    testing_hparams = None

    if runner_name in ["train", "test"]:
        # for eg. if model_name="cifar10_resnet" then we are not dealing with pretrained or finetuned benchmark models
        assert (
            model_name is not None
        ), "For train and test runner, a single model must be specified via --model_name"

        assert (
            model_type is None
        ), "Model type is currently only an option for multi runners".format(model_type)

        assert (
            threat_model is None
        ), "Threat model is currently only an option for multi runners".format(
            threat_model
        )

        model_hparams = models.registry.get_default_hparams(
            model_name, dataset_name, param_str="model"
        )
        training_hparams = models.registry.get_default_hparams(
            model_name, dataset_name, param_str="training"
        )
        if runner_name == "train":
            return TrainingDesc(
                dataset_hparams, augment_hparams, model_hparams, training_hparams
            )
        if runner_name == "test":
            testing_hparams = hparams.TestingHparams()
            return TestingDesc(
                dataset_hparams,
                augment_hparams,
                model_hparams,
                training_hparams,
                testing_hparams,
            )

    elif runner_name == "multi_test":
        # for multi_test runner,
        # python nonisotropic.py multi_test --dataset=cifar10 --threat_model-Linf --model_type=pretrained

        assert (
            model_type is not None
        ), "For multi_test runner, a valid model type needs to be specified via --model_type"

        threat_model = (
            threat_model or "Linf"
        )  # for now Linf is the default and only threat model

        if model_type == "multitrain":
            assert (
                model_name is not None
            ), "For testing previously multi-trained models via multi_test runner, a single model must be specified via --model_name"

            raise NotImplementedError
        elif model_type == "pretrained" or "finetuned":
            assert (
                threat_model == "Linf"
            ), "For multi_test runner with model_type : {}, currently only Linf threat_model is supported."

            testing_hparams = hparams.TestingHparams()

            selected_model_names = rb_registry[dataset_name][threat_model]
            defaults_list = []
            for model_name in selected_model_names:
                model_hparams = models.registry.get_default_hparams(
                    model_name,
                    runner_name,
                    dataset_name,
                    threat_model,
                    param_str="model",
                )

                if model_type == "finetuned":
                    training_hparams = models.registry.get_default_hparams(
                        model_name,
                        runner_name,
                        dataset_name,
                        threat_model,
                        param_str="training",
                    )
                    test_desc = TestingDesc(
                        dataset_hparams,
                        augment_hparams,
                        model_hparams,
                        training_hparams,
                        testing_hparams,
                    )
                    defaults_list.append(test_desc)
                else:
                    test_desc = TestingDesc(
                        dataset_hparams,
                        model_hparams,
                        testing_hparams,
                    )
                    defaults_list.append(test_desc)
            return defaults_list
    elif runner_name == "multi_train":
        raise NotImplementedError
    else:
        raise ValueError(
            "Cannot supply default hparams for an invalid runner - {}".format(
                runner_name
            )
        )