from dataclasses import dataclass, asdict
from typing import Optional

from cli import arg_utils
from foundations import hparams

from datasets.registry import registered_datasets

import models.registry
from models.robustbench_registry import rb_registry

from training.desc import TrainingDesc
from testing.desc import TestingDesc
from threat_specification.desc import ThreatDesc
from models.pretrained.desc import PretrainDesc


@dataclass
class JobArgs(hparams.Hparams):
    """Arguments shared across jobs"""

    dataset_name: str = None
    model_name: str = None
    quiet: bool = False
    evaluate_only_at_end: bool = False
    evaluate_every_few_epoch: int = 2  # 0
    evaluate_only_batch_test: bool = False
    threat_replicate: int = 1
    train_replicate: int = -1
    test_replicate: int = -1
    float_16: bool = False
    in_memory: bool = False

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
    _evaluate_every_few_epoch: str = (
        "Evaluate validation runner every few epochs, default is 10"
    )
    _threat_replicate: str = (
        "The replicate number for threat specification. default is 1"
    )
    _train_replicate: str = (
        "The replicate number for training. -1 means no replicate number is specified."
    )
    _test_replicate: str = (
        "The replicate number for testing. -1 means no replicate number is specified."
    )
    _float_16: str = (
        "Use float16 for training or testing (currently only for threat evaluation)"
    )
    _in_memory: str = "Compute common corruptions of a dataset in memory (rather than loading from disk)"
    # Why should evaluate_only_at_end , evaluate_only_batch_test and evalaute_every_few_epoch be listed here?


@dataclass
class ToggleArgs(hparams.Hparams):
    model_type: str = None
    toggle_N_aug: bool = False
    toggle_mixup: bool = False
    toggle_adv_train: bool = False
    toggle_N_adv_train: bool = False

    _name: str = "Multi runner toggle hyperparameters"
    _description: str = "Toggle options for multi-runners based on boolean sub attributes. These options help decide the set of sub-runners"
    _model_type: str = "Type of model to multi-train/multi-test : None or pretrained or finetuned as applicable"
    _toggle_N_aug: str = "Add non-isotropic augmentations for all sub-runners"
    _toggle_mixup: str = "Add mixup augmentation for all sub-runners"
    _toggle_adv_train: str = "Add adversarial training for all sub-runners"
    _toggle_N_adv_train: str = (
        "Add non-isotropic adversarial training for all sub-runners"
    )
    # resolve consistent naming toggle vs multi


@dataclass
class MultiTestArgs(hparams.Hparams):
    multi_standard_eval: bool = False
    multi_adv_eval: bool = False
    multi_N_adv_eval: bool = False

    _name: str = "Multi test runner hyperparameters"
    _description: str = "Common options for multi-test runner based on boolean sub attributes. These options are applied for evaluation in each test sub-runner"
    _multi_standard_eval: str = (
        "Evaluate models on clean test data for each test sub-runner"
    )
    _multi_robust_eval: str = (
        "Evaluate models on isotropic adversarial attacks for each test sub-runner"
    )
    _multi_N_robust_eval: str = (
        "Evaluate models on nonisotropic adversarial attacks for each test sub-runner"
    )


def maybe_get_default_hparams(runner_name: str = None):
    dataset_name = arg_utils.maybe_get_arg("dataset_name")
    if dataset_name not in registered_datasets:
        raise ValueError(
            "Cannot provide default runner hparams for invalid dataset : {}".format(
                dataset_name
            )
        )

    dataset_hparams = registered_datasets[
        dataset_name
    ].Dataset.default_dataset_hparams()

    model_name = arg_utils.maybe_get_arg("model_name")
    threat_model = arg_utils.maybe_get_arg("threat_model")
    model_type = arg_utils.maybe_get_arg("model_type")

    if runner_name in ["train", "test"]:
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
                dataset_hparams,
                model_hparams,
                hparams.AugmentationHparams(),
                training_hparams,
                hparams.ThreatHparams(),
            )
        if runner_name == "test":
            return TestingDesc(
                dataset_hparams,
                model_hparams,
                hparams.TestingHparams(),
                hparams.AugmentationHparams(),
                training_hparams,
            )

    elif runner_name == "multi_test":
        # for multi_test runner,
        # python nonisotropic.py multi_test --dataset_name=cifar10 --threat_model=Linf --model_type=pretrained

        assert model_type in [
            None,
            "pretrained",
            "finetuned",
        ], "Invalid model type {}".format(model_type)

        threat_model = (
            threat_model or "Linf"
        )  # for now Linf is the default and only threat model

        if model_type is None:
            assert (
                model_name is not None
            ), "For testing previously multi-trained models via multi_test runner, a single model must be specified via --model_name"

            model_hparams = models.registry.get_default_hparams(
                model_name, dataset_name, param_str="model"
            )
            training_hparams = models.registry.get_default_hparams(
                model_name, dataset_name, param_str="training"
            )
            testing_hparams = hparams.TestingHparams()
            return [
                TestingDesc(
                    dataset_hparams,
                    model_hparams,
                    testing_hparams,
                    hparams.AugmentationHparams(),
                    training_hparams,
                )
            ]

        else:  # model_type == "pretrained" or "finetuned"
            assert (
                threat_model == "Linf"
            ), "For multi_test runner with model_type : {}, currently only Linf threat_model is supported."

            testing_hparams = hparams.TestingHparams()

            selected_model_names = rb_registry[dataset_name][threat_model]
            defaults_list = []
            for model_name in selected_model_names:
                model_hparams = models.registry.get_default_hparams(
                    model_name,
                    dataset_name,
                    threat_model,
                    model_type,
                    param_str="model",
                )

                if model_type == "finetuned":
                    training_hparams = models.registry.get_default_hparams(
                        model_name,
                        dataset_name,
                        threat_model,
                        model_type,
                        param_str="training",
                    )
                    test_desc = TestingDesc(
                        dataset_hparams,
                        model_hparams,
                        testing_hparams,
                        hparams.AugmentationHparams(),
                        training_hparams,
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
        # for multi_train runner,
        # python nonisotropic.py multi_test --dataset_name=cifar10 --threat_model-Linf --model_type=pretrained
        # or python nonisotropic.py multi_test --dataset_name=cifar10 --model_name=cifar10_resnet_50 --toggle_N_aug

        assert model_type in [
            None,
            "pretrained",
        ], "Invalid model type {}".format(model_type)

        if model_type is None:
            assert (
                model_name is not None
            ), "For multi training regular models with different configurations, a single model name must be specified via --model_name"

            model_hparams = models.registry.get_default_hparams(
                model_name, dataset_name, param_str="model"
            )
            training_hparams = models.registry.get_default_hparams(
                model_name, dataset_name, param_str="training"
            )
            return [
                TrainingDesc(
                    dataset_hparams,
                    model_hparams,
                    hparams.AugmentationHparams(),
                    training_hparams,
                    hparams.ThreatHparams(),
                )
            ]

        else:  # model_type == "pretrained"
            # for finetuning pretrained models.
            threat_model = (
                threat_model or "Linf"
            )  # for now Linf is the default and only threat model

            assert (
                threat_model == "Linf"
            ), "For multi_test runner with model_type : {}, currently only Linf threat_model is supported."

            selected_model_names = rb_registry[dataset_name][threat_model]
            defaults_list = []
            for model_name in selected_model_names:
                model_hparams = models.registry.get_default_hparams(
                    model_name,
                    dataset_name,
                    threat_model,
                    model_type,
                    param_str="model",
                )
                training_hparams = models.registry.get_default_hparams(
                    model_name,
                    dataset_name,
                    threat_model,
                    model_type,
                    param_str="training",
                )
                if dataset_name != "imagenet":
                    dataset_dict = asdict(dataset_hparams)
                    dataset_dict["batch_size"] = 128
                    dataset_hparams = hparams.DatasetHparams(**dataset_dict)

                # training_dict = asdict(training_hparams)
                # training_dict["N_adv_train"] = True
                # training_hparams = hparams.TrainingHparams(**training_dict)
                # # replace batch_size in dataset_hparams
                train_desc = TrainingDesc(
                    dataset_hparams,
                    model_hparams,
                    hparams.AugmentationHparams(),
                    training_hparams,
                    hparams.ThreatHparams(),
                )
                defaults_list.append(train_desc)
            return defaults_list
    elif runner_name == "compute_threat":
        threat_hparams = hparams.ThreatHparams()
        return ThreatDesc(dataset_hparams, threat_hparams)
    elif runner_name == "evaluate_threat":
        threat_hparams = hparams.ThreatHparams()
        perturbation_hparams = hparams.PerturbationHparams()
        return ThreatDesc(dataset_hparams, threat_hparams, perturbation_hparams)
    elif runner_name == "download_pretrained":
        return PretrainDesc(dataset_hparams)
    else:
        raise ValueError(
            "Cannot supply default hparams for an invalid runner - {}".format(
                runner_name
            )
        )
