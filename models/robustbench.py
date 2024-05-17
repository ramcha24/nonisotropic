import os
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from foundations import hparams
from models import base
from models.robustbench_registry import rb_registry, default_rb_registry
from models.utils import load_model
from training.desc import TrainingDesc
from testing.desc import TestingDesc
from platforms.platform import get_platform


def invert_normalization(dataset_name):
    if dataset_name == "cifar10":
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
    elif dataset_name == "cifar100":
        mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
        std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
    elif dataset_name == "imagenet":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

    return torchvision.transforms.Normalize(
        mean=[-m / s for m, s in zip(mean, std)], std=[1 / s for s in std]
    )


class Model(base.Model):
    """A wrapper for all pretrained robustbenchmark models for Cifar10 dataset."""

    def __init__(self, dataset_name, model_name, pretrained_model, threat_model):
        super(Model, self).__init__()
        self.model_name = model_name
        self.pretrained_model = pretrained_model
        self.threat_model = threat_model
        self.criterion = nn.CrossEntropyLoss()
        self.invert_normalization = invert_normalization(dataset_name)

    def forward(self, x):
        x = self.invert_normalization(x)
        return self.pretrained_model(x)

    @staticmethod
    def is_valid_model_name(model_name, dataset_name, threat_model):
        assert (
            threat_model == "Linf"
        ), "Only Linf threat specification is allowed for pretrained models."
        assert dataset_name in ["cifar10", "cifar100", "imagenet"]

        return model_name in rb_registry[dataset_name][threat_model]

    @staticmethod
    def get_model_from_name(
        model_name,
        dataset_name,
        threat_model,
        outputs=10,
        initializer=None,
    ):
        assert initializer == None
        assert (
            threat_model == "Linf"
        ), "Only Linf threat specification is allowed for pretrained models."
        assert dataset_name in ["cifar10", "cifar100", "imagenet"]

        model_dir = os.path.join(
            get_platform().runner_root,
            dataset_name,
            "pretrained",
            "robustbenchmark",
            "Linf",
        )

        pretrained_model = load_model(
            model_name=model_name,
            dataset=dataset_name,
            model_dir=model_dir,
            threat_model=threat_model,
        )
        return Model(dataset_name, model_name, pretrained_model, threat_model)

    @property
    def loss_criterion(self):
        return self.criterion

    @staticmethod
    def default_model_hparams(
        model_name=None, dataset_name=None, threat_model=None, model_type=None
    ):
        assert (
            threat_model == "Linf"
        ), "Only Linf threat specification is allowed for pretrained models."
        assert dataset_name in ["cifar10", "cifar100", "imagenet"]

        return hparams.ModelHparams(
            model_name=default_rb_registry[dataset_name][threat_model]
            if model_name is None
            else model_name,
            model_type=model_type,
            model_source="robustbenchmark",
            threat_model=threat_model,
            model_init=None,
            batchnorm_init=None,
        )

    @staticmethod
    def default_training_hparams(
        model_name=None, dataset_name=None, threat_model=None, model_type=None
    ):
        if model_type in ["pretrained", "finetuned"]:
            return hparams.TrainingHparams(
                optimizer_name="sgd",
                momentum=0.9,
                milestone_steps="10ep",
                lr=0.01,
                gamma=0.1,
                weight_decay=1e-4,
                training_steps="20ep",
            )
        else:
            raise ValueError(
                "No default training hparams for invalid model_type : {}".format(
                    model_type
                )
            )
