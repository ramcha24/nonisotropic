import torch.nn as nn
import torch.nn.functional as F

from foundations import hparams
from models import base
from training.desc import TrainingDesc
from testing.desc import TestingDesc


class Model(base.Model):
    """A VGG-style neural network designed for CIFAR-10."""

    class ConvModule(nn.Module):
        """A single convolutional module in a VGG network."""

        def __init__(self, in_filters, out_filters):
            super(Model.ConvModule, self).__init__()
            self.conv = nn.Conv2d(in_filters, out_filters, kernel_size=3, padding=1)
            self.bn = nn.BatchNorm2d(out_filters)

        def forward(self, x):
            return F.relu(self.bn(self.conv(x)))

    def __init__(self, model_name, plan, initializer, outputs=10):
        super(Model, self).__init__()
        self.model_name = model_name

        layers = []
        filters = 3

        for spec in plan:
            if spec == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(Model.ConvModule(filters, spec))
                filters = spec

        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(512, outputs)
        self.criterion = nn.CrossEntropyLoss()

        self.apply(initializer)

    def forward(self, x):
        x = self.layers(x)
        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    @staticmethod
    def is_valid_model_name(model_name, dataset_name):
        return (
            dataset_name == "cifar10"
            and model_name.startswith("cifar10_vgg_")
            and len(model_name.split("_")) == 3
            and model_name.split("_")[2].isdigit()
            and int(model_name.split("_")[2]) in [11, 13, 16, 19]
        )

    @staticmethod
    def get_model_from_name(model_name, initializer, dataset_name=None, outputs=10):
        if not Model.is_valid_model_name(model_name, dataset_name):
            raise ValueError("Invalid model name: {}".format(model_name))

        outputs = outputs or 10

        num = int(model_name.split("_")[2])
        if num == 11:
            plan = [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512]
        elif num == 13:
            plan = [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512]
        elif num == 16:
            plan = [
                64,
                64,
                "M",
                128,
                128,
                "M",
                256,
                256,
                256,
                "M",
                512,
                512,
                512,
                "M",
                512,
                512,
                512,
            ]
        elif num == 19:
            plan = [
                64,
                64,
                "M",
                128,
                128,
                "M",
                256,
                256,
                256,
                256,
                "M",
                512,
                512,
                512,
                512,
                "M",
                512,
                512,
                512,
                512,
            ]
        else:
            raise ValueError("Unknown VGG model: {}".format(model_name))

        return Model(model_name, plan, initializer, outputs)

    @property
    def loss_criterion(self):
        return self.criterion

    @staticmethod
    def default_model_hparams(
        model_name=None, dataset_name=None, threat_model=None, model_type=None
    ):
        assert threat_model is None
        assert model_type is None
        if not Model.is_valid_model_name(model_name, dataset_name):
            raise ValueError("Invalid model name: {}".format(model_name))

        return hparams.ModelHparams(
            model_name="cifar10_vgg_16",
        )

    @staticmethod
    def default_training_hparams(
        model_name=None, dataset_name=None, threat_model=None, model_type=None
    ):
        assert (
            model_type is None
        ), "No default training hparams for invalid model_type : {}".format(model_type)
        assert (
            threat_model is None
        ), "No default training hparams for invalid model_type : {}".format(model_type)

        if not Model.is_valid_model_name(model_name, dataset_name):
            raise ValueError("Invalid model name: {}".format(model_name))

        return hparams.TrainingHparams(
            optimizer_name="sgd",
            momentum=0.9,
            milestone_steps="80ep,120ep",
            lr=0.1,
            gamma=0.1,
            weight_decay=1e-4,
            training_steps="160ep",
        )
