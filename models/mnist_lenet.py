import torch.nn as nn
import torch.nn.functional as F

from foundations import hparams
from training.desc import TrainingDesc
from testing.desc import TestingDesc
from models import base


class Model(base.Model):
    """A LeNet fully-connected model for MNIST"""

    def __init__(self, model_name, plan, initializer, outputs=10):
        super(Model, self).__init__()
        self.model_name = model_name

        self.layers = []
        current_size = 784  # 28 * 28 = number of pixels in MNIST image.

        for size in plan:
            self.layers.append(nn.Linear(current_size, size))
            current_size = size

        for i, layer in enumerate(self.layers):
            self.add_module(f"layer{i}", layer)

        self.fc = nn.Linear(current_size, outputs)
        self.criterion = nn.CrossEntropyLoss()

        self.apply(initializer)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten.

        for i, layer in enumerate(self.layers):
            x = F.relu(self.layers[i](x))

        return self.fc(x)

    @staticmethod
    def is_valid_model_name(model_name, dataset_name=None):
        return (
            dataset_name == "mnist"
            and model_name.startswith("mnist_lenet")
            and len(model_name.split("_")) > 2
            and all([x.isdigit() and int(x) > 0 for x in model_name.split("_")[2:]])
        )

    @staticmethod
    def get_model_from_name(model_name, initializer, dataset_name=None, outputs=None):
        """The name of a model is mnist_lenet_N1[_N2...].

        N1, N2, etc. are the number of neurons in each fully-connected layer excluding the
        output layer (10 neurons by default). A LeNet with 300 neurons in the first hidden layer,
        100 neurons in the second hidden layer, and 10 output neurons is 'mnist_lenet_300_100'.
        """

        outputs = outputs or 10

        if not Model.is_valid_model_name(model_name, dataset_name=dataset_name):
            raise ValueError("Invalid model name: {}".format(model_name))

        plan = [int(n) for n in model_name.split("_")[2:]]
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
            model_name="mnist_lenet_300_100",
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
            optimizer_name="sgd", lr=0.1, training_steps="40ep"
        )
