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

    @property
    def output_layer_names(self):
        return ['fc.weight', 'fc.bias']
    
    @staticmethod
    def is_valid_model_name(model_name):
        return (model_name.startswith('mnist_lenet') and
                len(model_name.split('_')) > 2 and
                all([x.isdigit() and int(x) > 0 for x in model_name.split('_')[2:]]))

    @staticmethod
    def get_model_from_name(model_name, initializer, outputs=None):
        """The name of a model is mnist_lenet_N1[_N2...].

        N1, N2, etc. are the number of neurons in each fully-connected layer excluding the
        output layer (10 neurons by default). A LeNet with 300 neurons in the first hidden layer,
        100 neurons in the second hidden layer, and 10 output neurons is 'mnist_lenet_300_100'.
        """

        outputs = outputs or 10
        
        if not Model.is_valid_model_name(model_name):
            raise ValueError('Invalid model name: {}'.format(model_name))

        plan = [int(n) for n in model_name.split('_')[2:]]
        return Model(model_name, plan, initializer, outputs)

    @property
    def loss_criterion(self):
        return self.criterion

    @staticmethod
    def default_hparams(runner_name):
        model_hparams = hparams.ModelHparams(
            model_name='mnist_lenet_300_100',
            model_init='kaiming_normal',
            batchnorm_init='uniform'
        )

        dataset_hparams = hparams.DatasetHparams(
            dataset_name='mnist',
            batch_size=128,
            num_classes=10
        )

        training_hparams = hparams.TrainingHparams(
            optimizer_name='sgd',
            lr=0.1,
            training_steps='40ep'
        )

        testing_hparams = hparams.TestingHparams(
        )

        if runner_name == 'train':
            return TrainingDesc(model_hparams, dataset_hparams, training_hparams)
        elif runner_name == 'test':
            return TestingDesc(model_hparams, dataset_hparams, training_hparams, testing_hparams)
        else:
            raise ValueError("Cannot supply default hparams for an invalid runner - {}".format(runner_name))

