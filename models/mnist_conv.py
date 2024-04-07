import torch.nn as nn
import torch.nn.functional as F

from foundations import hparams
from training.desc import TrainingDesc
from testing.desc import TestingDesc

from models import base


class Model(base.Model):
    """A convolutional network designed for CIFAR-10."""

    def __init__(self, model_name, plan, initializer, outputs=10):
        super(Model, self).__init__()
        self.model_name = model_name

        k_size = 3

        self.layers = []
        current_channels = 1
        for next_channels in plan:
            self.layers.append(nn.Conv2d(current_channels, next_channels, kernel_size=k_size))
            current_channels = next_channels

        for i, layer in enumerate(self.layers):
            self.add_module(f"convlayer{i}", layer)

        final_feature_dim = current_channels*(28-len(plan)*(k_size-1))*(28-len(plan)*(k_size-1))

        self.fc = nn.Linear(final_feature_dim, outputs)
        self.criterion = nn.CrossEntropyLoss()
        self.apply(initializer)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(self.layers[i](x))

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    @property
    def output_layer_names(self):
        return ['fc.weight', 'fc.bias']

    @staticmethod
    def is_valid_model_name(model_name):
        # length = len(model_name.split('_'))

        return (model_name.startswith('mnist_conv') and
                len(model_name.split('_')) > 2 and
                all([x.isdigit() and int(x) > 0 for x in model_name.split('_')[2:]]))
        #        length > 2 and
        #       [x.isdigit() for x in model_name.split('_')[2:]] == [True]*(length - 2))

    @staticmethod
    def get_model_from_name(model_name, initializer, outputs=10):
        if not Model.is_valid_model_name(model_name):
            raise ValueError('Invalid model name: {}'.format(model_name))

        outputs = outputs or 10
        plan = [int(x) for x in model_name.split('_')[2:]]

        return Model(model_name, plan, initializer, outputs)

    @property
    def loss_criterion(self):
        return self.criterion


    @staticmethod
    def default_hparams(runner_name):
        model_hparams = hparams.ModelHparams(
            model_name='mnist_conv_16_16_32',
            model_init='kaiming_normal',
            batchnorm_init='uniform',
        )

        dataset_hparams = hparams.DatasetHparams(
            dataset_name='mnist',
            batch_size=128
        )

        training_hparams = hparams.TrainingHparams(
            optimizer_name='sgd',
            momentum=0.9,
            milestone_steps='20ep',
            lr=0.001,
            gamma=0.1,
            weight_decay=1e-4,
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

