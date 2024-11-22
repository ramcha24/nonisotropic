from functools import partial
from typing import Any, Callable, List, Optional, Type, Union
import torch
import torch.nn as nn
import math
from torch import Tensor
import torchvision
from torch.utils.checkpoint import checkpoint_sequential as checkpoint

from foundations import hparams
from models import base
from training.desc import TrainingDesc
from testing.desc import TestingDesc


def conv3x3(
    in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1
) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv2d_init(m):
    assert isinstance(m, nn.Conv2d)
    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    m.weight.data.normal_(0, math.sqrt(2.0 / n))


def gn_init(m, zero_init=False):
    assert isinstance(m, nn.GroupNorm)
    m.weight.data.fill_(0.0 if zero_init else 1.0)
    m.bias.data.zero_()


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        width: int = 64,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes, eps=1e-5)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes, eps=1e-5)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.GroupNorm(32, planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.GroupNorm(32, planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.GroupNorm(32, planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        gn_init(self.bn1)
        gn_init(self.bn2)
        gn_init(self.bn3, zero_init=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_labels=1000,
        width: int = 64,
        preprocess=False,
        apply_checkpoint=True,
    ):
        """To make it possible to vary the width, we need to override the constructor of the torchvision resnet."""

        torch.nn.Module.__init__(self)  # super(ResNet, self).__init__()
        self.inplanes = 64
        self.dilation = 1
        self.groups = 1
        self.base_width = 64

        # The initial convolutional layer.
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.GroupNorm(32, self.inplanes, eps=1e-5)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # The subsequent blocks.
        self.layer1 = self.make(block, 64, layers[0])
        self.layer2 = self.make(block, 64 * 2, layers[1], stride=2, dilate=False)
        self.layer3 = self.make(block, 64 * 4, layers[2], stride=2, dilate=False)
        self.layer4 = self.make(block, 64 * 8, layers[3], stride=2, dilate=False)

        self.apply_checkpoint = apply_checkpoint

        # The last layers.
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool = nn.AvgPool2d(7, stride=1)

        self.fc = nn.Linear(64 * 8 * block.expansion, num_labels)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv2d_init(m)

        gn_init(self.bn1)
        # # Default init.
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(
        #             m.weight, mode="fan_out", nonlinearity="relu"
        #         )
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #          nn.init.constant_(m.bias, 0)

    def make(
        self,
        block,
        planes,
        blocks,
        dilate=False,
        stride=1,
        in_planes=64,
    ):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.GroupNorm(32, planes * block.expansion, eps=1e-5),
            )
            m = downsample[1]
            assert isinstance(m, nn.GroupNorm)
            gn_init(m)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        if self.apply_checkpoint:
            x = checkpoint(self.layer1, 3, x, use_reentrant=False)
        else:
            x = self.layer1(x)

        if self.apply_checkpoint:
            x = checkpoint(self.layer2, 4, x, use_reentrant=False)
        else:
            x = self.layer2(x)

        if self.apply_checkpoint:
            x = checkpoint(self.layer3, 4, x, use_reentrant=False)
        else:
            x = self.layer3(x)

        if self.apply_checkpoint:
            x = checkpoint(self.layer4, 3, x, use_reentrant=False)
        else:
            x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


class Model(base.Model):
    """A residual neural network as originally designed for ImageNet."""

    def __init__(self, model_name, model_fn, initializer, outputs=None):
        super(Model, self).__init__()
        self.model_name = model_name
        self.model = model_fn(num_labels=outputs or 1000)
        self.criterion = nn.CrossEntropyLoss()
        # self.apply(initializer)

    def forward(self, x):
        return self.model(x)  # self.model(x)

    @staticmethod
    def is_valid_model_name(model_name, dataset_name=None):
        return (
            dataset_name == "imagenet"
            and model_name.startswith("imagenet_resnet_")
            and 4 >= len(model_name.split("_")) >= 3
            and model_name.split("_")[2].isdigit()
            and int(model_name.split("_")[2]) in [18, 34, 50, 101, 152, 200]
        )

    @staticmethod
    def get_model_from_name(model_name, initializer, dataset_name=None, outputs=1000):
        """Name: imagenet_resnet_D[_W].

        D is the model depth (e.g., 50 for ResNet-50). W is the model width - the number of filters in the first
        residual layers. By default, this number is 64."""

        if not Model.is_valid_model_name(model_name, dataset_name):
            raise ValueError("Invalid model name: {}".format(model_name))

        num = int(model_name.split("_")[2])
        if num == 18:
            model_fn = partial(ResNet, BasicBlock, [2, 2, 2, 2])
        elif num == 34:
            model_fn = partial(ResNet, BasicBlock, [3, 4, 6, 3])
        elif num == 50:
            model_fn = partial(ResNet, Bottleneck, [3, 4, 6, 3])
        elif num == 101:
            model_fn = partial(ResNet, Bottleneck, [3, 4, 23, 3])
        elif num == 152:
            model_fn = partial(ResNet, Bottleneck, [3, 8, 36, 3])
        elif num == 200:
            model_fn = partial(ResNet, Bottleneck, [3, 24, 36, 3])
        elif num == 269:
            model_fn = partial(ResNet, Bottleneck, [3, 30, 48, 8])

        if len(model_name.split("_")) == 4:
            width = int(model_name.split("_")[3])
            model_fn = partial(model_fn, width=width)

        return Model(model_name, model_fn, initializer, outputs)

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
            model_name="imagenet_resnet_50",
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
            milestone_steps="30ep,60ep,80ep",
            lr=0.2,
            gamma=0.1,
            weight_decay=1e-4,
            training_steps="90ep",
            warmup_steps="5ep",
            grad_clipping_val=1.0,
            grad_accumulation_steps=4,
            adv_train_attack_iter=1,
            adv_train_attack_power_Linf=32 / 255,
            # N_anchor_chunks=24,
        )
