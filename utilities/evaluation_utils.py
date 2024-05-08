import torch
import math
import numpy as np
import matplotlib.pyplot as plt

from platforms.platform import get_platform
from foundations import paths


def compute_prob(data, steps=100, start=0.001, end=1, tail=True, log_scale=False):
    max_val = torch.max(data)
    min_val = torch.min(data)
    num = torch.numel(data)

    if log_scale:
        threshold = torch.logspace(math.log10(start), math.log10(end), steps)
    else:
        threshold = torch.linspace(start, end, steps)

    threshold = min_val + threshold * (max_val - min_val)

    prob = torch.zeros(steps, device="cpu")

    for i in range(steps):
        if tail:
            prob[i] = torch.sum(data > threshold[i]) / num
        else:
            prob[i] = torch.sum(data <= threshold[i]) / num

    return prob, threshold


def correct(labels, output):
    return torch.sum(torch.eq(labels, output.argmax(dim=1)))


def get_pointwise_margin(outputs, label):
    return outputs[label] - torch.max(
        outputs[torch.arange(len(outputs)) != label.item()]
    )


def get_soft_margin(model, examples, labels, max_margin):
    num_bins = 100
    num_examples = torch.tensor(len(labels), device=get_platform().torch_device).item()

    accuracies = torch.zeros(num_bins)
    actual_margins = torch.zeros(num_examples)

    margin_thresholds = torch.linspace(0, max_margin, steps=num_bins)

    outputs = model(examples)

    for index in range(0, num_examples):
        actual_margins[index] = get_pointwise_margin(outputs[index, :], labels[index])

    for j in range(0, num_examples):
        accuracies[j] = torch.sum(actual_margins >= margin_thresholds[j]) / num_examples

    return accuracies, actual_margins


def report_adv(model, examples, labels, delta):
    model.eval()
    i = 41
    x = examples[i : i + 1, :, :, :]
    true_y = labels[i : i + 1]
    pert = delta[i : i + 1, :, :, :]
    std_out = model(x)
    adv_out = model(x + pert)
    std_y = std_out.argmax(dim=1)
    adv_y = adv_out.argmax(dim=1)
    std_loss = model.loss_criterion(std_out, true_y)
    adv_loss = model.loss_criterion(adv_out, true_y)
    print("shape of std out is {}".format(std_out.shape))
    std_margin = std_out[0, true_y] - torch.max(
        std_out[0][torch.arange(10) != true_y.item()]
    )
    adv_margin = adv_out[0, true_y] - torch.max(
        adv_out[0][torch.arange(10) != true_y.item()]
    )
    print("norm of initial input is {}".format(torch.norm(x)))
    print("norm of delta is {}".format(torch.norm(pert)))
    print("true label is {}".format(true_y))
    print(
        "predicted label before attack is {} with loss {} at margin {}".format(
            std_y.item(), std_loss, std_margin.item()
        )
    )
    print(
        "predicted label after attack is {} with loss {} at margin {}".format(
            adv_y.item(), adv_loss, adv_margin.item()
        )
    )
