import torch
import numpy as np
import matplotlib.pyplot as plt

from platforms.platform import get_platform
from foundations import paths


def correct(labels, output):
    return torch.sum(torch.eq(labels, output.argmax(dim=1)))


def get_pointwise_margin(output, label):
    # assuming x and y are for one point in the batch
    return output[label] - torch.max(output[torch.arange(len(output)) != label.item()])


def get_soft_margin(model, batch_x, batch_y, Mrgs, Accs, Actual_Mrgs):
    batch_output = model(batch_x)
    num_examples = torch.tensor(len(batch_y), device=get_platform().torch_device).item()
    for index in range(0, num_examples):
        out = batch_output[index, :]
        Actual_Mrgs[index] = get_pointwise_margin(out, batch_y[index])

    for j in range(len(Accs)):
        margin_threshold = Mrgs[j]
        Accs[j] = torch.sum(Actual_Mrgs >= margin_threshold) / num_examples


def report_adv(model, examples, labels, delta):
    model.eval()
    i = 41
    x = examples[i:i + 1, :, :, :]
    true_y = labels[i:i + 1]
    pert = delta[i:i + 1, :, :, :]
    std_out = model(x)
    adv_out = model(x + pert)
    std_y = std_out.argmax(dim=1)
    adv_y = adv_out.argmax(dim=1)
    std_loss = model.loss_criterion(std_out, true_y)
    adv_loss = model.loss_criterion(adv_out, true_y)
    print('shape of std out is {}'.format(std_out.shape))
    std_margin = std_out[0, true_y] - torch.max(std_out[0][torch.arange(10) != true_y.item()])
    adv_margin = adv_out[0, true_y] - torch.max(adv_out[0][torch.arange(10) != true_y.item()])
    print('norm of initial input is {}'.format(torch.norm(x)))
    print('norm of delta is {}'.format(torch.norm(pert)))
    print('true label is {}'.format(true_y))
    print('predicted label before attack is {} with loss {} at margin {}'.format(std_y.item(), std_loss,
                                                                                 std_margin.item()))
    print('predicted label after attack is {} with loss {} at margin {}'.format(adv_y.item(), adv_loss,
                                                                                adv_margin.item()))
