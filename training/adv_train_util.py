import torch
import torch.nn as nn


def get_attack(training_hparams):
    adv_attack = None
    if training_hparams.adv_train_attack_type == "PGD":
        if training_hparams.adv_train_attack_norm == "L2":
            adv_attack = projected_gradient_descent_2
            attack_power = training_hparams.adv_train_attack_power_L2
        if training_hparams.adv_train_attack_norm == "Linf":
            adv_attack = projected_gradient_descent_linf
            attack_power = training_hparams.adv_train_attack_power_Linf
    return adv_attack, attack_power


def batch_norms(examples):
    """Compute norms over all but the first dimension"""
    return examples.view(examples.shape[0], -1).norm(dim=1)[:, None, None, None]


def projected_gradient_descent_2(model, examples, labels, epsilon, alpha, num_iter):
    random_pert = torch.zeros_like(examples).normal_(0.0, 0.1)
    examples += random_pert
    delta = torch.zeros_like(examples, requires_grad=True)
    # print('running pgd attack for {}'.format(num_iter))
    for t in range(num_iter):
        loss = nn.CrossEntropyLoss()(model(examples + delta), labels)
        loss.backward()
        delta.data += alpha * delta.grad.detach() / batch_norms(delta.grad.detach())
        delta.data = torch.min(
            torch.max(delta.detach(), -examples), 1 - examples
        )  # clip X+delta to [0,1]
        delta.data *= epsilon / batch_norms(delta.detach()).clamp(min=epsilon)
        delta.grad.zero_()

    return delta.detach()


def projected_gradient_descent_linf(model, examples, labels, epsilon, alpha, num_iter):
    """Construct FGSM adversarial examples on the examples X"""
    random_pert = torch.zeros_like(examples).normal_(0.0, 0.1)
    examples += random_pert
    delta = torch.zeros_like(examples, requires_grad=True)
    for t in range(num_iter):
        loss = nn.CrossEntropyLoss()(model(examples + delta), labels)
        loss.backward()
        delta.data = (delta + alpha * delta.grad.detach().sign()).clamp(
            -epsilon, epsilon
        )
        delta.grad.zero_()

    return delta.detach()
