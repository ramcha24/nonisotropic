import torch


def binary(w):
    if isinstance(w, torch.nn.Linear) or isinstance(w, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(w.weight)
        sigma = w.weight.data.std()
        w.weight.data = torch.sign(w.weight.data) * sigma


def kaiming_normal(w):
    if isinstance(w, torch.nn.Linear) or isinstance(w, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(w.weight)


def kaiming_uniform(w):
    if isinstance(w, torch.nn.Linear) or isinstance(w, torch.nn.Conv2d):
        torch.nn.init.kaiming_uniform_(w.weight)


def orthogonal(w):
    if isinstance(w, torch.nn.Linear) or isinstance(w, torch.nn.Conv2d):
        torch.nn.init.orthogonal_(w.weight)


def dummy(w):
    pass
