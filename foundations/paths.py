import os


def checkpoint(root, model_name=None):
    if model_name is not None:
        return os.path.join(root, model_name + ".pt")
    else:
        return os.path.join(root, "checkpoint.pth")


def logger(root):
    return os.path.join(root, "logger")


def mask(root):
    return os.path.join(root, "mask.pth")


def sparsity_report(root):
    return os.path.join(root, "sparsity_report.json")


def model(root, step):
    model_loc = "model_ep{}_it{}.pth".format(step.ep, step.it)
    return os.path.join(root, model_loc)


def plot_save(root, stat_str, layer_index):
    os.makedirs(root, exist_ok=True)
    if layer_index is not None:
        return os.path.join(root, stat_str + "_layer{}.pdf".format(layer_index))
    else:
        return os.path.join(root, stat_str)


def hparams(root):
    return os.path.join(root, "hparams.log")


def params_loc(root, type_str):
    return os.path.join(root, type_str + "_hparams.log")
