import os


def checkpoint(root, model_name=None):
    if model_name is not None:
        return os.path.join(root, model_name + ".pt")
    else:
        return os.path.join(root, "checkpoint.pth")


def feedback(root):
    return os.path.join(root, "feedback.pt")


def logger(root):
    return os.path.join(root, "logger")


def mask(root):
    return os.path.join(root, "mask.pth")


def sparsity_report(root):
    return os.path.join(root, "sparsity_report.json")


def model(root, step):
    model_loc = "model_ep{}_it{}.pth".format(step.ep, step.it)
    return os.path.join(root, model_loc)


def ema_model(root, step):
    model_loc = "ema_model_ep{}_it{}.pth".format(step.ep, step.it)
    return os.path.join(root, model_loc)


def dataset(root, dataset_name):
    return os.path.join(root, dataset_name)


def class_partition(root, label, train=True):
    dir_str = "train_" if train else "val_"
    return os.path.join(root, dir_str + "class_partition") + "/" + str(label) + ".pt"


def threat_specification(root, per_label):
    return os.path.join(root, "per_label_" + str(per_label), "anchor_points")


def anchor_points(
    root,
    label,
    first_half=True,
):
    half_str = "first_half_" if first_half else "second_half_"
    return os.path.join(root, half_str + str(label) + ".pt")


def threat_run_path(root, threat_dir, threat_replicate):
    return os.path.join(
        root,
        "threat_specification",
        threat_dir,
        "threat_replicate_" + str(threat_replicate),
    )


def threat_evaluation_path(root, per_label, perturbation_style=None, severity=None):
    out = os.path.join(
        root,
        "per_label_" + str(per_label),
        "threat_evaluation",
    )
    if perturbation_style:
        out = os.path.join(
            out,
            perturbation_style,
        )
        if severity:
            out = os.path.join(
                out,
                str(severity),
            )
    return out


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
