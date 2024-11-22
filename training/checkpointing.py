import torch

from foundations import paths
from foundations.step import Step

from platforms.platform import get_platform
from training.metric_logger import MetricLogger


def save_checkpoint_callback(
    output_location, step, model, optimizer, logger, scaler=None, ema_model=None
):
    if get_platform().is_primary_process:
        get_platform().save_model(
            {
                "epoch": step.ep,
                "iteration": step.it,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict() if scaler else None,
                "ema_model_state_dict": ema_model.state_dict() if ema_model else None,
                "logger": str(logger),
            },
            paths.checkpoint(output_location),
        )
    get_platform().barrier()


def restore_checkpoint(
    output_location, model, optimizer, scaler, ema_model, iterations_per_epoch
):
    checkpoint_location = paths.checkpoint(output_location)
    if not get_platform().exists(checkpoint_location):
        return None, None

    checkpoint = get_platform().load_model(
        checkpoint_location, map_location=torch.device("cpu")
    )

    # Handle DataParallel.
    module_in_name = get_platform().is_parallel
    if module_in_name and not all(
        k.startswith("module.") for k in checkpoint["model_state_dict"]
    ):
        checkpoint["model_state_dict"] = {
            "module." + k: v for k, v in checkpoint["model_state_dict"].items()
        }
    elif (
        all(k.startswith("module.") for k in checkpoint["model_state_dict"])
        and not module_in_name
    ):
        checkpoint["model_state_dict"] = {
            k[len("module.") :]: v for k, v in checkpoint["model_state_dict"].items()
        }

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if checkpoint["scaler_state_dict"]:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    if checkpoint["ema_model_state_dict"]:
        ema_model.load_state_dict(checkpoint["ema_model_state_dict"])
    step = Step.from_epoch(
        checkpoint["epoch"], checkpoint["iteration"], iterations_per_epoch
    )
    logger = MetricLogger.create_from_string(checkpoint["logger"])

    return step, logger
