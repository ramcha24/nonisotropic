import torch
import datetime
import builtins

from platforms.platform import get_platform


def sanity_check(args, resolve_val=None):
    if args is not None:
        for key in args.keys():
            if torch.is_tensor(args[key]) and args[key].isnan().any():
                if resolve_val is not None:
                    try:
                        temp = args[key]
                        temp[temp != temp] = resolve_val
                        args[key] = temp
                    except Exception as e:
                        print(e)
                else:
                    raise ValueError("argument {} has NaN values".format(key))
    return args


def normalize_batch(x, val, ord=2):
    if ord not in [2, float("inf")]:
        raise ValueError("Only L2 and Linf norms are supported")
    if val is None:
        raise ValueError("Provide a float value for val argument")
    if x is None:
        raise ValueError("Provide a batch to normalize")

    x_flattened = x.view(x.shape[0], -1)  # Flatten each vector to [B, C*H*W]
    x_norms = torch.linalg.norm(x_flattened, dim=1, ord=ord)  # Shape: [B, 1]
    # Avoid division by zero by clamping small norms to a minimum value
    x_norms = torch.clamp(x_norms, min=1e-8)
    x_norms = x_norms.unsqueeze(1)

    # Normalize each vector by its norm and multiply by the desired norm
    x_normalized = (
        x_flattened / x_norms
    )  # T.div(x_flattened, x_norms) # Normalize to unit norm
    x_normalized *= val  # Scale to the desired norm

    # Reshape the normalized vectors back to the original shape [B, C, H, W]
    x_normalized = x_normalized.view(x.shape)
    return x_normalized


def timeprint(*args, **kwargs):
    condition = kwargs.pop("condition", True)
    if condition and get_platform().is_primary_process:
        timestamp = datetime.datetime.now().strftime("[%I:%M %p, %B %d, %Y]: \t")
        builtins.print(timestamp, *args, **kwargs)


def _cast(_tensor, _dtype=torch.float32):
    if _tensor.dtype != _dtype:
        return _tensor.to(_dtype)
    else:
        return _tensor


def _move(_tensor, _device):
    if _tensor.device != _device:
        return _tensor.to(_device)
    else:
        return _tensor
