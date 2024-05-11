from foundations import paths
from foundations.hparams import ModelHparams
from foundations.step import Step
from models import (
    cifar10_resnet,
    mnist_lenet,
    mnist_conv,
    cifar10_vgg,
    imagenet_resnet,
    robustbench,
)
from models import bn_initializers, initializers
from platforms.platform import get_platform
from datasets.registry import registered_datasets

registered_models = [
    mnist_lenet.Model,
    mnist_conv.Model,
    cifar10_resnet.Model,
    cifar10_vgg.Model,
    imagenet_resnet.Model,
]


def get(model_hparams: ModelHparams, outputs=None):
    """Get the model for the corresponding hyperparameters."""

    # select the initializer.
    model_init = model_hparams.model_init
    batchnorm_init = model_hparams.batchnorm_init

    if model_init is not None and hasattr(initializers, model_init):
        initializer = getattr(initializers, model_init)
    else:
        raise ValueError("No model initializer: {}".format(model_init))

    # select the batchnorm initializer
    if batchnorm_init is not None and hasattr(bn_initializers, batchnorm_init):
        bn_initializer = getattr(bn_initializers, batchnorm_init)
    else:
        raise ValueError("No initializer: {}".format(model_init))

    # create the overall initializer function
    def init_fn(w):
        initializer(w)
        bn_initializer(w)

    # select the model.
    model = None
    for registered_model in registered_models:
        if registered_model.is_valid_model_name(model_hparams.model_name):
            model = registered_model.get_model_from_name(
                model_hparams.model_name,
                init_fn
                if ((model_init is not None) and (batchnorm_init is not None))
                else None,
                outputs,
            )
            break

    if model is None:
        raise ValueError("No such model: {}".format(model_hparams.model_name))

    return model


# def load(
#     save_location: str, save_step: Step, model_hparams: ModelHparams, outputs=None
# ):
#     state_dict = get_platform().load_model(paths.model(save_location, save_step))
#     model = get(model_hparams, outputs)
#     model.load_state_dict(state_dict)
#     return model


def exists(save_location, save_step):
    return get_platform().exists(paths.model(save_location, save_step))


def get_default_hparams(
    model_name, dataset_name, threat_model=None, model_type=None, param_str=None
):
    assert param_str in ["model", "training"]
    assert (
        model_name is not None
    ), "Cannot get model hparams without knowing the model name"
    """Get the default hyperparameters for a particular model."""
    if threat_model is None:
        assert model_type is None
        # retrieve default model hyper parameters for a regular model
        for registered_model in registered_models:
            if registered_model.is_valid_model_name(model_name, dataset_name):
                if param_str == "model":
                    params = registered_model.default_model_hparams(
                        model_name, dataset_name
                    )
                elif param_str == "training":
                    params = registered_model.default_training_hparams(
                        model_name, dataset_name
                    )
                return params
    elif threat_model in ["Linf", "L2"]:
        assert model_type in ["pretrained", "finetuned"]
        # the presence of threat model indicates robust benchmark
        if robustbench.Model.is_valid_model_name(
            model_name, dataset_name, threat_model
        ):
            if param_str == "model":
                params = registered_model.default_model_hparams(
                    model_name,
                    dataset_name,
                    threat_model,
                    model_type,
                )
            elif param_str == "training":
                params = registered_model.default_training_hparams(
                    model_name,
                    dataset_name,
                    threat_model,
                    model_type,
                )
            return params
    else:
        raise ValueError(
            "Cannot provide {}_hparams for invalid threat model : {}".format(
                param_str, threat_model
            )
        )

    raise ValueError("No such models: {}".format(model_name))
