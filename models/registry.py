from foundations import paths
from foundations.hparams import ModelHparams
from foundations.step import Step
from models import mnist_lenet, mnist_conv, cifar_resnet, cifar_vgg, imagenet_resnet
from models import bn_initializers, initializers
from platforms.platform import get_platform

registered_models = [mnist_lenet.Model, mnist_conv.Model, cifar_resnet.Model, cifar_vgg.Model, imagenet_resnet.Model]


def get(model_hparams: ModelHparams, outputs=None):
    """Get the model for the corresponding hyperparameters."""

    # select the initializer.
    if hasattr(initializers, model_hparams.model_init):
        initializer = getattr(initializers, model_hparams.model_init)
    else:
        raise ValueError("No model initializer: {}".format(model_hparams.model_init))

    # select the batchnorm initializer
    if hasattr(bn_initializers, model_hparams.batchnorm_init):
        bn_initializer = getattr(bn_initializers, model_hparams.batchnorm_init)
    else:
        raise ValueError("No initializer: {}".format(model_hparams.model_init))

    # create the overall initializer function
    def init_fn(w):
        initializer(w)
        bn_initializer(w)

    # select the model.
    model = None
    for registered_model in registered_models:
        if registered_model.is_valid_model_name(model_hparams.model_name):
            model = registered_model.get_model_from_name(
                model_hparams.model_name, init_fn, outputs
            )
            break

    if model is None:
        raise ValueError("No such model: {}".format(model_hparams.model_name))

    return model


def load(
    save_location: str, save_step: Step, model_hparams: ModelHparams, outputs=None
):
    state_dict = get_platform().load_model(paths.model(save_location, save_step))
    model = get(model_hparams, outputs)
    model.load_state_dict(state_dict)
    return model


def exists(save_location, save_step):
    return get_platform().exists(paths.model(save_location, save_step))


def get_default_hparams(model_name, runner_name):
    """Get the default hyperparameters for a particular model."""
    print("Model name is : " + str(model_name))
    print("Runner name is : " + str(runner_name))

    for registered_model in registered_models:
        if registered_model.is_valid_model_name(model_name):
            print("Reached here")
            params = registered_model.default_hparams(runner_name)
            params.model_hparams.model_name = model_name
            return params

    raise ValueError("No such models: {}".format(model_name))
