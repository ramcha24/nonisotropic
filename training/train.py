import typing
import warnings
import torch
from torchvision.transforms import v2
import numpy as np
from torch.autograd import Variable

from datasets.base import DataLoader
import datasets.registry

from foundations import hparams
from foundations import paths
from foundations.step import Step
from training.desc import TrainingDesc

from models.base import Model, DistributedDataParallel
import models.registry

from platforms.platform import get_platform
from training.checkpointing import restore_checkpoint
from training import optimizers
from training import standard_callbacks
from training.metric_logger import MetricLogger

from training.adv_train_util import get_attack
from utilities.evaluation_utils import report_adv

from threat_specification.projected_displacement import non_isotropic_projection
from threat_specification.greedy_subset import load_threat_specification


def mixup_data(x, y, alpha=1.0, use_cuda=True):

    """Compute the mixup data. Return mixed inputs, pairs of targets, and lambda"""
    if alpha > 0.0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).to(get_platform().torch_device)
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(
        pred, y_b
    )


def train(
    dataset_hparams: hparams.DatasetHparams,
    augment_hparams: hparams.AugmentationHparams,
    training_hparams: hparams.TrainingHparams,
    model: Model,
    train_loader: DataLoader,
    output_location: str,
    callbacks: typing.List[typing.Callable] = [],
    start_step: Step = None,
    end_step: Step = None,
):

    """The main training loop for this framework.

    Args:
      * dataset_hparams: The dataset hyperparameters whose schema is specified in hparams.py.
      * augment_hparams: The augmentation hyperparameters whose scheme is specified in hparams.py
      * training_hparams: The training hyperparameters whose schema is specified in hparams.py.
      * model: The model to train. Must be a models.base.Model
      * train_loader: The training data. Must be a datasets.base.DataLoader
      * output_location: The string path where all outputs should be stored.
      * callbacks: A list of functions that are called before each training step and once more
        after the last training step. Each function takes five arguments: the current step,
        the output location, the model, the optimizer, and the logger.
        Callbacks are used for running the test set, saving the logger, saving the state of the
        model, etc. They provide hooks into the training loop for customization so that the
        training loop itself can remain simple.
      * adv_attack: A function that carries out an adversarial attack on a batch of inputs.
      * start_step: The step at which the training data and learning rate schedule should begin.
        Defaults to step 0.
      * end_step: The step at which training should cease. Otherwise, training will go for the
        full `training_hparams.training_steps` steps.
    """
    # create the output location if it doesn't already exist.
    if not get_platform().exists(output_location) and get_platform().is_primary_process:
        get_platform().makedirs(output_location)

    # get the optimizer and learning rate schedule.
    model.to(get_platform().torch_device)
    optimizer = optimizers.get_optimizer(training_hparams, model)
    step_optimizer = optimizer
    lr_schedule = optimizers.get_lr_schedule(
        training_hparams, optimizer, train_loader.iterations_per_epoch
    )

    # Get the random seed for the data order.
    data_order_seed = training_hparams.data_order_seed

    # Restore the model from a saved checkpoint if the checkpoint exists
    cp_step, cp_logger = restore_checkpoint(
        output_location, model, optimizer, train_loader.iterations_per_epoch
    )

    # Handle parallelism if applicable.
    if get_platform().is_distributed:
        model = DistributedDataParallel(model, device_ids=[get_platform().local_rank])

    start_step = cp_step or start_step or Step.zero(train_loader.iterations_per_epoch)
    logger = cp_logger or MetricLogger()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        for _ in range(start_step.iteration):
            lr_schedule.step()

    # Determine when to end training
    end_step = end_step or Step.from_str(
        training_hparams.training_steps, train_loader.iterations_per_epoch
    )
    if end_step <= start_step:
        return

    if augment_hparams.N_aug or training_hparams.N_adv_train:
        threat_specification = load_threat_specification(dataset_hparams)

    if augment_hparams.mixup:
        mixup = v2.MixUp(num_classes=dataset_hparams.num_labels)

    # The training loop.
    for epoch in range(start_step.ep, end_step.ep + 1):
        train_loader.shuffle(
            None if data_order_seed is None else (data_order_seed + epoch)
        )

        for iteration, (examples, labels) in enumerate(train_loader):
            # if iteration % 10 == 0 and get_platform().is_primary_process:
            #     print("Epoch: {}, Iteration: {}".format(epoch, iteration))
            augmentation_flag = (
                # True  #
                torch.rand(1).item()
                <= (1 / augment_hparams.augmentation_frequency) + 1e-2
            )

            # Advance the data loader until the start epoch and iteration
            if epoch == start_step.ep and iteration < start_step.it:
                continue

            # Run the call backs
            step = Step.from_epoch(epoch, iteration, train_loader.iterations_per_epoch)
            for callback in callbacks:
                callback(output_location, step, model, optimizer, logger)

            # Exit at the end step
            if epoch == end_step.ep and iteration == end_step.it:
                return

            # Train!
            examples = examples.to(device=get_platform().torch_device)
            labels = labels.to(device=get_platform().torch_device)

            examples_N_aug = torch.empty((0,), dtype=torch.float32).to(
                device=get_platform().torch_device
            )
            examples_adv_train = torch.empty((0,), dtype=torch.float32).to(
                device=get_platform().torch_device
            )
            examples_N_adv_train = torch.empty((0,), dtype=torch.float32).to(
                device=get_platform().torch_device
            )

            labels_N_aug = torch.empty((0,), dtype=torch.int64).to(
                device=get_platform().torch_device
            )
            labels_adv_train = torch.empty((0,), dtype=torch.int64).to(
                device=get_platform().torch_device
            )
            labels_N_adv_train = torch.empty((0,), dtype=torch.int64).to(
                device=get_platform().torch_device
            )

            if augment_hparams.N_aug and augmentation_flag:
                # mixup augmentation
                shuffle_indices = torch.randperm(len(labels))
                examples_N_aug = non_isotropic_projection(
                    examples,
                    labels,
                    examples[shuffle_indices],
                    threat_specification,
                    threshold=augment_hparams.N_threshold,
                    # verbose=(epoch % 5 == 0 and iteration % 25 == 0),
                )
                labels_N_aug = labels.clone()

                # projection augmentation

                perturbation = torch.zeros_like(examples).to(
                    device=get_platform().torch_device
                )
                perturbation.normal_(
                    augment_hparams.gaussian_aug_mean,
                    10 * augment_hparams.gaussian_aug_std,
                )

                examples_N_aug = torch.cat(
                    (
                        examples_N_aug,
                        non_isotropic_projection(
                            examples,
                            labels,
                            perturbation,
                            threat_specification,
                            threshold=augment_hparams.N_threshold,
                            # verbose=(epoch % 5 == 0 and iteration % 25 == 0),
                        ),
                    ),
                    dim=0,
                )
                labels_N_aug = torch.cat((labels_N_aug, labels), dim=0)

                # perturbation = perturbed_examples - examples

                # perturbation = torch.min(
                #    torch.max(perturbation.detach(), -examples), 1 - examples
                # )  # clip examples+perturbation to [0,1]

                # examples_list.append(examples + perturbation)
                # del perturbed_examples, perturbation

            if (
                training_hparams.adv_train
                and epoch >= training_hparams.adv_train_start_epoch
            ):
                attack_fn, attack_power = get_attack(training_hparams)

                examples_adv_train = examples + attack_fn(
                    model,
                    examples,
                    labels,
                    attack_power,
                    attack_power / 10,
                    training_hparams.adv_train_attack_iter,
                )
                labels_adv_train = labels.clone()
                # perturbation = torch.min(
                #    torch.max(perturbation.detach(), -examples), 1 - examples
                # )  # clip examples+perturbation to [0,1]

                # examples_list.append(examples + perturbation)

                # del perturbation

            if (
                training_hparams.N_adv_train
                and epoch >= training_hparams.adv_train_start_epoch
            ):
                attack_fn, attack_power = get_attack(training_hparams)
                attack_power *= 4
                perturbation = attack_fn(
                    model,
                    examples,
                    labels,
                    attack_power,
                    attack_power / 10,
                    training_hparams.adv_train_attack_iter,
                )

                examples_N_adv_train = non_isotropic_projection(
                    examples,
                    labels,
                    examples + perturbation,
                    threat_specification,
                    threshold=training_hparams.N_threshold,
                    # verbose=(epoch % 5 == 0 and iteration % 25 == 0),
                )
                labels_N_adv_train = labels.clone()
                # perturbation = perturbed_examples - examples
                # perturbation[perturbation != perturbation] = 0

                # perturbation = torch.min(
                #    torch.max(perturbation.detach(), -examples), 1 - examples
                # )  # clip examples+perturbation to [0,1]

                # torch.cat(example, examples + perturbation, dim=0)
                # examples_list.append(examples + perturbation)
                # del perturbation, perturbed_examples
            if augment_hparams.mixup:
                mixup_examples, labels_a, labels_b, lambda_val = mixup_data(
                    examples.clone(), labels.clone()
                )

                # mixup_examples, mixup_labels = mixup(
                #     examples.detach().clone(), labels.detach().clone()
                # )

            examples = torch.cat(
                (examples, examples_N_aug, examples_adv_train, examples_N_adv_train),
                dim=0,
            )
            labels = torch.cat(
                (labels, labels_N_aug, labels_adv_train, labels_N_adv_train), dim=0
            )
            # examples = torch.cat((examples, perturbed_examples), dim=0)
            # labels = torch.cat((labels, labels), dim=0)
            step_optimizer.zero_grad()
            model.train()

            mixup_loss = 0.0
            if augment_hparams.mixup:
                mixup_examples, labels_a, labels_b = (
                    Variable(mixup_examples),
                    Variable(labels_a),
                    Variable(labels_b),
                )
                mixup_outputs = model(mixup_examples)
                loss_func = mixup_criterion(labels_a, labels_b, lambda_val)
                mixup_loss += loss_func(model.loss_criterion, mixup_outputs)

            if examples.isnan().any() and get_platform().is_primary_process:
                print(
                    "Warning!!! Examples have NaN values after augmentation/adversarial training! at epoch {}, iteration {}".format(
                        epoch, iteration
                    )
                )
                return

            loss = model.loss_criterion(model(examples), labels)
            overall_loss = loss + mixup_loss

            # if augment_hparams.mixup:
            #     loss += loss_func(model.loss_criterion, mixup_outputs)

            # loss = 0.0
            # for perturbed_examples in examples_list:
            #     loss += model.loss_criterion(model(perturbed_examples), labels)

            # if augment_hparams.mixup:
            #    loss += model.loss_criterion(model(mixup_examples), mixup_labels)

            # if overall_loss.isnan().any() and get_platform().is_primary_process:
            #     print(
            #         "Warning!!! Loss is NaN at epoch {}, iteration {}".format(
            #             epoch, iteration
            #         )
            #     )
            #     # Consider torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            #     return

            overall_loss.backward()

            # Step forward. Ignore warnings that the lr_schedule generates.
            step_optimizer.step()
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                lr_schedule.step()

    get_platform().barrier()


def standard_train(
    output_location: str,
    desc: TrainingDesc,
    start_step: Step = None,
    verbose: bool = True,
    evaluate_every_epoch: bool = False,
    evaluate_every_few_epoch: int = 10,
):
    """Train using the standard callbacks according to the provided hparams."""
    already_trained = False
    if desc.model_hparams.model_type == "pretrained":
        file_names = get_platform().listdir(output_location)
        for file in file_names:
            if file.endswith(".pt"):
                already_trained = True
    else:
        # If the model file for the end of training already exists in this location, do not train
        iterations_per_epoch = datasets.registry.iterations_per_epoch(
            desc.dataset_hparams
        )
        train_end_step = Step.from_str(
            desc.training_hparams.training_steps, iterations_per_epoch
        )
        if models.registry.exists(
            output_location, train_end_step
        ) and get_platform().exists(paths.logger(output_location)):
            already_trained = True

    if already_trained:
        return

    # need to train a model, get its initialized/pretrained weights
    model = models.registry.get(desc.dataset_hparams.dataset_name, desc.model_hparams)

    train_loader = datasets.registry.get(desc.dataset_hparams, train=True)
    test_loader = datasets.registry.get(desc.dataset_hparams, train=False)

    callbacks = standard_callbacks.standard_callbacks(
        desc.training_hparams,
        train_loader,
        test_loader,
        start_step=start_step,
        verbose=verbose,
        evaluate_every_epoch=evaluate_every_epoch,
        evaluate_every_few_epoch=evaluate_every_few_epoch,
    )

    train(
        desc.dataset_hparams,
        desc.augment_hparams,
        desc.training_hparams,
        model,
        train_loader,
        output_location,
        callbacks,
        start_step=start_step,
    )
