import typing
import warnings
import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn, update_bn
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

from threat_specification.projected_displacement_old import non_isotropic_projection
from threat_specification.subset_selection import load_threat_specification

from threat_specification.projected_displacement import ProjectedDisplacement

from utilities.miscellaneous import timeprint, _cast, _move


def filter_invalid(inputs, labels, range_min=0, range_max=None, target_range=None):
    # Type-checking
    if range_max and not isinstance(range_max, int):
        raise TypeError("range_max if provided must be an integer")
    if target_range and not isinstance(target_range, list):
        raise TypeError("target_labels if provided must be a list")

    # Initialize a default mask to include all elements
    mask = torch.ones(labels.shape, dtype=torch.bool)

    # Create masks based on conditions
    if range_max:
        range_mask = (labels >= range_min) & (labels < range_max)
        mask = mask & range_mask

    if target_range:
        target_mask = torch.isin(labels, torch.tensor(target_labels))
        mask = mask & target_mask

    # valid_indices = []
    # for i in range(len(labels)):
    #     if labels[i] >= 0 and labels[i] < num_labels:
    #         valid_indices.append(i)
    # return inputs[valid_indices], labels[valid_indices]

    return inputs[mask], labels[mask]


def train(
    dataset_hparams: hparams.DatasetHparams,
    augment_hparams: hparams.AugmentationHparams,
    training_hparams: hparams.TrainingHparams,
    threat_hparams: hparams.ThreatHparams,
    model: Model,
    train_loader: DataLoader,
    output_location: str,
    callbacks: typing.List[typing.Callable] = [],
    start_step: Step = None,
    end_step: Step = None,
):

    """
    The main training loop for this framework.

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

    # Prepare the model
    model.to(get_platform().torch_device)
    model.train()
    ema_model = (
        AveragedModel(
            model, multi_avg_fn=get_ema_multi_avg_fn(training_hparams.ema_decay)
        )
        if training_hparams.ema
        else None
    )

    # Prepare the optimizer
    optimizer = optimizers.get_optimizer(training_hparams, model)
    step_optimizer = optimizer
    lr_schedule = optimizers.get_lr_schedule(
        training_hparams, optimizer, train_loader.iterations_per_epoch
    )

    # Get the random seed for the data order.
    data_order_seed = training_hparams.data_order_seed

    # Prepare for automatic mixed precision training
    scaler = torch.GradScaler("cuda", enabled=training_hparams.use_amp)
    cast_type = torch.float16 if training_hparams.float_16 else torch.float32

    # Restore training from a saved checkpoint if the checkpoint exists
    cp_step, cp_logger = restore_checkpoint(
        output_location,
        model,
        optimizer,
        scaler,
        ema_model,
        train_loader.iterations_per_epoch,
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
        threat_model = ProjectedDisplacement(dataset_hparams, threat_hparams)
        threat_model.prepare(num_devices=torch.cuda.device_count())

    if augment_hparams.N_aug or augment_hparams.mixup:
        raise ValueError(
            "Data augmentation is not supported in this version of the code"
        )

    if training_hparams.adv_train or training_hparams.N_adv_train:
        attack_fn, attack_power, attack_step, attack_iters = get_attack(
            training_hparams
        )

    torch.set_float32_matmul_precision("medium")
    # torch.autograd.set_detect_anomaly(True, check_nan=True)

    # The training loop.
    for epoch in range(start_step.ep, end_step.ep + 1):
        # Advance the data loader until the start epoch
        if epoch < start_step.ep:
            continue

        train_loader.shuffle(
            None if data_order_seed is None else (data_order_seed + epoch)
        )

        for iteration, (examples, labels) in enumerate(train_loader):
            # Advance the data loader until start step iteration of the start step epoch
            if epoch == start_step.ep and iteration < start_step.it:
                continue

            # Run the call backs
            step = Step.from_epoch(epoch, iteration, train_loader.iterations_per_epoch)
            for callback in callbacks:
                callback(
                    output_location, step, model, optimizer, logger, scaler, ema_model
                )

            # Exit if at the end step
            if epoch == end_step.ep and iteration == end_step.it:
                return

            # Okay looks like we do have to train ... let's do it

            # Filter out invalid labels
            examples, labels = filter_invalid(examples, labels)

            # timeprint(
            #     "Epoch: {}, Iteration: {}".format(epoch, iteration),
            #     condition=iteration % 500 == 0,
            # )

            augmentation_flag = False  # True # (torch.rand(1).item() <= (1 / augment_hparams.augmentation_frequency) + 1e-2)

            # Run the forward pass with automatic mixed precision if required.
            with torch.autograd.detect_anomaly(check_nan=True):
                with torch.autocast(
                    device_type="cuda",  # get_platform().torch_device,
                    dtype=cast_type,
                    enabled=training_hparams.use_amp,
                ):
                    if examples.dtype != cast_type:
                        examples = _cast(examples, cast_type)

                    # To do. right a function to handle moving to devices that includes the if check.
                    examples = examples.clone().detach()
                    labels = labels.clone().detach()
                    examples_clone = examples.clone().detach()  # detach().clone()
                    labels_clone = labels.clone().detach()  # detach().clone()

                    examples = _move(examples, get_platform().torch_device)
                    labels = _move(labels, get_platform().torch_device)
                    examples_clone = _move(examples_clone, get_platform().torch_device)
                    labels_clone = _move(labels_clone, get_platform().torch_device)

                    if torch.isnan(examples).any():
                        raise ValueError(f"NaN in {name}")
                    print("examples are okay")
                    print(f"Examples average {examples.mean()}")
                    print(f"Examples std {examples.std()}")
                    print(f"Examples max {examples.max()}")
                    print(f"Examples min {examples.min()}")
                    if torch.isnan(labels).any():
                        raise ValueError(f"NaN in {name}")
                    print("labels are okay")

                    loss = (
                        model.loss_criterion(model(examples), labels)
                        / training_hparams.grad_accumulation_steps
                    )
                    for name, param in model.named_parameters():
                        if torch.isnan(param).any():
                            raise ValueError(f"NaN in {name}")
                        if torch.isinf(param).any():
                            raise ValueError(f"Inf in {name}")
                        if param.grad is not None and torch.isnan(param.grad).any():
                            raise ValueError(f"NaN gradient in {name}")
                    print("model parameters are okay")
                    output = model(examples)
                    if torch.isnan(output).any():
                        raise ValueError(f"NaN in output")
                    print("output is okay")

                    print(model.loss_criterion(output, labels))
                    print(loss)
                    if torch.isnan(loss).any():
                        raise ValueError(f"NaN in loss")
                    print("loss is okay")
                    # timeprint(examples.requires_grad, examples.grad_fn)
                    # timeprint("how is loss looking like?")
                    # timeprint(loss.requires_grad, loss.grad_fn)

                    if augmentation_flag and (
                        augment_hparams.N_aug or augment_hparams.mixup
                    ):
                        pass  # no-op right now

                    # epoch >= training_hparams.adv_train_start_epoch and
                    if training_hparams.adv_train or training_hparams.N_adv_train:
                        perturbed_examples = examples + attack_fn(
                            model,
                            examples_clone,
                            labels_clone,
                            attack_power
                            * (examples_clone.max() - examples_clone.min()),
                            attack_step * (examples_clone.max() - examples_clone.min()),
                            attack_iters,
                        )

                        if training_hparams.N_adv_train:
                            perturbed_examples = threat_model.project(
                                examples_clone,
                                labels_clone,
                                perturbed_examples,
                                threshold=training_hparams.N_threshold,
                                gray_scale=not training_hparams.N_multi_channel,
                                lazy_project=not threat_hparams.exact_project,
                            )
                        # timeprint("how are perturbed examples looking like?")
                        # timeprint(
                        #    perturbed_examples.requires_grad, perturbed_examples.grad_fn
                        # )
                        loss_adv = (
                            1.0
                            * model.loss_criterion(
                                model(perturbed_examples), labels_clone
                            )
                        ) / training_hparams.grad_accumulation_steps
                        # timeprint("how is adv loss looking like?")
                        # timeprint(loss_adv.requires_grad, loss_adv.grad_fn)

                        # del examples_clone, labels_clone

                for name, param in model.named_parameters():
                    if torch.isnan(param).any():
                        print(f"NaN in {name}")
                    if torch.isinf(param).any():
                        print(f"Inf in {name}")
                    if param.grad is not None and torch.isnan(param.grad).any():
                        print(f"NaN gradient in {name}")

                # loss.backward()
                scaler.scale(loss).backward()
                if training_hparams.adv_train or training_hparams.N_adv_train:
                    # print("backward for adv loss")
                    scaler.scale(loss_adv).backward()

                # Step forward at the frequency specified by gradient accumulation.
                if (
                    (iteration + 1) % training_hparams.grad_accumulation_steps == 0
                ) or (iteration + 1 == len(train_loader)):

                    # Gradient Value Clipping if needed
                    scaler.unscale_(step_optimizer)
                    # torch.nn.utils.clip_grad_norm_(
                    #     model.parameters(), max_norm=1e2, norm_type=2
                    # )
                    # Gradient Value Clipping
                    torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)

                    scaler.step(step_optimizer)
                    if training_hparams.ema:
                        ema_model.update_parameters(model)
                    scaler.update()

                    # Ignore warnings that the lr_schedule generates.
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=UserWarning)
                        lr_schedule.step()

                    step_optimizer.zero_grad()

    # get_platform().barrier()
    timeprint("Training complete at last.")


def standard_train(
    output_location: str,
    desc: TrainingDesc,
    start_step: Step = None,
    verbose: bool = True,
    evaluate_every_epoch: bool = False,
    evaluate_every_few_epoch: int = 5,
):
    """Train using the standard callbacks according to the provided hparams."""
    if desc.training_hparams.adv_train and desc.training_hparams.N_adv_train:
        timeprint("Will not do both adv_train and N_adv_train at the same time.")
        return

    if desc.augment_hparams.N_aug or desc.augment_hparams.mixup:
        timeprint("Data augmentation is not supported in this version of the code.")
        return

    # Don't reinvent the wheel.
    already_trained = False
    if get_platform().exists(output_location):
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
        timeprint("Training model already exists. Skipping.")
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
        desc.threat_hparams,
        model,
        train_loader,
        output_location,
        callbacks,
        start_step=start_step,
    )
