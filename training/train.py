import typing
import warnings
import torch
from torchvision.transforms import v2


from datasets.base import DataLoader
import datasets.registry

from foundations import hparams
from foundations import paths
from foundations.step import Step
from foundations.desc import TrainingDesc

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
from threat_specification.greedy_subset import load_greedy_subset


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

    if (
        augment_hparams.N_project
        or augment_hparams.N_mixup
        or training_hparams.N_adv_train
    ):
        greedy_subsets = load_greedy_subset(dataset_hparams)

    if augment_hparams.mixup:
        mixup = v2.transforms.MixUp(num_classes=dataset_hparams.num_labels)

    # The training loop.
    for epoch in range(start_step.ep, end_step.ep + 1):
        train_loader.shuffle(
            None if data_order_seed is None else (data_order_seed + epoch)
        )

        for iteration, (examples, labels) in enumerate(train_loader):
            augmentation_flag = True
            # (
            #     torch.rand(1).item() <= dataset_hparams.augmentation_frequency
            # )

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

            examples_list = []
            examples_list.append(examples)

            if augment_hparams.N_mixup and augmentation_flag:
                shuffle_indices = torch.randperm(len(labels))
                perturbed_examples = non_isotropic_projection(
                    examples,
                    labels,
                    examples[shuffle_indices],
                    greedy_subsets,
                    threshold=dataset_hparams.N_threshold,
                    # verbose=(epoch % 5 == 0 and iteration % 25 == 0),
                )
                examples.list(perturbed_examples)
                # examples = torch.cat((examples, perturbed_examples), dim=0)
                # labels = torch.cat((labels, labels), dim=0)

            if augmentation_flag and (
                dataset_hparams.gaussian_augment or augment_hparams.N_project
            ):
                perturbation = torch.zeros_like(examples).to(
                    device=get_platform().torch_device
                )
                perturbation.normal_(
                    dataset_hparams.gaussian_aug_mean,
                    10 * dataset_hparams.gaussian_aug_std,
                )

                if augment_hparams.N_project:
                    perturbed_examples = non_isotropic_projection(
                        examples,
                        labels,
                        perturbation,
                        greedy_subsets,
                        threshold=dataset_hparams.N_threshold,
                        # verbose=(epoch % 5 == 0 and iteration % 25 == 0),
                    )
                    perturbation = perturbed_examples - examples

                perturbation = torch.min(
                    torch.max(perturbation.detach(), -examples), 1 - examples
                )  # clip examples+perturbation to [0,1]

                examples_list.append(examples + perturbation)
                # examples = torch.cat((examples, examples + perturbation), dim=0)
                # labels = torch.cat((labels, labels), dim=0)

            if (
                training_hparams.N_adv_train or training_hparams.adv_train
            ) and epoch >= training_hparams.adv_train_start_epoch:
                attack_fn = get_attack(training_hparams)

                attack_power = training_hparams.adv_train_attack_power

                if training_hparams.N_adv_train:
                    attack_power *= 10

                perturbation = attack_fn(
                    model,
                    examples,
                    labels,
                    attack_power,
                    attack_power / 10,
                    training_hparams.adv_train_attack_iter,
                )
                perturbation[perturbation != perturbation] = 0

                # lets do some journalism to see if its faithful. and it works!
                # if (epoch % 2 == 0) and (iteration == 0):
                #    report_adv(model, examples, labels, perturbation)

                if training_hparams.N_adv_train:
                    perturbed_examples = non_isotropic_projection(
                        examples,
                        labels,
                        examples + perturbation,
                        greedy_subsets,
                        threshold=training_hparams.N_threshold,
                        # verbose=(epoch % 5 == 0 and iteration % 25 == 0),
                    )
                    perturbation = perturbed_examples - examples

                perturbation = torch.min(
                    torch.max(perturbation.detach(), -examples), 1 - examples
                )  # clip examples+perturbation to [0,1]

                examples_list.append(examples + perturbation)
                # examples = torch.cat((examples, examples + perturbation), dim=0)
                # labels = torch.cat((labels, labels), dim=0)

            step_optimizer.zero_grad()
            model.train()

            if examples.isnan().any() and get_platform().is_primary_process:
                print(
                    "Warning!!! Examples have NaN values after augmentation/adversarial training! at epoch {}, iteration {}".format(
                        epoch, iteration
                    )
                )
                return

            loss = 0.0
            for inputs in examples_list:
                loss += model.loss_criterion(model(inputs), labels)

            if augment_hparams.mixup:
                mixup_examples, mixup_labels = mixup(examples, labels)
                loss += model.loss_criterion(model(mixup_examples), mixup_labels)
            # loss = model.loss_criterion(model(examples), labels)

            if loss.isnan().any() and get_platform().is_primary_process:
                print(
                    "Warning!!! Loss is NaN at epoch {}, iteration {}".format(
                        epoch, iteration
                    )
                )
                return

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

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
    if desc.model_hparams.model_type == "pretrained":
        # check if model exists at the expected location.
        return
    else:
        # model_type is either None or finetuned
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
