import typing
import warnings
import torch

from datasets.base import DataLoader
import datasets.registry
from datasets.greedy_subset import load_greedy_subset

from foundations import hparams
from foundations import paths
from foundations.step import Step

from models.base import Model, DataParallel, DistributedDataParallel
import models.registry

from platforms.platform import get_platform
from training.checkpointing import restore_checkpoint
from training import optimizers
from training import standard_callbacks
from training.metric_logger import MetricLogger

from attacks.adv_train_util import get_attack
from attacks.projected_displacement import non_isotropic_projection
from utilities.capacity_utils import get_classifier_constant, get_feature_reg
from utilities.evaluation_utils import report_adv


def train(
    dataset_hparams: hparams.DatasetHparams,
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

    # Handle parallelism if applicable.
    if get_platform().is_distributed:
        model = DistributedDataParallel(model, device_ids=[get_platform().local_rank])
        print("In distributed!, rank id is " + str(get_platform().local_rank))
    elif get_platform().is_parallel:
        model = DataParallel(model)

    eta = 0.1  # regularization constant for orthogonal layer weights
    # Get the random seed for the data order.
    data_order_seed = training_hparams.data_order_seed

    # Restore the model from a saved checkpoint if the checkpoint exists
    cp_step, cp_logger = restore_checkpoint(
        output_location, model, optimizer, train_loader.iterations_per_epoch
    )
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
        dataset_hparams.non_isotropic_augment
        or dataset_hparams.non_isotropic_mixup
        or (training_hparams.adv_train and training_hparams.non_isotropic_adv)
    ):
        greedy_subsets = load_greedy_subset(dataset_hparams)

    # The training loop.
    # print_count = 0
    for epoch in range(start_step.ep, end_step.ep + 1):
        # Ensure the data order is different for each epoch.
        train_loader.shuffle(
            None if data_order_seed is None else (data_order_seed + epoch)
        )

        for iteration, (examples, labels) in enumerate(train_loader):
            if iteration % 100 <= 5 and get_platform().is_primary_process:
                print("At epoch, iteration ({} {})".format(epoch, iteration))

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
            labels_size = torch.tensor(len(labels), device=get_platform().torch_device)
            # if print_count < 10 and get_platform().is_primary_process:
            #     print(
            #         "\n batch size in {} gpu initially is {}".format(
            #             get_platform().local_rank, labels_size
            #         )
            #     )
            #     print_count += 1

            if dataset_hparams.non_isotropic_mixup:
                shuffle_indices = torch.randperm(len(labels))
                perturbed_examples = non_isotropic_projection(
                    examples,
                    labels,
                    examples[shuffle_indices],
                    greedy_subsets,
                    threshold=dataset_hparams.non_isotropic_projection_threshold,
                )
                examples = torch.cat((examples, perturbed_examples), dim=0)
                labels = torch.cat((labels, labels), dim=0)
                labels_size = torch.tensor(
                    len(labels), device=get_platform().torch_device
                )
                # if print_count < 10 and get_platform().is_primary_process:
                #     print(
                #         "\n batch size in {} gpu after mixup augmentation is {}".format(
                #             get_platform().local_rank, labels_size
                #         )
                #     )
                #     print_count += 1

            if (
                dataset_hparams.gaussian_augment
                or dataset_hparams.non_isotropic_augment
            ):
                perturbation = torch.zeros_like(examples).to(
                    device=get_platform().torch_device
                )
                perturbation.normal_(
                    dataset_hparams.gaussian_aug_mean, dataset_hparams.gaussian_aug_std
                )

                if dataset_hparams.non_isotropic_augment:
                    # project perturbation to be within epsilon sublevel set of the threat function
                    perturbed_examples = non_isotropic_projection(
                        examples,
                        labels,
                        perturbation,
                        greedy_subsets,
                        threshold=dataset_hparams.non_isotropic_projection_threshold,
                    )

                examples = torch.cat((examples, perturbed_examples), dim=0)
                labels = torch.cat((labels, labels), dim=0)
                labels_size = torch.tensor(
                    len(labels), device=get_platform().torch_device
                )
                # if print_count < 10 and get_platform().is_primary_process:
                #     print(
                #         "\n batch size in {} gpu after Non-isotropic projection augmentation is {}".format(
                #             get_platform().local_rank, labels_size
                #         )
                #     )
                #     print_count += 1

            if (
                training_hparams.non_isotropic_adv_train or training_hparams.adv_train
            ) and epoch >= training_hparams.adv_train_start_epoch:
                if (
                    epoch < training_hparams.adv_train_start_epoch + 6
                    and get_platform().is_primary_process
                ):
                    print("Starting adversarial training at epoch {}".format(epoch))
                attack_fn = get_attack(training_hparams)

                if training_hparams.non_isotropic_adv_train:
                    attack_power = training_hparams.adv_train_attack_power * 10

                perturbation = attack_fn(
                    model,
                    examples,
                    labels,
                    attack_power,
                    attack_power / 10,
                    training_hparams.adv_train_attack_iter,
                )
                # lets do some journalism to see if its faithful. and it works!
                if (epoch % 2 == 0) and (iteration == 0):
                    report_adv(model, examples, labels, perturbation)

                if training_hparams.non_isotropic_adv_train:
                    # project delta to be within epsilon sublevel set of the threat function
                    perturbed_examples = non_isotropic_projection(
                        examples,
                        labels,
                        examples + perturbation,
                        greedy_subsets,
                        threshold=training_hparams.non_isotropic_training_threshold,
                    )
                    perturbation = perturbed_examples - examples

                examples = torch.cat((examples, examples + perturbation), dim=0)
                labels = torch.cat((labels, labels), dim=0)
                labels_size = torch.tensor(
                    len(labels), device=get_platform().torch_device
                )
                # if print_count < 10 and get_platform().is_primary_process:
                #     print(
                #         "\n batch size in {} gpu after adversarial training is {}".format(
                #             get_platform().local_rank, labels_size
                #         )
                #     )
                #     print_count += 1

            step_optimizer.zero_grad()
            model.train()
            loss = model.loss_criterion(model(examples), labels)
            loss.backward()

            # Step forward. Ignore warnings that the lr_schedule generates.
            step_optimizer.step()
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                lr_schedule.step()

    get_platform().barrier()


def standard_train(
    model: Model,
    output_location: str,
    dataset_hparams: hparams.DatasetHparams,
    training_hparams: hparams.TrainingHparams,
    start_step: Step = None,
    verbose: bool = True,
    evaluate_every_epoch: bool = True,
):
    """Train using the standard callbacks according to the provided hparams."""

    # If the model file for the end of training already exists in this location, do not train
    iterations_per_epoch = datasets.registry.iterations_per_epoch(dataset_hparams)
    train_end_step = Step.from_str(
        training_hparams.training_steps, iterations_per_epoch
    )
    if models.registry.exists(
        output_location, train_end_step
    ) and get_platform().exists(paths.logger(output_location)):
        return

    # penultimate_ep = "38ep"
    # train_penultimate_step = Step.from_str(penultimate_ep, iterations_per_epoch)

    train_loader = datasets.registry.get(dataset_hparams, train=True)
    test_loader = datasets.registry.get(dataset_hparams, train=False)

    if get_platform().is_primary_process:
        print("Model name is : " + str(model.model_name))
        print("Dataset name is : " + str(dataset_hparams.dataset_name))

    callbacks = standard_callbacks.standard_callbacks(
        training_hparams,
        train_loader,
        test_loader,
        start_step=start_step,
        verbose=verbose,
        evaluate_every_epoch=evaluate_every_epoch,
    )

    train(
        dataset_hparams,
        training_hparams,
        model,
        train_loader,
        output_location,
        callbacks,
        start_step=start_step,
    )
