import torch
import torch.nn as nn
import time
from random import randrange
import math
import numpy as np
import eagerpy as ep

from foolbox import PyTorchModel, accuracy, samples


from platforms.platform import get_platform

from datasets.base import DataLoader
from foundations import hparams

from utilities.evaluation_utils import correct, get_soft_margin, get_pointwise_margin
from utilities.plotting_utils import plot_soft_margin, plot_hist, plot_security_curve

from lipschitz.lip_registry import registered_estimators
from attacks import attack_registry
from attacks.base import FixedSizeAttack, MinimizationAttack

# load attack
from autoattack import AutoAttack


def create_standard_eval(
    loader: DataLoader,
    data_str,
    verbose=False,
):

    time_of_last_call = None

    def standard_evaluation(model, feedback):
        if verbose and get_platform().is_primary_process:
            print(
                "-" * 20
                + "Running Standard Evaluation on {}".format(data_str)
                + "data"
                + "-" * 20
            )

        # output_location = test_output_location + "/standard_evaluation/"  # data_str +
        name = "standard_evaluation_" + data_str
        info = {}
        example_count = torch.tensor(0.0).to(get_platform().torch_device)
        total_loss = torch.tensor(0.0).to(get_platform().torch_device)
        total_correct = torch.tensor(0.0).to(get_platform().torch_device)

        model.eval()

        with torch.no_grad():
            for examples, labels in loader:
                examples = examples.to(get_platform().torch_device)
                labels = labels.squeeze().to(get_platform().torch_device)
                output = model(examples)
                labels_size = torch.tensor(
                    len(labels), device=get_platform().torch_device
                )
                example_count += labels_size
                total_loss += model.loss_criterion(output, labels) * labels_size
                total_correct += correct(labels, output)

        # Share the information if distributed.
        if get_platform().is_distributed:
            torch.distributed.reduce(total_loss, 0, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.reduce(
                total_correct, 0, op=torch.distributed.ReduceOp.SUM
            )
            torch.distributed.reduce(
                example_count, 0, op=torch.distributed.ReduceOp.SUM
            )

        total_loss = total_loss.cpu().item()
        total_correct = total_correct.cpu().item()
        example_count = example_count.cpu().item()

        loss = total_loss / example_count
        accuracy = 100 * total_correct / example_count

        info[data_str + "_loss"] = loss
        info[data_str + "_accuracy"] = accuracy

        if verbose and get_platform().is_primary_process:
            nonlocal time_of_last_call
            elapsed = (
                0 if time_of_last_call is None else time.time() - time_of_last_call
            )
            print(
                "Standard Testing on {} data: \t loss {:.3f} \t accuracy {:.2f}%\t examples {:d}\t time {:.2f}s".format(
                    data_str, loss, accuracy, int(example_count), elapsed
                )
            )
            time_of_last_call = time.time()
            info["elapsed"] = elapsed

        feedback[name] = info

    return standard_evaluation


def create_robust_eval(
    loader: DataLoader,
    test_output_location,
    data_str,
    random_batch_index,
    attack_norm,
    attack_size,
    non_isotropic=False,
    verbose=False,
):

    time_of_last_call = None

    def robust_evaluation(model, feedback):

        if verbose:
            iso_str = "" if not non_isotropic else "non_isotropic"
            print(
                "\n"
                + "-" * 20
                + "Running {} robust evaluation on {} data w.r.t L{}".format(
                    iso_str, data_str, attack_norm
                )
                + "-" * 20
            )

        name = "attack_evaluation_" + data_str
        output_location = (
            test_output_location
            + "/"
            + data_str
            + "_attack_evaluation/"
            + att.get_name()
        )

        if not non_isotropic:
            adversary = AutoAttack(
                model,
                norm=attack_norm,
                eps=attack_size,
                log_path=test_output_location,
            )
        else:
            adversary = AutoAttack(
                model,
                norm=attack_norm,
                eps=5 * attack_size,
                log_path=test_output_location,
            )

        l = [x for (x, y) in loader]

        x_test = torch.cat(l, 0)
        l = [y for (x, y) in test_loader]
        y_test = torch.cat(l, 0)

        # example of custom version
        if args.version == "custom":
            adversary.attacks_to_run = ["apgd-ce", "fab"]
            adversary.apgd.n_restarts = 2
            adversary.fab.n_restarts = 2

        # run attack and save images
        with torch.no_grad():
            if not args.individual:
                adv_complete = adversary.run_standard_evaluation(
                    x_test[: args.n_ex],
                    y_test[: args.n_ex],
                    bs=args.batch_size,
                    state_path=args.state_path,
                )

                torch.save(
                    {"adv_complete": adv_complete},
                    "{}/{}_{}_1_{}_eps_{:.5f}.pth".format(
                        args.save_dir,
                        "aa",
                        args.version,
                        adv_complete.shape[0],
                        args.epsilon,
                    ),
                )

        if name not in feedback.keys():
            feedback[name] = {}

        info = {}

        # loader.shuffle(0)
        current_batch_index = -1
        model.eval()

        for examples, labels in loader:
            current_batch_index += 1
            # only to want to evaluate for a single random batch (so not for the entire dataset?)
            if current_batch_index > random_batch_index:
                break
            if current_batch_index != random_batch_index:
                continue

            examples = examples.to(get_platform().torch_device)
            labels = labels.squeeze().to(get_platform().torch_device)
            batch_size = len(examples)

            output = model(examples)
            margins = []
            for index in range(0, len(examples)):
                margins.append(get_pointwise_margin(output[index, :], labels[index]))

            if verbose:
                print("Generating adversarial examples...")

            best_adv, best_adv_radius, found_adv = att.generate_adversarial_example(
                model, examples, labels, margins
            )

            success_rate = 100 * sum(found_adv) / batch_size
            adv_output = model(best_adv)
            robust_accuracy = 100 * correct(labels, adv_output).item() / batch_size
            is_correct = torch.eq(labels, adv_output.argmax(dim=1))
            check = torch.sum(torch.eq(torch.Tensor(found_adv).cuda(), is_correct))
            if success_rate != 100 - robust_accuracy or check != 0.0:
                print("issue with foolbox usage")
                print(
                    "success rate is {} but 100 - robust_accuracy is {}".format(
                        success_rate, 100 - robust_accuracy
                    )
                )
                print("check is {}".format(check))
            # rob_acc = accuracy(foolbox_model, best_adv, labels)
            # print(rob_acc)

            if verbose:
                print(
                    "The attack {} resulted in a robust accuracy of {}".format(
                        att.get_name(), robust_accuracy
                    )
                )
                print(
                    "It had a success rate of {} with a median corruption radius {}".format(
                        success_rate, torch.median(best_adv_radius)
                    )
                )
                print(best_adv_radius)

            nonlocal time_of_last_call
            elapsed = (
                0 if time_of_last_call is None else time.time() - time_of_last_call
            )

            info["elapsed"] = elapsed
            info["Best_Adversary"] = best_adv
            info["Best_Adversary_Radius"] = best_adv_radius
            info["Success"] = found_adv

            feedback[name][att.get_name()] = info

    return attack_evaluation


def evaluation_suite(
    testing_hparams: hparams.TestingHparams,
    test_output_location,
    train_set_loader: DataLoader,
    test_set_loader: DataLoader,
    eval_on_train: bool = False,
    evaluate_batch_only: bool = True,
    verbose: bool = True,
):
    evaluations = []
    # random_batch_index = randrange(50)
    random_batch_index = 10

    if testing_hparams.standard_eval:
        evaluations.append(
            create_standard_eval(
                test_set_loader,
                "test",
                verbose=True,
            )
        )
        if eval_on_train:
            evaluations.append(
                create_standard_eval(
                    train_set_loader,
                    "train",
                    verbose=True,
                )
            )

    if testing_hparams.adv_eval:
        attack_norm = testing_hparams.adv_test_attack_norm
        if attack_norm == "2":
            attack_size = testing_hparams.adv_attack_size_l2
        else:
            attack_size = testing_hparams.adv_attack_size_linf

        evaluations.append(
            create_robust_eval(
                test_set_loader,
                test_output_location,
                "test",
                random_batch_index,
                attack_norm,
                attack_size,
                verbose=True,
            )
        )
        if eval_on_train:
            evaluations.append(
                create_robust_eval(
                    train_set_loader,
                    test_output_location,
                    "train",
                    random_batch_index,
                    attack_norm,
                    attack_size,
                    verbose=True,
                )
            )

    if testing_hparams.N_adv_eval:
        attack_norm = testing_hparams.adv_test_attack_norm
        if attack_norm == "2":
            attack_size = 10 * testing_hparams.adv_attack_size_l2
        else:
            attack_size = 5 * testing_hparams.adv_attack_size_linf

        evaluations.append(
            create_robust_eval(
                test_set_loader,
                test_output_location,
                "test",
                random_batch_index,
                attack_norm,
                attack_size,
                non_isotropic=True,
                verbose=True,
            )
        )
        if eval_on_train:
            evaluations.append(
                create_robust_eval(
                    train_set_loader,
                    test_output_location,
                    "train",
                    random_batch_index,
                    attack_norm,
                    attack_size,
                    verbose=True,
                )
            )

    return evaluations
