import torch
import time
import os
from random import randrange
from autoattack import AutoAttack


from threat_specification.greedy_subset import load_greedy_subset
from threat_specification.projected_displacement import (
    non_isotropic_projection,
    non_isotropic_threat,
)

from platforms.platform import get_platform

from datasets.base import DataLoader
from foundations import hparams

from utilities.evaluation_utils import correct, get_soft_margin, get_pointwise_margin
from utilities.plotting_utils import plot_soft_margin, plot_hist, plot_security_curve


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
    dataset_hparams: hparams.DatasetHparams,
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
            iso_str = "isotropic" if not non_isotropic else "non isotropic"
            print(
                "\n"
                + "-" * 20
                + "Running {} robust evaluation on {} data w.r.t L{}".format(
                    iso_str, data_str, attack_norm
                )
                + "-" * 20
            )

        name = "robust_evaluation_" + data_str
        eval_dir = os.path.join(test_output_location, name)
        attack_str = ""
        if non_isotropic:
            attack_str += "non_isotropic_"
        else:
            attack_str += "isotropic_"
        attack_str += str(attack_norm)

        info = {}
        example_count = torch.tensor(0.0).to(get_platform().torch_device)
        total_correct = torch.tensor(0.0).to(get_platform().torch_device)

        model.eval()

        adversary = AutoAttack(
            model,
            norm=attack_norm,
            eps=attack_size,
            log_path=os.path.join(eval_dir, attack_str),
            device=get_platform().torch_device,
        )

        greedy_subsets = load_greedy_subset(dataset_hparams)

        current_batch_index = -1

        # run attack and save images
        with torch.no_grad():
            for examples, labels in loader:
                current_batch_index += 1
                if current_batch_index != random_batch_index:
                    continue

                examples = examples.to(get_platform().torch_device)
                labels = labels.squeeze().to(get_platform().torch_device)
                labels_size = torch.tensor(
                    len(labels), device=get_platform().torch_device
                )
                example_count += labels_size

                examples_adv, labels_adv = adversary.run_standard_evaluation(
                    examples, labels, bs=labels_size, return_labels=True
                )

                if non_isotropic:
                    # project examples_adv to the sublevel set
                    examples_adv = non_isotropic_projection(
                        examples,
                        labels,
                        examples_adv,
                        greedy_subsets,
                        threshold=dataset_hparams.N_threshold,
                    )
                    output = model(examples_adv)
                    total_correct += correct(labels, output)
                else:
                    # store Nonisotropic threats
                    total_correct += torch.sum(torch.eq(labels_adv.eq, labels))

                    threats = non_isotropic_threat(
                        examples,
                        labels,
                        examples_adv,
                        greedy_subsets,
                    )

        # Share the information if distributed.
        if get_platform().is_distributed:
            torch.distributed.reduce(
                total_correct, 0, op=torch.distributed.ReduceOp.SUM
            )
            torch.distributed.reduce(
                example_count, 0, op=torch.distributed.ReduceOp.SUM
            )

        total_correct = total_correct.cpu().item()
        example_count = example_count.cpu().item()
        robust_accuracy = 100 * total_correct / example_count

        info["robust_accuracy"] = robust_accuracy

        if verbose and get_platform().is_primary_process:
            nonlocal time_of_last_call
            elapsed = (
                0 if time_of_last_call is None else time.time() - time_of_last_call
            )
            print(
                "{} robust evaluation on {} data w.r.t L{}: \t robust_accuracy {:.2f}% \t examples {:d}\t time {:.2f}s".format(
                    iso_str.capitalize(),
                    data_str,
                    attack_norm,
                    robust_accuracy,
                    int(example_count),
                    elapsed,
                )
            )
            time_of_last_call = time.time()
            info["elapsed"] = elapsed

        feedback[name] = info

    return robust_evaluation


def image_generation_eval():
    # store a few isotropic and non isotropic adversarial images
    # store plots of robust accuracy vs threshold. (but this requires many robust-eval calls)
    # for isotropic threat : store histograms of the threats of adversarial examples.
    pass
    # if isotropic: # save images
    #     for i in range(len(examples_adv)):
    #         img = examples_adv[i].cpu().numpy()
    #         img = np.transpose(img, (1, 2, 0))
    #         img = (img + 1) / 2
    #         img = np.clip(img, 0, 1)
    #         img = (img * 255).astype(np.uint8)
    #         img = Image.fromarray(img)
    #         img.save(
    #             os.path.join(
    #                 eval_dir,
    #                 "adv_img_{}_{}.png".format(
    #                     current_batch_index, i
    #                 ),
    #             )
    #         )
    # maybe project the adversarial examples and store them. (for isotropic attacks)


def evaluation_suite(
    dataset_hparams: hparams.DatasetHparams,
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
                dataset_hparams,
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
                    dataset_hparams,
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
                dataset_hparams,
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
                    dataset_hparams,
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
