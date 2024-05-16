import torch
import time
import os
from random import randrange
from autoattack import AutoAttack


from threat_specification.greedy_subset import load_threat_specification
from threat_specification.projected_displacement import (
    non_isotropic_projection,
    non_isotropic_threat,
)

from platforms.platform import get_platform

from datasets.base import DataLoader
from foundations import hparams

from utilities.evaluation_utils import (
    correct,
    compute_prob,
    get_soft_margin,
    get_pointwise_margin,
)
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
    loader: DataLoader,
    data_str,
    dataset_hparams: hparams.DatasetHparams,
    test_output_location,
    random_batch_index,
    attack_norm,
    attack_power,
    N_threshold,
    non_isotropic=False,
    verbose=False,
):

    time_of_last_call = None

    def robust_evaluation(model, feedback):

        if verbose:
            iso_str = "nonisotropic" if non_isotropic else "isotropic"
            print(
                "\n"
                + "-" * 20
                + "Running {} robust evaluation on {} data w.r.t L{}".format(
                    iso_str, data_str, attack_norm
                )
                + "-" * 20
            )

        name = iso_str + "_robust_evaluation_" + data_str
        eval_dir = os.path.join(test_output_location, name)
        attack_str = iso_str + "_" + str(attack_norm)

        info = {}
        example_count = torch.tensor(0.0).to(get_platform().torch_device)
        total_correct = torch.tensor(0.0).to(get_platform().torch_device)

        model.eval()

        adversary = AutoAttack(
            model,
            norm=attack_norm,
            eps=attack_power,
            log_path=os.path.join(eval_dir, attack_str),
            device=get_platform().torch_device,
        )

        threat_specification = None
        num_bins = 50
        histogram_attack_statistics = torch.zeros(num_bins).to(
            get_platform().torch_device
        )
        attack_statistics = None

        if non_isotropic:
            threat_specification = load_threat_specification(dataset_hparams)
            if attack_norm == "L2":
                attack_statistics = torch.linspace(start=0.05, end=2.5, steps=num_bins)
                ord_val = 2
            if attack_norm == "Linf":
                attack_statistics = torch.linspace(
                    start=0.5 / 255, end=32 / 255, steps=num_bins
                )
                ord_val = float("inf")
        else:
            attack_statistics = torch.linspace(start=0.01, end=1.5, steps=num_bins)
        assert attack_statistics is not None

        current_batch_index = -1

        # run attack and save images
        with torch.no_grad():
            for examples, labels in loader:
                # unclear if the following works in distributed setting
                current_batch_index += 1
                if random_batch_index and current_batch_index != random_batch_index:
                    continue

                examples = examples.to(get_platform().torch_device)
                labels = labels.squeeze().to(get_platform().torch_device)
                labels_size = torch.tensor(
                    len(labels), device=get_platform().torch_device
                )
                example_count += labels_size

                examples_adv = adversary.run_standard_evaluation(
                    examples, labels, bs=labels_size, return_labels=False
                )

                if non_isotropic:
                    # project examples_adv to the sublevel set
                    examples_adv = non_isotropic_projection(
                        examples,
                        labels,
                        examples_adv,
                        threat_specification,
                        threshold=N_threshold,
                    )
                    # store attack statistics when evaluated with L2/Linf norm
                    threats = torch.linalg.norm(
                        torch.flatten(examples_adv - examples, start_dim=1),
                        dim=1,
                        ord=ord_val,
                    )
                    histogram_attack_statistics += compute_prob(
                        threats, threshold=attack_statistics, tail=True, raw_count=True
                    )
                else:
                    # store attack statistics when evaluated with nonisotropic threat
                    threats = non_isotropic_threat(
                        examples,
                        labels,
                        examples_adv,
                        threat_specification,
                    )
                    histogram_attack_statistics += compute_prob(
                        threats, threshold=attack_statistics, tail=True, raw_count=True
                    )
                output = model(examples_adv)
                total_correct += correct(labels, output)

        # Share the information if distributed.
        if get_platform().is_distributed:
            torch.distributed.reduce(
                total_correct, 0, op=torch.distributed.ReduceOp.SUM
            )
            torch.distributed.reduce(
                example_count, 0, op=torch.distributed.ReduceOp.SUM
            )
            torch.distributed.reduce(
                histogram_attack_statistics, 0, op=torch.distributed.ReduceOp.SUM
            )

        total_correct = total_correct.cpu().item()
        example_count = example_count.cpu().item()
        histogram_attack_statistics = 100 * (
            histogram_attack_statistics.cpu().item() / example_count
        )
        robust_accuracy = 100 * total_correct / example_count

        info["robust_accuracy"] = robust_accuracy
        info["histogram_attack_statistics"] = histogram_attack_statistics

        if verbose and get_platform().is_primary_process:
            nonlocal time_of_last_call
            elapsed = (
                0 if time_of_last_call is None else time.time() - time_of_last_call
            )
            print(
                "{} robust evaluation on {} data w.r.t {} attacks: \t robust_accuracy {:.2f}% \t examples {:d}\t time {:.2f}s".format(
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

    if evaluate_batch_only:
        # random_batch_index = randrange(50)
        random_batch_index = 10
    else:
        random_batch_index = None

    if testing_hparams.standard_eval:
        evaluations.append(
            create_standard_eval(
                test_set_loader,
                "test",
                verbose=verbose,
            )
        )
        if eval_on_train:
            evaluations.append(
                create_standard_eval(
                    train_set_loader,
                    "train",
                    verbose=verbose,
                )
            )

    attack_norm = None
    attack_power = None
    N_threshold = None

    if testing_hparams.adv_eval or testing_hparams.N_adv_eval:
        attack_norm = testing_hparams.adv_test_attack_norm
        N_threshold = testing_hparams.N_threshold
        if attack_norm == "L2":
            attack_power = testing_hparams.adv_test_attack_power_L2
        else:
            attack_power = testing_hparams.adv_test_attack_power_Linf

    if testing_hparams.adv_eval:
        evaluations.append(
            create_robust_eval(
                test_set_loader,
                "test",
                dataset_hparams,
                test_output_location,
                random_batch_index,
                attack_norm,
                attack_power,
                N_threshold,
                non_isotropic=False,
                verbose=verbose,
            )
        )
        if eval_on_train:
            evaluations.append(
                create_robust_eval(
                    train_set_loader,
                    "train",
                    dataset_hparams,
                    test_output_location,
                    random_batch_index,
                    attack_norm,
                    attack_power,
                    N_threshold,
                    non_isotropic=False,
                    verbose=verbose,
                )
            )

    if testing_hparams.N_adv_eval:
        # enlarge the attack power for non-isotropic attacks, the resulting adversarial perturbations will still undergo projection.
        if attack_norm == "L2":
            attack_power *= 10
        else:
            attack_power *= 4

        evaluations.append(
            create_robust_eval(
                test_set_loader,
                "test",
                dataset_hparams,
                test_output_location,
                random_batch_index,
                attack_norm,
                attack_power,
                N_threshold,
                non_isotropic=True,
                verbose=verbose,
            )
        )
        if eval_on_train:
            evaluations.append(
                create_robust_eval(
                    train_set_loader,
                    "train",
                    dataset_hparams,
                    test_output_location,
                    random_batch_index,
                    attack_norm,
                    attack_power,
                    N_threshold,
                    non_isotropic=True,
                    verbose=verbose,
                )
            )

    return evaluations
