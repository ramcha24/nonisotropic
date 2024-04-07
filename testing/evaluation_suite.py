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


def create_standard_eval(loader: DataLoader, test_output_location, data_str, random_batch_index, verbose=False):

    time_of_last_call = None

    def standard_evaluation(model, feedback):
        if verbose:
            print('-' * 20 + 'Running Standard Evaluation' + '-' * 20)

        output_location = test_output_location + '/' + data_str + '/standard_evaluation/'
        name = 'standard_evaluation_' + data_str
        info = {}
        example_count = torch.tensor(0.0).to(get_platform().torch_device)
        total_loss = torch.tensor(0.0).to(get_platform().torch_device)
        total_correct = torch.tensor(0.0).to(get_platform().torch_device)
        max_out = 0.0

        model.eval()

        with torch.no_grad():
            for examples, labels in loader:
                examples = examples.to(get_platform().torch_device)
                labels = labels.squeeze().to(get_platform().torch_device)
                output = model(examples)
                labels_size = torch.tensor(len(labels), device=get_platform().torch_device)
                example_count += labels_size
                total_loss += model.loss_criterion(output, labels) * labels_size
                total_correct += correct(labels, output)
                m_out = torch.max(output).item()
                if m_out > max_out:
                    max_out = m_out

            num_bins = 100
            current_batch_index = -1
            target_margin = 0.0
            target_accuracy = 0.0

            for examples, labels in loader:
                current_batch_index += 1
                # only to want to evaluate for a single random batch (so not for the entire dataset?)
                if current_batch_index > random_batch_index:
                    break
                if current_batch_index != random_batch_index:
                    continue

                examples = examples.to(get_platform().torch_device)
                labels = labels.squeeze().to(get_platform().torch_device)
                actual_margins = torch.zeros(len(examples))
                margins = torch.linspace(0, max_out, steps=num_bins)
                accuracies = torch.zeros(num_bins)

                get_soft_margin(model, examples, labels, margins, accuracies, actual_margins)
                plot_hist(output_location, 'pointwise_margin', 'Margin', actual_margins)
                target_accuracy, target_margin = plot_soft_margin(margins, accuracies, output_location)

            # for index in range(num_bins):
            #     accuracies[index] = accuracies[index] / example_count

        total_loss = total_loss.cpu().item()
        total_correct = total_correct.cpu().item()
        example_count = example_count.cpu().item()
        test_loss = total_loss / example_count
        test_accuracy = 100 * total_correct / example_count

        info['test_loss'] = test_loss
        info['test_accuracy'] = test_accuracy
        info['target_accuracy'] = target_accuracy
        info['target_margin'] = target_margin

        if verbose:
            nonlocal time_of_last_call
            elapsed = 0 if time_of_last_call is None else time.time() - time_of_last_call
            print('Standard Testing : \t loss {:.3f} \t accuracy {:.2f}%\t examples {:d}\t time {:.2f}s'.format(
                test_loss, test_accuracy, int(example_count), elapsed))
            print('Target Test Accuracy %.2f at Margin = %.4f' % (target_accuracy, target_margin))
            info['elapsed'] = elapsed

        feedback[name] = info

    return standard_evaluation


def create_lip_eval(loader: DataLoader,
                    test_output_location,
                    data_str,
                    random_batch_index,
                    type_str,
                    num_bins,
                    verbose=False):

    # return the smallest radius where every input can be certified via the lipschitz constant estimation should ideally be like
    # feedback[lipschitz_eval_test][type_str] = # tensor of size 100 with the maximum radius certifiable for each.

    def lipschitz_eval(model, feedback):
        if type_str in registered_estimators.keys():
            registered_estimators[type_str](model,
                                            feedback,
                                            loader,
                                            test_output_location,
                                            data_str,
                                            type_str,
                                            random_batch_index,
                                            num_bins,
                                            verbose=verbose)
        else:
            raise ValueError('No registered lipschitz estimator with name {}'.format(type_str))

    return lipschitz_eval


def create_attack_eval(loader: DataLoader,
                       test_output_location,
                       data_str,
                       random_batch_index,
                       attack,
                       attack_str,
                       attack_norm,
                       num_trials,
                       conv_flag,
                       verbose=False):

    time_of_last_call = None

    def attack_evaluation(model, feedback):

        if attack_str == 'minimum_size':
            att = MinimizationAttack(attack, attack_norm, num_trials)
        elif attack_str == 'fixed_l2' or attack_str == 'fixed_linf':
            att = FixedSizeAttack(attack, attack_registry.epsilons[attack_str], attack_norm, num_trials)
        else:
            raise ValueError('Please specify a valid type of adversarial attack')

        if verbose:
            print('\n' + '-' * 20 + 'Running Attack Evaluation : ' + att.get_name() + '-' * 20)

        name = 'attack_evaluation_' + data_str
        output_location = test_output_location + '/' + data_str + '/attack_evaluation/' + att.get_name()
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
                out = output[index, :]
                margins.append(get_pointwise_margin(out, labels[index]))

            if verbose:
                print('Generating adversarial examples...')

            best_adv, best_adv_radius, found_adv = att.generate_adversarial_example(model, examples, labels, margins, conv_flag)

            success_rate = 100 * sum(found_adv) / batch_size
            adv_output = model(best_adv)
            robust_accuracy = 100 * correct(labels, adv_output).item() / batch_size
            is_correct = torch.eq(labels, adv_output.argmax(dim=1))
            check = torch.sum(torch.eq(torch.Tensor(found_adv).cuda(), is_correct))
            if success_rate != 100 - robust_accuracy or check != 0.0:
                print('issue with foolbox usage')
                print('success rate is {} but 100 - robust_accuracy is {}'.format(success_rate, 100 - robust_accuracy))
                print('check is {}'.format(check))
            # rob_acc = accuracy(foolbox_model, best_adv, labels)
            # print(rob_acc)

            if verbose:
                print('The attack {} resulted in a robust accuracy of {}'.format(att.get_name(), robust_accuracy))
                print('It had a success rate of {} with a median corruption radius {}'.format(success_rate, torch.median(best_adv_radius)))
                print(best_adv_radius)

            nonlocal time_of_last_call
            elapsed = 0 if time_of_last_call is None else time.time() - time_of_last_call

            info['elapsed'] = elapsed
            info['Best_Adversary'] = best_adv
            info['Best_Adversary_Radius'] = best_adv_radius
            info['Success'] = found_adv

            feedback[name][att.get_name()] = info

    return attack_evaluation


def create_ensemble_eval(loader: DataLoader,
                         test_output_location,
                         data_str,
                         random_batch_index,
                         verbose=False):

    def ensemble_attack_eval(model, feedback):
        # pick the best advs among all the attacks
        # evaluate the accuracy of model on these inputs.
        # add to feedback dict
        if verbose:
            print('\n' + '-' * 20 + 'Running Ensemble Attack Evaluation : ' + '-' * 20)
        data_key = 'chosen_batch_' + data_str
        ensemble_key = 'ensemble_attack_evaluation_' + data_str
        name = 'attack_evaluation_' + data_str
        output_location = test_output_location + '/' + data_str + '/ensemble_attack_evaluation/'

        robust_dict = feedback[name]

        current_batch_index = -1

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
            clean_accuracy = 100 * correct(labels, output).item()/batch_size
            best_adversary_overall = examples.clone().detach()
            minimum_adversary_radius = torch.ones(batch_size).to(get_platform().torch_device)
            minimum_adversary_radius *= float("inf")
            success = [False]*batch_size
            if verbose:
                print('Combining adversarial examples...')

            # print(robust_dict)
            for att_name in robust_dict.keys():
                print('Processing {}'.format(att_name))
                info = robust_dict[att_name]
                best_adv = info['Best_Adversary']
                best_adv_radius = info['Best_Adversary_Radius']
                found_adv = info['Success']
                for j in range(0, batch_size):
                    if found_adv[j] and (minimum_adversary_radius[j] > best_adv_radius[j]):
                        minimum_adversary_radius[j] = best_adv_radius[j]
                        best_adversary_overall[j] = best_adv[j]
                        success[j] = True

            success_rate = sum(success)/batch_size
            adv_output = model(best_adversary_overall)
            robust_accuracy = correct(labels, adv_output).item()/batch_size

            assert success_rate == 1 - robust_accuracy

            clean_adversary_radius = []
            for j in range(0, batch_size):
                if minimum_adversary_radius[j] != float("inf"):
                    clean_adversary_radius.append(minimum_adversary_radius[j])
            clean_adversary_radius = torch.tensor(clean_adversary_radius).to(get_platform().torch_device)
            print(clean_adversary_radius)

            if verbose:
                print('The ensemble attack evaluation resulted in a robust accuracy of {}'.format(robust_accuracy))
                print('It had a success rate of {} with an average corruption radius {}'.format(success_rate, torch.mean(clean_adversary_radius)))

            plot_hist(output_location, 'minimum_adversary_radius', 'Minimum Radius of Succesful Adversary', clean_adversary_radius)

            feedback[data_key] = {
                'examples': examples,
                'labels': labels,
                'clean_acc': clean_accuracy
            }

            feedback[ensemble_key] = {'adversarial_examples': best_adversary_overall,
                                      'minimum_adversary_radius': minimum_adversary_radius,
                                      'success_rate': success_rate,
                                      'robust_accuracy': robust_accuracy}

    return ensemble_attack_eval


def create_security_eval(loader: DataLoader, test_output_location, data_str, random_batch_index, num_bins, conv_flag, verbose=False):

    def security_curve_eval(model, feedback):
        # take the underestimation from lipschitz evals and
        # the overestimation from the adversarial attacks and combine to form a plot.
        #num_bins_sc = 100 * num_bins
        num_bins_sc = 10000
        flaws = 0
        output_location = test_output_location + '/' + data_str + '/security_curve_evaluation/'

        upper = 1.0
        if conv_flag:
            upper = 0.1
        certified_radius_bucket = torch.linspace(0, upper, steps=num_bins_sc).to(device=get_platform().torch_device)
        # certified_accuracy_sparse = torch.zeros_like(certified_radius_bucket).to(device=get_platform().torch_device)
        certified_accuracy_red_op = torch.zeros_like(certified_radius_bucket).to(device=get_platform().torch_device)
        certified_accuracy_op_norm = torch.zeros_like(certified_radius_bucket).to(device=get_platform().torch_device)
        adversarial_accuracy = torch.zeros_like(certified_radius_bucket).to(device=get_platform().torch_device)

        data_key = 'chosen_batch_' + data_str
        ensemble_key = 'ensemble_attack_evaluation_' + data_str
        lipschitz_key = 'lipschitz_evaluation_' + data_str
        minimum_adversary_radius = feedback[ensemble_key]['minimum_adversary_radius']
        # maximum_certifiable_radius = feedback[lipschitz_key]['maximum_certifiable_radius']
        red_op_norm_radius = feedback[lipschitz_key]['reduced_op_norm_cert']
        operator_norm_radius = feedback[lipschitz_key]['operator_norm_cert']
        assert len(minimum_adversary_radius) == len(red_op_norm_radius)

        batch_size = len(minimum_adversary_radius)
        # sanity check :
        for index in range(batch_size):
            # print('\n At index {}, \n (max_op_norm, max_red_op_norm, min_adv) is ({}, {}, {})'.
            #      format(index, operator_norm_radius[index], red_op_norm_radius[index], minimum_adversary_radius[index]))
            print('\n At index {}, \n (max_red_op_norm, min_adv) is ({}, {})'.
                  format(index, red_op_norm_radius[index], minimum_adversary_radius[index]))

            if (operator_norm_radius is not None) and red_op_norm_radius[index] < operator_norm_radius[index]:
                raise ValueError('Conflict! why is the reduced op norm certificate {} worse than naive certificate {}?'.format(red_op_norm_radius[index], operator_norm_radius[index]))

            if red_op_norm_radius[index] > minimum_adversary_radius[index]:
                # something very wrong has happened
                flaws += 1
                print('Conflict! there appears to be an adversary of power {} within the radius!! {}'.format(minimum_adversary_radius[index], red_op_norm_radius[index]))
                if (operator_norm_radius is not None) and operator_norm_radius[index] < minimum_adversary_radius[index]:
                    red_op_norm_radius[index] = operator_norm_radius[index]
                else:
                    red_op_norm_radius[index] = 0.0

        for bin_id in range(num_bins_sc):
            test_radius = certified_radius_bucket[bin_id]
            # print('test_radius is {}'.format(test_radius))
            for index in range(0, batch_size):
                if red_op_norm_radius[index] == 0.0:
                    continue

                if (operator_norm_radius is not None) and test_radius <= operator_norm_radius[index]:
                    certified_accuracy_op_norm[bin_id] += 1.0

                if test_radius <= red_op_norm_radius[index]:
                    certified_accuracy_red_op[bin_id] += 1.0

                # if red_op_norm_radius[index] == minimum_adversary_radius[index] == 0:
                #    continue
                # if maximum_certifiable_radius[index] > minimum_adversary_radius[index]:
                #    continue

                # if test_radius <= maximum_certifiable_radius[index]:
                #    certified_accuracy_sparse[bin_id] += 1.0
                if test_radius <= minimum_adversary_radius[index]:
                    adversarial_accuracy[bin_id] += 1.0

            # certified_accuracy_sparse[bin_id] = certified_accuracy_sparse[bin_id] / batch_size
            certified_accuracy_op_norm[bin_id] = certified_accuracy_op_norm[bin_id] / batch_size
            certified_accuracy_red_op[bin_id] = certified_accuracy_red_op[bin_id] / batch_size
            adversarial_accuracy[bin_id] = adversarial_accuracy[bin_id] / batch_size

        # for bin_id in range(0, num_bins):
        #    certified_radius_bucket[bin_id] = math.log(certified_radius_bucket[bin_id]+1)

        print('Median reduced op_norm cert radius is {}'.format(torch.median(red_op_norm_radius).item()))
        if operator_norm_radius is not None:
            print('Median op_norm cert radius is {}'.format(torch.median(operator_norm_radius).item()))
        # print('Median certifiable radius is {}'.format(torch.median(maximum_certifiable_radius).item()))
        print('Median adversary radius is {}'.format(torch.median(minimum_adversary_radius).item()))
        print('total flaws {}'.format(flaws))
        plot_security_curve('security_curve_evaluation', output_location, certified_radius_bucket, certified_accuracy_op_norm, certified_accuracy_red_op, adversarial_accuracy, 'adv attack')

    return security_curve_eval


def evaluation_suite_mnist(testing_hparams: hparams.TestingHparams,
                           test_output_location,
                           train_set_loader: DataLoader,
                           test_set_loader: DataLoader,
                           eval_on_train: bool = False,
                           verbose: bool = True,
                           evaluate_batch_only: bool = True,
                           conv_flag: bool = False):
    evaluations = []
    attack_norm = testing_hparams.adv_test_attack_norm
    num_trials = testing_hparams.adv_test_num_trials
    # random_batch_index = randrange(50)
    random_batch_index = 10
    if conv_flag:
        kernel_shape = 3

    if testing_hparams.standard_eval:
        test_standard_eval = create_standard_eval(test_set_loader, test_output_location, 'test', random_batch_index, verbose=True)
        evaluations.append(test_standard_eval)

    if testing_hparams.adv_eval:
        attack_type = testing_hparams.adv_test_attack_type
        if attack_type == 'best_possible':
            for (attack_name, attack) in attack_registry.fixed_size_attacks_l2.items():
                evaluations.append(
                    create_attack_eval(test_set_loader, test_output_location, 'test', random_batch_index, attack,
                                       'fixed_l2', attack_norm, num_trials, conv_flag, verbose=True))
            for (attack_name, attack) in attack_registry.fixed_size_attacks_linf.items():
                evaluations.append(
                    create_attack_eval(test_set_loader, test_output_location, 'test', random_batch_index, attack,
                                       'fixed_linf', attack_norm, num_trials, conv_flag, verbose=True))
            for (attack_name, attack) in attack_registry.minimization_attacks.items():
                evaluations.append(
                    create_attack_eval(test_set_loader, test_output_location, 'test', random_batch_index, attack,
                                       'minimum_size', attack_norm, num_trials, conv_flag, verbose=True))
        elif attack_type in attack_registry.fixed_size_attacks_l2.keys():
            attack = attack_registry.fixed_size_attacks_l2[attack_type]
            evaluations.append(
                create_attack_eval(test_set_loader, test_output_location, 'test', random_batch_index, attack,
                                   'fixed_l2', attack_norm, num_trials, conv_flag, verbose=True))
        elif attack_type in attack_registry.fixed_size_attacks_linf.keys():
            attack = attack_registry.fixed_size_attacks_linf[attack_type]
            evaluations.append(
                create_attack_eval(test_set_loader, test_output_location, 'test', random_batch_index, attack,
                                   'fixed_linf', attack_norm, num_trials, conv_flag, verbose=True))
        elif attack_type in attack_registry.minimization_attacks.keys():
            attack = attack_registry.minimization_attacks[attack_type]
            evaluations.append(
                create_attack_eval(test_set_loader, test_output_location, 'test', random_batch_index, attack,
                                   'minimum_size', attack_norm, num_trials, conv_flag, verbose=True))
        else:
            possible_attack_str = 'best_possible \n' + attack_registry.get_possible_attacks()
            raise ValueError('Given adversarial attack type should be one of the following {}'.format(possible_attack_str))

        evaluations.append(create_ensemble_eval(test_set_loader, test_output_location, 'test', random_batch_index, verbose=True))

    if testing_hparams.lipschitz_eval:
        if conv_flag:
            lip_str = 'SLL_conv'
            evaluations.append(
                create_lip_eval(test_set_loader, test_output_location, 'test', random_batch_index, lip_str,
                                testing_hparams.security_curve_bins, verbose=True))
        else:
            if testing_hparams.lipschitz_estimator_type == 'all':
                for lip_str in registered_estimators.keys():
                    evaluations.append(create_lip_eval(test_set_loader, test_output_location, 'test', random_batch_index, lip_str,
                                    testing_hparams.security_curve_bins, verbose=True))
            elif testing_hparams.lipschitz_estimator_type in registered_estimators.keys():
                evaluations.append(create_lip_eval(test_set_loader, test_output_location, 'test', random_batch_index, testing_hparams.lipschitz_estimator_type,
                                testing_hparams.security_curve_bins, verbose=True))

    if testing_hparams.adv_eval and testing_hparams.lipschitz_eval and testing_hparams.security_eval:
        evaluations.append(create_security_eval(test_set_loader, test_output_location, 'test', random_batch_index, testing_hparams.security_curve_bins, conv_flag, verbose=True))

    return evaluations
