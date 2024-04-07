import matplotlib.pyplot as plt
import seaborn as sns
#sns.set(style="darkgrid")
import torch.nn.functional as F
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
from math import pi
import os
import torch
import numpy as np

from foundations import paths


def plot_hist(output_location, stat_str, info_str, info_tensor, marker_tensor=None,  hist_bins=10):
    # expecting type_str to be either local_lip_radius or margin or max_cert_radius
    path = paths.plot_save(output_location, stat_str, None)
    # hist_y = torch.histc(info_tensor, bins=hist_bins)
    # hist_y = hist_y.div(hist_y.sum())
    # hist_x = torch.linspace(torch.min(info_tensor).item(), torch.max(info_tensor).item(), steps=hist_bins)
    plt.figure(figsize=(6, 6))
#    figure()
    #  plt.plot(hist_x, hist_y)
    #sns.histplot(data=df, x="stat_str", kde=True)
    try:
        info_tensor = info_tensor.cpu().detach().numpy()
    except AttributeError:
        counts, bins, bars = plt.hist(info_tensor, bins=hist_bins*3, range=[0, 500], color='navy')
        plt.xlabel(info_str)
        plt.ylabel('Count')
        #fig, ax = plt.subplots()
        #sns.histplot(x=info_tensor, ax=ax, bins=5, color='limegreen', stat="probability", discrete=True)
        #ax.set_xlim(0, 500)
        plt.savefig(path)
        plt.close()

    else:
        if marker_tensor is not None:
            counts, bins, bars = plt.hist(info_tensor, bins=hist_bins * 3, color='navy')
            y = np.linspace(0, np.max(counts), 10)
            marker = marker_tensor
            x = [marker] * len(y)
            plt.plot(x, y, label='Global Lipschitz Constant', color='red')
        else:
            counts, bins, bars = plt.hist(info_tensor, bins=hist_bins * 3, range=[0, 500], color='navy')
        #if marker_tensor is not None:
    #    plt.hist(marker_tensor, bins=hist_bins, density=True)
        plt.xlabel(info_str)
        plt.ylabel('Count')
    #plt.title('Histogram of ' + title_str)
        plt.grid(True)
        plt.savefig(path)
        plt.close()


def plot_soft_margin(Mrgs, Accs, output_location):
    target_acc = 0.6
    target_margin = -1.0
    if torch.sum(Accs >= target_acc) != 0:
        target_margin = torch.max(Mrgs[Accs >= target_acc])

    path = paths.plot_save(output_location, 'softmargin', None)
    plt.figure()
    plt.plot(Mrgs, Accs, '.')
    if target_margin != -1.0:
        plt.plot(target_margin, target_acc, 'x')
    plt.xlabel('margin')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs Soft Margin')
    plt.grid(True)
    plt.savefig(path)
    plt.close()

    return target_acc, target_margin


def plot_encoder(stat_str, output_location, layer_encoder_gaps, layer_ds):
    # print(layer_encoder_gaps[0].shape)
    # print(layer_ds)
    num_layers = len(layer_encoder_gaps)
    locations = []
    for layer_index in range(0, num_layers):
        locations.append(paths.plot_save(output_location, stat_str, layer_index))

    # location = paths.plot_save(output_location, step, stat_str)

    for layer_index in range(0, num_layers):
        plt.grid(True)
        fig, ax = plt.subplots()
        ax.set_xlim(0, layer_ds[layer_index])
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))

        ax.grid(which='major', color='#CCCCCC', linestyle='--')
        ax.grid(which='minor', color='#CCCCCC', linestyle=':')
        # plt.figure()
        ax.plot(list(range(0, layer_ds[layer_index])), layer_encoder_gaps[layer_index].detach().numpy())
        ax.set_xlabel('Activity Level at Layer {}'.format(layer_index))
        ax.set_ylabel('Encoder Gap')
        if stat_str == 'worst':
            ax.set_title('Worst Case Encoder Gap at Layer ' + str(layer_index) + ' wrt random batch (128) - MNIST')
        elif stat_str == 'average':
            ax.set_title('Average Case Encoder Gap at Layer ' + str(layer_index) + ' wrt random batch (128) - MNIST')
        else:
            ax.set_title('Best Case Encoder Gap at Layer ' + str(layer_index) + ' wrt random batch (128) - MNIST')
        ax.grid(True)
        fig.savefig(locations[layer_index])
        plt.close()


def plot_security_curve(stat_str, output_location, certified_radius_bucket, certified_acc_op_norm, certified_acc_red_op, adversarial_accuracy, adv_type):
    certified_radius_bucket = certified_radius_bucket.cpu().detach().numpy()
    #certified_accuracy_sparse = certified_accuracy_sparse.cpu().detach().numpy()
    certified_acc_op_norm = certified_acc_op_norm.cpu().detach().numpy()
    certified_acc_red_op = certified_acc_red_op.cpu().detach().numpy()

    adversarial_accuracy = adversarial_accuracy.cpu().detach().numpy()
    path = paths.plot_save(output_location, stat_str, None)  # stat_str = security_curve

    plt.style.use('bmh')

    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111)

    # plt.plot(certified_radius_bucket, certified_accuracy_sparse, label='sparse-loc certification')
    plt.semilogx(certified_radius_bucket, certified_acc_op_norm, label='patch-global-Lip-cert')
    plt.semilogx(certified_radius_bucket, certified_acc_red_op, label='patch-local-Lip-cert')
    plt.semilogx(certified_radius_bucket, adversarial_accuracy, '--', label=adv_type)

    #plt.plot(certified_radius_bucket, certified_acc_op_norm, label='op-norm')
    #plt.plot(certified_radius_bucket, certified_acc_red_op, label='reduced-op-norm')
    #plt.plot(certified_radius_bucket, adversarial_accuracy, '--', label=adv_type)
    plt.legend(loc='upper right')
    plt.ylabel('Robust Accuracy')
    plt.ylim([0, 1])
    plt.xlim([0.00001, 0.1])  # upper used to be 1.5
    # plt.xlabel(r'Adversarial Corruption $\log(1+\nu)$')
    plt.xlabel(r'Adversarial Corruption $\nu$ in patch 2->infinity norm')

    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
              fancybox=True, shadow=True, ncol=5)
    plt.savefig(path)
    plt.close()


def plot_certified(eval_name, stat_str, output_location, c_acc, c_rad, sigmas, c_acc_sp, c_rad_sp):
    plt_location = output_location + '/certification/' + eval_name + '/'
    path = paths.plot_save(plt_location, stat_str, None)
    # stat_str = randomized_smoothing or generic or just certified accuracy

    plt.style.use('bmh')

    plt.figure(figsize=(4, 4))
    plt.plot(c_rad_sp, c_acc_sp, label='sparse')
    for k in range(len(c_rad)):
        plt.plot(c_rad[k], c_acc[k], '--',
                 label=r'$\sigma$ = %.2f' % np.round(sigmas[k], 2))
    plt.legend(loc='upper right')
    plt.ylabel('Certified Accuracy')
    plt.ylim([0, 1])
    plt.xlabel('Certified Radius')
    plt.savefig(path)
    plt.close()


def line(m,x):
    for i in range(len(x)):
        if m*x[i] > np.sqrt(1-x[i]**2):
            x[i] = 0
        elif m*x[i] < 0:
            x[i] = 0
    return m*x


def circle(x):
    return np.sqrt(1-x**2)


def generate_layer_angles(layer_matrix, layer_bias, layer_input):
    layer_input_norm = np.linalg.norm(layer_input, ord=2)
    tan_theta_array = []
    for i in range(layer_matrix.shape[0]):
        row = layer_matrix[i]
        #print(row.shape)
        row_norm = np.linalg.norm(layer_matrix[i], ord=2)
        cos_theta =(np.dot(row, layer_input) + layer_bias[i])/(row_norm*layer_input_norm)
        print('cos theta is {} and angle is {} degrees'.format(cos_theta, 180*(np.arccos(cos_theta)/pi)))
        tan_theta = np.sqrt(1-cos_theta*cos_theta)/cos_theta
        tan_theta_array.append(tan_theta)
        #print(np.tan(np.arccos(cos_theta)))
#        print(tan_theta)
        #assert np.tan(np.arccos(cos_theta)) == tan_theta
    return tan_theta_array


def plot_critical_angle(output_location, model, example):
    # for each layer matrix and layer input generate the theta angles.
    stat_str = 'angular_distance'
    print('Inside critical angle plot')
    print(example.shape)
    layer_input = example.view(example.size(0), -1)  # Flatten.
    print(layer_input.shape)
    #if layer_input.shape != 784:
    #    raise ValueError('Incorrect dimension handling')
    plt.rcParams['figure.figsize'] = [5, 5]
    for i, layer in enumerate(model.layers):
        next_layer_input = F.relu(model.layers[i](layer_input))
        #plt.hist(model.layers[i](layer_input).cpu().detach().numpy(), bins=10, density=True)
        #plt.show()
        num_active = next_layer_input[next_layer_input > 0].shape
        print('The base amount of activity is {}'.format(num_active))
        tan_theta_array = generate_layer_angles(model.layers[i].weight.cpu().detach().numpy(), model.layers[i].bias.cpu().detach().numpy(),layer_input[0].cpu().detach().numpy())
        #print(tan_theta_array)
        plt_location = paths.plot_save(output_location, stat_str, i)
        fig, ax = plt.subplots()
        x = np.linspace(-1, 1, 500)
        x2 = np.linspace(-1, 1, 500)
        ax.axis('off')
        ax.plot(x, circle(x), c='r')
        ax.plot(x, -circle(x), c='r')
        for j in range(len(tan_theta_array)):
            m = tan_theta_array[j]
            # label=r'$\tan(\theta) = ${:.2f}'.format(m)
            plt.plot(x2, line(m, x2), c='b')
        plt.legend()
        plt.title("Critical Angle Threshold at Layer {}".format(i))
        fig.savefig(plt_location)
        plt.close()
        layer_input = next_layer_input
