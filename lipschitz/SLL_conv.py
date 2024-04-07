import torch
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
from collections import defaultdict
from random import randrange
from math import sqrt
from joblib import Parallel, delayed
import copy
from datasets.base import DataLoader

from models.forward_hook_model import ForwardHookModel

from platforms.platform import get_platform

from utilities.capacity_utils import get_2_inf_norm, get_classifier_constant, get_naive_lip, get_reduced_classifier_constant
from utilities.evaluation_utils import get_pointwise_margin
from utilities.plotting_utils import plot_hist, plot_critical_angle


def extract_conv_patches(batch_inputs, kernel_shape):
    # returns shape : (b, c , kernel, kernel, h_new, w_new) where (c, kernel, kernel) is the dimension of an individual patch of which there are h_new*w_new in number.
    batch_size, c_old, h_old, w_old = batch_inputs.shape
    h_new = h_old - kernel_shape + 1
    w_new = w_old - kernel_shape + 1

    patches = torch.zeros(batch_size, c_old, kernel_shape, kernel_shape, h_new, w_new).cuda()

    for i in range(0, h_new):
        for j in range(0, w_new):
            patches[:, :, :, :, i, j] = batch_inputs[:, :, i:i + kernel_shape, j:j + kernel_shape]

    return patches


def tensor_mul(weight, local_patch):
    c_out = weight.shape[0]
    assert weight[0].shape == local_patch.shape
    temp = torch.zeros(c_out).cuda()
    for c in range(0, c_out):
        temp[c] = torch.mul(weight[c], local_patch).sum()

    return temp

# checks if I have the computational model of nn.Conv2d correct.
# Manually computes layer outputs and checks if it is the same as torch's layer outputs.
def sanity_check(layer_index, layer_names, layer_out, inp, model):
    assert layer_index < 5
    assert layer_index > -2

    if layer_index != -1:
        layer_name_inp = layer_names[layer_index]
        input_ = F.relu(layer_out[layer_name_inp])
    else:
        input_ = inp

    layer_name_out = layer_names[layer_index + 1]
    output_ = layer_out[layer_name_out]
    print(input_.shape)
    print(output_.shape)

    W = getattr(model, layer_name_out).weight.cuda()
    b = getattr(model, layer_name_out).bias.cuda()
    # W_flat = W.reshape(W.shape[0], W.shape[1]*W.shape[2]*W.shape[3])
    print(W.shape)
    print(b.shape)

    kernel_shape = W.shape[2]
    # print(kernel_shape)
    patches = extract_conv_patches(input_, kernel_shape)
    print(patches.shape)
    batch_size = patches.shape[0]

    assert kernel_shape == W.shape[3]
    assert batch_size == output_.shape[0]
    assert output_[0, 0].shape == patches[0, 0, 0, 0].shape

    manual_out = torch.zeros_like(output_)
    for index in range(0, batch_size):
        patch_inp = patches[index]
        for i in range(0, output_.shape[2]):
            for j in range(0, output_.shape[3]):
                local_patch = patch_inp[:, :, :, i, j].cuda()

                temp = tensor_mul(W, local_patch) + b
                # print(torch.linalg.vector_norm(temp).item())
                diff = output_[index, :, i, j] - temp
                # print(diff)
                # print(temp)
                # print(output_[index,:,i,j])
                diff_norm = torch.linalg.vector_norm(diff, ord=2).item()
                print("At index " + str(index) + " spatial coordinates (" + str(i) + "," + str(
                    j) + "), the difference in norm between channel vectors is " + str(diff_norm))
                manual_out[index, :, i, j] = temp
                print("next")

    print(torch.norm(output_ - manual_out, p='fro'))


# Given two tensor collections of patches - one original and another perturbed, computes distances between patches of the same index
# and provides both total and aggregate distance.
def get_patch_diff(patch1, patch2, agg_ord, patch_ord):
    assert patch1.shape == patch2.shape
    # expecting a batch of 3 D shapes where dimensions 4,5 indicate spatial coordinate of each patch.
    # patch_ord can be float("inf") or 2
    # agg_ord = "max", "mean", "sum", "fro"

    patch1 = patch1.view(patch1.shape[0], -1, patch1.shape[-2], patch1.shape[-1]).cuda()
    patch2 = patch2.view(patch2.shape[0], -1, patch2.shape[-2], patch2.shape[-1]).cuda()

    diff_norms = torch.zeros(patch1.shape[0], patch1.shape[2], patch1.shape[3]).cuda()
    agg_norms = torch.zeros(patch1.shape[0]).cuda()
    patch1_norms = torch.zeros(patch1.shape[0], patch1.shape[2], patch1.shape[3]).cuda()
    patch2_norms = torch.zeros(patch1.shape[0], patch1.shape[2], patch1.shape[3]).cuda()

    for index in range(0, patch1.shape[0]):
        for i in range(0, patch1.shape[2]):
            for j in range(0, patch1.shape[3]):
                patch1_norms[index, i, j] = torch.linalg.vector_norm(patch1[index, :, i, j], ord=patch_ord).item()
                patch2_norms[index, i, j] = torch.linalg.vector_norm(patch2[index, :, i, j], ord=patch_ord).item()
                diff_norms[index, i, j] = torch.linalg.vector_norm(patch1[index, :, i, j] - patch2[index, :, i, j],
                                                                   ord=patch_ord).item()

        if agg_ord == "max":
            agg_norms[index] = torch.max(diff_norms[index]).item()
        elif agg_ord == "mean":
            agg_norms[index] = torch.mean(diff_norms[index]).item()
            # print("hit mean")
        elif agg_ord == "sum":
            agg_norms[index] = torch.sum(diff_norms[index]).item()
        elif agg_ord == "fro":
            agg_norms[index] = torch.linalg.vector_norm(diff_norms[index], ord=2).item()
        else:
            raise ValueError("invalid agg_ord")

    return diff_norms, agg_norms, patch1_norms, patch2_norms


# given input and noise, manual computes original and perturbed layer outputs.
def get_layer_outs(layer_names, inp, noise, model, layer_out):
    manual_outs = OrderedDict()
    manual_outs_n = OrderedDict()
    num_layers = len(layer_names)

    for layer_index in range(-1, num_layers - 1):
        print("Computing layer outputs for layer " + str(layer_index))
        assert layer_index < 4
        assert layer_index > -2

        if layer_index != -1:
            input_ = F.relu(manual_outs[layer_names[layer_index]])
            input_n = F.relu(manual_outs_n[layer_names[layer_index]])
        else:
            input_ = inp
            input_n = inp + noise

        layer_name_out = layer_names[layer_index + 1]
        output_ = layer_out[layer_name_out]

        W = getattr(model, layer_name_out).weight.cuda()
        b = getattr(model, layer_name_out).bias.cuda()

        kernel_shape = W.shape[2]
        patches = extract_conv_patches(input_, kernel_shape)
        patches_n = extract_conv_patches(input_n, kernel_shape)
        batch_size = patches.shape[0]

        manual_out = torch.zeros_like(output_)
        manual_out_n = torch.zeros_like(output_)
        for index in range(0, batch_size):
            patch_inp = patches[index]
            patch_inp_n = patches_n[index]

            for i in range(0, output_.shape[2]):
                for j in range(0, output_.shape[3]):
                    local_patch = patch_inp[:, :, :, i, j].cuda()
                    local_patch_n = patch_inp_n[:, :, :, i, j].cuda()

                    manual_out[index, :, i, j] = tensor_mul(W, local_patch) + b
                    manual_out_n[index, :, i, j] = tensor_mul(W, local_patch_n) + b
        manual_outs[layer_names[layer_index + 1]] = manual_out
        manual_outs_n[layer_names[layer_index + 1]] = manual_out_n

    return manual_outs, manual_outs_n


def noise_propagation(layer_index, layer_names, manual_outs, inp, noise, manual_outs_n, model):
    assert layer_index < 4
    assert layer_index > -2

    if layer_index != -1:
        layer_name_inp = layer_names[layer_index]
        input_ = F.relu(manual_outs[layer_name_inp])
        input_n_ = F.relu(manual_outs_n[layer_name_inp])
    else:
        input_ = inp
        input_n_ = inp + noise

    layer_name_out = layer_names[layer_index + 1]
    output_ = manual_outs[layer_name_out]
    output_n_ = manual_outs_n[layer_name_out]

    print(input_.shape)
    print(output_.shape)

    W = getattr(model, layer_name_out).weight.cuda()
    b = getattr(model, layer_name_out).bias.cuda()
    W_flat = W.view(W.shape[0], -1)
    W_channel_norms = torch.linalg.matrix_norm(W)
    W_flat_op_norm = torch.linalg.matrix_norm(W_flat, ord=2)
    W_flat_op_norm_fake = torch.linalg.matrix_norm(W_flat[0:int(W_flat.shape[0] // (1.5)), 0:W_flat.shape[1] // 2],
                                                   ord=2)
    print(W.shape)
    print(b.shape)
    kernel_shape = W.shape[2]

    patches = extract_conv_patches(input_, kernel_shape)
    patches_n = extract_conv_patches(input_n_, kernel_shape)
    print(patches.shape)
    batch_size = patches.shape[0]

    assert kernel_shape == W.shape[3]
    assert patches.shape[0] == output_.shape[0]
    assert output_[0, 0].shape == patches[0, 0, 0, 0].shape

    diff_norms, agg_norms, patches_norms, patches_n_norms = get_patch_diff(patches, patches_n, agg_ord="mean",
                                                                           patch_ord=2)

    W_next = getattr(model, layer_names[layer_index + 2]).weight.cuda()
    kernel_shape_next = W_next.shape[2]
    out_patches = extract_conv_patches(F.relu(output_), kernel_shape_next)
    out_patches_n = extract_conv_patches(F.relu(output_n_), kernel_shape_next)

    out_diff_norms, out_agg_norms, out_patches_norms, out_patches_n_norms = get_patch_diff(out_patches, out_patches_n,
                                                                                           agg_ord="mean", patch_ord=2)
    # out_diff_norms is the actual difference in norms in the output patches.
    # i now need to estimate this via the global Lipschtiz constant using W_flat_op_norm and patch norm for each patch.
    # each channel vector is W_flat * a specific patch (i,j) and the norm of this is bounded by W_flat_op_norm * patch_norm(i,j)

    global_lip_diff_norms = torch.zeros_like(out_diff_norms).cuda()
    global_lip_agg_norms = torch.zeros_like(out_agg_norms).cuda()
    sparse_lip_diff_norms = torch.zeros_like(out_diff_norms).cuda()
    sparse_lip_agg_norms = torch.zeros_like(out_agg_norms).cuda()
    diff_norms_sq = torch.square(diff_norms)
    fake_mask = torch.cuda.FloatTensor(kernel_shape_next, kernel_shape_next).uniform_() > 0.35
    print(diff_norms.shape)
    print(out_diff_norms.shape)
    for index in range(0, out_diff_norms.shape[0]):
        for i in range(0, out_diff_norms.shape[1]):
            for j in range(0, out_diff_norms.shape[2]):
                # sum all the diff_norms squared from i:i+kernel and j:j+kernel
                # if j + kernel_shape-1 == diff_norms.shape[2]:
                #    print(j)
                temp = diff_norms_sq[index, i:i + kernel_shape_next, j:j + kernel_shape_next]
                t1 = torch.sum(temp).item()

                # print("temp")
                # print(temp.shape)
                # print(fake_mask.shape)
                t2 = torch.sum(torch.mul(temp, fake_mask)).item()

                global_lip_diff_norms[index, i, j] = torch.sqrt((W_flat_op_norm ** 2) * t1)
                sparse_lip_diff_norms[index, i, j] = torch.sqrt((W_flat_op_norm_fake ** 2) * t2)
        global_lip_agg_norms[index] = torch.mean(global_lip_diff_norms[index]).item()
        sparse_lip_agg_norms[index] = torch.mean(sparse_lip_diff_norms[index]).item()

    return out_diff_norms, out_agg_norms, out_patches_norms, out_patches_n_norms, global_lip_diff_norms, global_lip_agg_norms, sparse_lip_diff_norms, sparse_lip_agg_norms


def record_diffs(layer_index, layer_names, manual_outs, inp, noise, manual_outs_n, model, num_channels):
    out_diff_norms = OrderedDict()
    out_agg_norms = OrderedDict()
    out_patches_norms = OrderedDict()
    out_patches_n_norms = OrderedDict()
    global_lip_diff_norms = OrderedDict()
    global_lip_agg_norms = OrderedDict()
    sparse_lip_diff_norms = OrderedDict()
    sparse_lip_agg_norms = OrderedDict()
    patch_radius = OrderedDict()
    optimal_patch_radius = OrderedDict()
    optimal_patch_sparsity = OrderedDict()

    layer_index = -1

    assert layer_index < 4
    assert layer_index > -2

    input_ = inp
    input_n_ = inp + noise

    W = getattr(model, 'conv1').weight
    kernel_shape = W.shape[2]

    patches = extract_conv_patches(input_, kernel_shape)
    patches_n = extract_conv_patches(input_n_, kernel_shape)
    print(patches.shape)
    batch_size = patches.shape[0]

    dn, an, pn, pnn = get_patch_diff(patches, patches_n, agg_ord="mean", patch_ord=2)
    out_diff_norms['input'] = dn
    out_agg_norms['input'] = an
    out_patches_norms['input'] = pn
    out_patches_n_norms['input'] = pnn

    # manual_outs, manual_outs_n = get_layer_outs()

    while layer_index < 3:
        print(layer_index)
        key = layer_names[layer_index + 1]
        odn, oan, opn, opnn, gldn, glan, sldn, slan = noise_propagation(layer_index)
        out_diff_norms[key] = odn
        out_agg_norms[key] = oan
        out_patches_norms[key] = opn
        out_patches_n_norms[key] = opnn
        global_lip_diff_norms[key] = gldn
        global_lip_agg_norms[key] = glan
        sparse_lip_diff_norms[key] = sldn
        sparse_lip_agg_norms[key] = slan

        pre_act = manual_outs[key]
        num_patches_x = pre_act.shape[2]
        num_patches_y = pre_act.shape[3]
        radius = torch.zeros_like(pre_act)
        assert radius.shape[1] == num_channels[key]

        radius[:, 0, :, :] = float("inf") * torch.ones(batch_size, num_patches_x, num_patches_y)
        temp = torch.stack(
            ([torch.min(torch.topk(-pre_act, k, dim=1).values, dim=1).values for k in range(1, num_channels[key])]))
        print(temp.shape)
        radius[:, 1:, :, :] = temp.permute(1, 0, 2, 3)
        patch_radius[key] = radius

        if layer_index == -1:
            diff_norms = out_diff_norms['input']
        else:
            diff_norms = out_diff_norms[layer_names[layer_index]]

        radius = radius.cuda()
        diff_norms = diff_norms.cuda()
        assert radius[:, 0, :, :].shape == diff_norms.shape

        optimal_sparsity = torch.ones_like(diff_norms)
        optimal_radius = float("inf") * torch.ones_like(diff_norms)
        s = [0] * num_channels[key]

        for index in range(0, batch_size):
            for i in range(0, num_patches_x):
                for j in range(0, num_patches_y):
                    for k in range(0, radius.shape[1]):
                        if radius[index, k, i, j] >= diff_norms[index, i, j]:
                            optimal_sparsity[index, i, j] = k
                            optimal_radius[index, i, j] = radius[index, k, i, j]

        optimal_patch_radius[key] = optimal_radius
        optimal_patch_sparsity[key] = optimal_sparsity

        layer_index += 1

    return out_diff_norms, out_agg_norms, out_patches_norms, out_patches_n_norms, global_lip_diff_norms, global_lip_agg_norms, sparse_lip_diff_norms, sparse_lip_agg_norms, patch_radius, optimal_patch_radius, optimal_patch_sparsity


def sll_conv_parallel(i, example, label, model, output, layer_outputs, patch_radius, stable_index_sets, cw):

    print('Processing example {} in batch'.format(i))

    margin = get_pointwise_margin(output, label).clone().detach()
    # print("Point wise margin for this example is : " + str(point_wise_margins[i]))

    if margin <= 0:
        # print('Aborted due to negative margin at index {}'.format(i))
        return

    # maximum_certifiable_radius_naive_op[i] = point_wise_margins[i] / (cw_batch[i] * naive_operator_norm_lip)

    layer_names = []
    for index, layer_name in enumerate(list(model._modules.keys())):
        layer_names.append(layer_name)

    delta_shape = tuple(layer_outputs[layer_names[0]].shape[2:])

    eps_low = 0.0
    eps_high = 0.1
    tol = 0.00001
    while eps_high > eps_low + tol:
        # print("Example : " + str(i) + ", Running binary search with eps_low : " + str(eps_low) + " and eps_high : " + str(eps_high))
        eps_mid = (eps_high + eps_low)/2
        delta_curr = torch.ones(delta_shape) * eps_mid
        # delta_curr = torch.abs(torch.randn(delta_shape)) * eps_mid

        greedy_sparsity = OrderedDict()
        greedy_radius = OrderedDict()
        greedy_lipschitz = OrderedDict()

        old_key = None
        for key in layer_names[:-2]:
            # print("Propagating delta to layer : " + key)
            # key here gives the output layer's key (all conv layers before fc and criterion)
            W = getattr(model, key).weight
            kernel_shape = tuple(W.shape[2:])
            W_flat = W.view(W.shape[0], -1)
            W_flat_norm = torch.linalg.matrix_norm(W_flat, ord=2)

            l_out = layer_outputs[key][i]

            radius = patch_radius[key][i]
            stable_ind_set = stable_index_sets[key][i]

            num_patches_x = radius.shape[1]
            num_patches_y = radius.shape[2]

            if not key.endswith('0'):
                # print("Aggregating current delta")
                delta_old = delta_curr
                delta_curr = torch.zeros(tuple(l_out.shape[1:]))
                chosen_lipschitz_prev = greedy_lipschitz[old_key]
                temp_del = chosen_lipschitz_prev * delta_old

                for u in range(delta_curr.shape[0]):
                    for v in range(delta_curr.shape[1]):
                        delta_curr[u, v] = torch.linalg.matrix_norm(temp_del[u:u+kernel_shape[0], v:v+kernel_shape[1]], ord='fro')

            chosen_sparsity = torch.zeros_like(delta_curr)
            chosen_radius = float("inf") * torch.ones_like(delta_curr)
            chosen_lipschitz = torch.ones_like(delta_curr) * W_flat_norm

            # print("Choosing sparsity, radius and lipschitz scales")
            num_reduced = 0
            for u in range(0, num_patches_x):
                for v in range(0, num_patches_y):
                    for c in range(0, radius.shape[0]):
                        if radius[c, u, v] >= delta_curr[u, v]:
                            chosen_sparsity[u, v] = c
                            chosen_radius[u, v] = radius[c, u, v]
                    row_index_set = stable_ind_set[int(chosen_sparsity[u, v]):, u, v]

                    W_flat_red = W_flat

                    # print("Before reducing flattened shape is : " + str(W_flat_red.shape))
                    if not key.endswith('0'):
                        W_old = getattr(model, old_key).weight.cpu()
                        kernel_shape_old = tuple(W_old.shape[2:])
                        chosen_sparsity_old = greedy_sparsity[old_key]
                        stable_ind_set_old = stable_index_sets[old_key][i]

                        col_mask = torch.ones((W_old.shape[0], kernel_shape_old[0], kernel_shape_old[1]), dtype=torch.bool)

                        for k1 in range(0, kernel_shape_old[0]):
                            for k2 in range(0, kernel_shape_old[1]):
                                # print("Reduced sparsity of older channel vector is : " + str(int(chosen_sparsity_old[u+k1, v+k2])))
                                temp_set = stable_ind_set_old[:int(chosen_sparsity_old[u+k1, v+k2]), u+k1, v+k2]
                                col_mask[temp_set, k1, k2] = False
                        W_flat_red = W[:, col_mask]

                    if W_flat_red.shape[1] == 0 or len(row_index_set) == 0:
                        chosen_lipschitz[u, v] = 0.0
                        num_reduced += 1
                    elif len(row_index_set) < radius.shape[0]:
                        num_reduced += 1
                        chosen_lipschitz[u, v] = torch.linalg.matrix_norm(W_flat_red[row_index_set], ord=2)

            # print("Number of reduced lipschitz scales is : " + str(num_reduced) + " / " + str(num_patches_x*num_patches_y))
            greedy_sparsity[key] = chosen_sparsity
            greedy_radius[key] = chosen_radius
            greedy_lipschitz[key] = chosen_lipschitz
            old_key = key

        # print("Aggregating delta for final fc layer")
        delta_old = delta_curr
        chosen_lipschitz_prev = greedy_lipschitz[old_key]
        delta_curr = torch.linalg.matrix_norm(chosen_lipschitz_prev * delta_old, ord='fro')

        # print("The perturbation in the final output layer is : " + str(delta_curr))

        safe_threshold = margin/cw
        # print("Safety threshold is : " + str(safe_threshold))

        if delta_curr <= safe_threshold:
            eps_low = eps_mid
        else:
            eps_high = eps_mid

    print('Results for example {}'.format(i))
    print('Best sparse certificate for patch 2->inf norm is {} '.format(eps_low))
    eps_red_op = eps_low
    eps_op = None

    eps_low = 0.0
    eps_high = 0.1
    tol = 0.00001
    while eps_high > eps_low + tol:
        # print("Example : " + str(i) + ", Running binary search with eps_low : " + str(eps_low) + " and eps_high : " + str(eps_high))
        eps_mid = (eps_high + eps_low) / 2
        delta_curr = torch.ones(delta_shape) * eps_mid
        # delta_curr = torch.abs(torch.randn(delta_shape)) * eps_mid

        old_key = None
        wnorms = OrderedDict()
        for key in layer_names[:-2]:
            # print("Propagating delta to layer : " + key)
            # key here gives the output layer's key (all conv layers before fc and criterion)
            W = getattr(model, key).weight
            kernel_shape = tuple(W.shape[2:])
            W_flat = W.view(W.shape[0], -1)
            wnorms[key] = torch.linalg.matrix_norm(W_flat, ord=2)

            l_out = layer_outputs[key][i]

            num_patches_x = l_out.shape[1]
            num_patches_y = l_out.shape[2]

            if not key.endswith('0'):
                # print("Aggregating current delta")
                delta_old = delta_curr
                delta_curr = torch.zeros(tuple(l_out.shape[1:]))

                for u in range(delta_curr.shape[0]):
                    for v in range(delta_curr.shape[1]):
                        delta_curr[u, v] = wnorms[old_key] * torch.linalg.matrix_norm(delta_old[u:u+kernel_shape[0], v:v+kernel_shape[1]], ord='fro')

            old_key = key

        # print("Aggregating delta for final fc layer")
        delta_old = delta_curr
        delta_curr = wnorms[old_key] * torch.linalg.matrix_norm(delta_old, ord='fro')
        # print("The perturbation in the final output layer is : " + str(delta_curr))

        safe_threshold = margin / cw
        # print("Safety threshold is : " + str(safe_threshold))

        if delta_curr <= safe_threshold:
            eps_low = eps_mid
        else:
            eps_high = eps_mid

    print('Results for example {}'.format(i))
    print('Best naive certificate for patch 2->inf norm is {} '.format(eps_low))

    eps_op = eps_low

    return [eps_red_op, eps_op]


def SLL_conv_eval(model,
                  feedback,
                  loader: DataLoader = None,
                  test_output_location: str = None,
                  data_str: str = None,
                  type_str: str = None,
                  random_batch_index: int = 0,
                  num_bins: int = 20,
                  verbose: bool = False):
    if verbose:
        print('\n' + '-' * 20 + 'Running Patch SLL Lipschitz Evaluation for Convolutional Networks: ' + '-' * 20)

    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    name = 'lipschitz_evaluation_' + data_str
    info = {}
    output_location = test_output_location + '/' + data_str + '/lipschitz_evaluation/patch_lip/'
    # omitting final fully connected layer and loss criterion layer from module list
    num_layers = len(list(model._modules.keys())) - 2
    selection = list(range(0, num_layers))
    channel_dims = OrderedDict()  # getting layer wise channel dimensions
    for index, layer_name in enumerate(list(model._modules.keys())):
        print(layer_name)
        if index in selection:
            channel_dims[layer_name] = getattr(model, layer_name).out_channels

    forward_hook_model = ForwardHookModel(model, output_layers=selection)
    forward_hook_model.eval()

    n_batches = 1
    batch_size = 128
    total_size = batch_size * n_batches

    total_maximum_certifiable_radius_patch_local_op = torch.zeros(total_size)
    total_maximum_certifiable_radius_patch_global_op = torch.zeros(total_size)

    current_batch_index = -1
    current_batch_sub_index = -1
    for examples, labels in loader:
        current_batch_index += 1
        if current_batch_index > random_batch_index+n_batches-1:
            break
        if current_batch_index < random_batch_index:
            continue

        current_batch_sub_index += 1

        print("INFERENCE for batch number - {}".format(current_batch_index - random_batch_index))
        examples = examples.to(get_platform().torch_device)
        labels = labels.squeeze().to(get_platform().torch_device)
        output, layer_outputs = forward_hook_model(examples)
        predicted_labels = output.argmax(dim=1)
        assert batch_size == torch.tensor(len(labels), device=get_platform().torch_device).item()

        patch_radius = OrderedDict()
        stable_index_sets = OrderedDict()

        for i in range(0, len(selection)):
            key = f'convlayer{i}'
            l_out = layer_outputs[key]

            assert batch_size == l_out.shape[0]
            assert l_out.shape[1] == channel_dims[key]
            num_patches_x = l_out.shape[2]
            num_patches_y = l_out.shape[3]
            # print(l_out.shape)

            W = getattr(model, key).weight
            W_flat = W.view(W.shape[0], -1)
            w_norms = torch.norm(W_flat, p=2, dim=1)

            # computing normalized activation vectors
            l_out = l_out.permute(0,2,3,1)
            l_out = l_out.div(w_norms)
            l_out = l_out.permute(0,3,1,2)

            radius = torch.zeros_like(l_out,  device=get_platform().torch_device)

            radius[:, 0, :, :] = float("inf") * torch.ones(batch_size, num_patches_x, num_patches_y)
            radius[:, 1:, :, :] = torch.stack(
                (
                    [
                        torch.min(torch.topk(-l_out, k, dim=1).values, dim=1).values for k in range(1, channel_dims[key])
                    ]
                )
            ).permute(1, 0, 2, 3)
            patch_radius[key] = radius
            stable_index_sets[key] = torch.argsort(-l_out, dim=1, descending=True)

        # maximum_certifiable_radius_naive_op = torch.zeros(batch_size)
        # naive_operator_norm_lip = get_naive_lip(model) # this should be modified to something that computes the singular values of convolutional layers.

        maximum_certifiable_radius_patch_global_op = torch.zeros(batch_size)
        maximum_certifiable_radius_patch_local_op = torch.zeros(batch_size)

        cw_batch = get_classifier_constant(model, predicted_labels, True)

        model_copy = copy.deepcopy(model)
        model_copy.to('cpu')
        rad = Parallel(n_jobs=2, verbose=20, require='sharedmem')(
            delayed(sll_conv_parallel)(i, examples[i], labels[i], model_copy, output[i, :], layer_outputs, patch_radius, stable_index_sets, cw_batch[i])
            for i in range(int(batch_size)))

        # now need to append the single batch statistics to the total statistics
        for i in range(batch_size):
            if rad[i] is not None:
                maximum_certifiable_radius_patch_local_op[i] = rad[i][0]
                maximum_certifiable_radius_patch_global_op[i] = rad[i][1]
        total_maximum_certifiable_radius_patch_local_op[current_batch_sub_index * batch_size: current_batch_sub_index * batch_size + batch_size] = maximum_certifiable_radius_patch_local_op
        total_maximum_certifiable_radius_patch_global_op[current_batch_sub_index * batch_size: current_batch_sub_index * batch_size + batch_size] = maximum_certifiable_radius_patch_global_op

    if verbose:
        print('The average local certificate radius for the total batch using just the reduced spectral norm is {}'.
              format(torch.mean(total_maximum_certifiable_radius_patch_local_op).item()))

    info['lipschitz_type'] = type_str
    info['operator_norm_cert'] = total_maximum_certifiable_radius_patch_global_op
    info['operator_norm_lip'] = None
    info['reduced_op_norm_cert'] = total_maximum_certifiable_radius_patch_local_op
    info['reduced_op_norm_lip'] = None
    feedback[name] = info


