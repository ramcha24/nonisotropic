import torch
import numpy as np
from collections import defaultdict
from random import randrange

from joblib import Parallel, delayed

from datasets.base import DataLoader

from models.forward_hook_model import ForwardHookModel

from platforms.platform import get_platform

from utilities.capacity_utils import get_2_inf_norm, get_classifier_constant, get_naive_lip, get_reduced_classifier_constant
from utilities.evaluation_utils import get_pointwise_margin
from utilities.plotting_utils import plot_hist, plot_critical_angle


def check_nontrivial_activity(sel, layer_outs, layer_ds):
    maximum_activity_batch = [1] * len(sel)
    #print('printing layer shapes')
    #for i in range(0, len(sel)):
    #    print(layer_outs[f'layer{i}'].shape)
    #    print(type(layer_outs[f'layer{i}']))
    batch_shape = layer_outs['layer0'].shape[0]

    # rewrite below into a single batch function call
    for i in range(0, batch_shape):
        for j in range(0, len(sel)):
            l_out = layer_outs[f'layer{j}'][i, :]
            num_active = l_out[l_out > 0].shape[0]
            if num_active > maximum_activity_batch[j]:
                maximum_activity_batch[j] = num_active

    e_flag = True
    j = 0
    while e_flag and (j < len(sel)):
        e_flag = e_flag and (maximum_activity_batch[j] < layer_ds[j])
        j = j + 1
    return e_flag, maximum_activity_batch


def dummy(batch_shape , sel, layer_outs):
    layerwise_encoder_gaps_batch = []  # for each sample in batch, should contain an array of length L
    layerwise_permuted_indices_batch = []
    for i in range(0, batch_shape):
        layerwise_tau = []
        layerwise_permuted_indices = []

        for j in range(0, len(sel)):
            # the raw layer output pre-activation
            l_out = layer_outs[f'layer{j}'][i, :]
            l_reverse_relu_copy = torch.Tensor([torch.abs(x) if x < 0 else 0 for x in l_out])
            tau, permuted_indices = l_reverse_relu_copy.sort()

            # the size of activity level
            # sbar = l_out[l_out > 0].shape[0]
            # the encoder gap tau for this layer across activity levels
            # tau = torch.zeros(layer_ds[j])
            # sorted value of layer output among the inactive atoms
            # l_inactive_sorted = torch.abs(l_out[l_out <= 0]).sort().values
            # the encoder gap from s-bar to p is set based on b
            # tau[sbar:layer_ds[j]] = l_inactive_sorted
            layerwise_tau.append(tau.detach().cpu())
            layerwise_permuted_indices.append(permuted_indices.detach().cpu())

        layerwise_encoder_gaps_batch.append(layerwise_tau)
        layerwise_permuted_indices_batch.append(layerwise_permuted_indices)


def get_encoder_gap_statistics(sel, layer_outs, layer_ds, layerwise_encoder_gaps_batch):
    # print('what is the layer out shape {}'.format(layer_outs['layer0'].shape))
    batch_shape = layer_outs['layer0'].shape[0]

    worst_layerwise_encoder_gaps = []  # across the whole batch
    average_layerwise_encoder_gaps = []  # across the whole batch
    best_layerwise_encoder_gaps = []  # across the whole batch

    for j in range(0, len(sel)):
        worst_layerwise_encoder_gaps.append(torch.ones(layer_ds[j]) * float("inf"))
        average_layerwise_encoder_gaps.append(torch.ones(layer_ds[j]) * float("inf"))
        best_layerwise_encoder_gaps.append(torch.ones(layer_ds[j]) * float(0.0))

    for j in range(0, len(sel)):
        for k in range(0, layer_ds[j]):
            total = 0.0
            for i in range(0, batch_shape):
                total += layerwise_encoder_gaps_batch[i][j][k]

                if best_layerwise_encoder_gaps[j][k] < layerwise_encoder_gaps_batch[i][j][k]:
                    best_layerwise_encoder_gaps[j][k] = layerwise_encoder_gaps_batch[i][j][k]

            average_layerwise_encoder_gaps[j][k] = total / batch_shape

            if worst_layerwise_encoder_gaps[j][k] == 0.0:
                continue

            for i in range(0, batch_shape):
                if worst_layerwise_encoder_gaps[j][k] > layerwise_encoder_gaps_batch[i][j][k]:
                    worst_layerwise_encoder_gaps[j][k] = layerwise_encoder_gaps_batch[i][j][k]
        best_layerwise_encoder_gaps[j].detach().numpy()
        average_layerwise_encoder_gaps[j].detach().numpy()
        worst_layerwise_encoder_gaps[j].detach().numpy()

    return worst_layerwise_encoder_gaps, average_layerwise_encoder_gaps, best_layerwise_encoder_gaps


class DfsLookup:
    def __init__(self,
                 model,
                 margin,
                 layer_outputs,
                 label,
                 layer_encoder_gaps_batch,
                 layer_permuted_indices_batch,
                 chosen_activity_levels,
                 naive_certificate):
        self.model = model
        self.best_overall_cert = 0.0
        self.chosen_red_op = float("inf")
        self.label = label
        self.chosen_activity_levels = chosen_activity_levels
        self.chosen_activation_gaps = layer_encoder_gaps_batch
        self.layer_outputs = layer_outputs
        self.margin = margin.detach().cpu()                                    # Point-wise margin for the given sample x
        self.naive_certificate = naive_certificate.detach().cpu()              # certificate for the given sample obtained by product of operator norms.

    def set_cert(self, val):
        self.best_overall_cert = val

    def get_cert(self): return self.best_overall_cert

    def set_lip(self, val):
        self.chosen_red_op = val

    def set_chosen_activity(self, j, val):
        self.chosen_activity_levels[j] = val
        # print('should have set the activity levels at layer {} to {} and it is now {}'.format(j, val, self.get_chosen_activity()))

    def set_chosen_activation_gap(self, layer_index, inp_index, val):
        self.chosen_activation_gaps[inp_index][layer_index] = val

    def get_chosen_activity(self): return self.chosen_activity_levels

    def get_lip(self): return self.chosen_red_op

    def get_label(self): return self.label

    # def get_tau(self, layer_index, activity_level): return self.taus[layer_index][activity_level]

    def get_margin(self): return self.margin

    def get_layer_output(self, layer_index, inp_index):
        return self.layer_outputs[f'layer{layer_index}'][inp_index, :]

    def get_layer_weight(self, layer_index):
        return self.model.layers[layer_index].weight

    def get_fc_weight(self):
        return self.model.fc.weight

    # def get_layer_weight(self, layer_index): return self.layer_permuted_weights[layer_index]

    # def get_cw(self): return self.cw

    def get_naive_cert(self): return self.naive_certificate

    #def get_reduced_layer_row_norm(self, layer_index, row_activity_level, col_activity_level):
    #    return torch.max(torch.norm(self.layer_permuted_weights[layer_index][0:row_activity_level, 0:col_activity_level], p=2, dim=1))


def explore_cert_paths(src,                                     # Tuple (layer_index, current_activity_level)
                       graph,                                   # graph dict with vertices as possible activity levels in each layer and all possible edges connections
                       visited_nodes,                           # boolean array to keep track of visited nodes
                       previous_activity_level,                 # activity level chosen for permuted layer weight at layer src[0]-1
                       previous_row_perm,                       # permutation of rows in previous layer based on chosen activity levels
                       layer_reduced_weights_so_far,            # Product of reduced weights up till layer : src[0]-1
                       reduced_op_norm_so_far,                  # Operator norm of layer_reduced_weights_so_far computed in previous layer
                       minimum_certificate_so_far,              # minimum certificate across all layers till layer src[0]-1
                       activity_levels_so_far,                  # List of activity levels chosen so far. Gives the path.
                       activation_gaps_so_far,
                       lookup,                                  # DfsLookup class instance to hold global parameter values that can be shared or accessed between different branches of the recursion.
                       i):
    visited_nodes[src] = True ##  == False
    if src == 's':
        for (layer, activity_level) in graph[src]:
            if visited_nodes[(layer, activity_level)] is False:
                explore_cert_paths((layer, activity_level),
                                   graph,
                                   visited_nodes,
                                   previous_activity_level,
                                   previous_row_perm,
                                   layer_reduced_weights_so_far,
                                   reduced_op_norm_so_far,
                                   minimum_certificate_so_far,
                                   activity_levels_so_far,
                                   activation_gaps_so_far,
                                   lookup,
                                   i)
    else:
        naive_cert = lookup.get_naive_cert()
        margin = lookup.get_margin()
        # cw = lookup.get_cw()

        current_layer_index = src[0]
        current_activity_level = src[1]

        #print('Trying Layer {} and activity level {}'.format(current_layer_index, current_activity_level))

        if minimum_certificate_so_far < max(lookup.get_cert(), naive_cert):
            print('WTF')
            print(minimum_certificate_so_far)
            print(lookup.get_cert())
            print(naive_cert)
            raise ValueError('shit')

        assert previous_activity_level != 0
        assert current_activity_level != 0

        current_layer_weight = lookup.get_layer_weight(current_layer_index)
        col_perm = previous_row_perm
        col_permuted_layer_weight = torch.zeros_like(current_layer_weight)
        col_permuted_layer_weight[:, col_perm] = current_layer_weight
        current_layer_output = lookup.get_layer_output(current_layer_index, i)
        current_layer_dim = len(current_layer_output)
        current_activation_gap = torch.zeros_like(current_layer_output)

        for k in range(current_layer_dim):
            if current_layer_output[k] < 0:
                current_activation_gap[k] = torch.abs(current_layer_output[k]) / torch.norm(col_permuted_layer_weight[k, 0: previous_activity_level])

        sorted_activation_gap, current_row_perm = current_activation_gap.sort()
        tau_val = sorted_activation_gap[current_activity_level-1]

        #l_out = layer_outs[f'layer{j}'][i, :]
        #l_reverse_relu_copy = torch.Tensor([torch.abs(x) if x < 0 else 0 for x in current_layer_output])
        #tau, permuted_indices = l_reverse_relu_copy.sort()
        #orig_reduced_norm = torch.max(torch.norm(current_layer_weights[layer_index][0:row_activity_level, 0:col_activity_level], p=2, dim=1))
        #print('The original encoder gap was - ')

        # col_permuted_weight = torch.zeros_like(layer_weight)
        # col_perm_index = torch.arange(0, layer_weight.shape[1])
        # if j > 0:
        #   col_perm_index = layer_permuted_indices_batch[i][j-1]
        #   Col_permuted_weight[:, col_perm_index] = layer_weight
        #   row_col_permuted_weight = torch.zeros_like(layer_weight)
        #   row_perm_index = layer_permuted_indices_batch[i][j]
        #   row_col_permuted_weight[row_perm_index] = col_permuted_weight
        #   layer_permuted_weights.append(row_col_permuted_weight.detach().cpu())

        # col permutation using previous row permutation.

        #tau_val = lookup.get_tau(current_layer_index, current_activity_level-1).detach()
        #reduced_layer_row_norm = lookup.get_reduced_layer_row_norm(current_layer_index, current_activity_level, previous_activity_level).detach()
        # print('The reduced layer row norm at layer {} and activity level {} is {}'.format(current_layer_index, current_activity_level, reduced_layer_row_norm))
        #print('tau is')
        #print(tau_val)
        #print('reduced_op_norm is')
        #print(reduced_op_norm_so_far)
        # tau_val = taus[layer_index][chosen_activity_level].detach().cpu()
        current_cert = tau_val / reduced_op_norm_so_far
        current_cert = current_cert.detach().cpu()
        #print('naive_cert is')
        #print(naive_cert)
        #print('look up cert is')
        #print(lookup.get_cert())
        #print('current cert is')
        #print(current_cert)
        #print('minimum cert is')
        #print(minimum_certificate_so_far)
        cert_so_far = min(current_cert, minimum_certificate_so_far)

        #if current_activity_level % 40 == 0:
        #    print('processing layer {} and activity_level {}'.format(current_layer_index, current_activity_level))

        # if cert_so_far > max(best_overall_cert, naive_cert):
        if cert_so_far > max(lookup.get_cert(), naive_cert):
            #print('initial breakthrough')
            activity_levels_so_far.append(current_activity_level)
            activation_gaps_so_far.append(sorted_activation_gap)
            row_col_permuted_layer_weight = torch.zeros_like(current_layer_weight)
            row_col_permuted_layer_weight[current_row_perm] = col_permuted_layer_weight
            new_reduced_weights = row_col_permuted_layer_weight[0:current_activity_level, 0:previous_activity_level]
            try:
                new_reduced_op_norm = torch.linalg.norm(new_reduced_weights, ord=2).item()
            except:
                new_reduced_op_norm = torch.linalg.norm(new_reduced_weights + 1e-4 * new_reduced_weights.mean() * torch.randn_like(new_reduced_weights), ord=2).item()

            reduced_op_norm_so_far = new_reduced_op_norm * reduced_op_norm_so_far

            # if this path has only 'd' left then we first check for the margin condition ,then we have a new best_cert!
            if len(graph[src]) == 1 and graph[src][0] == 'd':
                fc_weight = lookup.get_fc_weight()
                col_permuted_fc_weight = torch.zeros_like(fc_weight)
                col_permuted_fc_weight[:, current_row_perm] = fc_weight
                cw = get_reduced_classifier_constant(margin, col_permuted_fc_weight, current_activity_level, lookup.get_label())
                margin_cert = margin / (reduced_op_norm_so_far * cw)
                #print('margin_cert is {}'.format(margin_cert))
                #print('look up cert is {} and naive_cert is {}'.format(lookup.get_cert(), naive_cert))
                if margin_cert > max(lookup.get_cert(), naive_cert):
                    print('[{}] Found a viable path! with activiation_levels {} where cert_so_far is {} and margin_cert is {}, lookup.get_cert() is {}'.format(i, activity_levels_so_far, cert_so_far, margin_cert, lookup.get_cert()))
                    print('[{}] Found a vaiable path! with activation_levels {} where margin is {} and reduced_cw is {}'.format(i, activity_levels_so_far, margin, cw))
                    best_overall_cert = max(min(cert_so_far, margin_cert), lookup.get_cert())
                    lookup.set_cert(best_overall_cert)
                    lookup.set_lip(reduced_op_norm_so_far)
                    for l_index in range(len(activity_levels_so_far)):
                        lookup.set_chosen_activity(l_index, activity_levels_so_far[l_index])
                        lookup.set_chosen_activation_gap(l_index, i, activation_gaps_so_far[l_index])

                    print('[{}] Found a viable path! with activity_levels {} with new best certificate of {} with chosen reduced operator norm {}'.format(i, activity_levels_so_far, lookup.get_cert(), lookup.get_lip()))
                else:
                    if current_activity_level % 20 == 0:
                        yolo = False
                        # print('failed in margin cert with activity levels {} as margin_cert {} less than lookup {}'.format(activity_levels_so_far, margin_cert, lookup.get_cert()))
            else:
                for item in graph[src]:
                    #print('Going from src {} to item {}'.format(src, item))
                    new_layer_index = item[0]
                    new_activity_level = item[1]
                    if visited_nodes[(new_layer_index, new_activity_level)] is False:
                        explore_cert_paths((new_layer_index, new_activity_level),
                                           graph,
                                           visited_nodes,
                                           current_activity_level,
                                           current_row_perm,
                                           new_reduced_weights,
                                           reduced_op_norm_so_far,
                                           cert_so_far,
                                           activity_levels_so_far,
                                           activation_gaps_so_far,
                                           lookup,
                                           i)
            activity_levels_so_far.pop()
            activation_gaps_so_far.pop()
        else:
            if previous_activity_level % 20 == 0 and current_activity_level % 20 == 0:
                yolo = False
            #print('terminating recursion in path {} with extension {} as cert_so_far {} less than min of best_cert {} and naive_cert {}'.format(activity_levels_so_far, current_activity_level, cert_so_far, lookup.get_cert(), naive_cert))
    # Remove current vertex from path[] and mark it as unvisited
    visited_nodes[src] = False


def sparse_local_lip_parallel(i,
                              model,
                              output,
                              layer_outputs,
                              labels,
                              layer_encoder_gaps_batch,
                              layer_permuted_indices_batch,
                              layer_dims,
                              point_wise_margins,
                              maximum_certifiable_radius_naive_op,
                              cw_batch,
                              naive_operator_norm_lip,
                              num_layers,
                              maximum_certifiable_radius_reduced_op,
                              chosen_local_lip_reduced_op,
                              chosen_activity_levels_layer_wise
                              ):
    #if i > 1 :
    #    return

    print('Processing example {} in batch'.format(i))

    out = output[i, :]
    point_wise_margins[i] = get_pointwise_margin(out, labels[i]).clone().detach()

    if point_wise_margins[i] <= 0:
        print('Aborted due to negative margin at index {}'.format(i))
        return

    maximum_certifiable_radius_naive_op[i] = point_wise_margins[i] / (cw_batch[i] * naive_operator_norm_lip)

    minimum_activity_level = []
    layer_resolution = []
    for j in range(num_layers):
        # attempt to find an appropriate sparsity level layer by layer
        # first valid sparsity level across batch
        l_out = layer_outputs[f'layer{j}'][i, :]
        # current_tau = taus[j].detach().cpu()
        # first_positive = torch.sum(current_tau <= 0).detach().cpu().data + 1
        # resolution = int(len(current_tau)/25)  # for unregularized network i dont need to be as fine
        # resolution = int(len(current_tau)/75)
        if j == 0:
            resolution = 5
        else:
            resolution = 10
        layer_resolution.append(max(resolution, 1))
        minimum_activity_level.append(int(torch.sum(l_out >= 0).data))
        # layer_weight = model.layers[j].weight
        # col_permuted_weight = torch.zeros_like(layer_weight)
        # col_perm_index = torch.arange(0, layer_weight.shape[1])
        # if j > 0:
        #   col_perm_index = layer_permuted_indices_batch[i][j-1]
        #   Col_permuted_weight[:, col_perm_index] = layer_weight
        #   row_col_permuted_weight = torch.zeros_like(layer_weight)
        #   row_perm_index = layer_permuted_indices_batch[i][j]
        #   row_col_permuted_weight[row_perm_index] = col_permuted_weight
        #   layer_permuted_weights.append(row_col_permuted_weight.detach().cpu())

    # form the encoder gap graph
    graph = defaultdict(list)
    visited_nodes = defaultdict(bool)
    prev_dim = layer_dims[0]
    prev_minimum_activity_level = minimum_activity_level[0]
    # temp = np.linalg.norm(permuted_weight[0:chosen_activity_level].clone().detach().cpu().numpy(), ord=2)

    for j in range(num_layers):
        if j == 0:
            graph['s'] = []
            visited_nodes['s'] = False
            for u in range(prev_minimum_activity_level, prev_dim, layer_resolution[0]):
                graph['s'].append((j, u))
                # print('adding connection from s to ({},{})'.format(j, u))
        if j > 0:
            current_dim = layer_dims[j]
            current_minimum_activity_level = minimum_activity_level[j]
            upper_0 = prev_dim + 1
            upper_1 = current_dim + 1
            lower_0 = prev_minimum_activity_level
            lower_1 = current_minimum_activity_level

            for u in range(lower_0, upper_0, layer_resolution[j]):
                if (j, u) not in graph.keys():
                    graph[(j, u)] = []
                if (j, u) not in visited_nodes.keys():
                    visited_nodes[(j, u)] = False

                for v in range(lower_1, upper_1, layer_resolution[j]):
                    # print('adding connection ({},{}) to ({},{})'.format(j-1, u, j, v))
                    graph[(j - 1, u)].append((j, v))

            if j == num_layers - 1:
                # print("in here too")
                visited_nodes['d'] = False
                for v in range(lower_1, upper_1, layer_resolution[j]):
                    # print('adding connection ({},{}) to d)'.format(num_layers-1, v))
                    graph[(num_layers - 1, v)].append('d')

            prev_dim = current_dim
            prev_minimum_activity_level = current_minimum_activity_level

    lookup = DfsLookup(model,
                       point_wise_margins[i],
                       layer_outputs,
                       labels[i],
                       layer_encoder_gaps_batch,
                       layer_permuted_indices_batch,
                       layer_dims.copy(),
                       maximum_certifiable_radius_naive_op[i])

    explore_cert_paths('s',
                       graph,
                       visited_nodes,
                       28 * 28,
                       torch.arange(0, 28*28),
                       torch.eye(28 * 28).cuda(),
                       1.0,
                       float("inf"),
                       [],
                       [],
                       lookup,
                       i)
    best_cert = lookup.get_cert()
    if best_cert == 0.0:
        best_cert = maximum_certifiable_radius_naive_op[i]

    reduced_op_lip = lookup.get_lip()
    if reduced_op_lip == float("inf"):
        reduced_op_lip = naive_operator_norm_lip

    activity_levels = lookup.get_chosen_activity()
    # print(len(lookup.get_chosen_activity()))
    print('Results for example {}'.format(i))
    print('naive certificate is {} and naive lip is {}'.format(maximum_certifiable_radius_naive_op[i],
                                                                       naive_operator_norm_lip))
    print('best sparse certificate is {} with local lip {}'.format(best_cert, reduced_op_lip))
    print('the chosen activity level is {}'.format(activity_levels))
    for j in range(num_layers):
        chosen_activity_levels_layer_wise[j][i] = activity_levels[j]
        # print('Did i set the activity level for example {} at layer {} to {} correctly - {}'.format(i, j, activity_levels[j], chosen_activity_levels_layer_wise[j][i]))
        maximum_certifiable_radius_reduced_op[i] = best_cert
        chosen_local_lip_reduced_op[i] = reduced_op_lip


def SLL_feedforward_eval(model,
                         feedback,
                         loader: DataLoader = None,
                         test_output_location: str = None,
                         data_str: str = None,
                         type_str: str = None,
                         random_batch_index: int = 0,
                         num_bins: int = 20,
                         verbose: bool = False):
    if verbose:
        print('\n' + '-' * 20 + 'Running SLL Lipschitz Evaluation for Feedforward Networks: ' + '-' * 20)

    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    name = 'lipschitz_evaluation_' + data_str
    info = {}
    output_location = test_output_location + '/' + data_str + '/lipschitz_evaluation/sparse_local_lip/'
    # omitting final fully connected layer and loss criterion layer from module list
    num_layers = len(list(model._modules.keys())) - 2
    selection = list(range(0, num_layers))
    layer_dims = []  # getting layer wise output dimensions
    for index, layer_name in enumerate(list(model._modules.keys())):
        if index in selection:
            layer_dims.append(getattr(model, layer_name).out_features)

    forward_hook_model = ForwardHookModel(model, output_layers=selection)
    forward_hook_model.eval()

    ########################################################################################################
    # here calculate the worst-case activity in each layer across any input in the entire test set.
    layer_wise_max_activity = [0]*len(layer_dims)
    all_data_exists_flag = True

    layer_wise_data_activity = [[]] * num_layers

    for examples, labels in loader:
        examples = examples.to(get_platform().torch_device)
        output, layer_outputs = forward_hook_model(examples)

        layer_wise_batch_activity = [[]] * num_layers
        batch_shape = layer_outputs['layer0'].shape[0]

        # rewrite below into a single batch function call
        for j in range(0, num_layers):
            for i in range(0, batch_shape):
                l_out = layer_outputs[f'layer{j}'][i, :]
                num_active = l_out[l_out > 0].shape[0]
                layer_wise_batch_activity[j].append(num_active)
            layer_wise_data_activity[j] = layer_wise_data_activity[j] + layer_wise_batch_activity[j]

        exists_flag, activity_levels = check_nontrivial_activity(selection, layer_outputs, layer_dims)

        for layer_index in range(0, len(layer_dims)):
            layer_wise_max_activity[layer_index] = max(layer_wise_max_activity[layer_index], activity_levels[layer_index])

        all_data_exists_flag = all_data_exists_flag and exists_flag

    print('sanity check : size of layer 0 activity across dataset is {}'.format(len(layer_wise_data_activity[0])))

    if not all_data_exists_flag:
        if verbose:
            print(
                'No luck, there exists at least one example in the {} data which saturates activity'.format(data_str))
            print(layer_wise_max_activity)
    else:
        if verbose:
            print('Good case! There is non trivial max activity in the {} data'.format(data_str))
            print(layer_wise_max_activity)
            num = 0.0
            den = 0.0
            for j in range(0, len(selection)):
                num += layer_wise_max_activity[j]
                den += layer_dims[j]
            print('Only {:.3f}% of neurons are overall worst case active in {} data'.format(num / den, data_str))

    for j in range(num_layers):
        plot_hist(output_location, 'num_active_neurons_at_layer ' + str(j + 1) + '.pdf',
                  'Number of Active Neurons at Layer ' + str(j + 1),
                  layer_wise_data_activity[j], hist_bins=40)

    ###############################################################################################
    print("resuming")
    total_size = 128*10

    total_layerwise_chosen_activity_levels = []
    for j in range(num_layers):
        total_layerwise_chosen_activity_levels.append([layer_dims[j]] * total_size)

    total_chosen_local_lip_reduced_op = torch.ones(total_size) * float("inf")

    total_maximum_certifiable_radius_naive_op = torch.zeros(total_size)
    total_maximum_certifiable_radius_reduced_op = torch.zeros(total_size)

    total_clean_red_op_lip = []


    current_batch_index = -1
    current_batch_sub_index = -1
    for examples, labels in loader:
        current_batch_index += 1
        if current_batch_index > random_batch_index+9:
            break
        if current_batch_index < random_batch_index:
            continue
        current_batch_sub_index += 1

        print("INFERENCE for batch number - {}".format(current_batch_index - random_batch_index))
        examples = examples.to(get_platform().torch_device)
        labels = labels.squeeze().to(get_platform().torch_device)
        output, layer_outputs = forward_hook_model(examples)
        predicted_labels = output.argmax(dim=1)
        batch_size = torch.tensor(len(labels), device=get_platform().torch_device).item()

        exists_flag, activity_levels = check_nontrivial_activity(selection, layer_outputs, layer_dims)

        if not exists_flag:
            if verbose:
                print('No luck, there exists at least one example in {} batch which saturates activity'.format(data_str))
                print(activity_levels)
        else:
            if verbose:
                print('Ooh! lets see the max layer wise activity in {} batch'.format(data_str))
                print(activity_levels)
                num = 0.0
                den = 0.0
                for j in range(0, len(selection)):
                    num += activity_levels[j]
                    den += layer_dims[j]
                print('Only {:.3f}% of neurons are overall worst case active in {} batch'.format(num/den, data_str))
        layer_norms = get_2_inf_norm(model)
        point_wise_margins = torch.zeros(batch_size)

        maximum_certifiable_radius_naive_op = torch.zeros(batch_size)
        maximum_certifiable_radius_reduced_op = torch.zeros(batch_size)
        layerwise_encoder_gaps_batch = [[torch.zeros(layer_dims[j]) for j in range(num_layers)] for i in range(0, batch_size)]
        layerwise_permuted_indices_batch = [[] for i in range(0,batch_size)]

        layerwise_chosen_activity_levels = []
        for j in range(num_layers):
            layerwise_chosen_activity_levels.append([layer_dims[j]] * batch_size)

        chosen_local_lip_reduced_op = torch.ones(batch_size) * float("inf")
        naive_operator_norm_lip = get_naive_lip(model)

        if verbose:
            print('operator norm is {}'.format(naive_operator_norm_lip))

        cw_batch = get_classifier_constant(model, predicted_labels, True)
        if verbose:
            print("Average Classifier constant is {}".format(np.average(cw_batch)))
            print(cw_batch)
            print("Layer norms :")
            print(layer_norms)

        if verbose:
            print('-' * 5 + 'Certifying Robustness via Local Lipschitzness' + '-' * 5)

        total_dims = 0
        for dim in layer_dims:
            total_dims += dim

        Parallel(n_jobs=20, verbose=20, prefer="threads", require='sharedmem')(
            delayed(sparse_local_lip_parallel)(i,
                                               model,
                                               output,
                                               layer_outputs,
                                               labels,
                                               layerwise_encoder_gaps_batch,
                                               layerwise_permuted_indices_batch,
                                               layer_dims,
                                               point_wise_margins,
                                               maximum_certifiable_radius_naive_op,
                                               cw_batch,
                                               naive_operator_norm_lip,
                                               num_layers,
                                               maximum_certifiable_radius_reduced_op,
                                               chosen_local_lip_reduced_op,
                                               layerwise_chosen_activity_levels)
            for i in range(batch_size))
        # need to append these single batch statistics to the total batch statistics
        for j in range(num_layers):
            total_layerwise_chosen_activity_levels[j][current_batch_sub_index * batch_size: current_batch_sub_index * batch_size + batch_size] = layerwise_chosen_activity_levels[j]
        total_chosen_local_lip_reduced_op[current_batch_sub_index * batch_size: current_batch_sub_index * batch_size + batch_size] = chosen_local_lip_reduced_op

        total_maximum_certifiable_radius_reduced_op[current_batch_sub_index * batch_size: current_batch_sub_index * batch_size + batch_size] = maximum_certifiable_radius_reduced_op

        total_maximum_certifiable_radius_naive_op[current_batch_sub_index * batch_size: current_batch_sub_index * batch_size + batch_size] = maximum_certifiable_radius_naive_op

    means = []
    for j in range(num_layers):
        total_layerwise_chosen_activity_levels[j] = torch.Tensor(total_layerwise_chosen_activity_levels[j]).to(device=get_platform().torch_device)
        means.append(torch.mean(total_layerwise_chosen_activity_levels[j]).item())

    for i in range(0, total_size):
        if total_chosen_local_lip_reduced_op[i] != float("inf"):
            total_clean_red_op_lip.append(total_chosen_local_lip_reduced_op[i])

    total_clean_red_op_lip = torch.Tensor(total_clean_red_op_lip).to(device=get_platform().torch_device)
    total_maximum_certifiable_radius_reduced_op = total_maximum_certifiable_radius_reduced_op.to(device=get_platform().torch_device)

    if verbose:
        #print('The average pointwise margin for the batch is {}'.format(torch.mean(point_wise_margins).item()))
        print('The average local certificate radius for the total batch using just the reduced spectral norm is {}'.
              format(torch.mean(total_maximum_certifiable_radius_reduced_op).item()))
        print('The mean global Lipschitz constant via product of spectral norms is {}'.format(naive_operator_norm_lip))
        print('The mean local Lipschitz constant for the total batch using just the reduced spectral norm  is {}'.format(
            torch.mean(total_clean_red_op_lip).item()))
        print('The average layer wise activity levels are - {}'.format(means))

    #plot_hist(output_location, 'Max_Cert_Radius_Reduced_Op', 'Maximum Certifiable Radius (Reduced Op Norm)', total_maximum_certifiable_radius_reduced_op,  hist_bins=40)
    plot_hist(output_location, 'Local_Lipschitz_Constant', 'Local Lipschitz Constant (Reduced Op Norm)', total_clean_red_op_lip, marker_tensor=naive_operator_norm_lip,  hist_bins=20)
    location_str = 'chosen_activity_level_at_layer '
    title_str = 'Reduced Activity Level at layer '
    for j in range(num_layers):
        plot_hist(output_location, location_str + str(j+1), title_str + str(j+1),
                  total_layerwise_chosen_activity_levels[j],  hist_bins=30)

    info['lipschitz_type'] = type_str
    info['operator_norm_cert'] = total_maximum_certifiable_radius_naive_op
    info['operator_norm_lip'] = naive_operator_norm_lip
    info['reduced_op_norm_cert'] = total_maximum_certifiable_radius_reduced_op
    info['reduced_op_norm_lip'] = total_chosen_local_lip_reduced_op
    feedback[name] = info
