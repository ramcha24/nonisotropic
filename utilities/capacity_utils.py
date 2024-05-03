import torch
import numpy as np

from platforms.platform import get_platform


def batch_norms(examples):
    """Compute norms over all but the first dimension"""
    return examples.view(examples.shape[0], -1).norm(dim=1)[:, None, None, None]


def get_2_inf_norm(model):
    layer_norms = []
    for i, layer in enumerate(model.layers):
        layer_weight = model.layers[i].weight
        layer_row_norm = torch.norm(layer_weight, p=2, dim=1)
        layer_norms.append(torch.max(layer_row_norm))

    return layer_norms


def get_feature_reg(model, epoch, iteration, ortho=True, euclidean=True):
    num_layers = 0
    loss_ortho = 0.0
    loss_euclidean = 0.0
    layer_norms = get_2_inf_norm(model)

    for i, layer in enumerate(model.layers):
        layer_weight = model.layers[i].weight
        layer_row_norm = torch.norm(layer_weight, p=2, dim=1)
        layer_row_normalized = layer_weight.div(layer_row_norm.expand_as(layer_weight.t()).t())
        layer_gram_matrix = torch.matmul(layer_row_normalized, layer_row_normalized.t())
        sq_dim = layer_gram_matrix.shape[0]

        identity = torch.Tensor(np.eye(sq_dim)).type(layer_weight.type()).to(device=get_platform().torch_device)
        loss_ortho += 1 * torch.norm(identity - layer_gram_matrix) ** 2

        loss_euclidean += torch.max(layer_row_norm)

        num_layers += 1

        layer_mu = np.max(np.abs(layer_gram_matrix.detach().cpu().numpy()) - np.eye(sq_dim))
        if iteration == 0:
            print('At layer {}, the row mutual coherence is {}'.format(i, layer_mu))

    if iteration == 0:
        print('Layer norms')
        print(layer_norms)

    return loss_ortho, loss_euclidean, num_layers


def get_babel_mu(model):
    layer_babel_mus = []
    num_layers = 0

    for i, layer in enumerate(model.layers):
        layer_weight = model.layers[i].weight
        layer_row_norm = torch.norm(layer_weight, p=2, dim=1)
        layer_row_normalized = layer_weight.div(layer_row_norm.expand_as(layer_weight.t()).t())
        layer_gram_matrix = torch.matmul(layer_row_normalized, layer_row_normalized.t())
        sq_dim = layer_gram_matrix.shape[0]

        (GS, _) = torch.sort(torch.abs(layer_gram_matrix), descending=True, dim=1)
        mus = torch.zeros(sq_dim)  # mus[0] = 0 by default
        for j in range(1, sq_dim):
            mus[j] = torch.max(torch.sum(GS[:, 1:j+1], dim=1)).item()
        layer_babel_mus.append(mus)
        num_layers += 1

    return layer_babel_mus


def get_reduced_classifier_constant(margin, permuted_fc_weight, col_activity_level, true_label):
    w = permuted_fc_weight
    num_labels = w.shape[0]  # number of classes
    w_diff = torch.zeros(num_labels)
    activity_level = col_activity_level
    if margin < 1:
        activity_level = w.shape[1]
    for j in range(num_labels):
        w_diff[j] = torch.norm(w[true_label, 0:activity_level] - w[j, 0:activity_level])

    return torch.max(w_diff)


def get_classifier_constant(model, labels, npy=False):
    assert labels is not None

    W = model.fc.weight
    W_row_norm = torch.norm(W, p=2, dim=1)
    #W_euclidean = torch.max(W_row_norm)
    W_loss = 0.0

    # computing pairwise distances
    num_labels = W.shape[0]  # number of classes

    PD = torch.zeros(num_labels, num_labels)
    for i in range(num_labels):
        for j in range(num_labels):
            PD[i, j] = torch.norm(W[i, :] - W[j, :])

    W_norm = torch.norm(W)
    # C = torch.max(PD)
    #print('printing the entire PD')
    #print(PD)

    batch_size = len(labels)
    cw_batch = []
    # cw_batch = torch.zeros_like(labels)
    for i in range(0, batch_size):
        true_label = int(labels[i].item())
        pdtl = PD[true_label]
        # print('true label is {}'.format(true_label))
        # print('corresponding PD')
        # print(pdtl)
        # print('max value of corresponding PD')
        # print(torch.max(pdtl))
        cw_batch.append(torch.max(pdtl))
        # print('chosen cw is therefore')
        # print(cw_batch[i])
        W_loss += W_row_norm[true_label]
    # print(cw_batch)
    cw_batch = torch.tensor(cw_batch).to(device=get_platform().torch_device)
    # cw_loss = cw_batch.float().mean()
    W_loss = W_loss/batch_size
    if npy:
        #print(cw_batch)
        return cw_batch.cpu().detach().numpy()
    else:
        return W_loss



def get_naive_lip(model):
    num_layers = 0
    operator_norm_lip = 1.0

    for i, layer in enumerate(model.layers):
        layer_weight = model.layers[i].weight
        operator_norm_lip *= np.linalg.norm(layer_weight.clone().detach().cpu().numpy(), ord=2)

    #W = model.fc.weight
    #operator_norm_lip *= np.linalg.norm(W.clone().detach().cpu().numpy(), ord='2')
    return operator_norm_lip
