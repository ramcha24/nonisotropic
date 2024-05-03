from platforms.platform import get_platform

import pathlib
import os
import torch
from torchmetrics.functional import pairwise_cosine_similarity


def get_greedy_subset_partition(domain, num_points):
    domain = domain.to(device=get_platform().torch_device)
    domain_flat = torch.flatten(domain, start_dim=1)

    # flat_shape = domain_flat.shape[1:]
    # subset_shape = [num_points, flat_shape]
    subset_shape = [num_points] + input_shape

    subset_domain = torch.zeros(subset_shape, device=get_platform().torch_device)

    # random initialization
    rand_index = torch.randint(0, len(domain), (1,)).item()
    subset_domain[0] = domain[rand_index]

    for index in range(1, num_points):
        sim = pairwise_cosine_similarity(
            domain_flat, torch.flatten(subset_domain[:index], start_dim=1)
        )
        max_sim = torch.max(sim, dim=1).values
        selected_index = torch.argmin(max_sim).item()
        subset_domain[index] = domain[selected_index]
    return subset_domain


def save_greedy_partition(image_partition, per_label, dataset_loc=None):
    dir_path = dataset_loc + "/train"
    dir_path += "/greedy"
    dir_path += "/per_label_" + str(per_label)
    pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)

    file_path = "/greedy_partition_"

    greedy_class_partition_first_half = dict()
    greedy_class_partition_second_half = dict()

    max_data_size = int(0.5 * per_label * 10)
    half = int(0.5 * len(image_partition[0]))
    assert max_data_size <= half

    num_labels = len(image_partition)
    for label in range(num_labels):
        if label % 10 == 0:
            print("Finding greedy partition for label " + str(label))

        greedy_class_partition_first_half[label] = get_greedy_subset_partition(
            image_partition[label][:max_data_size], per_label // 2
        )
        greedy_class_partition_second_half[label] = get_greedy_subset_partition(
            image_partition[label][half : half + max_data_size], per_label // 2
        )

        torch.save(
            greedy_class_partition_first_half[label],
            dir_path + file_path + "first_half_" + str(label) + ".pt",
        )

        torch.save(
            greedy_class_partition_second_half[label],
            dir_path + file_path + "second_half_" + str(label) + ".pt",
        )
    return greedy_class_partition_first_half, greedy_class_partition_second_half


def load_greedy_partition(
    per_label, num_labels, input_shape, label=None, dataset_loc=None
):
    dir_path = dataset_loc + "/train"
    dir_path += "/greedy"
    dir_path += "/per_label_" + str(per_label)

    file_path_first_half = "/greedy_partition_first_half_"
    file_path_second_half = "/greedy_partition_second_half_"

    if label is not None:
        first_half = torch.load(
            dir_path + file_path_first_half + str(label) + ".pt", map_location="cpu"
        )
        second_half = torch.load(
            dir_path + file_path_second_half + str(label) + ".pt", map_location="cpu"
        )
        return first_half, second_half
    else:
        shape = [num_labels, per_label // 2] + input_shape
        class_partition_first_half = torch.zeros(shape)
        class_partition_second_half = torch.zeros(shape)

        for label in range(num_labels):
            class_partition_first_half[label] = torch.load(
                dir_path + file_path_first_half + str(label) + ".pt", map_location="cpu"
            )
            class_partition_second_half[label] = torch.load(
                dir_path + file_path_second_half + str(label) + ".pt",
                map_location="cpu",
            )

        return class_partition_first_half, class_partition_second_half


def load_greedy_subset(dataset_hparams):
    dataset_loc = os.path.join(
        get_platform().dataset_root, dataset_hparams.dataset_name
    )
    input_shape = [dataset_hparams.num_channels] + [
        dataset_hparams.num_spatial_dims
    ] * 2
    per_label = dataset_hparams.greedy_per_label
    (first_half, second_half) = load_greedy_partition(
        per_label,
        dataset_hparams.num_labels,
        input_shape,
        dataset_loc=dataset_loc,
    )
    greedy_subsets = torch.cat((first_half, second_half), dim=1)

    assert (
        greedy_subsets.shape[1] == per_label
    ), "Greedy subset shape mismatch expected {} got {}".format(
        per_label, greedy_subsets.shape[1]
    )

    return greedy_subsets
