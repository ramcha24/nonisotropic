from platforms.platform import get_platform

from pathlib import Path
import os
import torch
from torchmetrics.functional import pairwise_cosine_similarity


def get_greedy_subset_partition(domain, num_points):
    domain = domain.to(device=get_platform().torch_device)
    input_shape = list(domain.shape[1:])
    domain_flat = torch.flatten(domain, start_dim=1)

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


def save_greedy_partition(
    run_path,
    dataset_hparams,
    per_label,
    domain_expansion_factor=10,
    subset_selection_seed=0,
    verbose=False,
):
    dataset_name = dataset_hparams.dataset_name
    dataset_loc = os.path.join(get_platform().dataset_root, dataset_name)

    dir_path = os.path.join(run_path, "per_label_" + str(per_label))
    get_platform().makedirs(dir_path)
    num_labels = dataset_hparams.num_labels

    # set the random seed
    torch.manual_seed(subset_selection_seed)

    for label in range(num_labels):
        image_partition = torch.load(
            os.path.join(dataset_loc, "train_class_partition")
            + "/"
            + str(label)
            + ".pt"
        )
        half = int(0.5 * len(image_partition))
        max_data_size = min(int(0.5 * per_label * domain_expansion_factor), half)

        if (
            verbose
            and get_platform().is_primary_process
            and per_label * domain_expansion_factor <= half
        ):
            print(
                f"Warning! Not enough data points in label {label} to create a greedy subset of size {per_label}, needed {per_label * domain_expansion_factor} examples but only {half} examples are available"
            )

        # here I could use a different logic to find my points.
        start_1 = 0
        end_1 = max_data_size
        start_2 = half
        end_2 = half + max_data_size

        if verbose and get_platform().is_primary_process and label % 10 == 0:
            print("Finding greedy partition for label " + str(label))

        greedy_class_partition_first_half = get_greedy_subset_partition(
            image_partition[start_1:end_1], per_label // 2
        )
        greedy_class_partition_second_half = get_greedy_subset_partition(
            image_partition[start_2:end_2], per_label // 2
        )

        # always storing by halfs. Its easier to compare later, and also lesser code.
        torch.save(
            greedy_class_partition_first_half,
            dir_path + "first_half_" + str(label) + ".pt",
        )

        torch.save(
            greedy_class_partition_second_half,
            dir_path + "second_half_" + str(label) + ".pt",
        )
        del greedy_class_partition_first_half, greedy_class_partition_second_half


def load_greedy_partition(
    per_label, num_labels, input_shape, label=None, dataset_loc=None
):
    dir_path = os.path.join(dataset_loc, "per_label_" + str(per_label))

    if not get_platform().exists(dir_path):
        raise FileNotFoundError("Greedy subsets not found at " + dir_path)

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


def load_greedy_subset(dataset_hparams, per_label=50):
    dataset_loc = os.path.join(
        get_platform().threat_specification_root, dataset_hparams.dataset_name
    )
    input_shape = [
        dataset_hparams.num_channels,
        dataset_hparams.num_spatial_dims,
        dataset_hparams.num_spatial_dims,
    ]
    num_labels = dataset_hparams.num_labels
    (first_half, second_half) = load_greedy_partition(
        per_label,
        num_labels,
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
