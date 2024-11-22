from platforms.platform import get_platform

from pathlib import Path
import os
import torch
from torchmetrics.functional import pairwise_cosine_similarity

from foundations import hparams
from utilities.miscellaneous import timeprint


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


def compute_threat_specification(
    run_path,
    dataset_hparams,
    threat_hparams,
    verbose=False,
):
    subset_selection = threat_hparams.subset_selection
    if subset_selection != "greedy":
        raise ValueError(
            "Subset selection method not supported. Only greedy subset selection is currently supported."
        )
    # in the future we can add more subset selection methods and have a registry to fetch the appropriate method

    per_label = threat_hparams.per_label
    domain_expansion_factor = threat_hparams.domain_expansion_factor
    subset_selection_seed = threat_hparams.subset_selection_seed
    num_labels = dataset_hparams.num_labels
    dataset_name = dataset_hparams.dataset_name

    dataset_loc = os.path.join(get_platform().dataset_root, dataset_name)
    dir_path = os.path.join(run_path, "per_label_" + str(per_label), "anchor_points")
    get_platform().makedirs(dir_path)

    # set the random seed
    torch.manual_seed(subset_selection_seed)

    for label in range(num_labels):
        image_partition = torch.load(
            os.path.join(dataset_loc, "train_class_partition")
            + "/"
            + str(label)
            + ".pt"
        )

        image_partition = image_partition.to(device=get_platform().torch_device)

        half = int(0.5 * len(image_partition))
        max_data_size = min(int(0.5 * per_label * domain_expansion_factor), half)

        if (
            verbose
            and get_platform().is_primary_process
            and 0.5 * per_label * domain_expansion_factor > half
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
            print("Finding threat specification for label " + str(label))

        shuffle_indices = torch.randperm(len(image_partition))
        shuffle_partition = image_partition[shuffle_indices]

        first_half = shuffle_partition[start_1:end_1]
        second_half = shuffle_partition[start_2:end_2]

        threat_specification_first_half = get_greedy_subset_partition(
            first_half, per_label // 2
        )
        threat_specification_second_half = get_greedy_subset_partition(
            second_half, per_label // 2
        )

        # always storing by halfs. Its easier to compare later, and also lesser code.
        torch.save(
            threat_specification_first_half,
            os.path.join(dir_path, "first_half_" + str(label) + ".pt"),
        )

        torch.save(
            threat_specification_second_half,
            os.path.join(dir_path, "second_half_" + str(label) + ".pt"),
        )
        del threat_specification_first_half, threat_specification_second_half


def load_threat_specification(
    dataset_hparams,
    label=None,
    threat_hparams=None,
    threat_replicate=1,
):
    if threat_hparams is None:
        threat_hparams = (
            hparams.ThreatHparams()
        )  # this should eventually be in the desc object for each runner
    else:
        subset_selection = threat_hparams.subset_selection
        if subset_selection != "greedy":
            raise ValueError(
                "Subset selection method not supported. Only greedy subset selection is currently supported."
            )

    input_shape = [
        dataset_hparams.num_channels,
        dataset_hparams.num_spatial_dims,
        dataset_hparams.num_spatial_dims,
    ]
    num_labels = dataset_hparams.num_labels

    dataset_loc = os.path.join(get_platform().runner_root, dataset_hparams.dataset_name)
    threat_dir = "greedy"
    #    threat_hparams.dir_path(identifier_name="subset_selection")  # greedy_xx

    threat_hparams_path = os.path.join(
        dataset_loc, "threat_specification", threat_dir
    )  # nonisotropic/runner_data/cifar10/threat_specification/greedy_xx

    threat_run_path = os.path.join(
        threat_hparams_path, "threat_replicate_" + str(threat_replicate)
    )  # nonisotropic/runner_data/cifar10/threat_specification/greedy_xx/threat_replicate_1

    per_label = threat_hparams.per_label
    per_label_path = os.path.join(
        threat_run_path, "per_label_" + str(per_label), "anchor_points"
    )

    file_path_first_half = "/first_half_"
    file_path_second_half = "/second_half_"

    if label is not None:
        first_half_path = per_label_path + file_path_first_half + str(label) + ".pt"
        second_half_path = per_label_path + file_path_second_half + str(label) + ".pt"

        if not get_platform().exists(first_half_path) or not get_platform().exists(
            second_half_path
        ):
            raise FileNotFoundError(
                f"No threat specification files found at {first_half_path}, {second_half_path} for label {label}"
            )
        first_half = torch.load(
            first_half_path,
            map_location="cpu",
        )
        second_half = torch.load(
            second_half_path,
            map_location="cpu",
        )
        return first_half, second_half
    else:
        shape = [num_labels, per_label // 2] + input_shape
        threat_specification_first_half = torch.zeros(shape)
        threat_specification_second_half = torch.zeros(shape)

        for label in range(num_labels):
            first_half_path = per_label_path + file_path_first_half + str(label) + ".pt"
            second_half_path = (
                per_label_path + file_path_second_half + str(label) + ".pt"
            )

            if not get_platform().exists(first_half_path) or not get_platform().exists(
                second_half_path
            ):
                raise FileNotFoundError(
                    f"No threat specification files found at {first_half_path}, {second_half_path} for label {label}"
                )
            # if label % 200 == 0:
            #     timeprint(f"Loading threat specification for label {label}")

            threat_specification_first_half[label] = torch.load(
                first_half_path,
                map_location="cpu",
            )
            threat_specification_second_half[label] = torch.load(
                second_half_path,
                map_location="cpu",
            )

    threat_specification = torch.cat(
        (threat_specification_first_half, threat_specification_second_half), dim=1
    )

    assert (
        threat_specification.shape[1] == per_label
    ), "Threat specification shape mismatch expected {} got {}".format(
        per_label, threat_specification.shape[1]
    )

    return threat_specification
