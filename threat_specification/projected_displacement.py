from platforms.platform import get_platform
from utilities.miscellaneous import sanity_check

import torch


def partial_threat_fn(
    reference_inputs,
    reference_labels,
    perturbed_inputs,
    anchor_points,
    anchor_labels,
    all_pairs=False,
    return_all=False,
    beta=0.5,
):
    sanity_check(locals())

    assert (
        reference_inputs.shape[1] == anchor_points.shape[1]
    ), "Reference and threat specification shapes do not match got {} and {}".format(
        reference_inputs.shape[1], anchor_points.shape[1]
    )

    if all_pairs:
        assert len(anchor_labels) == len(anchor_points)
        perturbations = -(reference_inputs.unsqueeze(1) - perturbed_inputs.unsqueeze(0))
    else:
        assert len(anchor_labels) == 1  # only one label for all anchor points.
        perturbations = -(reference_inputs - perturbed_inputs).unsqueeze(1)

    # assuming batch of flat inputs, perturbations and threats
    unsafe_directions = -(reference_inputs.unsqueeze(1) - anchor_points)

    unsafe_norms = (
        beta * (torch.linalg.norm(unsafe_directions, dim=2, ord=2) ** 2) + 1e-2
    )

    unsafe_directions = unsafe_directions / unsafe_norms.unsqueeze(-1)

    scaled_projections = torch.bmm(perturbations, unsafe_directions.permute(0, 2, 1))

    if not all_pairs:
        scaled_projections[reference_labels == anchor_labels[0]] = 0.0

    partial_threats = torch.max(scaled_projections, dim=2).values

    if return_all:
        return (
            partial_threats.squeeze(1),
            unsafe_directions,
            unsafe_norms,
            scaled_projections,
        )
    else:
        return partial_threats.squeeze(1)


def non_isotropic_threat(
    reference_inputs,
    reference_labels,
    perturbed_inputs,
    greedy_subsets,
):
    sanity_check(locals())
    # ref_input : B x input_shape
    # Perturbations : B x input_shape
    # one perturbation for each reference input.
    # iterate through threat specification tensors for each threat_label
    # store the maximum projected displacement val - PL(ref_input, perturbation)
    # account for labels of reference input when passing through maximum.

    reference_inputs = reference_inputs.to(device=get_platform().torch_device)
    reference_inputs = torch.flatten(reference_inputs, start_dim=1)

    perturbed_inputs = perturbed_inputs.to(device=get_platform().torch_device)
    perturbed_inputs = torch.flatten(perturbed_inputs, start_dim=1)

    assert (
        reference_inputs.shape[1] == perturbed_inputs.shape[1]
    ), "Reference and perturbed input shapes do not match"

    assert (
        reference_inputs.shape[0] == perturbed_inputs.shape[0]
    ), "Reference and perturbed input batch sizes do not match"

    num_labels = greedy_subsets.shape[0]

    threats = torch.zeros(len(reference_inputs)).to(device=get_platform().torch_device)

    for anchor_label in range(0, num_labels):
        anchor_points = greedy_subsets[anchor_label].to(
            device=get_platform().torch_device
        )
        anchor_points = torch.flatten(anchor_points, start_dim=1)

        threats = torch.maximum(
            partial_threat_fn(
                reference_inputs,
                reference_labels,
                perturbed_inputs,
                anchor_points,
                [anchor_label],
            ),
            threats,
        )
    return threats


def non_isotropic_threat_all_pairs(
    reference_inputs,
    reference_labels,
    perturbed_inputs,
    perturbed_labels,
    greedy_subsets,
):
    sanity_check(locals())
    # ref_input : B1 x input_shape
    # Perturbations : B2 x input_shape
    # Each perturbed input is compared to each reference input. so B1 x B2 threat values.
    # iterate through threat specification tensors for each threat_label
    # store the maximum projected displacement val - PL(ref_input, perturbation)
    # account for labels of reference input when passing through maximum.

    reference_inputs = reference_inputs.to(device=get_platform().torch_device)
    reference_inputs = torch.flatten(reference_inputs, start_dim=1)

    perturbed_inputs = perturbed_inputs.to(device=get_platform().torch_device)
    perturbed_inputs = torch.flatten(perturbed_inputs, start_dim=1)

    assert (
        reference_inputs.shape[1] == perturbed_inputs.shape[1]
    ), "Reference and perturbed input shapes do not match"

    assert (
        len(reference_labels.unique()) == 1
    )  # only one label for all reference inputs.
    assert (
        len(perturbed_labels.unique()) == 1
    )  # only one label for all perturbed inputs.
    assert (
        reference_labels[0] != perturbed_labels[0]
    ), "Reference and perturbed input labels are the same"

    num_labels = greedy_subsets.shape[0]
    per_label = greedy_subsets.shape[1]

    threats = torch.zeros(len(reference_inputs), len(perturbed_inputs)).to(
        device=get_platform().torch_device
    )

    label_step = 4

    for threat_label in range(0, num_labels, label_step):
        label_list = torch.arange(threat_label, threat_label + label_step)

        anchor_points = greedy_subsets[
            label_list[label_list != reference_labels[0]]
        ].to(device=get_platform().torch_device)

        anchor_points = torch.flatten(anchor_points, start_dim=2)
        anchor_points = anchor_points.view(
            anchor_points.shape[0] * anchor_points.shape[1], anchor_points.shape[2]
        )

        anchor_labels = []

        for _label in label_list[label_list != reference_labels[0]]:
            anchor_labels.append(
                _label * torch.ones(per_label).to_device(get_platform().torch_device)
            )

        threats = torch.maximum(
            partial_threat_fn(
                reference_inputs,
                reference_labels,
                perturbed_inputs,
                anchor_points,
                anchor_labels,
                all_pairs=True,
            ),
            threats,
        )
    return threats


def non_isotropic_projection(
    reference_inputs,
    reference_labels,
    perturbed_inputs,
    greedy_subsets,
    threshold,
    num_iterations=10,
    verbose=False,
):
    sanity_check(locals())
    # ref_input : B x input_shape
    # Perturbations : B x input_shape
    # one perturbation for each reference input.
    # project each perturbation onto threshold-sublevel set i.e. until PL(ref_input, perturbation) <= threshold

    assert reference_inputs.shape == perturbed_inputs.shape
    input_shape = list(reference_inputs[0].shape)

    reference_inputs = reference_inputs.to(device=get_platform().torch_device)
    reference_inputs = torch.flatten(reference_inputs, start_dim=1)

    perturbed_inputs = perturbed_inputs.to(device=get_platform().torch_device)
    perturbed_inputs = torch.flatten(perturbed_inputs, start_dim=1)

    threats = non_isotropic_threat(
        reference_inputs, reference_labels, perturbed_inputs, greedy_subsets
    )

    # artificial sanitizing
    max_threat_bp = threats[threats == threats].max()
    threats[threats != threats] = max_threat_bp

    if max_threat_bp <= threshold:
        return torch.unflatten(perturbed_inputs, 1, input_shape)

    num_labels = greedy_subsets.shape[0]

    current_perturbations = -(reference_inputs - perturbed_inputs)

    for t in range(num_iterations):
        for threat_label in range(0, num_labels):
            # assuming batch of flat inputs, perturbations and threats
            anchor_points = greedy_subsets[threat_label].to(
                device=get_platform().torch_device
            )
            anchor_points = torch.flatten(anchor_points, start_dim=1)

            (
                partial_threats,
                unsafe_directions,
                unsafe_norms,
                scaled_projections,
            ) = partial_threat_fn(
                reference_inputs,
                reference_labels,
                reference_inputs + current_perturbations,
                anchor_points,
                [threat_label],
                return_all=True,
            )

            partial_threats[partial_threats != partial_threats] = 1.0
            if partial_threats.max() <= threshold:
                continue  # no need to project

            max_selection = list(
                enumerate(
                    torch.argmax(scaled_projections, dim=2).detach().cpu().numpy()
                )
            )

            max_unsafe_directions = torch.stack(
                [
                    unsafe_directions[input_index][unsafe_index].squeeze(0)
                    for (input_index, unsafe_index) in max_selection
                ],
                dim=0,
            ).to(device=get_platform().torch_device)

            max_unsafe_norms = torch.stack(
                [
                    unsafe_norms[input_index][unsafe_index]
                    for (input_index, unsafe_index) in max_selection
                ],
                dim=0,
            ).to(device=get_platform().torch_device)

            residuals = partial_threats - threshold
            residuals[residuals < 0.0] = 0.0

            step_size = residuals * (max_unsafe_norms.squeeze(1))

            current_perturbations -= 2.0 * max_unsafe_directions * step_size[:, None]

    current_perturbations = current_perturbations.squeeze(1)

    threats = non_isotropic_threat(
        reference_inputs,
        reference_labels,
        reference_inputs + current_perturbations,
        greedy_subsets,
    )
    max_threat_ap = threats[threats == threats].max()
    threats[threats != threats] = max_threat_ap

    if get_platform().is_primary_process and verbose:
        print(
            "Max threat : Before projection was {} and after projection is {}".format(
                max_threat_bp, max_threat_ap
            )
        )

    # undo flattening
    reference_inputs = torch.unflatten(reference_inputs, 1, input_shape)
    current_perturbations = torch.unflatten(current_perturbations, 1, input_shape)

    if max_threat_ap <= threshold:
        return reference_inputs + current_perturbations
    else:
        return reference_inputs + current_perturbations * (threshold / max_threat_ap)
