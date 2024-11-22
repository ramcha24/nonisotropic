from platforms.platform import get_platform
from utilities.miscellaneous import sanity_check
from torchvision.transforms import v2
import numpy as np
from ortools.linear_solver import pywraplp
import torch
import os


@torch.no_grad()
def partial_threat_fn(
    reference_inputs,
    reference_labels,
    perturbed_inputs,
    anchor_points,
    anchor_labels,
    all_pairs=False,
    return_unsafe=False,
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
        # perturbations shape is B x 1 x flat_input_shape
        perturbations = -(reference_inputs - perturbed_inputs).unsqueeze(1)

    # assuming batch of flat inputs, perturbations and threats
    # unsafe direcrtions shape is B x num_anchors x flat_input_shape
    unsafe_directions = -(reference_inputs.unsqueeze(1) - anchor_points)

    # unsafe norms shape is B x num_anchors
    unsafe_norms = (
        beta * (torch.linalg.norm(unsafe_directions, dim=2, ord=2) ** 2) + 1e-5
    )

    # scaled_projections shape is B x 1 x num_anchor
    scaled_projections = torch.bmm(
        perturbations, (unsafe_directions / unsafe_norms.unsqueeze(-1)).permute(0, 2, 1)
    )

    scaled_projections = torch.clamp(scaled_projections, min=0.0)

    if not all_pairs:
        scaled_projections[reference_labels == anchor_labels[0]] = 0.0

    # Get the max values and indices of the max projections
    max_values, max_indices = torch.max(scaled_projections.squeeze(1), dim=1)
    # max_indices shape is B x 1 where each value is the index of the max projection within the anchor points
    # max_values shape is B x 1

    if return_unsafe:
        # Gather the unsafe directions corresponding to the max projections using the indices
        # top_unsafe_directions shape is  B x flat_input_shape
        top_unsafe_directions = unsafe_directions[
            torch.arange(unsafe_directions.size(0)), max_indices
        ]

        return (
            max_values,
            top_unsafe_directions,
        )
    else:
        return max_values, None


@torch.no_grad()
def non_isotropic_threat(
    reference_inputs,
    reference_labels,
    perturbed_inputs,
    greedy_subsets,
    return_unsafe=False,
):
    sanity_check(locals())
    # ref_input : B x input_shape
    # Perturbations : B x input_shape
    # one perturbation for each reference input.
    # iterate through threat specification tensors for each threat_label
    # store the maximum projected displacement val - PL(ref_input, perturbation)
    # account for labels of reference input when passing through maximum.
    if reference_inputs.device != get_platform().torch_device:
        reference_inputs = reference_inputs.to(device=get_platform().torch_device)
    if reference_labels.device != get_platform().torch_device:
        reference_labels = reference_labels.to(device=get_platform().torch_device)
    if perturbed_inputs.device != get_platform().torch_device:
        perturbed_inputs = perturbed_inputs.to(device=get_platform().torch_device)

    # reference_inputs = reference_inputs.to(device=get_platform().torch_device)
    reference_inputs = torch.flatten(reference_inputs, start_dim=1)

    perturbed_inputs = torch.flatten(perturbed_inputs, start_dim=1)

    assert (
        reference_inputs.shape[1] == perturbed_inputs.shape[1]
    ), "Reference and perturbed input shapes do not match"

    assert (
        reference_inputs.shape[0] == perturbed_inputs.shape[0]
    ), "Reference and perturbed input batch sizes do not match"

    num_labels = greedy_subsets.shape[0]

    threats = torch.zeros(len(reference_inputs), requires_grad=False).to(
        device=get_platform().torch_device
    )
    if return_unsafe:
        top_unsafe_directions = torch.zeros(
            reference_inputs.shape[0], reference_inputs.shape[1], requires_grad=False
        ).to(device=get_platform().torch_device)
        top_unsafe_labels = torch.zeros_like(reference_labels)

    for anchor_label in range(0, num_labels):
        anchor_points = greedy_subsets[anchor_label].to(
            device=get_platform().torch_device
        )
        anchor_points = torch.flatten(anchor_points, start_dim=1)

        partial_threats, partial_unsafe_directions = partial_threat_fn(
            reference_inputs,
            reference_labels,
            perturbed_inputs,
            anchor_points,
            [anchor_label],
            return_unsafe=return_unsafe,
        )
        mask = threats < partial_threats
        threats[mask] = partial_threats[mask]
        if return_unsafe:
            top_unsafe_directions[mask] = partial_unsafe_directions[mask]
            top_unsafe_labels[mask] = anchor_label

        del anchor_points, partial_threats, partial_unsafe_directions

    if return_unsafe:
        return threats, top_unsafe_directions, top_unsafe_labels
    else:
        return threats


@torch.no_grad()
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

    threats = torch.zeros(
        len(reference_inputs), len(perturbed_inputs), requires_grad=False
    ).to(device=get_platform().torch_device)

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


@torch.no_grad()
def non_isotropic_projection(
    reference_inputs,
    reference_labels,
    perturbed_inputs,
    greedy_subsets,
    threshold,
    num_iterations=10,
    verbose=False,
):
    # print("Inside non-isotropic projection 1")
    # print("requires_grad check")
    # print(f"reference_inputs : {reference_inputs.requires_grad}")
    # print(f"reference_labels : {reference_labels.requires_grad}")
    # print(f"perturbed_inputs : {perturbed_inputs.requires_grad}")
    # print(f"greedy_subsets : {greedy_subsets.requires_grad}")

    reference_inputs = reference_inputs.detach().clone()
    reference_labels = reference_labels.detach().clone()
    perturbed_inputs = perturbed_inputs.detach().clone()
    greedy_subsets = greedy_subsets.detach().clone()

    # print("Inside non-isotropic projection 2")
    # print("requires_grad check")
    # print(f"reference_inputs : {reference_inputs.requires_grad}")
    # print(f"reference_labels : {reference_labels.requires_grad}")
    # print(f"perturbed_inputs : {perturbed_inputs.requires_grad}")
    # print(f"greedy_subsets : {greedy_subsets.requires_grad}")

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

    if max_threat_ap > threshold:
        current_perturbations = current_perturbations * (threshold / max_threat_ap)

    perturbed_inputs = reference_inputs + current_perturbations

    return perturbed_inputs.detach().clone()


def check_nonisotropic_conflict(
    examples_1, examples_2, greedy_subsets, epsilon, beta=0.5, verbose=False
):
    examples_1 = examples_1.detach().clone()
    examples_2 = examples_2.detach().clone()
    assert examples_1.shape == examples_2.shape
    examples_1 = torch.flatten(examples_1, start_dim=1)
    examples_2 = torch.flatten(examples_2, start_dim=1)

    num_labels = greedy_subsets.shape[0]
    per_label = greedy_subsets.shape[1]
    batch_size, flat_dim = examples_1.shape

    for _index in range(batch_size):
        if _index > 0:
            return
        ref_input_1 = examples_1[_index]
        ref_input_2 = examples_2[_index]

        # Create the linear solver with GLOP (Google Linear Optimization Package) PDLP
        solver = pywraplp.Solver.CreateSolver("GLOP")

        if not solver:
            raise ValueError("Solver not available.")

        # print(solver)

        solver.SetTimeLimit(10000)

        if verbose:
            solver.EnableOutput()
            # solver.set_solver_specific_parameters("{verbosity_level: 3}")
            # solver.SetSolver
            # 3

        # Define the decision variables
        delta = [
            solver.NumVar(-epsilon, epsilon, f"delta_{i}") for i in range(flat_dim)
        ]

        # Initialize constraints. warning : potentially very large.
        print("Initializing constraints")
        num_constraints = num_labels * per_label
        constraints_1 = np.zeros((num_constraints, flat_dim))
        constraints_2 = np.zeros((num_constraints, flat_dim))

        for anchor_label in range(0, num_labels):
            # if anchor_label % 100 == 0:
            #     print(anchor_label)
            start_index = anchor_label * per_label
            end_index = start_index + per_label

            anchor_points = greedy_subsets[
                anchor_label
            ]  # .to(device=get_platform().torch_device)
            anchor_points = torch.flatten(anchor_points, start_dim=1)
            # unsafe direcrtions shape is num_anchors x flat_input_shape
            unsafe_directions_1 = -(ref_input_1.unsqueeze(0) - anchor_points)
            unsafe_directions_2 = -(ref_input_2.unsqueeze(0) - anchor_points)

            # unsafe norms shape is B x num_anchors
            unsafe_norms_1 = (
                beta * (torch.linalg.norm(unsafe_directions_1, dim=1, ord=2) ** 2)
                + 1e-5
            )
            unsafe_norms_2 = (
                beta * (torch.linalg.norm(unsafe_directions_2, dim=1, ord=2) ** 2)
                + 1e-5
            )
            unsafe_directions_1 = unsafe_directions_1 / unsafe_norms_1.unsqueeze(1)
            unsafe_directions_2 = unsafe_directions_2 / unsafe_norms_2.unsqueeze(1)
            constraints_1[start_index:end_index] = unsafe_directions_1.cpu().numpy()
            constraints_2[start_index:end_index] = unsafe_directions_2.cpu().numpy()

        print("Adding Constraints")
        # Add the constraints: <unsafe_i, delta> <= epsilon for each constraint i
        for _i in range(num_constraints):
            constraint_expr_1 = sum(
                constraints_1[_i, _j] * delta[_j] for _j in range(flat_dim)
            )  # <unsafe_1_i, x>
            constraint_expr_2 = sum(
                constraints_2[_i, _j] * delta[_j] for _j in range(flat_dim)
            )  # <unsafe_2_i, x>
            solver.Add(constraint_expr_1 <= epsilon)
            solver.Add(constraint_expr_2 <= epsilon)

        solver.Minimize(0)
        # Solve the problem
        print("Starting solver")
        status = solver.Solve()

        # Check if the solution was successful
        if status == pywraplp.Solver.OPTIMAL:
            print("Solution found.")
            solution = np.array([delta[j].solution_value() for j in range(flat_dim)])
            print(f"Optimal value: {solver.Objective().Value()}")
            print(f"Solution vector delta: {solution}")
        else:
            print("No optimal solution found.")

        print("\nAdvanced usage:")
        print(f"Problem solved in {solver.wall_time():d} milliseconds")
        print(f"Problem solved in {solver.iterations():d} iterations")


@torch.no_grad()
def non_isotropic_threat_full(
    reference_inputs,
    reference_labels,
    perturbed_inputs,
    return_unsafe=False,
):
    sanity_check(locals())
    # ref_input : B x input_shape
    # Perturbations : B x input_shape
    # one perturbation for each reference input.
    # iterate through threat specification tensors for each threat_label
    # store the maximum projected displacement val - PL(ref_input, perturbation)
    # account for labels of reference input when passing through maximum.
    if reference_inputs.device != get_platform().torch_device:
        reference_inputs = reference_inputs.to(device=get_platform().torch_device)
    if reference_labels.device != get_platform().torch_device:
        reference_labels = reference_labels.to(device=get_platform().torch_device)
    if perturbed_inputs.device != get_platform().torch_device:
        perturbed_inputs = perturbed_inputs.to(device=get_platform().torch_device)

    # reference_inputs = reference_inputs.to(device=get_platform().torch_device)
    reference_inputs = torch.flatten(reference_inputs, start_dim=1)

    perturbed_inputs = torch.flatten(perturbed_inputs, start_dim=1)

    assert (
        reference_inputs.shape[1] == perturbed_inputs.shape[1]
    ), "Reference and perturbed input shapes do not match"

    assert (
        reference_inputs.shape[0] == perturbed_inputs.shape[0]
    ), "Reference and perturbed input batch sizes do not match"

    if reference_inputs.shape[1] == 3 * 224 * 224:
        num_labels = 1000
    else:
        num_labels = 10

    threats = torch.zeros(len(reference_inputs), requires_grad=False).to(
        device=get_platform().torch_device
    )
    if return_unsafe:
        top_unsafe_directions = torch.zeros(
            reference_inputs.shape[0], reference_inputs.shape[1], requires_grad=False
        ).to(device=get_platform().torch_device)
        top_unsafe_labels = torch.zeros_like(reference_labels)

    for anchor_label in range(0, num_labels):
        # for now only imagenet
        dataset_loc = "./datasets/imagenet"
        train_partition_folder_path = os.path.join(dataset_loc, "train_class_partition")
        partition_file_path = "/" + str(anchor_label) + ".pt"

        anchor_points = torch.load(
            train_partition_folder_path + partition_file_path,
        ).to(device=get_platform().torch_device)

        if anchor_label % 100 == 0:
            print(f"Computing threats w.r.t {anchor_label}")
        anchor_points = torch.flatten(anchor_points, start_dim=1)

        partial_threats, partial_unsafe_directions = partial_threat_fn(
            reference_inputs,
            reference_labels,
            perturbed_inputs,
            anchor_points,
            [anchor_label],
            return_unsafe=return_unsafe,
        )
        mask = threats < partial_threats
        threats[mask] = partial_threats[mask]
        if return_unsafe:
            top_unsafe_directions[mask] = partial_unsafe_directions[mask]
            top_unsafe_labels[mask] = anchor_label

        del anchor_points, partial_threats, partial_unsafe_directions

    if return_unsafe:
        return threats, top_unsafe_directions, top_unsafe_labels
    else:
        return threats


# Define the operation for a single batch element
@torch.no_grad()
def alt_compute_unsafe_directions(reference_input, anchor_points):
    # Subtract anchor_points from a single reference_input and negate
    return -(anchor_points - reference_input.unsqueeze(0))  # Broadcasting happens here


@torch.no_grad()
def NIT_alt_partial_one_at_a_time(
    reference_input,
    reference_label,
    perturbed_input,
    anchor_points,
    anchor_labels,
    beta=0.5,
    device=None,
    float_16=False,
):
    device = device or reference_input.device.type
    dtype = torch.float16 if float_16 else torch.float32
    with torch.autocast(device_type=device, dtype=dtype):
        perturbation = perturbed_input - reference_input
        unsafe_perturbations = anchor_points - reference_input.unsqueeze(0)

        unsafe_norms = (
            beta
            * (
                torch.linalg.norm(unsafe_perturbations, dim=1, ord=2, keepdim=True).pow(
                    2
                )
            )
            + 1e-5
        )

        unsafe_perturbations.div_(unsafe_norms)
        # unsafe perturbations shape is 50000, T
        # perturbations shape is 1xT
        # projections shape should be 50000, 1
        # print("Unsafe perturbations shape: ", unsafe_perturbations.shape)
        # print("perturbation shape: ", perturbation.shape)

        scaled_projections = torch.matmul(
            unsafe_perturbations, perturbation.unsqueeze(0).T
        )

        mask = (reference_label.unsqueeze(0) != anchor_labels.unsqueeze(1)).to(
            reference_input.device
        )

        scaled_projections = torch.clamp(scaled_projections, min=0.0)

        scaled_projections = scaled_projections * mask.float()

        partial_threats = torch.max(scaled_projections, dim=0).values

        return partial_threats  # scaled_projections # partial_threats.squeeze(1)


@torch.no_grad()
def NIT_alt_partial_batch(
    reference_input,
    reference_label,
    perturbed_input,
    anchor_points,
    anchor_labels,
    beta=0.5,
    device=None,
    float_16=False,
):
    device = device or reference_input.device.type
    dtype = torch.float16 if float_16 else torch.float32
    # print(anchor_points.shape)
    # print(reference_input.shape)
    with torch.autocast(device_type=device, dtype=dtype):
        perturbation = (perturbed_input - reference_input).unsqueeze(1)
        unsafe_perturbations = -(
            reference_input.unsqueeze(1) - anchor_points
        )  # anchor_points - reference_input.unsqueeze(0)

        unsafe_norms = (
            beta
            * (
                torch.linalg.norm(unsafe_perturbations, dim=2, ord=2, keepdim=True).pow(
                    2
                )
            )
            + 1e-5
        )

        unsafe_perturbations.div_(unsafe_norms)
        # unsafe perturbations shape is 50000, T
        # perturbations shape is 1xT
        # projections shape should be 50000, 1
        # print("Unsafe perturbations shape: ", unsafe_perturbations.shape)
        # print("perturbation shape: ", perturbation.shape)

        scaled_projections = torch.bmm(
            perturbation, unsafe_perturbations.permute(0, 2, 1)
        )
        # print("Scaled projections shape: ", scaled_projections.shape)
        mask = (reference_label.unsqueeze(0) != anchor_labels.unsqueeze(1)).to(
            reference_input.device
        )
        # print("Mask shape: ", mask.shape)
        scaled_projections = torch.clamp(scaled_projections, min=0.0)

        scaled_projections = scaled_projections.squeeze(1) * mask.T.float()

        partial_threats = torch.max(scaled_projections, dim=1).values

        return partial_threats  # scaled_projections # partial_threats.squeeze(1)


@torch.no_grad()
def NIT_alt(
    reference_inputs,
    reference_labels,
    perturbed_inputs,
    anchor_points,
    anchor_labels,
    float_16=False,
    num_devices=1,
    num_label=1000,
    per_label=50,
    gray_scale=False,
):
    # ref_input : B x input_shape
    # Perturbations : B x input_shape

    assert (
        reference_inputs.shape == perturbed_inputs.shape
    ), "Reference input shape {} and perturbed input shape {} does not match".format(
        reference_inputs.shape, perturbed_inputs.shape
    )
    num_channels = 3
    if gray_scale:
        num_channels = 1
        transform = v2.Grayscale()
        reference_inputs = transform(reference_inputs)
        perturbed_inputs = transform(perturbed_inputs)
        for _gpu_index in range(num_devices):
            anchor_points[_gpu_index] = transform(anchor_points[_gpu_index])
            anchor_labels[_gpu_index] = transform(anchor_labels[_gpu_index])

    if float_16:
        reference_inputs = reference_inputs.to(torch.float16)
        perturbed_inputs = perturbed_inputs.to(torch.float16)
        # expecting float 16 from anchor points

    reference_inputs = torch.flatten(
        reference_inputs.to(device=get_platform().torch_device),
        start_dim=1,
    )

    reference_labels = reference_labels.to(device=get_platform().torch_device)
    perturbed_inputs = torch.flatten(
        perturbed_inputs.to(device=get_platform().torch_device),
        start_dim=1,
    )

    threats = torch.zeros(len(reference_inputs), requires_grad=False).to(
        get_platform().torch_device
    )
    # print("Length of reference inputs: ", len(reference_inputs))
    # assume len(reference_inputs) is a multiple of 4
    input_step = len(reference_inputs) // 8  # // 5  #
    for _gpu_index in range(num_devices):

        for _index in range(0, len(reference_inputs), input_step):
            # only find threats for this input (for now)
            ref_inputs = reference_inputs[_index : _index + input_step]
            ref_labels = reference_labels[_index : _index + input_step]
            pert_inputs = perturbed_inputs[_index : _index + input_step]

            threats[_index : _index + input_step] = torch.maximum(
                NIT_alt_partial_batch(
                    ref_inputs.to(torch.device(_gpu_index)),
                    ref_labels.to(torch.device(_gpu_index)),
                    pert_inputs.to(torch.device(_gpu_index)),
                    anchor_points[_gpu_index],
                    anchor_labels[_gpu_index],
                    float_16=float_16,
                ).to(get_platform().torch_device),
                threats[_index : _index + input_step],
            )
    return threats
