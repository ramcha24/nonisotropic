import torch
from torchvision.transforms import v2
import os
import pickle
import numpy as np
import math

from segment_anything import sam_model_registry, SamPredictor

from threat_specification import base
from platforms.platform import get_platform
from foundations import hparams
from foundations import paths

from threat_specification.subset_selection import get_greedy_subset_partition
from utilities.miscellaneous import timeprint, _cast, _move


class ProjectedDisplacement(base.ThreatModel):
    """The projected displacement threat model."""

    def __init__(
        self,
        dataset_hparams: hparams.DatasetHparams,
        threat_hparams: hparams.ThreatHparams,
        threat_replicate: int = 1,
        weighted: bool = False,
        segmented: bool = False,
    ):
        self.dataset_hparams = dataset_hparams
        self.threat_hparams = threat_hparams
        self.threat_replicate = threat_replicate
        self.weighted = weighted
        self.segmented = segmented

        if self.threat_hparams.subset_selection != "greedy":
            raise ValueError(
                "Subset selection method not supported. Only greedy subset selection is currently supported."
            )
        # in the future we can add more subset selection methods and have a registry to fetch the appropriate method

    def prepare(self, num_devices: int):
        """Prepare the threat for use on a model and dataset."""
        timeprint(f"Preparing threat specification")

        num_labels = self.dataset_hparams.num_labels
        per_label = self.threat_hparams.per_label
        num_chunks = self.threat_hparams.num_chunks
        anchor_points = torch.zeros(self._get_threat_specification_shape())
        num_channels = anchor_points.shape[2]
        spatial_dims = anchor_points.shape[3]

        for label in range(num_labels):
            anchor_points[label] = self.load_threat_specification(label)
        timeprint("Threat specification loaded")

        anchor_points = anchor_points.view(-1, num_channels, spatial_dims, spatial_dims)
        anchor_labels = (
            torch.arange(num_labels).unsqueeze(1).repeat(1, per_label).view(-1)
        )

        """
        threat_specification should be arranged as 
        {
        
        chunk_1: {
        gpu_index_1: (anchor_point_1_1, anchor_labels_1_1), 
        gpu_index_2: (anchor_point_1_2, anchor_labels_1_2),
        ...},

        chunk_2: {
        gpu_index_1: (anchor_point_2_1, anchor_labels_2_1),
        gpu_index_2: (anchor_point_2_2, anchor_labels_2_2),
        ...},
        
        ...
        }
        """
        self.anchor_points = {}
        self.anchor_labels = {}

        _anchor_step = len(anchor_points) // (num_devices * num_chunks)
        anchor_ids = torch.arange(0, len(anchor_points), _anchor_step)

        for dict_index, _anchor_start_index in enumerate(anchor_ids):
            _gpu_index = dict_index % num_devices
            _chunk_index = dict_index // num_devices
            if _chunk_index not in self.anchor_points.keys():
                self.anchor_points[_chunk_index] = {}
                self.anchor_labels[_chunk_index] = {}

            self.anchor_points[_chunk_index][_gpu_index] = anchor_points[
                _anchor_start_index : _anchor_start_index + _anchor_step
            ].clone()

            self.anchor_labels[_chunk_index][_gpu_index] = anchor_labels[
                _anchor_start_index : _anchor_start_index + _anchor_step
            ].clone()

        self.num_chunks = len(list(self.anchor_points.keys()))
        self.num_device_list = [
            len(list(self.anchor_points[_chunk_index].keys()))
            for _chunk_index in range(self.num_chunks)
        ]
        timeprint(
            f"Threat computation will be split across {num_devices} gpus with {self.threat_hparams.num_chunks} chunks"
        )
        # timeprint("Number of chunks: ", self.num_chunks)
        # timeprint("Number of devices: ", self.num_device_list)

        if self.dataset_hparams.dataset_name == "imagenet":
            self.mini_batch_scaling = 8
            self.mini_batch_additional_scaling = 4
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]
        elif self.dataset_hparams.dataset_name == "cifar10":
            self.mini_batch_scaling = 1
            self.mini_batch_additional_scaling = 1
            self.mean = [0.4914, 0.4822, 0.4465]
            self.std = [0.2471, 0.2435, 0.2616]
        elif self.dataset_hparams.dataset_name == "cifar100":
            self.mini_batch_scaling = 1
            self.mini_batch_additional_scaling = 1
            self.mean = [0.5071, 0.4865, 0.4409]
            self.std = [0.2673, 0.2564, 0.2762]

        if self.weighted:
            assert (
                self.dataset_hparams.dataset_name == "imagenet"
            ), "Weighted threat only supported for ImageNet"
            file_name = os.path.join(
                get_platform().runner_root,
                "imagenet",
                "threat_specification",
                "imagenet_PD_weights.pkl",
            )

            with open(file_name, "rb") as f:
                info = pickle.load(f)
                self.raw_weights = info["weighted_betas"] + 1e-4
                assert self.raw_weights.shape == (num_labels, num_labels)
                self.weights = torch.zeros(num_labels, num_labels)

                for i in range(num_labels):
                    for j in range(num_labels):
                        self.weights[i][j] = self.raw_weights[i][j] ** 2 + 1e-4

                assert self.weights.min() > 0, "Weights should be positive"

        if self.segmented:
            device = get_platform().torch_device
            sam_checkpoint = os.path.join(
                get_platform().model_root, "sam_vit_h_4b8939.pth"
            )  # Change path accordingly #sam_vit_b_01ec64
            model_type = "vit_h"  # You can use vit_l, vit_h for larger models
            sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)
            self.segmentation_predictor = SamPredictor(sam)

        del anchor_points, anchor_labels

    @torch.no_grad()
    def get_masks(self, examples):
        if not self.segmented:
            raise ValueError(
                "Threat specification was initialized without segmentation option"
            )
        if self.segmentation_predictor is None:
            raise ValueError("Segmentation model not loaded in initialization")

        boolean_masks = []
        for i in range(examples.shape[0]):
            # Convert and preprocess image
            image = self.denormalize(examples[i]).permute(1, 2, 0).numpy() * 255
            image = image.astype(np.uint8)
            self.segmentation_predictor.set_image(image)

            # Generate masks (center point as prompt, for instance)
            input_point = np.array(
                [[image.shape[1] // 2, image.shape[0] // 2]]
            )  # Center pixel
            input_label = np.array([1])  # Foreground prompt
            masks, scores, _ = self.segmentation_predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )

            # Take the mask with the highest confidence
            best_mask = masks[np.argmax(scores)]
            boolean_masks.append(best_mask.astype(bool))

        examples_mask = torch.stack(
            [torch.tensor(mask, dtype=torch.bool) for mask in boolean_masks]
        ).unsqueeze(1)
        examples_mask = examples_mask.repeat(1, 3, 1, 1)

        return examples_mask

    @torch.no_grad()
    def evaluate(
        self,
        examples: torch.Tensor,
        labels: torch.Tensor,
        perturbed_examples: torch.Tensor,
        gray_scale: bool = False,
        weighted: bool = False,
        segmented: bool = False,
        examples_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Evaluate the threat on a model and dataset."""
        if weighted and not self.weighted:
            raise ValueError("Threat model is not weighted during preparation")

        """"
        segmentation support 

        """

        batch_threat_stats = torch.zeros(examples.shape[0])

        if segmented and examples_mask is None:
            examples_mask = self.get_masks(examples)
            # boolean_masks = []
            # for i in range(examples.shape[0]):
            #     # Convert and preprocess image
            #     image = self.denormalize(examples[i]).permute(1, 2, 0).numpy() * 255
            #     image = image.astype(np.uint8)
            #     self.segmentation_predictor.set_image(image)

            #     # Generate masks (center point as prompt, for instance)
            #     input_point = np.array(
            #         [[image.shape[1] // 2, image.shape[0] // 2]]
            #     )  # Center pixel
            #     input_label = np.array([1])  # Foreground prompt
            #     masks, scores, _ = self.segmentation_predictor.predict(
            #         point_coords=input_point,
            #         point_labels=input_label,
            #         multimask_output=True,
            #     )

            #     # Take the mask with the highest confidence
            #     best_mask = masks[np.argmax(scores)]
            #     boolean_masks.append(best_mask.astype(bool))

            # examples_mask = torch.stack(
            #     [torch.tensor(mask, dtype=torch.bool) for mask in boolean_masks]
            # ).unsqueeze(1)
            # examples_mask = examples_mask.repeat(1, 3, 1, 1)

        transform = v2.Grayscale() if gray_scale else v2.Lambda(lambda x: x)
        examples = transform(examples)
        perturbed_examples = transform(perturbed_examples)

        for _chunk_index in range(self.num_chunks):
            chunk_threat_stats = torch.zeros(examples.shape[0])
            num_devices = self.num_device_list[_chunk_index]
            transformed_anchor_points = {}
            transformed_anchor_labels = {}
            for _gpu_index in range(num_devices):
                transformed_anchor_points[_gpu_index] = torch.flatten(
                    transform(self.anchor_points[_chunk_index][_gpu_index]),
                    start_dim=1,
                )
                if not self.threat_hparams.full_precision:
                    transformed_anchor_points[_gpu_index] = _move(
                        _cast(transformed_anchor_points[_gpu_index], torch.float16),
                        torch.device(_gpu_index),
                    )

                transformed_anchor_labels[_gpu_index] = _move(
                    self.anchor_labels[_chunk_index][_gpu_index],
                    torch.device(_gpu_index),
                )
            mini_batch_size = examples.shape[0] // self.mini_batch_scaling  # // 1
            if mini_batch_size == 0:
                mini_batch_size = 1

            # evaluate threats for this chunk and keep a running max.
            for start_index in range(0, examples.shape[0], mini_batch_size):
                end_index = min(start_index + mini_batch_size, examples.shape[0])
                if end_index == start_index:
                    continue  # skip empty batch

                chunk_threat_stats[
                    start_index:end_index
                ] = self._projected_displacement_threat(
                    examples[start_index:end_index],
                    labels[start_index:end_index],
                    perturbed_examples[start_index:end_index],
                    transformed_anchor_points,
                    transformed_anchor_labels,
                    num_devices=num_devices,
                    float_16=not self.threat_hparams.full_precision,
                    weighted=weighted,
                    mask=examples_mask[start_index:end_index]
                    if examples_mask is not None
                    else None,
                ).cpu()

            del transformed_anchor_points, transformed_anchor_labels

            batch_threat_stats = torch.max(batch_threat_stats, chunk_threat_stats)

        return batch_threat_stats

    @torch.no_grad()
    def project(
        self,
        examples: torch.Tensor,
        labels: torch.Tensor,
        perturbed_examples: torch.Tensor,
        threshold: float = 0.1,
        gray_scale: bool = False,
        lazy_project: bool = True,
        return_threats: bool = False,
        weighted: bool = False,
        segmented: bool = False,
    ) -> torch.Tensor:
        """Project the perturbed examples onto the permissible set of the threat specification."""
        assert (
            lazy_project
        ), "Projected displacement threat model only supports lazy projection for now"

        if weighted and not self.weighted:
            raise ValueError("Threat model is not weighted during preparation")

        if segmented and not self.segmented:
            raise ValueError(
                "Threat specification was initialized without segmentation option"
            )

        if segmented:
            examples_mask = self.get_masks(examples)
        else:
            examples_mask = None

        threats = self.evaluate(
            examples,
            labels,
            perturbed_examples,
            gray_scale=gray_scale,
            weighted=weighted,
            examples_mask=examples_mask,
        )

        if threats.isnan().any():
            print(threats)
            raise ValueError("Threats have NaN values")

        perturbation = perturbed_examples - examples
        perturbation = perturbation.clone().detach()

        if examples_mask is not None:
            masked_fg_perturbation = perturbation * examples_mask
            masked_bg_perturbation = perturbation * (~examples_mask)
            # only perturbation in the mask selected fg region should be projected.
            for _index in range(examples.shape[0]):
                if threats[_index] > threshold:
                    masked_fg_perturbation[_index] *= threshold / threats[_index]
            perturbation = masked_fg_perturbation + masked_bg_perturbation
        else:
            for _index in range(examples.shape[0]):
                if threats[_index] > threshold:
                    perturbation[_index] *= threshold / threats[_index]

        projected_examples = examples + perturbation
        if projected_examples.isnan().any():
            raise ValueError("NaNs in perturbed examples")

        if return_threats:
            return projected_examples.clone().detach(), threats.clone().detach()

        return projected_examples.clone().detach()

    def __call__(
        self,
        examples: torch.Tensor,
        labels: torch.Tensor,
        perturbed_examples: torch.Tensor,
    ) -> torch.Tensor:
        return self.evaluate(examples, labels, perturbed_examples)

    @torch.no_grad()
    def _projected_displacement_threat(
        self,
        examples,
        labels,
        perturbed_examples,
        anchor_points,
        anchor_labels,
        num_devices=1,
        float_16=False,
        weighted=False,
        mask: torch.Tensor = None,
    ):

        assert (
            examples.shape == perturbed_examples.shape
        ), "Reference input shape {} and perturbed input shape {} does not match".format(
            examples.shape, perturbed_examples.shape
        )

        assert len(examples) == len(
            labels
        ), "Number of examples and labels do not match"

        assert len(examples) > 0, "Empty batch"

        num_channels = examples.shape[1]

        if float_16:
            examples = _cast(examples, torch.float16)
            perturbed_examples = _cast(perturbed_examples, torch.float16)

        if mask is not None:
            assert (
                mask.shape == examples.shape
            ), f"Mask shape {mask.shape} does not match input shape {examples.shape}"
            mask = _move(torch.flatten(mask, start_dim=1), get_platform().torch_device)

        examples = _move(
            torch.flatten(examples, start_dim=1), get_platform().torch_device
        )
        perturbed_examples = _move(
            torch.flatten(perturbed_examples, start_dim=1),
            get_platform().torch_device,
        )

        labels = _move(labels, get_platform().torch_device)

        threats = torch.zeros(len(examples), requires_grad=False).to(
            get_platform().torch_device
        )
        input_step = len(examples) // self.mini_batch_additional_scaling  # // 8
        if input_step > 4:
            input_step = 4
        if input_step == 0:
            input_step = 1

        for _gpu_index in range(num_devices):

            for _index in range(0, len(examples), input_step):
                end_index = min(_index + input_step, len(examples))
                if end_index == _index:
                    continue  # skip empty batch

                ref_inputs = _move(
                    examples[_index : _index + input_step], torch.device(_gpu_index)
                )
                ref_labels = _move(
                    labels[_index : _index + input_step], torch.device(_gpu_index)
                )
                pert_inputs = _move(
                    perturbed_examples[_index : _index + input_step],
                    torch.device(_gpu_index),
                )
                ref_mask = (
                    _move(mask[_index : _index + input_step], torch.device(_gpu_index))
                    if mask is not None
                    else None
                )

                threats[_index : _index + input_step] = torch.maximum(
                    _move(
                        self._PD_partial_batch(
                            ref_inputs,
                            ref_labels,
                            pert_inputs,
                            anchor_points[_gpu_index],
                            anchor_labels[_gpu_index],
                            float_16=float_16,
                            weighted=weighted,
                            mask=ref_mask,
                        ),
                        get_platform().torch_device,
                    ),
                    threats[_index : _index + input_step],
                )
        return threats

    @torch.no_grad()
    def _PD_partial_batch(
        self,
        reference_input,
        reference_label,
        perturbed_input,
        anchor_points,
        anchor_labels,
        beta=0.5,
        device=None,
        float_16=False,
        weighted=False,
        mask=None,
    ):
        device = device or str(reference_input.device)
        dtype = torch.float16 if float_16 else torch.float32
        # print("Inside PD partial batch")
        # print(anchor_points.shape)
        # print(reference_input.shape)

        with torch.autocast(device_type="cuda", dtype=dtype):
            perturbation = (perturbed_input - reference_input).unsqueeze(1)
            unsafe_perturbations = -(
                reference_input.unsqueeze(1) - anchor_points
            )  # anchor_points - reference_input.unsqueeze(0)

            # i have mask for each reference input of the same shape as reference input.
            # i need to apply this to unsafe perturbations
            if mask is not None:
                unsafe_perturbations = unsafe_perturbations * mask.unsqueeze(1)

            unsafe_norms = (
                beta
                * (
                    torch.linalg.norm(
                        unsafe_perturbations, dim=2, ord=2, keepdim=True
                    ).pow(2)
                )
                + 1e-5
            )

            if mask is not None:
                unsafe_perturbations = unsafe_perturbations * mask.unsqueeze(1)
                # check
                # print("Reference input shape: ", reference_input.shape)
                # print("Mask shape: ", mask.shape)
                # print("Unsafe perturbations shape: ", unsafe_perturbations.shape)
                # explicit_unsafe = torch.zeros_like(unsafe_perturbations)
                # for i in range(len(reference_input)):
                #     explicit_unsafe[i] = unsafe_perturbations[i] * mask[i]
                # # if not torch.allclose(unsafe_perturbations, explicit_unsafe):
                #     raise ValueError("Mask not applied successfully")
                # else:
                #     print("Mask applied successfully")

            if weighted:
                self.weights = _move(self.weights, device)
                # Perform advanced indexing
                # print("Reference label shape: ", reference_label.shape)
                # print("Anchor labels shape: ", anchor_labels.shape)
                N = reference_label.shape[0]
                M = anchor_labels.shape[0]
                # Expand labels and anchor_labels to perform batched indexing
                # expanded_labels = reference_label.expand(N, M)  # NxM
                # expanded_anchor_labels = anchor_labels.T.expand(N, M)  # NxM

                # Use torch.meshgrid to create index pairs
                row_indices, col_indices = torch.meshgrid(
                    reference_label, anchor_labels, indexing="ij"
                )  # Shape: NxM

                # Select corresponding weights
                # selected_weights = self.weights[
                #    expanded_labels, expanded_anchor_labels
                # ]  # NxM

                # Perform advanced indexing into weights
                selected_weights = self.weights[row_indices, col_indices]

                # Ensure shape compatibility for multiplication
                selected_weights = selected_weights.unsqueeze(2)  # NxMx1

                # # Extract weights based on labels and anchor_labels
                # selected_weights = self.weights[reference_label, anchor_labels.T]  # NxM

                # # Ensure selected_weights matches unsafe_norms for broadcasting
                # selected_weights = selected_weights.unsqueeze(2)  # NxMx1

                # scale selected weights by a factor of 50 for comparability
                selected_weights = selected_weights * 50

                # Multiply unsafe_norms by corresponding weights
                unsafe_norms = unsafe_norms * selected_weights

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
            # print("Anchor labels shape: ", anchor_labels.shape)
            # print("Reference labels shape: ", reference_label.shape)

            mask = _move(
                reference_label.unsqueeze(0) != anchor_labels.unsqueeze(1),
                reference_input.device,
            )
            # print("Mask shape: ", mask.shape)
            scaled_projections = torch.clamp(scaled_projections, min=0.0)

            scaled_projections = scaled_projections.squeeze(1) * mask.T.float()

            partial_threats = torch.max(scaled_projections, dim=1).values

            return partial_threats

    def _get_path(self):
        dataset_name = self.dataset_hparams.dataset_name
        num_labels = self.dataset_hparams.num_labels
        per_label = self.threat_hparams.per_label
        subset_selection = self.threat_hparams.subset_selection
        input_shape = [
            self.dataset_hparams.num_channels,
            self.dataset_hparams.num_spatial_dims,
            self.dataset_hparams.num_spatial_dims,
        ]

        dataset_loc = os.path.join(get_platform().runner_root, dataset_name)
        threat_dir = "greedy"  # self.threat_hparams.dir_path(identifier_name="subset_selection")  # greedy_xx
        # greedy_800b81

        threat_hparams_path = os.path.join(
            dataset_loc, "threat_specification", threat_dir
        )  # nonisotropic/runner_data/cifar10/threat_specification/greedy_xx

        threat_run_path = paths.threat_run_path(
            dataset_loc, threat_dir, self.threat_replicate
        )  # nonisotropic/runner_data/cifar10/threat_specification/greedy_xx/threat_replicate_1
        threat_specification_path = paths.threat_specification(
            threat_run_path, per_label
        )
        get_platform().makedirs(threat_specification_path)

        return threat_specification_path

    def _get_anchor_path(self, label):
        threat_specification_path = self._get_path()
        first_half_path = paths.anchor_points(
            threat_specification_path, label, first_half=True
        )
        second_half_path = paths.anchor_points(
            threat_specification_path, label, first_half=False
        )

        return first_half_path, second_half_path

    def _get_input_shape(self):
        return [
            self.dataset_hparams.num_channels,
            self.dataset_hparams.num_spatial_dims,
            self.dataset_hparams.num_spatial_dims,
        ]

    def _per_label_shape(self):
        return [self.threat_hparams.per_label] + self._get_input_shape()

    def _get_threat_specification_shape(self):
        return [
            self.dataset_hparams.num_labels,
            self.threat_hparams.per_label,
        ] + self._get_input_shape()

    def save_threat_specification(self, label, first_half=True):
        """Compute and save the threat."""
        if verbose and get_platform().is_primary_process and label % 10 == 0:
            timeprint("Finding threat specification for label " + str(label))

        subset_selection = self.threat_hparams.subset_selection
        per_label = self.threat_hparams.per_label
        domain_expansion_factor = self.threat_hparams.domain_expansion_factor
        subset_selection_seed = self.threat_hparams.subset_selection_seed
        dataset_name = self.dataset_hparams.dataset_name
        dataset_loc = paths.dataset(get_platform().dataset_root, dataset_name)
        threat_specification_path = self._get_path()

        # set the random seed
        torch.manual_seed(subset_selection_seed)

        image_partition = (
            get_platform()
            .load_model(paths.class_partition(dataset_loc, label, train=True))
            .to(device=get_platform().torch_device)
        )

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

        get_platform().save_model(
            threat_specification_first_half,
            paths.anchor_points(threat_specification_path, label, first_half=True),
        )

        get_platform().save_model(
            threat_specification_second_half,
            paths.anchor_points(threat_specification_path, label, first_half=False),
        )

        del image_partition
        return torch.cat(
            (threat_specification_first_half, threat_specification_second_half), dim=0
        ).cpu()

    def load_threat_specification(self, label):
        num_labels = self.dataset_hparams.num_labels
        per_label = self.threat_hparams.per_label

        first_half_path, second_half_path = self._get_anchor_path(label)
        per_label_threat_specification = torch.zeros(self._per_label_shape())

        if get_platform().exists(first_half_path) and get_platform().exists(
            second_half_path
        ):
            per_label_threat_specification[
                : per_label // 2
            ] = get_platform().load_model(first_half_path, map_location="cpu")
            per_label_threat_specification[
                per_label // 2 :
            ] = get_platform().load_model(second_half_path, map_location="cpu")
        else:
            print(first_half_path)
            print(second_half_path)
            raise ValueError("Threat specification not found for label ", label)
            # need to compute it.
            per_label_threat_specification = self.save_threat_specification(label)

        return per_label_threat_specification

    # Function to denormalize an image
    def denormalize(self, image_tensor):
        image = image_tensor.clone()  # Clone to avoid altering original tensor
        for t, m, s in zip(image, self.mean, self.std):
            t.mul_(s).add_(m)  # Apply reverse normalization: image * std + mean
        return image

    @property
    def display(self):
        return "\n".join(
            [
                self.dataset_hparams.display,
                self.threat_hparams.display,
            ]
        )
