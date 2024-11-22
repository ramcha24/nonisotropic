from torchvision.transforms import v2
from PIL import Image
import torch

from corruptions.ImageNetC.create_c import make_imagenet_c as make_imagenet_c
from corruptions.ImageNetC.imagenetc.imagenet_c import (
    corruption_dict as imagenet_corruption_dict,
)
from corruptions.ImageNetC.imagenetc.imagenet_c.corruptions import *

from corruptions.imagenet_c_bar.transform_finder import transform_dict as cbar_dict
from corruptions.imagenet_c_bar.transform_finder import (
    build_transform as corruption_cbar,
)
from corruptions.imagenet_c_bar.utils.converters import PilToNumpy, NumpyToTensor

# functional in memory perturbations
# load from disk perturbations

transformations = {
    "Perspective": v2.RandomPerspective(distortion_scale=0.6, p=1.0),
    "Rotation": v2.RandomRotation(degrees=(0, 180)),
    "Affine": v2.RandomAffine(
        degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75)
    ),
    "Elastic": v2.ElasticTransform(alpha=250.0),
    "Gray": v2.Grayscale(num_output_channels=3),
    "Jitter": v2.ColorJitter(brightness=0.5, hue=0.3),
    "Gaussian Blur": v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.0)),
    "Invert": v2.RandomInvert(p=1.0),
    "Posterize": v2.RandomPosterize(bits=2, p=1.0),
    "Solarize": v2.RandomSolarize(threshold=0.77, p=1.0),
    "Adjust Sharpness": v2.RandomAdjustSharpness(sharpness_factor=2, p=1.0),
    "Auto Contrast": v2.RandomAutocontrast(p=1.0),
    "Equalize": v2.RandomEqualize(p=1.0),
    "RandAugment": v2.RandAugment(),
    "AugMix": v2.AugMix(),
    "Horizontal Flip": v2.RandomHorizontalFlip(p=1.0),
    "Vertical Flip": v2.RandomVerticalFlip(p=1.0),
}

common_corruptions_2d = {
    "Gaussian Noise": "gaussian_noise",
    "Shot Noise": "shot_noise",
    "Impulse Noise": "impulse_noise",
    "Defocus Blur": "defocus_blur",
    "Glass Blur": "glass_blur",
    "Motion Blur": "motion_blur",
    "Zoom Blur": "zoom_blur",
    "Snow": "snow",
    "Frost": "frost",
    "Fog": "fog",
    "Brightness": "brightness",
    "Contrast": "contrast",
    "Elastic Transform": "elastic_transform",
    "Pixelate": "pixelate",
    "JPEG Compression": "jpeg_compression",
    "Speckle Noise": "speckle_noise",
    "Gaussian Blur": "gaussian_blur",
    "Spatter": "spatter",
    "Saturate": "saturate",
}


class CC2DTransform:
    def __init__(
        self, perturbation_style="Gaussian Noise", severity=1, dataset_name="imagenet"
    ):
        self.severity = severity
        self.perturbation_style = perturbation_style
        self.dataset_name = dataset_name

    def transform(self, image):
        # print("Transforming image in memory")
        if self.dataset_name == "imagenet":
            # image = torch.permute(image, (1, 2, 0))
            # image = (image * 255).byte().numpy()
            # image = Image.fromarray(image)
            # print(
            #     "Transforming image in memory for corruption name ",
            #     self.perturbation_style,
            # )

            corrupted_image = imagenet_corruption_dict[
                common_corruptions_2d[self.perturbation_style]
            ](
                image,
                severity=self.severity,
            )
            if isinstance(corrupted_image, Image.Image):
                return corrupted_image
            else:
                return Image.fromarray(
                    corrupted_image.astype(np.uint8)
                )  # torch.from_numpy(corrupted_image).permute(2, 0, 1).float() / 255.0
        elif self.dataset_name == "cifar10":
            raise NotImplementedError(
                "CIFAR-10 in-memory perturbation has not been implemented yet."
            )
        else:
            raise ValueError(
                f"Dataset name {self.dataset_name} not recognized for in-memory perturbation. Must be 'imagenet' or 'cifar10'."
            )

    def __call__(self, image):
        return self.transform(image)


# expensive corruptions
# blue noise
# brownish noise
# caustic refraction
common_corruptions_2d_bar = {
    "Blue Noise Sample": "blue_noise_sample",
    "Brownish Noise": "brownish_noise",
    # "Caustic Refraction": "caustic_refraction",
    # "Checkerboard Cutout": "checkerboard_cutout",
    "Cocentric Sine Waves": "cocentric_sine_waves",
    "Inverse Sparkles": "inverse_sparkles",
    "Perlin Noise": "perlin_noise",
    # "Plasma Noise": "plasma_noise",
    # "Single Frequency Greyscale": "single_frequency_greyscale",
    # "Sparkles": "sparkles",
}

common_corruptions_2d_bar_severities = {
    "Blue Noise Sample": [0.8, 1.6, 2.4, 4.0, 5.6],
    "Brownish Noise": [1.0, 2.0, 3.0, 4.0, 5.0],
    "Caustic Refraction": [2.35, 3.2, 4.9, 6.6, 9.15],
    "Checkerboard Cutout": [2.0, 3.0, 4.0, 5.0, 6.0],
    "Cocentric Sine Waves": [3.0, 5.0, 8.0, 9.0, 10.0],
    "Inverse Sparkles": [1.0, 2.0, 4.0, 9.0, 10.0],
    "Perlin Noise": [4.6, 5.2, 5.8, 7.6, 8.8],
    "Plasma Noise": [4.75, 7.0, 8.5, 9.25, 10.0],
    "Single Frequency Greyscale": [1.0, 1.5, 2.0, 4.5, 5.0],
    "Sparkles": [1.0, 2.0, 3.0, 5.0, 6.0],
}

# 3D simple - Nah 3D is out of the question. We can't do that.
# non3d : quantization iso noise low light noise
# near and far focus requires depth data which is downloaded.
# xy and z motion blur requires some coding.

common_corruptions_3d = {
    "Color Quantization": "color_quant",
    "Near Focus": "near_focus",
    "Far Focus": "far_focus",
    "Flash": "flash",
    "Fog 3D": "fog_3d",
    "ISO Noise": "iso_noise",
    "Low-Light Noise": "low_light",
    "XY-Motion Blur": "xy_motion_blur",
    "Z-Motion Blur": "z_motion_blur",
    "Bit Error": "bit_error",
    "CRF Compress": "h265_crf",
    "ABR Compress": "h265_abr",
}

backgrounds = {
    "FG Mask": "fg_mask",
    "Mixed-Next": "mixed_next",
    "Mixed-Rand": "mixed_rand",
    "Mixed-Same": "mixed_same",
    "No FG": "no_fg",
    "Only BG-B": "only_bg_b",
    "Only BG-T": "only_bg_t",
    "Only FG": "only_fg",
    "Original": "original",
}
