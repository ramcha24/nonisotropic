cifar10_Linf_RB_model_names = [
    "Peng2023Robust",
    "Wang2023Better_WRN-70-16",
    # "Bai2024MixedNUTS",
    # "Bai2023Improving_edm",
    "Cui2023Decoupled_WRN-28-10",
    "Wang2023Better_WRN-28-10",
    "Rebuffi2021Fixing_70_16_cutmix_extra",
    "Gowal2021Improving_70_16_ddpm_100m",
    "Gowal2020Uncovering_70_16_extra",
    "Huang2022Revisiting_WRN-A4",
    "Rebuffi2021Fixing_106_16_cutmix_ddpm",
    "Rebuffi2021Fixing_70_16_cutmix_ddpm",
    "Kang2021Stable",
    "Xu2023Exploring_WRN-28-10",
    "Gowal2021Improving_28_10_ddpm_100m",
]
cifar10_dict = {"Linf": cifar10_Linf_RB_model_names}

default_cifar10_dict = {"Linf": cifar10_Linf_RB_model_names[7]}


cifar100_Linf_RB_model_names = [
    "Wang2023Better_WRN-70-16",
    # "Bai2024MixedNUTS",
    "Cui2023Decoupled_WRN-28-10",
    "Wang2023Better_WRN-28-10",
    # "Bai2023Improving_edm",
    "Gowal2020Uncovering_extra",
    "Bai2023Improving_trades",
    "Debenedetti2022Light_XCiT-L12",
    "Rebuffi2021Fixing_70_16_cutmix_ddpm",
    "Debenedetti2022Light_XCiT-M12",
    "Pang2022Robustness_WRN70_16",
    "Cui2023Decoupled_WRN-34-10_autoaug",
    "Debenedetti2022Light_XCiT-S12",
    "Rebuffi2021Fixing_28_10_cutmix_ddpm",
    "Jia2022LAS-AT_34_20",
]
cifar100_dict = {"Linf": cifar100_Linf_RB_model_names}
default_cifar100_dict = {"Linf": cifar100_Linf_RB_model_names[7]}


imagenet_Linf_RB_model_names = [
    "Liu2023Comprehensive_Swin-L",
    # "Bai2024MixedNUTS",
    "Liu2023Comprehensive_ConvNeXt-L",
    "Singh2023Revisiting_ConvNeXt-L-ConvStem",
    "Liu2023Comprehensive_Swin-B",
    "Singh2023Revisiting_ConvNeXt-B-ConvStem",
    "Liu2023Comprehensive_ConvNeXt-B",
    "Singh2023Revisiting_ViT-B-ConvStem",
    "Singh2023Revisiting_ConvNeXt-S-ConvStem",
    "Singh2023Revisiting_ConvNeXt-T-ConvStem",
    "Peng2023Robust",
    "Singh2023Revisiting_ViT-S-ConvStem",
    "Debenedetti2022Light_XCiT-L12",
    "Debenedetti2022Light_XCiT-M12",
    "Debenedetti2022Light_XCiT-S12",
]
imagenet_dict = {"Linf": imagenet_Linf_RB_model_names}
default_imagenet_dict = {"Linf": imagenet_Linf_RB_model_names[7]}

rb_registry = {
    "cifar10": cifar10_dict,
    "cifar100": cifar100_dict,
    "imagenet": imagenet_dict,
}

default_rb_registry = {
    "cifar10": default_cifar10_dict,
    "cifar100": default_cifar100_dict,
    "imagenet": default_imagenet_dict,
}
