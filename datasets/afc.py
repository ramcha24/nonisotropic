from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import numpy as np
from PIL import Image
import os
import torch

from platforms.platform import get_platform


def get_class_id(class_name):
    file_path = os.path.join(
        get_platform().dataset_root, "imagenet", "imagenet1000_clsidx_to_labels.txt"
    )

    class_id = None

    with open(file_path, "r") as file:
        for line in file:
            # Split each line into class_id and word label
            parts = line.strip().split(
                ": ", 1
            )  # Split only on the first occurrence of ': '
            if len(parts) == 2:
                current_class_id, word_label = parts
                # Clean up the word label (remove quotes)
                word_label = word_label.strip("'")
                # Check if the target label is a substring of the word label
                if class_name in word_label:
                    class_id = current_class_id
                    break
    if class_id is None:
        raise ValueError(f"Could not find class id for class name: {class_name}")

    return int(class_id)


def get_preprocess(model_type):
    if "lpips" in model_type:
        return "LPIPS"
    elif "dists" in model_type:
        return "DISTS"
    elif "psnr" in model_type:
        return "PSNR"
    elif "ssim" in model_type:
        return "SSIM"
    elif "clip" in model_type or "open_clip" in model_type:
        return "CLIP"
    elif "dino" in model_type or "mae" in model_type:
        return "DEFAULT"
    else:
        return "DEFAULT"


def get_preprocess_fn(preprocess, load_size, interpolation):
    if preprocess == "LPIPS":
        t = transforms.ToTensor()
        return lambda pil_img: t(pil_img.convert("RGB")) / 0.5 - 1.0
    else:
        if preprocess == "DEFAULT":
            t = transforms.Compose(
                [
                    transforms.Resize(
                        (load_size, load_size), interpolation=interpolation
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
        elif preprocess == "CLIP":
            t = transforms.Compose(
                [
                    transforms.Resize(
                        (load_size, load_size), interpolation=interpolation
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.48145466, 0.4578275, 0.40821073],
                        [0.26862954, 0.26130258, 0.27577711],
                    ),
                ]
            )

        elif preprocess == "DISTS":
            t = transforms.Compose(
                [transforms.Resize((256, 256)), transforms.ToTensor()]
            )
        elif preprocess == "SSIM" or preprocess == "PSNR":
            t = transforms.ToTensor()
        else:
            raise ValueError(f"Unknown preprocessing method: {preprocess}")
        return lambda pil_img: t(pil_img.convert("RGB"))


class TwoAFCDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str = "test_imagenet",
        load_size: int = 224,
        interpolation: transforms.InterpolationMode = transforms.InterpolationMode.BICUBIC,
        preprocess: str = "DEFAULT",
        **kwargs,
    ):
        self.root_dir = root_dir
        self.csv = pd.read_csv(os.path.join(self.root_dir, "data.csv"))
        self.csv = self.csv[
            self.csv["votes"] >= 6
        ]  # Filter out triplets with less than 6 unanimous votes
        self.split = split
        self.load_size = load_size
        self.interpolation = interpolation
        self.preprocess_fn = get_preprocess_fn(
            preprocess, self.load_size, self.interpolation
        )

        if self.split == "train" or self.split == "val" or self.split == "test":
            self.csv = self.csv[self.csv["split"] == split]
        elif split == "test_imagenet":
            self.csv = self.csv[self.csv["split"] == "test"]
            self.csv = self.csv[self.csv["is_imagenet"] == True]
        elif split == "test_no_imagenet":
            self.csv = self.csv[self.csv["split"] == "test"]
            self.csv = self.csv[self.csv["is_imagenet"] == False]
        else:
            raise ValueError(f"Invalid split: {split}")

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        id = self.csv.iloc[idx, 0]
        p = self.csv.iloc[idx, 2].astype(np.float32)
        img_ref = self.preprocess_fn(
            Image.open(os.path.join(self.root_dir, self.csv.iloc[idx, 4]))
        )
        img_left = self.preprocess_fn(
            Image.open(os.path.join(self.root_dir, self.csv.iloc[idx, 5]))
        )
        img_right = self.preprocess_fn(
            Image.open(os.path.join(self.root_dir, self.csv.iloc[idx, 6]))
        )
        # class_name = self.csv.iloc[idx, 9]
        class_id = np.array(get_class_id(self.csv.iloc[idx, 9])).astype(np.int64)
        return img_ref, img_left, img_right, class_id, p, id
