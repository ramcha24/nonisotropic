import argparse
from dataclasses import dataclass, fields
import os

from foundations import hparams
from foundations import desc
from platforms.platform import get_platform


@dataclass
class PretrainDesc(desc.Desc):
    """The hyperparameters necessary to download pretrained robustbenchmark models"""

    dataset_hparams: hparams.DatasetHparams

    @staticmethod
    def name_prefix() -> str:
        return "download_pretrained_robustbenchmark"

    @staticmethod
    def add_args(
        parser: argparse.ArgumentParser, defaults: "PretrainDesc" = None, prefix=None
    ) -> None:
        hparams.DatasetHparams.add_args(
            parser,
            defaults=defaults.dataset_hparams if defaults else None,
            prefix=prefix,
        )

    @staticmethod
    def create_from_args(args: argparse.Namespace) -> "PretrainDesc":
        dataset_hparams = hparams.DatasetHparams.create_from_args(args)
        return PretrainDesc(dataset_hparams)

    def run_path(self, verbose: bool = False) -> dict:
        """_summary_

        Args:
            verbose (bool, optional): _description_. Defaults to False.

        Returns:
            logger_paths (dict): Containing paths where hparams and threat specification will be stored

        Example run paths
        nonisotropic/runner_data/cifar10/pretrained/robustbenchmark/Linf/Peng2013robust/*.pt
        """

        logger_paths = dict()
        root_location = get_platform().runner_root  # nonisotropic/runner_data/cifar10
        dataset_dir = self.dataset_hparams.dataset_name
        logger_paths["dataset_path"] = os.path.join(root_location, dataset_dir)

        logger_paths["pretrain_run_path"] = os.path.join(
            logger_paths["dataset_path"], "pretrained", "robustbenchmark", "Linf"
        )

        if get_platform().is_primary_process and verbose:
            print(
                "\n All runner data for the specified dataset will be found at {}".format(
                    logger_paths["dataset_path"]
                )
            )
            print(
                "\n Pretrained models will be downloaded and stored at {}".format(
                    logger_paths["pretrain_run_path"]
                )
            )

        if not get_platform().exists(logger_paths["pretrain_run_path"]) and verbose:
            print("A job with this configuration has not been run yet.")

        return logger_paths

    @property
    def display(self):
        return "\n".join(
            [
                self.dataset_hparams.display,
            ]
        )

    def save_param(self):
        pass
