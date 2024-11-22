import abc
import argparse
from dataclasses import dataclass, fields
import hashlib
import os

from foundations import hparams
from foundations import paths
from foundations import desc
from platforms.platform import get_platform


@dataclass
class ThreatDesc(desc.Desc):
    """The hyperparameters necessary to compute threat specifications"""

    dataset_hparams: hparams.DatasetHparams
    threat_hparams: hparams.ThreatHparams
    perturbation_hparams: hparams.PerturbationHparams

    @staticmethod
    def name_prefix() -> str:
        return "threat_specification"

    @staticmethod
    def add_args(
        parser: argparse.ArgumentParser, defaults: "ThreatDesc" = None, prefix=None
    ) -> None:
        hparams.DatasetHparams.add_args(
            parser,
            defaults=defaults.dataset_hparams if defaults else None,
            prefix=prefix,  # prefix="threat_specification_",
        )
        hparams.ThreatHparams.add_args(
            parser,
            defaults=defaults.threat_hparams if defaults else None,
            prefix=prefix,
        )
        hparams.PerturbationHparams.add_args(
            parser,
            defaults=defaults.perturbation_hparams if defaults else None,
            prefix=prefix,
        )

    @staticmethod
    def create_from_args(args: argparse.Namespace) -> "ThreatDesc":
        dataset_hparams = hparams.DatasetHparams.create_from_args(args)
        threat_hparams = hparams.ThreatHparams.create_from_args(args)
        perturbation_hparams = hparams.PerturbationHparams.create_from_args(args)
        return ThreatDesc(dataset_hparams, threat_hparams, perturbation_hparams)

    def run_path(self, threat_replicate: int = 1, verbose: bool = False) -> dict:
        """_summary_

        Args:
            threat_replicate (_type_): _description_
            verbose (bool, optional): _description_. Defaults to False.

        Returns:
            logger_paths (dict): Containing paths where hparams and threat specification will be stored

        Example run paths
        nonisotropic/runner_data/cifar10/threat_specification/greedy_xx/threat_replicate_T/per_label_m/first_half_labelindex.pt

        threat hparams will be stored at nonisotropic/runner_data/cifar10/threat_specification/greedy/threat_hparams.log

        _xx indicates appropriate hashstring
        Currently not storing hash strings for dataset or model hparams under the assumption these are the same for a given model name dataset_name for all runs

        threat replicate T indexes multiple runs for the same hyperparameter configurations
        """

        assert (
            threat_replicate is not None
        ), "Threat replicate must be specified or inferred"

        assert (
            self.threat_hparams.subset_selection is not None
        ), "Subset selection must be specified"

        logger_paths = dict()
        root_location = get_platform().runner_root  # nonisotropic/runner_data/cifar10
        dataset_dir = self.dataset_hparams.dataset_name
        logger_paths["dataset_path"] = os.path.join(root_location, dataset_dir)

        threat_dir = "greedy"
        # self.threat_hparams.dir_path(
        #     identifier_name="subset_selection"
        # )  # greedy_xx

        logger_paths["threat_hparams_path"] = os.path.join(
            logger_paths["dataset_path"],
            "threat_specification",
            threat_dir,
        )  # nonisotropic/runner_data/cifar10/threat_specification/greedy_xx

        logger_paths["threat_run_path"] = os.path.join(
            logger_paths["threat_hparams_path"],
            "threat_replicate_" + str(threat_replicate),
        )  # nonisotropic/runner_data/cifar10/threat_specification/greedy_xx/threat_replicate_1

        if get_platform().is_primary_process and verbose:
            print(
                "\n All runner data for the specified dataset will be found at {}".format(
                    logger_paths["dataset_path"]
                )
            )
            print(
                "\n Threat hparams will be stored at {}".format(
                    logger_paths["threat_hparams_path"]
                )
            )

        if not get_platform().exists(logger_paths["threat_run_path"]) and verbose:
            print("A job with this configuration has not been run yet.")

        return logger_paths

    @property
    def display(self):
        return "\n".join(
            [
                self.dataset_hparams.display,
                self.threat_hparams.display,
                self.perturbation_hparams.display,
            ]
        )

    def save_param(self, threat_replicate: int = 1):

        logger_paths = self.run_path(threat_replicate)
        full_run_path = logger_paths["threat_run_path"]
        if not get_platform().is_primary_process:
            return
        if not get_platform().exists(full_run_path):
            get_platform().makedirs(full_run_path)
            for type_str in ["threat"]:
                hparams_strs = self.get_hparams_str(type_str=type_str)
                with get_platform().open(
                    paths.params_loc(
                        logger_paths[type_str + "_hparams_path"], type_str
                    ),
                    "w",
                ) as fp:
                    fp.write("\n".join(hparams_strs))
        else:
            print(
                "A job with this configuration may have already been initiated."  # Stale. Delete the existing job results to run again.
            )

        """
        To Do
        1. In the future I want the flexibility to have different algorithms for selecting points (currently the only method is greedy_partition)
        2. I want to be able to specify the number of points to select for each label (currently this is hardcoded to the per_label_array [10,20,30,40,50])
        3. I want to specify the number of points to be considered for choosing each greedy partition (currently this is hardcoded to 10*per_label)
        4. Since there is randomness in the greedy partitioning, I want to be able to specify the threat_replicate to index each run (currently this is hardcoded to 1)
        """
