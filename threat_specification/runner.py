import abc
import argparse
from dataclasses import dataclass

from cli import shared_args
from platforms.platform import get_platform

from foundations.runner import Runner
from foundations import paths

from datasets.partition import save_class_partition

from threat_specification.desc import ThreatDesc
from threat_specification.subset_selection import compute_threat_specification
from threat_specification.evaluate_threat import evaluate_threat_specification


@dataclass
class ComputeThreatRunner(Runner):
    desc: ThreatDesc
    threat_replicate: int = 1
    verbose: bool = True

    @staticmethod
    def description() -> str:
        return "Compute threat specifications."

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        ThreatDesc.add_args(
            parser,
            shared_args.maybe_get_default_hparams(runner_name="compute_threat"),
        )

    @staticmethod
    def create_from_args(args: argparse.Namespace) -> "Runner":
        desc = ThreatDesc.create_from_args(args)
        threat_replicate = (
            args.threat_replicate if args.threat_replicate is not None else 1
        )
        verbose = not args.quiet
        return ComputeThreatRunner(desc, threat_replicate, verbose)

    def display_output_location(self) -> None:
        logger_paths = self.desc.run_path(self.threat_replicate, verbose=self.verbose)
        print("\n Output Location : " + logger_paths["threat_run_path"])

    def run(self) -> None:
        logger_paths = self.desc.run_path(self.threat_replicate)
        threat_run_path = logger_paths["threat_run_path"]
        dataset_name = self.desc.dataset_hparams.dataset_name

        if self.verbose and get_platform().is_primary_process:
            print(
                "=" * 82
                + f"\nComputing threat specification for dataset : {dataset_name} (Replicate : {self.threat_replicate})\n"
                + "=" * 82
            )
            print(self.desc.display)
            print("\n Output Location : " + threat_run_path + "\n" + "-" * 82 + "\n")

        self.desc.save_param(self.threat_replicate)

        # check if there is class-wise partitioning for the dataset. If not, create one
        save_class_partition(self.desc.dataset_hparams, dataset_type="train")

        # run the threat specification
        compute_threat_specification(
            threat_run_path,
            self.desc.dataset_hparams,
            self.desc.threat_hparams,
            verbose=self.verbose,
        )


@dataclass
class EvaluateThreatRunner(Runner):
    desc: ThreatDesc
    threat_replicate: int = 1
    verbose: bool = True
    float_16: bool = False
    in_memory: bool = False

    @staticmethod
    def description() -> str:
        return "Evaluate threat specifications."

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        ThreatDesc.add_args(
            parser,
            shared_args.maybe_get_default_hparams(runner_name="evaluate_threat"),
        )

    @staticmethod
    def create_from_args(args: argparse.Namespace) -> "Runner":
        desc = ThreatDesc.create_from_args(args)
        threat_replicate = (
            args.threat_replicate if args.threat_replicate is not None else 1
        )
        verbose = not args.quiet
        float_16 = args.float_16
        in_memory = args.in_memory
        return EvaluateThreatRunner(
            desc, threat_replicate, verbose, float_16, in_memory
        )

    def display_output_location(self) -> None:
        logger_paths = self.desc.run_path(self.threat_replicate, verbose=self.verbose)
        print("\n Output Location : " + logger_paths["threat_run_path"])

    def run(self) -> None:
        logger_paths = self.desc.run_path(self.threat_replicate)
        threat_run_path = logger_paths["threat_run_path"]
        dataset_name = self.desc.dataset_hparams.dataset_name

        if self.verbose and get_platform().is_primary_process:
            print(
                "=" * 82
                + f"\nComputing threat specification for dataset : {dataset_name} (Replicate : {self.threat_replicate})\n"
                + "=" * 82
            )
            print(self.desc.display)
            print("\n Output Location : " + threat_run_path + "\n" + "-" * 82 + "\n")

        self.desc.save_param(self.threat_replicate)

        # run the threat specification
        evaluate_threat_specification(
            threat_run_path,
            self.desc.dataset_hparams,
            self.desc.threat_hparams,
            self.desc.perturbation_hparams,
            verbose=self.verbose,
            float_16=self.float_16,
            in_memory=self.in_memory,
        )
