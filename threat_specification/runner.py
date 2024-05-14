import abc
import argparse

from cli import shared_args

from foundations.runner import Runner
from threat_specification.desc import ThreatDesc
from threat_specification.greedy_subset import save_greedy_partition
from platforms.platform import get_platform
from foundations import paths
from datasets.partition import save_class_partition


class ThreatRunner(Runner):
    desc: ThreatDesc
    verbose: bool = True
    threat_replicate: int = 1

    @staticmethod
    def description() -> str:
        return "Compute threat specifications."

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        ThreatDesc.add_args(
            parser,
            shared_args.maybe_get_default_hparams(runner_name="threat_specification"),
        )

    @staticmethod
    def create_from_args(args: argparse.Namespace) -> "Runner":
        desc = ThreatDesc.create_from_args(args)
        verbose = not args.quiet
        threat_replicate = (
            args.threat_replicate if args.threat_replicate is not None else 1
        )
        return ThreatRunner(desc, verbose, threat_replicate)

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
                + f"Computing threat specification for dataset : {dataset_name} (Replicate : {self.threat_replicate})"
                + "=" * 82
            )
            self.desc.display()
            print("\n Output Location : " + threat_run_path + +"\n" + "-" * 82 + "\n")

        self.desc.save_param(self.threat_replicate)

        # check if there is class-wise partitioning for the dataset. If not, create one
        save_class_partition(dataset_name)

        # run the threat specification
        per_label_array = self.desc.threat_hparams.per_label_array
        subset_selection = self.desc.threat_hparams.subset_selection
        assert (
            subset_selection == "greedy"
        ), "Only greedy subset selection is currently supported"
        # in the future we can add more subset selection methods and have a registry to fetch the appropriate method

        for per_label in per_label_array:
            save_greedy_partition(threat_run_path, self.desc.dataset_hparams, per_label)
