import argparse
from dataclasses import dataclass

from cli import shared_args
from foundations.runner import Runner
import models.registry
from platforms.platform import get_platform

from testing.runner import TestingRunner

from multi_testing import multi_test
from multi_testing.desc import MultiTestingDesc


@dataclass
class MultiTestingRunner(Runner):
    replicate: int
    multi_desc: MultiTestingDesc
    # multi_run: list[TestingRunner]
    verbose: bool = True
    evaluate_batch_only: bool = True
    num_sub_runners: int = 16

    @staticmethod
    def description():
        return "Test multiple models."

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        # shared_args.JobArgs.add_args(parser)
        defaults_list = shared_args.maybe_get_default_hparams(runner_name="multi_test")
        MultiTestingDesc.add_args(parser, defaults_list)

    @staticmethod
    def create_from_args(args: argparse.Namespace) -> "MultiTestingRunner":
        return MultiTestingRunner(
            args.replicate,
            MultiTestingDesc.create_from_args(args),
            not args.quiet,
            args.evaluate_only_batch_test,
        )

    def create_sub_runners(self):
        sub_runner_list = []
        for desc in self.multi_desc.desc_list:
            sub_runner_list.append(
                TestingRunner(
                    self.replicate, desc, self.verbose, self.evaluate_batch_only
                )
            )
        return sub_runner_list

    def get_num_sub_runners(self):
        return len(self.multi_desc.desc_list)

    def display_output_location(self):
        for desc_index, desc in enumerate(self.multi_desc.desc_list):
            full_run_path, _ = desc.run_path(self.replicate, verbose=self.verbose)
            print(
                "\n Output Location for subrunner {} : {} ".format(
                    desc_index, full_run_path
                )
            )

    def run(self):
        if self.verbose and get_platform().is_primary_process:
            print(
                "=" * 82
                + f"\n Multi-Testing with {self.get_num_sub_runners()} sub-runners (Replicate {self.replicate})\n"
                + "-" * 82
            )

        sub_runner_list = self.create_sub_runner()
        for sub_runner_index, sub_runner in enumerate(sub_runner_list):
            print(
                "=" * 82
                + f"\n Testing Runner {sub_runner_index} (Replicate {self.replicate})\n"
                + "-" * 82
            )
            sub_runner.run()
