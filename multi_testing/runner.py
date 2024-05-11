import argparse
from dataclasses import dataclass

from cli import shared_args
from foundations.runner import Runner
import models.registry
from platforms.platform import get_platform

from multi_testing import multi_test
from multi_testing.desc import MultiTestingDesc


@dataclass
class MultiTestingRunner(Runner):
    replicate: int
    multi_desc: MultiTestingDesc
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

    def display_output_location(self):
        for desc in self.multi_desc.desc_list:
            full_run_path, _ = desc.run_path(self.replicate, verbose=self.verbose)
            print("\n Output Location for subrunner : " + full_run_path)
        # print(self.desc.run_path(self.replicate))

    def run(self):
        full_run_path, logger_paths = self.desc.run_path(self.replicate)

        train_run_path = logger_paths["train_run_path"]
        if self.verbose and get_platform().is_primary_process:
            print(
                "=" * 82
                + f"\n Testing a trained Model (Replicate {self.replicate})\n"
                + "-" * 82
            )
            print(self.desc.display)
            print(f"Output Location: {full_run_path}/test\n" + "-" * 82 + "\n")

        self.desc.save_param(self.replicate)
        # self.desc.save(self.desc.run_path(self.replicate))
        test.standard_test(
            models.registry.get(self.desc.model_hparams),
            train_run_path,
            full_run_path,
            self.desc.dataset_hparams,
            self.desc.testing_hparams,
            verbose=self.verbose,
            evaluate_batch_only=self.evaluate_batch_only,
        )
