import argparse
from dataclasses import dataclass

from cli import shared_args
from foundations.runner import Runner
import models.registry
from platforms.platform import get_platform

from testing import test
from testing.desc import TestingDesc


@dataclass
class TestingRunner(Runner):
    replicate: int
    desc: TestingDesc
    verbose: bool = True
    evaluate_batch_only: bool = True
    num_sub_runners: int = 1

    @staticmethod
    def description():
        return "Test a trained model."

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        # shared_args.JobArgs.add_args(parser)
        TestingDesc.add_args(
            parser, shared_args.maybe_get_default_hparams(runner_name="test")
        )

    @staticmethod
    def create_from_args(args: argparse.Namespace) -> "TestingRunner":
        return TestingRunner(
            args.replicate,
            TestingDesc.create_from_args(args),
            not args.quiet,
            args.evaluate_only_batch_test,
        )

    def num_sub_runners(self) -> int:
        return self.num_sub_runners

    def display_output_location(self):
        logger_paths = self.desc.run_path(verbose=self.verbose)
        print("\n Output Location : " + logger_paths["test_path"])

    def run(self):
        logger_paths = self.desc.run_path()

        train_run_path = logger_paths["train_path"]
        full_run_path = logger_paths["test_path"]
        if self.verbose and get_platform().is_primary_process:
            print(
                "=" * 82
                + f"\n Testing a trained Model (Replicate {self.desc.test_replicate})\n"
                + "-" * 82
            )
            print(self.desc.display)
            print(f"Output Location: {full_run_path}/test\n" + "-" * 82 + "\n")

        self.desc.save_param()
        test.standard_test(
            models.registry.get(self.desc.model_hparams),
            train_run_path,
            full_run_path,
            self.desc.dataset_hparams,
            self.desc.testing_hparams,
            verbose=self.verbose,
            evaluate_batch_only=self.evaluate_batch_only,
        )
