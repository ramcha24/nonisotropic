import argparse
from dataclasses import dataclass
from typing import Optional

from cli import shared_args
from foundations.runner import Runner
import models.registry
from platforms.platform import get_platform

from testing import test
from testing.desc import TestingDesc


@dataclass
class TestingRunner(Runner):
    desc: TestingDesc
    test_replicate: int = 1
    verbose: bool = True
    evaluate_batch_only: bool = True
    train_replicate: Optional[int] = None

    @staticmethod
    def description():
        return "Test a trained model."

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        TestingDesc.add_args(
            parser, shared_args.maybe_get_default_hparams(runner_name="test")
        )

    @staticmethod
    def create_from_args(args: argparse.Namespace) -> "TestingRunner":
        desc = TestingDesc.create_from_args(args)

        model_type = desc.model_hparams.model_type
        # infer train replicate number if not provided
        if args.train_replicate == -1:
            if model_type is None:
                train_replicate = 1
            elif model_type == "pretrained":
                train_replicate = None  # No training replicate for pretrained models
            elif model_type == "finetuned":
                train_replicate = 1
            else:
                raise ValueError(
                    f"Cannot infer train replicate number for unknown model type {model_type}"
                )
        else:
            train_replicate = args.train_replicate

        test_replicate = args.test_replicate if args.test_replicate != -1 else 1
        verbose = not args.quiet
        evaluate_batch_only = args.evaluate_only_batch_test

        return TestingRunner(
            desc,
            test_replicate,
            verbose,
            evaluate_batch_only,
            train_replicate,
        )

    def display_output_location(self):
        logger_paths = self.desc.run_path(
            self.train_replicate, self.test_replicate, verbose=self.verbose
        )
        if self.verbose and get_platform().is_primary_process:
            print("\n Output Location : " + logger_paths["test_run_path"])

    def run(self):
        logger_paths = self.desc.run_path(self.train_replicate, self.test_replicate)

        train_run_path = logger_paths["train_run_path"]
        full_run_path = logger_paths["test_run_path"]
        if self.verbose and get_platform().is_primary_process:

            print(
                "=" * 82
                + f"\n (Replicate {self.test_replicate}) : Testing model at {train_run_path}\n"
                + "-" * 82
            )
            print(self.desc.display)
            print(f"Output Location: {full_run_path}/test\n" + "-" * 82 + "\n")

        self.desc.save_param(self.train_replicate, self.test_replicate)
        feedback = test.standard_test(
            models.registry.get(
                self.desc.dataset_hparams.dataset_name, self.desc.model_hparams
            ),
            train_run_path,
            full_run_path,
            self.desc.dataset_hparams,
            self.desc.testing_hparams,
            model_type=self.desc.model_hparams.model_type,
            verbose=self.verbose,
            evaluate_batch_only=self.evaluate_batch_only,
        )
        
        return full_run_path, feedback
