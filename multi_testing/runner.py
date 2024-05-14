import argparse
from dataclasses import dataclass

from cli import shared_args
from foundations.runner import Runner
import models.registry
from platforms.platform import get_platform

from testing.runner import TestingRunner

from multi_testing.desc import MultiTestingDesc


@dataclass
class MultiTestingRunner(Runner):
    multi_desc: MultiTestingDesc
    multi_test_replicate: int = 1
    verbose: bool = True
    evaluate_batch_only: bool = True
    multi_train_replicate: int = 1

    @staticmethod
    def description():
        return "Test multiple models."

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        defaults_list = shared_args.maybe_get_default_hparams(runner_name="multi_test")
        MultiTestingDesc.add_args(parser, defaults_list)

    @staticmethod
    def create_from_args(args: argparse.Namespace) -> "MultiTestingRunner":
        multi_desc = MultiTestingDesc.create_from_args(args)
        multi_test_replicate = args.test_replicate if args.test_replicate != -1 else 1
        verbose = not args.quiet
        evaluate_batch_only = args.evaluate_only_batch_test
        multi_train_replicate = (
            args.train_replicate if args.train_replicate != -1 else 1
        )
        # infer train replicate number if not provided
        # print(multi_desc.desc_list)
        desc_list = getattr(multi_desc, "desc_list")

        model_type = desc_list[
            0
        ].model_hparams.model_type  # assuming homogenous model type
        if args.train_replicate == -1:
            if model_type is None:
                multi_train_replicate = 1
            elif model_type == "pretrained":
                multi_train_replicate = (
                    None  # No training replicate for pretrained models
                )
            elif model_type == "finetuned":
                multi_train_replicate = 1
            else:
                raise ValueError(
                    f"Cannot infer common train replicate number for unknown model type {model_type}"
                )
        else:
            multi_train_replicate = args.train_replicate

        return MultiTestingRunner(
            multi_desc,
            multi_test_replicate,
            verbose,
            evaluate_batch_only,
            multi_train_replicate,
        )

    def create_sub_runners(self):
        sub_runner_list = []
        desc_list = getattr(self.multi_desc, "desc_list")
        for desc in desc_list:
            sub_runner_list.append(
                TestingRunner(
                    desc,
                    self.multi_test_replicate,
                    self.verbose,
                    self.evaluate_batch_only,
                    self.multi_train_replicate,
                )
            )
        return sub_runner_list

    def get_num_sub_runners(self):
        desc_list = getattr(self.multi_desc, "desc_list")
        return len(desc_list)

    def display_output_location(self):
        desc_list = getattr(self.multi_desc, "desc_list")
        for desc_index, desc in enumerate(desc_list):
            logger_paths = desc.run_path(
                self.multi_train_replicate, self.multi_test_replicate, verbose=False
            )
            print(
                "\n Output Location for subrunner {} : {} ".format(
                    desc_index, logger_paths["test_run_path"]
                )
            )

    def run(self):
        if self.verbose and get_platform().is_primary_process:
            print(
                "=" * 82
                + f"\n Multi-Testing with {self.get_num_sub_runners()} sub-runners (Replicate {self.multi_test_replicate})\n"
                + "-" * 82
            )

        sub_runner_list = self.create_sub_runners()
        for sub_runner_index, sub_runner in enumerate(sub_runner_list):
            print(
                "=" * 82
                + f"\n Testing Runner {sub_runner_index} (Replicate {sub_runner.test_replicate})\n"
                + "-" * 82
            )
            sub_runner.run()
