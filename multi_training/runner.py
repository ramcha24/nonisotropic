import argparse
from dataclasses import dataclass

from cli import shared_args
from foundations.runner import Runner
import models.registry
from platforms.platform import get_platform
from training import train
from training.runner import TrainingRunner
from multi_training.desc import MultiTrainingDesc


@dataclass
class MultiTrainingRunner(Runner):
    multi_desc: MultiTrainingDesc
    verbose: bool = True
    evaluate_every_epoch: bool = True
    evaluate_every_few_epoch: int = 10
    multi_train_replicate: int = 1

    @staticmethod
    def description():
        return "Train or finetune multiple models."

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        defaults_list = shared_args.maybe_get_default_hparams(runner_name="multi_train")
        MultiTrainingDesc.add_args(parser, defaults_list)

    @staticmethod
    def create_from_args(args: argparse.Namespace) -> "MultiTrainingRunner":
        multi_desc = MultiTrainingDesc.create_from_args(args)
        verbose = not args.quiet
        evaluate_every_epoch = not args.evaluate_only_at_end
        evaluate_every_few_epoch = (
            args.evaluate_every_few_epoch if args.evaluate_every_few_epoch else 10
        )
        multi_train_replicate = args.train_replicate if args.train_replicate != -1 else 1

        return MultiTrainingRunner(
            multi_desc,
            verbose,
            evaluate_every_epoch,
            evaluate_every_epoch,
            evaluate_every_few_epoch,
            multi_train_replicate,
        )

    def create_sub_runners(self):
        sub_runner_list = []
        for desc in self.multi_desc.desc_list:
            sub_runner_list.append(
                TrainingRunner(
                    desc,
                    self.verbose,
                    self.evaluate_every_epoch,
                    self.evaluate_every_few_epoch,
                    self.multi_train_replicate,
                )
            )
        return sub_runner_list

    def get_num_sub_runners(self):
        return len(self.multi_desc.desc_list)

    def display_output_location(self):
        desc_list = getattr(self.multi_desc, "desc_list")
        for desc_index, desc in enumerate(desc_list):
            logger_paths = desc.run_path(self.multi_train_replicate, verbose=False)
            print(
                "\n Output Location for subrunner {} : {} ".format(
                    desc_index, logger_paths["train_run_path"]
                )
            )

    def run(self):
        if self.verbose and get_platform().is_primary_process:
            print(
                "=" * 82
                + f"\n Multi-Training with {self.get_num_sub_runners()} sub-runners (Replicate {self.multi_train_replicate})\n"
                + "-" * 82
            )

        sub_runner_list = self.create_sub_runners()
        for sub_runner_index, sub_runner in enumerate(sub_runner_list):
            print(
                "=" * 82
                + f"\n Training Runner {sub_runner_index} (Replicate {sub_runner.train_replicate})\n"
                + "-" * 82
            )
            sub_runner.run()
