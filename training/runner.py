import argparse
from dataclasses import dataclass

from cli import shared_args
from foundations.runner import Runner
import models.registry
from platforms.platform import get_platform
from training import train
from training.desc import TrainingDesc


@dataclass
class TrainingRunner(Runner):
    desc: TrainingDesc
    verbose: bool = True
    evaluate_every_epoch: bool = True
    evaluate_every_few_epoch: int = 10
    num_sub_runners: int = 1

    @staticmethod
    def description():
        return "Train a model."

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        print("Inside training runner")
        TrainingDesc.add_args(
            parser, shared_args.maybe_get_default_hparams(runner_name="train")
        )
        print(parser.parse_args())

    @staticmethod
    def create_from_args(args: argparse.Namespace) -> "TrainingRunner":
        return TrainingRunner(
            args.replicate,
            TrainingDesc.create_from_args(args),
            not args.quiet,
            not args.evaluate_only_at_end,
        )

    def num_sub_runners(self) -> int:
        return self.num_sub_runners

    def display_output_location(self):
        logger_paths = self.desc.run_path(verbose=self.verbose)
        print("\n Output Location : " + logger_paths["train_path"])

    def run(self):
        logger_paths = self.desc.run_path()
        full_run_path = logger_paths["train_path"]

        if self.verbose and get_platform().is_primary_process:
            print(
                "=" * 82
                + f"\nTraining a Model (Replicate {self.desc.training_hparams.train_replicate})\n"
                + "-" * 82
            )
            print(self.desc.display)
            print(f"Output Location: {full_run_path}" + "\n" + "-" * 82 + "\n")

        self.desc.save_param()

        train.standard_train(
            full_run_path,
            self.desc,
            evaluate_every_epoch=self.evaluate_every_epoch,
            evaluate_every_few_epoch=self.evaluate_every_few_epoch,
        )
