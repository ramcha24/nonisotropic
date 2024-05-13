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
    train_replicate: int = 1

    @staticmethod
    def description():
        return "Train a model."

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        TrainingDesc.add_args(
            parser, shared_args.maybe_get_default_hparams(runner_name="train")
        )

    @staticmethod
    def create_from_args(args: argparse.Namespace) -> "TrainingRunner":
        desc = TrainingDesc.create_from_args(args)

        model_type = desc.model_hparams.model_type
        assert model_type in ["pretrained", None]

        # infer train replicate number if not provided
        train_replicate = args.train_replicate if args.train_replicate != -1 else 1
        verbose = not args.quiet
        evaluate_every_epoch = not args.evaluate_only_at_end
        evaluate_every_few_epoch = (
            args.evaluate_every_few_epoch if args.evaluate_every_few_epoch else 10
        )

        return TrainingRunner(
            desc,
            verbose,
            evaluate_every_epoch,
            evaluate_every_few_epoch,
            train_replicate,
        )

    def display_output_location(self):
        logger_paths = self.desc.run_path(self.train_replicate, verbose=self.verbose)
        print("\n Output Location : " + logger_paths["train_run_path"])

    def run(self):
        logger_paths = self.desc.run_path(self.train_replicate)
        full_run_path = logger_paths["train_run_path"]

        if self.verbose and get_platform().is_primary_process:
            print(
                "=" * 82
                + f"\nTraining a Model (Replicate {self.train_replicate})\n"
                + "-" * 82
            )
            print(self.desc.display)
            print(f"Output Location: {full_run_path}" + "\n" + "-" * 82 + "\n")

        self.desc.save_param(self.train_replicate)

        train.standard_train(
            full_run_path,
            self.desc,
            evaluate_every_epoch=self.evaluate_every_epoch,
            evaluate_every_few_epoch=self.evaluate_every_few_epoch,
        )
