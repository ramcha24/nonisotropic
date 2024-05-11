import argparse
from dataclasses import dataclass

from cli import shared_args
from foundations.runner import Runner
import models.registry
from platforms.platform import get_platform
from training import train
from training.desc import TrainingDesc


@dataclass
class MultiTrainingRunner(Runner):
    replicate: int
    desc: TrainingDesc
    verbose: bool = True
    evaluate_every_epoch: bool = True
    evaluate_every_few_epoch: int = 10

    @staticmethod
    def description():
        return "Train or finetune multiple models."

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        # shared_args.JobArgs.add_args(parser)
        print(parser.parse_args())
        # store the list of hparams from maybe_get_default_hparams
        # do add_args for each training/testing desc in list(Desc)
        TrainingDesc.add_args(
            parser, shared_args.maybe_get_default_hparams(runner_name="train")
        )
        print(parser.parse_args())

    @staticmethod
    def create_from_args(args: argparse.Namespace) -> "TrainingRunner":
        # create a list of Runners from a list of Desc and then create a overall BenchmarkRunner
        return TrainingRunner(
            args.replicate,
            TrainingDesc.create_from_args(args),
            not args.quiet,
            not args.evaluate_only_at_end,
        )

    def display_output_location(self):
        # here list all the different output locations for each desc in list(desc)
        full_run_path, _ = self.desc.run_path(self.replicate, verbose=self.verbose)
        print("\n Output Location : " + full_run_path)

    def run(self):
        full_run_path, _ = self.desc.run_path(self.replicate)

        if self.verbose and get_platform().is_primary_process:
            print(
                "=" * 82
                + f"\nTraining a Model (Replicate {self.replicate})\n"
                + "-" * 82
            )
            print(self.desc.display)
            print(f"Output Location: {full_run_path}" + "\n" + "-" * 82 + "\n")

        self.desc.save_param(self.replicate)

        # run a for loop over all the individual runners and call their run function.

        train.standard_train(
            models.registry.get(self.desc.model_hparams),
            full_run_path,
            self.desc.dataset_hparams,
            self.desc.training_hparams,
            evaluate_every_epoch=self.evaluate_every_epoch,
            evaluate_every_few_epoch=self.evaluate_every_few_epoch,
        )
