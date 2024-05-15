import abc
import argparse
from dataclasses import dataclass

from cli import shared_args
from platforms.platform import get_platform

from foundations.runner import Runner
from foundations import paths

from models.pretrained.desc import PretrainDesc
from models.utils import load_model
from models.robustbench_registry import rb_registry


@dataclass
class PretrainRunner(Runner):
    desc: PretrainDesc
    verbose: bool = True

    @staticmethod
    def description() -> str:
        return "Download pretrained models."

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        PretrainDesc.add_args(
            parser,
            shared_args.maybe_get_default_hparams(runner_name="download_pretrained"),
        )

    @staticmethod
    def create_from_args(args: argparse.Namespace) -> "Runner":
        desc = PretrainDesc.create_from_args(args)
        verbose = not args.quiet
        return PretrainRunner(desc, verbose)

    def display_output_location(self) -> None:
        logger_paths = self.desc.run_path(verbose=self.verbose)
        print("\n Output Location : " + logger_paths["pretrain_run_path"])

    def run(self) -> None:
        logger_paths = self.desc.run_path(verbose=self.verbose)
        pretrain_run_path = logger_paths["pretrain_run_path"]
        dataset_name = self.desc.dataset_hparams.dataset_name

        if self.verbose and get_platform().is_primary_process:
            print(
                "=" * 82
                + f"\nDownloading Pretrained models from robustbenchmark for dataset : {dataset_name}\n"
                + "=" * 82
            )
            print(self.desc.display)
            print("\n Output Location : " + pretrain_run_path + "\n" + "-" * 82 + "\n")

        threat_model = "Linf"  # default threat model
        model_names = rb_registry[dataset_name][threat_model]
        for model_name in model_names:
            if self.verbose and get_platform().is_primary_process:
                print("Downloading model : ", model_name)
            model = load_model(
                model_name=model_name,
                model_dir=pretrain_run_path,
                dataset=dataset_name,
                threat_model=threat_model,
            )
