import argparse
from dataclasses import dataclass
import torch
import os

from cli import shared_args
from foundations.runner import Runner
import models.registry
from platforms.platform import get_platform

from testing.runner import TestingRunner

from multi_testing.desc import MultiTestingDesc
from utilities.plotting_utils import scatter_plot


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
                self.multi_train_replicate,
                self.multi_test_replicate,
                verbose=self.verbose,
            )
            if self.verbose and get_platform().is_primary_process:
                print(
                    "\n Output Location for subrunner {} : {} ".format(
                        desc_index, logger_paths["test_run_path"]
                    )
                )
        multi_logger_paths = self.multi_desc.run_path(
            self.multi_train_replicate, self.multi_test_replicate, verbose=self.verbose
        )
        if self.verbose and get_platform().is_primary_process:
            print(
                "\n Final output location for multi runner : {}".format(
                    multi_logger_paths["feedback_path"]
                )
            )

    def run(self):
        num_sub_runners = self.get_num_sub_runners()

        if self.verbose and get_platform().is_primary_process:
            print(
                "=" * 82
                + f"\n Multi-Testing with {num_sub_runners} sub-runners (Replicate {self.multi_test_replicate})\n"
                + "-" * 82
            )

        sub_runner_list = self.create_sub_runners()
        desc_list = getattr(self.multi_desc, "desc_list")
        return_dict = {}

        standard_eval = desc_list[0].testing_hparams.standard_eval
        isotropic_robust_eval = desc_list[0].testing_hparams.adv_eval
        nonisotropic_robust_eval = desc_list[0].testing_hparams.N_adv_eval

        standard_accuracy = [] if standard_eval else None
        isotropic_robust_accuracy = [] if isotropic_robust_eval else None
        nonisotropic_robust_accuracy = [] if nonisotropic_robust_eval else None
        output_locations = []
        for sub_runner_index, sub_runner in enumerate(sub_runner_list):
            sub_runner_dict = {}
            if self.verbose and get_platform().is_primary_process:
                print(
                    "=" * 82
                    + f"\n Testing Runner {sub_runner_index}/{num_sub_runners} (Replicate {sub_runner.test_replicate})\n"
                    + "-" * 82
                )
            logger_paths = desc_list[sub_runner_index].run_path(
                self.multi_train_replicate,
                self.multi_test_replicate,
                verbose=False,
            )
            full_run_path, feedback = sub_runner.run()
            if standard_eval:
                standard_accuracy.append(
                    feedback["standard_evaluation_test"]["test_accuracy"]
                )
            if isotropic_robust_eval:
                isotropic_robust_accuracy.append(
                    feedback["isotropic_robust_evaluation_test"]["robust_accuracy"]
                )
            if nonisotropic_robust_eval:
                nonisotropic_robust_accuracy.append(
                    feedback["nonisotropic_robust_evaluation_test"]["robust_accuracy"]
                )
            # return_dict["full_run_path"] = full_run_path
            sub_runner_dict["feedback"] = feedback
            sub_runner_dict["full_run_path"] = full_run_path
            sub_runner_dict["logger_paths"] = logger_paths
            if standard_eval:
                sub_runner_dict["standard_accuracy"] = feedback[
                    "standard_evaluation_test"
                ]["test_accuracy"]
            else:
                sub_runner_dict["standard_accuracy"] = None

            if isotropic_robust_eval:
                sub_runner_dict["isotropic_robust_accuracy"] = feedback[
                    "isotropic_robust_evaluation_test"
                ]["robust_accuracy"]
            else:
                sub_runner_dict["isotropic_robust_accuracy"] = None

            if nonisotropic_robust_eval:
                sub_runner_dict["nonisotropic_robust_accuracy"] = feedback[
                    "nonisotropic_robust_evaluation_test"
                ]["robust_accuracy"]
            else:
                sub_runner_dict["nonisotropic_robust_accuracy"] = None

            return_dict[sub_runner_index] = sub_runner_dict

        multi_logger_paths = self.multi_desc.run_path(
            self.multi_train_replicate, self.multi_test_replicate, verbose=self.verbose
        )
        if self.verbose and get_platform().is_primary_process:
            print("Multi logger paths is : ", multi_logger_paths)
            print(
                "\n Saving return dicts at {}".format(
                    multi_logger_paths["feedback_path"]
                )
            )
            if not get_platform().exists(multi_logger_paths["feedback_path"]):
                get_platform().makedirs(multi_logger_paths["feedback_path"])

            torch.save(
                return_dict,
                os.path.join(
                    multi_logger_paths["feedback_path"], "evaluation_return_dict.pt"
                ),
            )

        if self.verbose and get_platform().is_primary_process:
            print(
                "=" * 82
                + f"\n Result of Test Evaluations (Replicate {sub_runner.test_replicate})\n"
                + "-" * 82
            )

        for sub_runner_index, sub_runner in enumerate(sub_runner_list):
            logger_paths = desc_list[sub_runner_index].run_path(
                self.multi_train_replicate,
                self.multi_test_replicate,
                verbose=False,
            )
            if self.verbose and get_platform().is_primary_process:
                print(
                    "\n Output Location for subrunner {} : {} ".format(
                        sub_runner_index, return_dict[sub_runner_index]["full_run_path"]
                    )
                )
                if standard_eval:
                    print(
                        "\n Standard Accuracy for subrunner {} : {} ".format(
                            sub_runner_index, standard_accuracy[sub_runner_index]
                        )
                    )
                if isotropic_robust_eval:
                    print(
                        "\n Isotropic Robust Accuracy for subrunner {} : {} ".format(
                            sub_runner_index,
                            isotropic_robust_accuracy[sub_runner_index],
                        )
                    )
                if nonisotropic_robust_eval:
                    print(
                        "\n Nonisotropic Robust Accuracy for subrunner {} : {} ".format(
                            sub_runner_index,
                            nonisotropic_robust_accuracy[sub_runner_index],
                        )
                    )

        if standard_eval and nonisotropic_robust_eval:
            scatter_plot(
                nonisotropic_robust_accuracy,
                standard_accuracy,
                multi_logger_paths["feedback_path"],
                "standard_vs_nonisotropic",
                "nonisotropic_robust_accuracy",
                "standard_accuracy",
                "Standard Accuracy vs Non isotropic Robust Accuracy",
            )

        if isotropic_robust_eval and nonisotropic_robust_eval:
            scatter_plot(
                nonisotropic_robust_accuracy,
                isotropic_robust_accuracy,
                multi_logger_paths["feedback_path"],
                "isotropic_vs_nonisotropic",
                "nonisotropic_robust_accuracy",
                "isotropic_accuracy",
                "Isotropic Robust Accuracy vs Non isotropic Robust Accuracy",
            )

        if standard_eval and isotropic_robust_eval:
            scatter_plot(
                isotropic_robust_accuracy,
                standard_accuracy,
                multi_logger_paths["feedback_path"],
                "standard_vs_isotropic",
                "isotropic_robust_accuracy",
                "standard_accuracy",
                "Standard Accuracy vs Isotropic Robust Accuracy",
            )

        # what should I save
        # Each subrunner has feedback, and a full_run_path
        # Each subrunner corresponds to a specific model - either learned with standard/adv/N-adv training or a pretrained model from robustbenchmark
        # feedback saves the output of evaluations from evaluation suite in a nested dictionary format
        # feedback[standard_evaluation_test][test_accuracy] gives the standard accuracy
        # feedback[standard_evaluation_test][test_loss] gives the standard loss

        # feedback[isotropic_robust_evaluation_test][robust_accuracy] gives the isotropic robust accuracy
        # feedback[isotropic_robust_evaluation_test][histogram_attack_statistics] stores the distribution of nonisotropic threat for isotropic lp adversarial corruptions

        # feedback[non_isotropic_robust_evaluation_test][robust_accuracy] gives the nonisotropic robust accuracy
        # feedback[non_isotropic_robust_evaluation_test][histogram_attack_statistics] stores the distribution of nonisotropic threat for isotropic lp adversarial corruptions. This bit is only to sanity check the constraint.

        # multi-runner collects the different feedbacks from subrunners and saves them in a dictionary called return_dict

        # for every
