import argparse
import sys
import os

from cli import runner_registry, arg_utils
from cli.shared_args import JobArgs, ToggleArgs, MultiTestArgs

import platforms.registry
from torch.distributed import init_process_group, destroy_process_group
import torch


from dataclasses import dataclass, fields, MISSING


def main():
    # Welcome message
    welcome = (
        "-" * 120
        + "\n Non-isotropic Robustness : Measuring adversarial robustness against non-isotropic threat specifications\n"
        + "-" * 120
    )

    # choose an initial command
    helptext = welcome + "\n Choose a command to run:"
    for name, runner in runner_registry.registered_runners.items():
        helptext += f"\n * nonisotropic.py {name} [...] => {runner.description()}"
    helptext += "\n" + "-" * 82

    runner_name = arg_utils.maybe_get_arg("subcommand", positional=True)
    if runner_name not in runner_registry.registered_runners:
        print(helptext)
        sys.exit(1)

    # Add the arguments for that command.
    usage = "\n" + welcome + "\n"
    usage += f"nonisotropic.py {runner_name} [...] => {runner_registry.get(runner_name).description()}"
    usage += "\n" + "-" * 120 + "\n"

    parser = argparse.ArgumentParser(usage=usage, conflict_handler="resolve")
    parser.add_argument("subcommand")
    parser.add_argument(
        "--platform", default="local", help="The platform on which the job will run."
    )
    parser.add_argument(
        "--display_output_location",
        action="store_true",
        help="Display the output location for this job.",
    )

    rank_flag = "LOCAL_RANK" in os.environ
    default_platform_name = "local" if not rank_flag else "distributed"

    platform_name = arg_utils.maybe_get_arg("platform") or default_platform_name

    if "OMP_NUM_THREADS" not in os.environ:
        os.environ["OMP_NUM_THREADS"] = "48"  # default value

    if platform_name == "distributed" and not rank_flag:
        print(
            "\nLocal rank environment variable has not been set, For a distributed job, try \n'torchrun --nnodes=? --nproc_per_node=? --rdzv-endpoint=localhost:???? nonisotropic.py ...'\n"
        )
        sys.exit(1)
    if platform_name == "local" and rank_flag:
        print(
            "\nLocal rank environment variable indicates a distributed job but the chosen platform is local. Try setting --platform distributed or ignore --platform argument when running with torchrun launcher\n"
        )
        sys.exit(1)
        
    if platform_name and platform_name in platforms.registry.registered_platforms:
        platforms.registry.get(platform_name).add_args(parser)
    else:
        print(f"Invalid platform name: {platform_name}")
        sys.exit(1)

    # Add the arguments for various runners
    JobArgs.add_args(parser)

    if runner_name.startswith("multi"):
        ToggleArgs.add_args(parser)

    if runner_name == "multi_test":
        MultiTestArgs.add_args(parser)

    runner_registry.get(runner_name).add_args(parser)

    args = parser.parse_args()
    platform = platforms.registry.get(platform_name).create_from_args(args)

    if args.display_output_location:
        platform.run_job(
            runner_registry.get(runner_name)
            .create_from_args(args)
            .display_output_location
        )
        sys.exit(0)

    if platform.is_distributed:
        init_process_group(backend="nccl")

    platform.run_job(runner_registry.get(runner_name).create_from_args(args).run)

    if platform.is_distributed:
        destroy_process_group()


if __name__ == "__main__":
    main()
