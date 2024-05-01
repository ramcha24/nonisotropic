import argparse
import sys

from cli import runner_registry
from cli import arg_utils
import platforms.registry
from torch.distributed import init_process_group, destroy_process_group
import torch


def main():
    # Welcome message
    welcome = (
        "-" * 100
        + "\n Non-isotropic Robustness : Measuring adversarial robustness against non-isotropic threat specifications\n"
        + "-" * 100
    )

    # choose an initial command
    helptext = welcome + "\n Choose a command to run:"
    for name, runner in runner_registry.registered_runners.items():
        helptext += f"\n * {sys.argv[0]} {name} [...] => {runner.description()}"
    helptext += "\n" + "-" * 82

    runner_name = arg_utils.maybe_get_arg("subcommand", positional=True)
    if runner_name not in runner_registry.registered_runners:
        print(helptext)
        sys.exit(1)

    # Add the arguments for that command.
    usage = "\n" + welcome + "\n"
    usage += f"nonisotropic.py {runner_name} [...] => {runner_registry.get(runner_name).description()}"
    usage += "\n" + "-" * 82 + "\n"

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

    # get the platform arguments
    platform_name = arg_utils.maybe_get_arg("platform") or "local"
    if platform_name and platform_name in platforms.registry.registered_platforms:
        platforms.registry.get(platform_name).add_args(parser)
    else:
        print(f"Invalid platform name: {platform_name}")
        sys.exit(1)

    # Add the arguments for various runners
    runner_registry.get(runner_name).add_args(parser)

    args = parser.parse_args()
    platform = platforms.registry.get(platform_name).create_from_args(args)
    print("platform rank is")
    print(platform.local_rank)
    print(platform.torch_device)
    print("platform global rank is")
    print(platform.global_rank)

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
