import abc
import argparse


class Runner(abc.ABC):
    """An instance of a training run of some kind."""

    @staticmethod
    @abc.abstractmethod
    def description() -> str:
        """A description of this runner."""

        pass

    @staticmethod
    @abc.abstractmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        """Add all command line flags necessary for this runner."""
        pass

    @staticmethod
    @abc.abstractmethod
    def create_from_args(args: argparse.Namespace) -> "Runner":
        """Create a runner for command line arguments."""

        pass

    # @staticmethod
    @abc.abstractmethod
    def display_output_location(self) -> None:
        """Print the output location for the job."""

        pass

    @abc.abstractmethod
    def num_sub_runners(self) -> int:
        """Print the number of sub-runners if this class represents a collection of runners"""
        pass

    # @staticmethod
    @abc.abstractmethod
    def run(self) -> None:
        """Run the job."""

        pass
