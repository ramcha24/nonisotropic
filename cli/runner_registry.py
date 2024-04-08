from foundations.runner import Runner
from training.runner import TrainingRunner
from testing.runner import TestingRunner

registered_runners = {"train": TrainingRunner, "test": TestingRunner}


def get(runner_name: str) -> Runner:
    if runner_name not in registered_runners:
        raise ValueError("No such runner : {}".format(runner_name))
    else:
        return registered_runners[runner_name]
