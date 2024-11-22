from foundations.runner import Runner
from training.runner import TrainingRunner
from testing.runner import TestingRunner
from multi_testing.runner import MultiTestingRunner
from multi_training.runner import MultiTrainingRunner
from threat_specification.runner import ComputeThreatRunner, EvaluateThreatRunner
from models.pretrained.runner import PretrainRunner

registered_runners = {
    "train": TrainingRunner,
    "test": TestingRunner,
    "multi_test": MultiTestingRunner,
    "multi_train": MultiTrainingRunner,
    "compute_threat": ComputeThreatRunner,
    "download_pretrained": PretrainRunner,
    "evaluate_threat": EvaluateThreatRunner,
}

# To Do: runners
# evaluate_threat_specification


def get(runner_name: str) -> Runner:
    if runner_name not in registered_runners:
        raise ValueError("No such runner : {}".format(runner_name))
    else:
        return registered_runners[runner_name]
