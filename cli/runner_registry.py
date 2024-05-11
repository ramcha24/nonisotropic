from foundations.runner import Runner
from training.runner import TrainingRunner
from testing.runner import TestingRunner
from multi_testing.runner import MultiTestingRunner
from multi_training.runner import MultiTrainingRunner

registered_runners = {
    "train": TrainingRunner,
    "test": TestingRunner,
    "multi_test": MultiTestingRunner,
    "multi_train": MultiTrainingRunner,
    # load_pretrained,
    # compute_threat_specification,
    # evaluate_threat_specification
}

# {"benchmark": BenchmarkRunner, "compute_threat_specification": TSRunner, "evaluate_threat_specification": EvalTSRunner,
# "finetune": FinetuningRunner, "multi-train": MultiTrainingRunner}
# multi-train : specifies a list of training jobs. do this last.
# finetuning runner contains a list of training runners
# benchmark contains a list of testing runners
# benchmark can have default hparams - pretrained, finetuned or multi-trained
# manual refers to the trained combos that I will run.
# evaluate refers to the
# i need two more runners - one that computes threats specification and one that downloads pretrained models


def get(runner_name: str) -> Runner:
    if runner_name not in registered_runners:
        raise ValueError("No such runner : {}".format(runner_name))
    else:
        return registered_runners[runner_name]
