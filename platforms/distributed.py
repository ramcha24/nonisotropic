import os
import pathlib

from platforms import base
import torch


class Platform(base.Platform):
    @property
    def torch_device(self):
        return torch.device(int(os.environ["LOCAL_RANK"]))

    @property
    def is_distributed(self):
        return True

    @property
    def local_rank(self):
        return int(os.environ["LOCAL_RANK"])

    @property
    def is_primary_process(self):
        return self.local_rank == 0

    @property
    def runner_root(self):
        return os.path.join(pathlib.Path.home(), "nonisotropic/runner_data")

    @property
    def multi_runner_root(self):
        return os.path.join(pathlib.Path.home(), "nonisotropic/multi_runner_data")

    @property
    def dataset_root(self):
        return os.path.join(pathlib.Path.home(), "nonisotropic/datasets")

    @property
    def model_root(self):
        return os.path.join(pathlib.Path.home(), "nonisotropic/models")

    @property
    def threat_specification_root(self):
        return os.path.join(pathlib.Path.home(), "nonisotropic/threat_specification")
