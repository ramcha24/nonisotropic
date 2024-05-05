import abc
from dataclasses import dataclass
import os
import torch

from foundations.hparams import Hparams
import platforms.platform


@dataclass
class Platform(Hparams):
    num_workers: int = 0

    _name: str = "Platform Hyper parameters"
    _description: str = (
        "Hyper parameters that control the platform on which the job is run"
    )
    _num_workers: str = (
        "The number of worker threads to use for data loading (currently just 1)"
    )

    @property
    def device_str(self):

        # GPU device
        if torch.cuda.is_available():
            return "cuda"

        # CPU device
        else:
            print("Not connecting to the GPU!")
            return "cpu"

    @property
    @abc.abstractmethod
    def torch_device(self):
        """
        The torch device to use for computation.
        """
        pass
        # return torch.device(int(os.environ["LOCAL_RANK"]))

    # torch.cuda.current_device()

    @property
    def is_parallel(self):
        """
        Should Pytorch use DataParallel computation.
        """

        return torch.cuda.is_available() and torch.cuda.device_count() > 1

    @property
    @abc.abstractmethod
    def is_distributed(self):
        """
        Should Pytorch use DistributedDataParallel computation.
        """

        pass

    @property
    @abc.abstractmethod
    def local_rank(self):
        """
        Local rank of the process.
        """
        pass  # return 0 # int(os.environ["LOCAL_RANK"])

    @property
    @abc.abstractmethod
    def is_primary_process(self):
        """
        Is the process with rank 0?
        """
        pass

    # return not self.is_distributed or (self.local_rank == 0)

    def barrier(self):
        pass

    # manage the location of files

    @property
    @abc.abstractmethod
    def root(self):
        """The root directory where data will be stored"""
        pass

    @property
    @abc.abstractmethod
    def dataset_root(self):
        """The root directory where datasets will be stored"""
        pass

    @property
    @abc.abstractmethod
    def imagenet_root(self):
        """The directory where imagenet will be stored."""
        pass

    # Mediate access to files

    @staticmethod
    def open(file, mode="r"):
        return open(file, mode)

    @staticmethod
    def exists(file):
        return os.path.exists(file)

    @staticmethod
    def makedirs(path):
        return os.makedirs(path)

    @staticmethod
    def isdir(path):
        return os.path.isdir(path)

    @staticmethod
    def listdir(path):
        return os.listdir(path)

    @staticmethod
    def save_model(model, path, *args, **kwargs):
        return torch.save(model, path, *args, **kwargs)

    @staticmethod
    def load_model(path, *args, **kwargs):
        return torch.load(path, *args, **kwargs)

    # Run jobs. Called by the command line interface
    def run_job(self, f):
        """Run a function that trains a network."""
        old_platform = platforms.platform._PLATFORM
        platforms.platform._PLATFORM = self
        f()
        platforms.platform._PLATFORM = old_platform
