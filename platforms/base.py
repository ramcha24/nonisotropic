import abc
from dataclasses import dataclass
import os
import torch

from foundations.hparams import Hparams
import platforms.platform


@dataclass
class Platform(Hparams):
    num_workers: int = 0
    
    _name: str = 'Platform Hyper parameters'
    _description: str = 'Hyper parameters that control the platform on which the job is run'
    _num_workers: str = 'The number of worker threads to use for data loading (currently just 1)'

    @property
    def device_str(self):

        # GPU device
        if torch.cuda.is_available():
            return 'cuda'
        
        # CPU device
        else:
            print("Not connecting to the GPU!")
            return 'cpu'
    
    @property
    def torch_device(self):
        return torch.device(self.device_str)
    
    @property
    def is_parallel(self):
        # currently this is always false
        return torch.cuda.is_available() and torch.cuda.device_count() > 1
    
    @property
    def rank(self):
        return 0
    
    @property
    def world_size(self):
        return 1
    
    @property 
    def is_primary_process(self):
        return not False or (self.rank == 0)  # so just true?
    
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
    
    # Mediate access to files
    
    @staticmethod 
    def open(file, mode='r'):
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
        return os.path.listdir(path)
    
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
