import os
import pathlib

from platforms import base


class Platform(base.Platform):
    @property
    def root(self):
        return os.path.join(pathlib.Path.home(), "nonisotropic/runner_data")

    @property
    def dataset_root(self):
        return os.path.join(pathlib.Path.home(), "nonisotropic/datasets")

    @property
    def imagenet_root(self):
        return os.path.join(pathlib.Path.home(), "nonisotropic/datasets/imagenet")
