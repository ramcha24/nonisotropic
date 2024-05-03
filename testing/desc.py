from dataclasses import dataclass, fields
import os
import argparse
import hashlib

from datasets import registry as datasets_registry
from foundations import desc
from foundations import hparams
from foundations import paths
from platforms.platform import get_platform


@dataclass
class TestingDesc(desc.Desc):
    """ the hyperparameters necessary to describe a testing run """

    model_hparams: hparams.ModelHparams
    dataset_hparams: hparams.DatasetHparams
    training_hparams: hparams.TrainingHparams
    testing_hparams: hparams.TestingHparams

    @staticmethod
    def name_prefix(): return 'test'

    def hashname(self, type_str) -> str:
        fields_dict = {f.name: getattr(self, f.name) for f in fields(self)}
        if type_str == 'base':
            hparams_strs = [str(fields_dict[k]) for k in sorted(fields_dict) if (
                        isinstance(fields_dict[k], hparams.DatasetHparams) or isinstance(fields_dict[k],
                                                                                         hparams.ModelHparams))]
            hash_str = hashlib.md5(';'.join(hparams_strs).encode('utf-8')).hexdigest()
            return f'base_{hash_str}'
        elif type_str == 'train':
            hparams_strs = [str(fields_dict[k]) for k in sorted(fields_dict) if
                            isinstance(fields_dict[k], hparams.TrainingHparams)]
            hash_str = hashlib.md5(';'.join(hparams_strs).encode('utf-8')).hexdigest()
            return f'train_{hash_str}'
        elif type_str == 'test':
            hparams_strs = [str(fields_dict[k]) for k in sorted(fields_dict) if
                            isinstance(fields_dict[k], hparams.TestingHparams)]
            hash_str = hashlib.md5(';'.join(hparams_strs).encode('utf-8')).hexdigest()
            return f'test_{hash_str}'
        else:
            raise ValueError('Invalid type string of hparam : {}'.format(type_str))

    @staticmethod
    def add_args(parser: argparse.ArgumentParser, defaults: 'TestingDesc' = None):
        hparams.ModelHparams.add_args(parser, defaults=defaults.model_hparams if defaults else None)
        hparams.DatasetHparams.add_args(parser, defaults=defaults.dataset_hparams if defaults else None)
        hparams.TrainingHparams.add_args(parser, defaults=defaults.training_hparams if defaults else None)
        hparams.TestingHparams.add_args(parser, defaults=defaults.testing_hparams if defaults else None)

    @staticmethod
    def create_from_args(args: argparse.Namespace) -> 'TestingDesc':
        model_hparams = hparams.ModelHparams.create_from_args(args)
        dataset_hparams = hparams.DatasetHparams.create_from_args(args)
        training_hparams = hparams.TrainingHparams.create_from_args(args)
        testing_hparams = hparams.TestingHparams.create_from_args(args)

        return TestingDesc(model_hparams, dataset_hparams, training_hparams, testing_hparams)

    @property
    def test_outputs(self):
        return datasets_registry.num_labels(self.dataset_hparams)

    def train_checkpoint_path(self, replicate):
        base_hash = self.hashname(type_str='base')
        train_hash = self.hashname(type_str='train')
        return os.path.join(get_platform().root, base_hash, train_hash, f'replicate_{replicate}')

    def run_path(self, replicate):
        base_hash = self.hashname(type_str='base')
        train_hash = self.hashname(type_str='train')
        test_hash = self.hashname(type_str='test')
        return os.path.join(get_platform().root, base_hash, train_hash, f'replicate_{replicate}', test_hash)

    @property
    def display(self):
        return '\n'.join([self.dataset_hparams.display, self.model_hparams.display, self.training_hparams.display, self.testing_hparams.display])

    def save_param(self, replicate, type_str):
        if type_str != 'test':
            raise ValueError('Invalid parameter string {} inside Test Runner'.format(type_str))

        base_hash = self.hashname(type_str='base')
        train_hash = self.hashname(type_str='train')
        test_hash = self.hashname(type_str='test')
        train_output_location = os.path.join(get_platform().root, base_hash, train_hash, f'replicate_{replicate}')

        test_output_location = os.path.join(get_platform().root, base_hash, train_hash, f'replicate_{replicate}', test_hash)

        if not get_platform().is_primary_process:
            return
        if not get_platform().exists(train_output_location):
            raise ValueError('The train output location does not exist, please train the model first')
        if type_str == 'test' and not get_platform().exists(test_output_location):
            get_platform().makedirs(test_output_location)

        fields_dict = {f.name: getattr(self, f.name) for f in fields(self)}
        hparams_strs = [fields_dict[k].display for k in sorted(fields_dict) if isinstance(fields_dict[k], hparams.TestingHparams)]
        with get_platform().open(paths.params_loc(test_output_location, type_str), 'w') as fp:
            fp.write('\n'.join(hparams_strs))
