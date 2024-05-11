import torch.nn as nn
import torch.nn.functional as F

from foundations import hparams
from models import base
from models.robustbench_registry import rb_registry, default_rb_registry
from models.utils import load_model
from training.desc import TrainingDesc
from testing.desc import TestingDesc


class Model(base.Model):
    """A wrapper for all pretrained robustbenchmark models for Cifar10 dataset."""

    def __init__(self, model_name, pretrained_model, threat_model):
        super(Model, self).__init__()
        self.model_name = model_name
        self.pretrained_model = pretrained_model
        self.threat_model = threat_model
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.pretrained_model(x)

    @staticmethod
    def is_valid_model_name(model_name, dataset_name, threat_model):
        assert threat_model == "Linf", "Only Linf threat specification is allowed for pretrained models."
        assert dataset_name in ["cifar10", "cifar100", "imagenet"]

        return model_name in rb_registry[dataset_name][threat_model]

    @staticmethod
    def get_model_from_name(
        model_name,
        dataset_name,
        threat_model
        outputs=10,
        initializer=None,
    ):
        assert initializer == None
        assert threat_model == "Linf", "Only Linf threat specification is allowed for pretrained models."
        assert dataset_name in ["cifar10", "cifar100", "imagenet"]

        #threat_model = model_name.split("_")[0]
        # pretrained_model_name = model_name.split("_")[1] 

        pretrained_model = load_model(
            model_name=model_name,
            dataset=dataset_name,
            threat_model=threat_model,
        )
        return Model(model_name, pretrained_model, threat_model)

    @property
    def loss_criterion(self):
        return self.criterion

    @staticmethod
    def default_model_hparams(model_name = None, dataset_name=None, threat_model=None, model_type=None):
        assert threat_model == "Linf", "Only Linf threat specification is allowed for pretrained models."
        assert dataset_name in ["cifar10", "cifar100", "imagenet"]

        return hparams.ModelHparams(
            model_name = default_rb_registry[dataset_name][threat_model] if model_name is None else model_name,
            model_type = model_type,
            model_source = "robustbenchmark"
            threat_model = threat_model
        )
    
    # THREAT MODEL IS SUPERFICIAL. ONLY Linf. 
        
    @staticmethod
    def default_training_hparams(model_name = None, dataset_name=None, threat_model=None, model_type=None):
        if model_type == "pretrained" or  model_type is None:
            return None 
        elif model_type == "finetuned":
            return hparams.TrainingHparams(
            optimizer_name="sgd",
            momentum=0.9,
            milestone_steps="10ep",
            lr=0.01,
            gamma=0.1,
            weight_decay=1e-4,
            training_steps="20ep",
        )
        else:
            raise ValueError("No default training hparams for invalid model_type : {}".format(model_type))