# Nonisotropic Adversarial Robustness

There are 5 experiments that are currently written.

## Local vs Distributed Jobs

To run a job on a single gpu, simply call,

```
python nonisotropic.py runner_name --dataset_name=cifar10 ... 
```

Common to every job, is the requirement to specify a dataset name as indicated above. Some runners may require more essential information.

The current framework allows one to run experiments over multiple GPUs (and/or multiple nodes). To run a job on multiple GPUS, (say 4 gpus with device ids - 0,1,2,3)

```
OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc_per_node=4 --rdzv-endpoint=localhost:5001 nonisotropic.py runner_name --dataset_name=cifar10 ....
```

Play with the `OMP_NUM_THREADS` environemnt variable to find the optimal choice for your computing server. The code uses PyTorch's `DistributedDataParallel` functionality to automatically handles distributed computing of gradient computation, loss statistcs etc.

## Runners

In the above, each runner/job/workflow is identified by the _runner\_name_.  There are currently 6 runners identified in `cli/runner_registry.py`. Below I describe each one briefly. For all run jobs one can display the output location to check where hyperparameter configurations, model weights and any runner output will be stored.

```
python nonisotropic.py runner_name --dataset_name=cifar10 --display_output_location
```

### Threat Specification Runner

To compute and store a nonisotropic threat specification for a given dataset,

```
python nonisotropic.py compute_threat --dataset_name=cifar10 --threat_replicate=1
```

Currently the default (and only) algorithm to compute threat specification is `greedy_subsets`.

### Training Runner

To train a model, specify a dataset, a model name and any augmentation/training options/hyperparameters

```
python nonisotropic.py train --dataset_name=cifar10 --model_name=cifar10_resnet_50 
```

The `train_replicate` indexes multiple training runs with the same hyper-parameter configuration. One can specify augmentation or adversarial training options as follows,  

```
python nonisotropic.py train --dataset_name=cifar10 --model_name=cifar10_resnet_50 --N_aug
python nonisotropic.py train --dataset_name=cifar10 --model_name=cifar10_resnet_50 --N_aug --adv_train
```

To see the list of options, consult the dataclasses `AugmentationHparams` and `TrainingHparams` in `foundations/hparams.py`.

### Multi-Training Runner

To enable multiple training runs with a variety of options,

```
python nonisotropic.py multi_train --dataset_name=cifar10 --model_name=cifar10_resnet_50 --toggle_N_aug --toggle_adv_train --toggle_N_adv_train
```

Here the command line flag `toggle_N_aug` indicates that two jobs should be run both with and without N_aug. In the above 3 boolean options are presented, thus this job sets up 8 possible combinations and runs them sequentially. To check the possible toggle options, see `ToggleHparams` in `cli/shared_args.py`

### Pretrained Models

The run jobs facilitate downloading, fine-tuning and testing pretrained robust models maintained by [robust benchmark](https://github.com/RobustBench/robustbench). Currently the list of models accounted for are identified in `models/robustbench_registry.py`. The only threat model considered currently is "Linf".

#### Downloading

To download pretrained models,

```
python nonisotropic.py download_pretrained --dataset_name=cifar10 
```

#### Finetuning

To finetune pretrained models, one can use `multi_train` runner,

```
python nonisotropic.py multi_train --dataset_name=cifar10 --model_type=pretrained --toggle_N_aug --toggle_mixup --toggle_N_adv --toggle_adv_train
```

### Testing Runner

To perform evaluate trained models,

```

python nonisotropic.py test --dataset_name=cifar10 --model_name=cifar10_resnet_50 --standard_eval

```

One can also specify the options `--adv_eval` for an isotropic robustness evaluation and `N_adv_eval` for nonisotropic robustness evaluation.

### Multi-Testing Runner

To test multiple models identified by a model_type ("pretrained/finetuned") or a single model_name

```

python nonisotropic.py multi_test --dataset_name=cifar10 --model_name=cifar10_resnet_50 --toggle_N_aug --toggle_N_adv_train --standard_eval
```

### Threat specification functionality

```

threat_model := ProjectedDisplacement(threat_hparams(num_chunks=True), weighted=True, segmented=False)

threat_model.prepare(num_devices=5)

threat_model.evaluate(examples, labels, perturbed_examples)

threat_model.project(examples, labels, perturbed_examples)
```
