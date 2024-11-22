#!/bin/bash
# incomplete script 

# Define the path to the Python script and the log file
main_script="../nonisotropic.py"
runner_name="multi_train"
dataset_name_1=("cifar10" "cifar100")
dataset_name_2=("imagenet")
model_name_1=("cifar10_resnet_50" "cifar100_resnet_50")
model_name_2=("imagenet_resnet_50")
runner_options="toggle_adv_train \
                toggle_N_adv_train \
                train_replicate=777 \
                display_output_location"
log_file="output_train.log"

# Define the torchrun options
torchrun_opt="--nnodes=1 --nproc_per_node=4 --rdzv-endpoint=localhost:5001"

# Define options for the training run
experiment_opt="multi_train \
        --dataset_name=cifar10 \
        --model_name=cifar10_resnet_50 \
        --toggle_adv_train \
        --toggle_N_adv_train \
        --train_replicate=12 \
        --display_output_location"  

# Define the options for parallel and sequential executions
PARALLEL_OPTIONS=(
    "--dataset_name=cifar10 --model_name=cifar10_resnet_50 --toggle_adv_train"
    "--dataset_name=cifar100 --model_name=cifar100_resnet_50 --toggle_N_adv_train"
)

SEQUENTIAL_OPTIONS=(
    "--dataset_name=imagenet --model_name=imagenet_resnet_50 --train_replicate=12"
    "--dataset_name=mnist --model_name=mnist_resnet_18 --display_output_location"
)

OMP_NUM_THREADS=64 CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun $torchrun_opt -u $main_script $experiment_opt 2>&1 | tee  $log_file

# Optional: Print a message indicating the output was saved
echo "Execution complete. Output saved to $log_file."        
