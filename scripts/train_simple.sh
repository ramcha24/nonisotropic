#!/bin/bash

# Define the path to the Python script and the log file
main_script="../nonisotropic.py"
log_file="output_train.log"

# Define the torchrun options
torchrun_opt="--nnodes=1 --nproc_per_node=8 --rdzv-endpoint=localhost:5001"


# Define options for the training run
experiment_opt="train \
        --dataset_name=cifar10 \
        --model_name=cifar10_resnet_50 \
        --train_replicate=50"
#  "  
# \--display_output_location"
#       --N_adv_train \
#         --use_amp \
#        --num_chunks=16 \
#        --float_16 \


PYTHONUNBUFFERED=1 OMP_NUM_THREADS=64 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun $torchrun_opt $main_script $experiment_opt 2>&1 | tee  $log_file

# echo "Trying."        
#python -u $main_script $experiment_opt 2>&1 | tee  $log_file

# Optional: Print a message indicating the output was saved
echo "Execution complete. Output saved to $log_file."        