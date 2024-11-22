#!/bin/bash

# Define the path to the Python script and the log file
main_script="../nonisotropic.py"
log_file="output_evaluate_threat.log"

# Define the options you want to pass to the Python script
options="evaluate_threat \
        --dataset_name=imagenet \
        --batch_size=96 \
        --float_16 \
        --in_memory \
        --v2_transformations \
        --common_2d \
        --common_2d_bar
        --num_chunks=25"  

# Execute the Python script with options and save output to the log file
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u $main_script $options 2>&1 | tee $log_file

# Optional: Print a message indicating the output was saved
echo "Execution complete. Output saved to $log_file."         