#!/bin/bash

# SBATCH options (adjust based on your needs)
#SBATCH --job-name=array_job             # Job name
#SBATCH --output=output_%A_%a.log        # Output file name (%A for job ID, %a for array index)
#SBATCH --time=01:00:00                  # Time limit (adjust as needed)
#SBATCH --partition=standard             # Partition (adjust as needed)
#SBATCH --array=0-47                     # Job array, 48 combinations (8 datasets * 6 sets)

# Define the dataset and set arrays
datasets=("CrossNews_Article.csv" "CrossNews_Tweet.csv" "c" "d" "e" "f" "g" "h")  # 8 values
sets=("1" "2" "3" "4" "5" "6")               # 6 values

# Total number of datasets and sets
num_datasets=${#datasets[@]}
num_sets=${#sets[@]}

# Calculate dataset and set indices based on SLURM_ARRAY_TASK_ID
dataset_index=$(( SLURM_ARRAY_TASK_ID / num_sets ))
set_index=$(( SLURM_ARRAY_TASK_ID % num_sets ))

# Get the actual dataset and set values
dataset=${datasets[$dataset_index]}
set=${sets[$set_index]}

# Echo the variables to verify
echo "Running job for dataset: $dataset, set: $set"

# Your main command here, using $dataset and $set
# Example:
# python my_script.py --dataset $dataset --set $set
