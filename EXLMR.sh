#!/usr/bin/env bash

#SBATCH --nodes=1                   # Number of nodes to request
#SBATCH --cpus-per-task=32           # Number of CPUs per node to request
#SBATCH --mem=120G                  # Maximum amount of memory this job will be given
#SBATCH --job-name="EXLMR"    # A nice readable name of your job, to see it in the queue, instead of numbers
#SBATCH --output=Ex.out     # Store the output console text to a file called jobName.<assigned job number>.out
#SBATCH --error=Ex.err      # Store the error messages to a file called jobName.<assigned job number>.err
#SBATCH --gres=gpu:a100m40:4       # Request 4 A100 GPUs

source /opt/conda/bin/activate /home/teklehaymanot/.conda/envs/Gllm

# Change to the directory where your script is located
python /home/teklehaymanot/EXLMR/EXLMR.py
