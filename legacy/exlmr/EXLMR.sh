#!/usr/bin/env bash

#SBATCH --nodes=1                   # Number of nodes to request
#SBATCH --cpus-per-task=32           # Number of CPUs per node to request
#SBATCH --mem=120G                  # Maximum amount of memory this job will be given
#SBATCH --job-name="EXLMR"    # A nice readable name of your job, to see it in the queue, instead of numbers
#SBATCH --output=Ex.out     # Store the output console text to a file called jobName.<assigned job number>.out
#SBATCH --error=Ex.err      # Store the error messages to a file called jobName.<assigned job number>.err
#SBATCH --partition=standby        # Queue normally, no preemption of others
#SBATCH --gres=gpu:a100_80gb:1     # Request 1 A100-80GB (earlier queue slot than plain a100)

source /opt/conda/bin/activate /homes/neumann/teklehaymanot/.conda/envs/Gllm

# Change to the directory where your script is located
python /homes/neumann/teklehaymanot/EXLMR/EXLMR.py
