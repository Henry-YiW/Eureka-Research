#!/bin/bash
#SBATCH --job-name="a.out_symmetric"
#SBATCH --output="a.out.%j.%N.out"
#SBATCH --partition=gpuA100x4
#SBATCH --mem=60G
#SBATCH --ntasks-per-node=1  # could be 1 for py-torch
#SBATCH --cpus-per-task=16   # spread out to use 1 core per numa, set to 64 if tasks is 1
#SBATCH --constraint="projects"
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest   # select a cpu close to gpu on pci bus topology
#SBATCH --account=bdes-delta-gpu    # <- match to a "Project" returned by the "accounts" command
#SBATCH --exclusive  # dedicated node for this job
#SBATCH --requeue
#SBATCH -t 24:00:00
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out

export LD_LIBRARY_PATH=/home/jensenyuan/miniconda3/envs/eureka/lib:$LD_LIBRARY_PATH
# Configuration
TASK="ShadowHandRope"
HEADLESS="False"  # Set to "False" for visualization
FORCE_RENDER="True"
NUM_ENVS=16     # Number of parallel environments
MAX_ITERATIONS=3000  # Total training iterations



# Execute training
python train.py \
    task=$TASK \
    headless=$HEADLESS \
    force_render=$FORCE_RENDER \
    num_envs=$NUM_ENVS \
    max_iterations=$MAX_ITERATIONS
