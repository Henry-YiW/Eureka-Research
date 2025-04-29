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
#SBATCH --account=bdpp-delta-gpu    # <- match to a "Project" returned by the "accounts" command
#SBATCH --exclusive  # dedicated node for this job
#SBATCH --requeue
#SBATCH -t 24:00:00
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out

# Set the environment variable
export OPENAI_API_KEY="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"

ENVIRONMENT="allegro_hand"
NUM_ITERATIONS=10
NUM_SAMPLES=6

# Run the Python script
srun python eureka.py env=$ENVIRONMENT iteration=$NUM_ITERATIONS sample=$NUM_SAMPLES