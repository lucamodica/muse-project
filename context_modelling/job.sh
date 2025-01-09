#!/bin/env bash

# The necessary flags to send to sbatch can either be included when calling
# sbatch or more conveniently it can be specified in the jobscript as follows

#SBATCH -A NAISS2024-22-578     # find your project with the "projinfo" command
#SBATCH -p alvis                # what partition to use (usually not necessary)
#SBATCH -t 0-05:00:00           # how long time it will take to run
#SBATCH --gpus-per-node=A100:1   # choosing no. GPUs and their type
#SBATCH -J NLP                  # the jobname (not necessary)

ml purge
ml load virtualenv/20.23.1-GCCcore-12.3.0
ml load torchvision/0.16.0-foss-2023a-CUDA-12.1.1
ml load librosa/0.10.1-foss-2023a
ml load tqdm/4.66.1-GCCcore-12.3.0
ml load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
ml load Transformers/4.39.3-gfbf-2023a
ml load matplotlib/3.7.2-gfbf-2023a
ml load Seaborn/0.13.2-gfbf-2023a

source ../my_venv/bin/activate

python main.py --lr 0.0001 --epochs 50