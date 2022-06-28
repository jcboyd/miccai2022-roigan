#!/bin/bash
#SBATCH --job-name=stain_transfer
#SBATCH --output=./outputs/%x_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --mem=120000

# Activate anaconda environment code
source activate $WORKDIR/miniconda3/envs/pytorch

python main.py ${SLURM_JOBID} ./config/config.yml
