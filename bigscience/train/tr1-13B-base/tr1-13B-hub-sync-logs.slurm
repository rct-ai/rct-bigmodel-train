#!/bin/bash
#SBATCH --job-name=tr1-13B-hub-sync-logs  # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --nodes=1                    # number of nodes
#SBATCH --cpus-per-task=1            # number of cores per task
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=1:00:00               # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name
#SBATCH --partition=compil
#SBATCH --account=six@cpu

echo "START TIME: $(date)"

source $six_ALL_CCFRWORK/code/tr1-13B/bigscience/train/tr1-13B-base/start-tr1-13B
module load git-lfs

DATA_OUTPUT_PATH=$six_ALL_CCFRSCRATCH/checkpoints/tr1-13B
LOGS_PATH=$DATA_OUTPUT_PATH/logs
BIG_SCIENCE_REPO_PATH=$six_ALL_CCFRWORK/code/tr1-13B/bigscience

$BIG_SCIENCE_REPO_PATH/tools/hub-sync.py --repo-path $LOGS_PATH --patterns '*.txt' -d

echo "END TIME: $(date)"
