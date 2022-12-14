#!/bin/bash
#SBATCH --job-name=openwebtext-jsonl-to-meg-t5   # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=100:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name
#SBATCH --account=six@cpu
#SBATCH --partition=cpu_p1

set -x -e

source $six_ALL_CCFRWORK/start-prod

cd $six_ALL_CCFRWORK/code/megatron-lm
python tools/preprocess_data.py \
       --input $six_ALL_CCFRWORK/datasets-custom/openwebtext/openwebtext.jsonl \
       --output-prefix $six_ALL_CCFRWORK/datasets-custom/openwebtext/meg-t5 \
       --vocab $six_ALL_CCFRWORK/datasets-custom/vocabs/bert-large-uncased-vocab.txt \
       --dataset-impl mmap \
       --tokenizer-type BertWordPieceLowerCase \
       --split-sentences \
       --workers 8
