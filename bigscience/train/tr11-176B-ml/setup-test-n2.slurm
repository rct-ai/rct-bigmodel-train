#!/bin/bash
#SBATCH --job-name=setup-test-n2
#SBATCH --partition=gpu_p5
#SBATCH --constraint=a100
#SBATCH --reservation=hug
#SBATCH --qos=qos_gpu-gc             # up to 100h
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=64           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --gres=gpu:8                 # number of gpus
#SBATCH --time 0:30:00               # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name
#SBATCH --account=six@a100

set -x -e

#source $six_ALL_CCFRWORK/start-py38-pt110
#source $six_ALL_CCFRWORK/start-py38-pt111
source $six_ALL_CCFRWORK/code/tr11-176B-ml/bigscience/train/tr11-176B-ml/start-tr11-176B-ml

echo "START TIME: $(date)"

variant=main-test-setup

DATA_OUTPUT_PATH=$six_ALL_CCFRSCRATCH/checkpoints/tr11-176B-ml-test-setup
CHECKPOINT_PATH=$DATA_OUTPUT_PATH/checkpoints/$variant
REPO_PATH=$DATA_OUTPUT_PATH/tr11-176B-ml-logs
TENSORBOARD_PATH=$REPO_PATH/tensorboard/$variant
LOGS_PATH=$REPO_PATH/logs/$variant
mkdir -p $LOGS_PATH

MEGATRON_DEEPSPEED_REPO=$six_ALL_CCFRWORK/code/tr11-176B-ml/Megatron-DeepSpeed
cd $MEGATRON_DEEPSPEED_REPO

KILL_SWITCH_PATH=$MEGATRON_DEEPSPEED_REPO/kill-switch-tr11-setup-test-n2

BIGSCIENCE_REPO=$six_ALL_CCFRWORK/code/tr11-176B-ml/bigscience
TRAIN_DATA_PATH=$MEGATRON_DEEPSPEED_REPO/data/train-splits.txt
VALID_DATA_PATH=$MEGATRON_DEEPSPEED_REPO/data/valid-splits.txt
CATALOGUE_JSON_PATH=$BIGSCIENCE_REPO/data/catalogue/training_dataset_ratios_merged_nigercongo_v3.json
LOAD_RATIOS_SCRIPT=$BIGSCIENCE_REPO/data/catalogue/load_ratios_meg_ds_format.py
python $LOAD_RATIOS_SCRIPT --dataset-ratios-path $CATALOGUE_JSON_PATH --split train --output-meg-ds-ratio-file $TRAIN_DATA_PATH
python $LOAD_RATIOS_SCRIPT --dataset-ratios-path $CATALOGUE_JSON_PATH --split valid --output-meg-ds-ratio-file $VALID_DATA_PATH

TOKENIZER_NAME_OR_PATH=bigscience-catalogue-data-dev/byte-level-bpe-tokenizer-no-norm-250k-whitespace-and-eos-regex-alpha-v3-dedup-lines-articles

# defining the right environment variables
export TRANSFORMERS_CACHE=$six_ALL_CCFRWORK/models
export HF_DATASETS_CACHE=$six_ALL_CCFRWORK/datasets
export HF_MODULES_CACHE=$six_ALL_CCFRWORK/modules
export HF_METRICS_CACHE=$six_ALL_CCFRWORK/metrics
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# testing for potential faulty nodes
# srun --jobid $SLURM_JOBID bash -c 'python -c "import torch, socket; print(socket.gethostname(), torch.cuda.is_available())"'

# so processes know who to talk to
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000

GPUS_PER_NODE=8
NNODES=$SLURM_NNODES

TP_SIZE=4
PP_SIZE=2

MICRO_BATCH_SIZE=2
GLOBAL_BATCH_SIZE=2048

NHIDDEN=$((1024*1))
NLAYERS=2
NHEADS=64
SEQ_LEN=2048

SAVE_INTERVAL=250

TRAIN_SAMPLES=220_000_000  # 450B tokens
LR_DECAY_SAMPLES=200_000_000  # Decay for the first 410B tokens then continue at fixed --min-lr
LR_WARMUP_SAMPLES=183_105  # 375M tokens. We used to use 216_320, not clear why


OPTIMIZER_ARGS=" \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-8 \
    --lr 6e-5 \
    --min-lr 6e-6 \
    --lr-decay-style cosine \
    --lr-decay-samples $LR_DECAY_SAMPLES \
    --lr-warmup-samples $LR_WARMUP_SAMPLES \
    --clip-grad 1.0 \
    --weight-decay 1e-1 \
    "
# for 20h 1190, for 100h 5990:
#    --exit-duration-in-mins 1190 \
# for quick auto-exit:
#    --exit-interval 10 \
EXIT_OPTS=" \
    --exit-duration-in-mins 5990 \
    "

GPT_ARGS=" \
    --pp-partition-method 'type:transformer|embedding' \
    --num-layers $NLAYERS \
    --hidden-size $NHIDDEN \
    --num-attention-heads $NHEADS \
    --seq-length $SEQ_LEN \
    --max-position-embeddings $SEQ_LEN \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --rampup-batch-size 192 16 9_765_625 \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --train-samples $TRAIN_SAMPLES \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path $TOKENIZER_NAME_OR_PATH \
    --init-method-std 0.0048 \
    --embed-layernorm \
    --bf16 \
    --seed 42 \
    --position-embedding-type alibi \
    --checkpoint-activations \
    --abort-on-unmet-fused-kernel-constraints \
    --kill-switch-path $KILL_SWITCH_PATH \
    --pad-vocab-size-to 250880 \
    $OPTIMIZER_ARGS \
    $EXIT_OPTS \
    "

# TODO: decide on efficient eval-interval + eval-iters

OUTPUT_ARGS=" \
    --log-interval 1 \
    --save-interval $SAVE_INTERVAL \
    --eval-interval 1000 \
    --eval-iters 1 \
    --tensorboard-dir $TENSORBOARD_PATH \
    --tensorboard-queue-size 5 \
    --log-timers-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
    "

ZERO_STAGE=0 # important: bf16 must use z0! it implements its own zero stage 1 equivalent

config_json="./ds_config.$SLURM_JOBID.json"

# Deepspeed figures out GAS dynamically from dynamic GBS via set_train_batch_size()
cat <<EOT > $config_json
{
  "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
  "train_batch_size": $GLOBAL_BATCH_SIZE,
  "gradient_clipping": 1.0,
  "zero_optimization": {
    "stage": $ZERO_STAGE
  },
  "bf16": {
    "enabled": true
  },
  "steps_per_print": 2000,
  "wall_clock_breakdown": false
}
EOT


DEEPSPEED_ARGS=" \
    --deepspeed \
    --deepspeed_config ${config_json} \
    --zero-stage ${ZERO_STAGE} \
    --deepspeed-activation-checkpointing \
    "

export LAUNCHER="python -u -m torch.distributed.run \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --tee 3 \
    "

export CMD=" \
    `pwd`/pretrain_gpt.py \
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
    $GPT_ARGS \
    $OUTPUT_ARGS \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    --train-weighted-split-paths-path $TRAIN_DATA_PATH \
    --valid-weighted-split-paths-path $VALID_DATA_PATH \
    --data-impl mmap \
    --distributed-backend nccl \
     $DEEPSPEED_ARGS \
    "

echo $CMD

# do not remove or the training will hang and nodes will be lost w/o this workaround
export CUDA_LAUNCH_BLOCKING=1

# hide duplicated errors using this hack - will be properly fixed in pt-1.12
export TORCHELASTIC_ERROR_FILE=/tmp/torch-elastic-error.json

clear; srun --jobid $SLURM_JOBID bash -c "$LAUNCHER --node_rank \$SLURM_PROCID $CMD" 2>&1 | tee -a $LOGS_PATH/main_log.txt

echo "END TIME: $(date)"
