#!/bin/bash
#SBATCH --job-name=meg_ds_3d_gpt2_perf_n16
#SBATCH --constraint=v100-32g
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=40           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --gres=gpu:4                 # number of gpus
#SBATCH --time 00:20:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name
#SBATCH --error=%x-%j.out            # error file name (same to watch just one file)
#SBATCH --account=six@gpu

set -x -e

MYENV=stas

if   [[ ${MYENV} == stas ]];    then
    source $six_CCFRWORK/start-user
    cd ~/stas/code/megatron-jeffra
    SAVE_CHECKPOINT_PATH=$six_CCFRSCRATCH/checkpoints/gpt2-meg-ds
else
    source $six_ALL_CCFRWORK/start-prod
    cd $six_ALL_CCFRWORK/code/DeepSpeedExamples/Megatron-LM-v1.1.5-3D_parallelism
    SAVE_CHECKPOINT_PATH=$six_ALL_CCFRSCRATCH/checkpoints/gpt2-meg-ds
fi


nvidia-smi


CHECKPOINT_PATH=$six_ALL_CCFRWORK/models-custom/megatron-gpt2/megatron_lm_345m_v0.0/release
VOCAB_FILE=$CHECKPOINT_PATH/gpt2-vocab.json
MERGE_FILE=$CHECKPOINT_PATH/gpt2-merges.txt
DATA_PATH=$six_ALL_CCFRWORK/datasets-custom/openwebtext-10k/meg-gpt2_text_document


MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000

# adjust depending on the number of the nodes

NNODES=16
PP_SIZE=16 # NLAYERS must be a multiple of PP_SIZE here
MICRO_BATCH_SIZE=4 # works at 4 (for meg alone) 8 (for meg+ds/3d), OOMs at
PP_CHUNKS=256 # GAS

USE_DEEPSPEED=1   # Meg-alone baseline
#USE_DEEPSPEED=0   # Meg+DS/3D

EXIT_EARLY=0
#EXIT_EARLY=1

MSIZE=52

if   [[ ${MSIZE} == 7 ]];    then NHIDDEN=4096;  NLAYERS=36
elif [[ ${MSIZE} == 14 ]];   then NHIDDEN=6144;  NLAYERS=32
elif [[ ${MSIZE} == 18 ]];   then NHIDDEN=6144;  NLAYERS=40
elif [[ ${MSIZE} == 25 ]];   then NHIDDEN=7168;  NLAYERS=40
elif [[ ${MSIZE} == 30 ]];   then NHIDDEN=7168;  NLAYERS=48
elif [[ ${MSIZE} == 39 ]];   then NHIDDEN=8192;  NLAYERS=48
elif [[ ${MSIZE} == 52 ]];   then NHIDDEN=8192;  NLAYERS=64
elif [[ ${MSIZE} == 65 ]];   then NHIDDEN=9216;  NLAYERS=64
elif [[ ${MSIZE} == 81 ]];   then NHIDDEN=10240; NLAYERS=64
elif [[ ${MSIZE} == 97 ]];   then NHIDDEN=11264; NLAYERS=64
elif [[ ${MSIZE} == 116 ]];  then NHIDDEN=12288; NLAYERS=64
elif [[ ${MSIZE} == 136 ]];  then NHIDDEN=13312; NLAYERS=64
elif [[ ${MSIZE} == 158 ]];  then NHIDDEN=14336; NLAYERS=64
elif [[ ${MSIZE} == 181 ]];  then NHIDDEN=15360; NLAYERS=64
elif [[ ${MSIZE} == 206 ]];  then NHIDDEN=16384; NLAYERS=64
else echo "invalid MSIZE: $MSIZE"
fi

GPUS_PER_NODE=4
NHEADS=32
SEQ_LEN=1024
VOCAB_SIZE=50257

DP_SIZE=1
TP_SIZE=4 # always fixed to the size of a single node

# --exit-duration-in-mins
# --exit-interval - exit after this many iterations
if [[ ${EXIT_EARLY} == 1 ]]; then EXIT_OPTS="--exit-interval 2"
else EXIT_OPTS=""
fi

GLOBAL_BATCH_SIZE=$(($MICRO_BATCH_SIZE*$PP_CHUNKS*$DP_SIZE))
GPT_ARGS=" \
    --num-layers $NLAYERS \
    --hidden-size $NHIDDEN \
    --num-attention-heads $NHEADS \
    --seq-length $SEQ_LEN \
    --max-position-embeddings $SEQ_LEN \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --loss-scale 12 \
    --lr 1.5e-4 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --train-iters 1000 \
    --lr-decay-iters 800 \
    --lr-warmup-fraction .01 \
    --weight-decay 1e-2 \
    --clip-grad 1.0 \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --fp16 \
    --checkpoint-activations \
    $EXIT_OPTS \
    "

OUTPUT_ARGS=" \
    --log-interval 1 \
    --save-interval 500 \
    --eval-interval 100 \
    --eval-iters 10 \
    "

#ZeRO Configs
#train_batch_size=$(($DP_SIZE*$MICRO_BATCH_SIZE*$gradient_accumulation_steps))
#  "train_batch_size": $train_batch_size,

ZERO_STAGE=0

config_json="./ds_config.json"

cat <<EOT > $config_json
{
  "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
  "gradient_accumulation_steps": $PP_CHUNKS,
  "gradient_clipping": 1.0,
  "prescale_gradients": true,
  "zero_optimization": {
    "stage": $ZERO_STAGE
  },
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 500,
    "hysteresis": 2,
    "min_loss_scale": 1,
    "initial_scale_power": 12
  },
  "steps_per_print": 2000,
  "wall_clock_breakdown": false
}
EOT


if [[ ${USE_DEEPSPEED} == 1 ]]; then
    DEEPSPEED_ARGS=" \
    --deepspeed \
    --deepspeed_config ${config_json} \
    --zero-stage ${ZERO_STAGE} \
    --deepspeed-activation-checkpointing \
    "
else
    DEEPSPEED_ARGS=""
fi

export LAUNCHER="python -u -m torch.distributed.launch \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    "

export CMD=" \
    `pwd`/pretrain_gpt.py \
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
    $GPT_ARGS \
    $OUTPUT_ARGS \
    --save $SAVE_CHECKPOINT_PATH \
    --load $SAVE_CHECKPOINT_PATH \
    --data-path $DATA_PATH \
    --data-impl mmap \
    --split 949,50,1 \
    --distributed-backend nccl \
     $DEEPSPEED_ARGS \
    "


# clear old checkpoint as it'd mismatch while we sort things out
if   [[ ${MYENV} == stas ]];    then
    rm -rf $six_CCFRSCRATCH/checkpoints/gpt2-meg-ds/*
else
    rm -rf $six_ALL_CCFRSCRATCH/checkpoints/gpt2-meg-ds/*
fi

# model size
python -c "h=$NHIDDEN; l=$NLAYERS; s=$SEQ_LEN; v=$VOCAB_SIZE; print(f'Model size: {(l * (12*h**2 + 13*h) + (v * h) + (s * h) ) / 10**9 :.0f}B')"

echo $CMD

# to debug - add echo (it exits and prints what it would have launched)
clear; srun --jobid $SLURM_JOBID bash -c '$LAUNCHER --node_rank $SLURM_PROCID $CMD' 2>&1 | tee meg_ds_3d_gpt2_perf_n16.out
