
six_ALL_CCFRWORK=/data/cache
export TRANSFORMERS_CACHE=$six_ALL_CCFRWORK/models
export HF_DATASETS_CACHE=$six_ALL_CCFRWORK/datasets
export HF_MODULES_CACHE=$six_ALL_CCFRWORK/modules
export HF_METRICS_CACHE=$six_ALL_CCFRWORK/metrics
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1


variant=test
DATA_OUTPUT_PATH=/data/pengjun/Megatron-DeepSpeed/checkpoints/tr11b-1B3-ml
CHECKPOINT_PATH=$DATA_OUTPUT_PATH/checkpoints/$variant
REPO_PATH=$DATA_OUTPUT_PATH/tr11f-1B3-ml-logs
TENSORBOARD_PATH=$REPO_PATH/tensorboard/$variant
LOGS_PATH=$REPO_PATH/logs/$variant
mkdir -p $LOGS_PATH

MEGATRON_DEEPSPEED_REPO=/data/pengjun/Megatron-DeepSpeed
cd $MEGATRON_DEEPSPEED_REPO

BIGSCIENCE_REPO=/data/pengjun/bigscience
TRAIN_DATA_PATH=$MEGATRON_DEEPSPEED_REPO/data/train-splits-1B3.txt
VALID_DATA_PATH=$MEGATRON_DEEPSPEED_REPO/data/valid-splits-1B3.txt
CATALOGUE_JSON_PATH=$MEGATRON_DEEPSPEED_REPO/preprocess_data/$variant.json
LOAD_RATIOS_SCRIPT=$BIGSCIENCE_REPO/data/catalogue/load_ratios_meg_ds_format.py
python $LOAD_RATIOS_SCRIPT --dataset-ratios-path $CATALOGUE_JSON_PATH --split train --output-meg-ds-ratio-file $TRAIN_DATA_PATH
python $LOAD_RATIOS_SCRIPT --dataset-ratios-path $CATALOGUE_JSON_PATH --split valid --output-meg-ds-ratio-file $VALID_DATA_PATH

TOKENIZER_NAME_OR_PATH=/data/pengjun/Megatron-DeepSpeed/byte-level-bpe-tokenizer-no-norm-250k-whitespace-and-eos-regex-alpha-v3-dedup-lines-articles

# N_GPUS=1
MICRO_BATCH_SIZE=2
GLOBAL_BATCH_SIZE=4
NNODES=1
TP_SIZE=2
PP_SIZE=2
GPUS_PER_NODE=2


NLAYERS=24
NHIDDEN=2048
NHEADS=16
SEQ_LEN=2048
MASTER_ADDR=localhost
MASTER_PORT=6002

SAVE_INTERVAL=1

TRAIN_SAMPLES=10_000

LR_DECAY_SAMPLES=8_000  # Decay for the first 410B tokens then continue at fixed --min-lr
LR_WARMUP_SAMPLES=2_000  # 375M tokens

OPTIMIZER_ARGS=" \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-8 \
    --lr 1.2e-4 \
    --min-lr 1e-5 \
    --lr-decay-style cosine \
    --lr-decay-samples $LR_DECAY_SAMPLES \
    --lr-warmup-samples $LR_WARMUP_SAMPLES \
    --clip-grad 1.0 \
    --weight-decay 1e-1 \
    "

EXIT_OPTS=" \
    --exit-duration-in-mins 5990 \
    "

GPT_ARGS=" \
    --pp-partition-method type:transformer|embedding \
    --num-layers $NLAYERS \
    --hidden-size $NHIDDEN \
    --num-attention-heads $NHEADS \
    --seq-length $SEQ_LEN \
    --max-position-embeddings $SEQ_LEN \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --rampup-batch-size 2 2 2000 \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --train-samples $TRAIN_SAMPLES \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path $TOKENIZER_NAME_OR_PATH \
    --init-method-std 0.0048 \
    --embed-layernorm \
    --fp16 \
    --seed 42 \
    --position-embedding-type alibi \
    --checkpoint-activations \
    --abort-on-unmet-fused-kernel-constraints \
    --pad-vocab-size-to 250880 \
    $OPTIMIZER_ARGS \
    $EXIT_OPTS \
    "


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

config_json="./ds_config_1.json"

# Deepspeed figures out GAS dynamically from dynamic GBS via set_train_batch_size()
cat <<EOT > $config_json
{
  "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
  "train_batch_size": $GLOBAL_BATCH_SIZE,
  "gradient_clipping": 1.0,
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

$LAUNCHER  $CMD 2>&1 | tee -a $LOGS_PATH/main_log.txt

echo "END TIME: $(date)"
