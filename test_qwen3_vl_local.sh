#!/bin/bash
# Train Qwen3-VL MoE model with 2 layers for single GPU testing
# Based on config for Qwen3VLMoeForConditionalGeneration

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=1
MASTER_ADDR=localhost
MASTER_PORT=6000
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

# Set output paths
# CHECKPOINT_PATH=${CHECKPOINT_PATH:-"./checkpoints/qwen3_vl_2layer_test"}
TENSORBOARD_LOGS_PATH=${TENSORBOARD_LOGS_PATH:-"./logs/qwen3_vl_2layer_test"}
# TOKENIZER_PATH=${TOKENIZER_PATH:-"Qwen/Qwen3-VL-30B-A3B-Instruct"}

# mkdir -p ${CHECKPOINT_PATH}
mkdir -p ${TENSORBOARD_LOGS_PATH}

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

# Text model configuration from config.json
# Original has 48 layers, reduced to 2 for testing
TEXT_MODEL_ARGS=(
    --decoder-num-layers 2
    --vocab-size 151936
    --hidden-size 2048
    --num-attention-heads 32
    --num-query-groups 4
    --ffn-hidden-size 6144
    --max-position-embeddings 262144
    --seq-length 2048
    --decoder-seq-length 4096
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --disable-bias-linear
    --position-embedding-type rope
    --rotary-percent 1.0
    --rotary-base 5000000
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --swiglu
    --untie-embeddings-and-output-weights
    --group-query-attention
    --no-masked-softmax-fusion
    --attention-softmax-in-fp32
    --qk-layernorm
)

# MoE configuration from text_config
MOE_ARGS=(
    # --num-experts 128
    --num-experts 16 # Reduced for single GPU testing
    --moe-router-topk 8
    --moe-ffn-hidden-size 768
    --moe-grouped-gemm
    --moe-token-dispatcher-type alltoall
)

# Vision model configuration from vision_config
# Original has 27 layers, keeping proportional reduction (not specified in args, handled by encoder-num-layers)
VISION_MODEL_ARGS=(
    --encoder-num-layers 2
    --img-h 1024
    --img-w 1024
    --patch-dim 16
    --disable-vision-class-token
)

# Training configuration for single GPU testing
TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 1
    --train-iters 100
    --weight-decay 0.01
    --adam-beta1 0.9
    --adam-beta2 0.95
    --init-method-std 0.02
    --clip-grad 1.0
    --bf16
    --lr 1.0e-4
    --lr-decay-style cosine
    --min-lr 1.0e-5
    --lr-warmup-fraction 0.01
    --lr-decay-iters 100
    --no-build-tokenizer
)

# Model parallelism - single GPU
MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    # --encoder-pipeline-model-parallel-size 1
    --sequence-parallel
)

# Data and tokenizer configuration
DATA_ARGS=(
    # --tokenizer-type MultimodalTokenizer
    # --tokenizer-model ${TOKENIZER_PATH}
    --split 98,2,0
    --dataloader-seq-length 256
)

# Logging and checkpointing
EVAL_AND_LOGGING_ARGS=(
    --log-interval 10
    --save-interval 50
    --eval-interval 50
    # --save ${CHECKPOINT_PATH}
    # --load ${CHECKPOINT_PATH}
    # --eval-iters 10
    --tensorboard-dir ${TENSORBOARD_LOGS_PATH}
    --log-params-norm
    --log-num-zeros-in-grad
)

# Other arguments
OTHER_ARGS=(
    --transformer-impl transformer_engine
    --use-flash-attn
    --apply-layernorm-1p
    --use-distributed-optimizer
    --num-workers 2
    --seed 1234
)

# Combine all arguments
OPTIONS=" \
    ${TEXT_MODEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${VISION_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    ${OTHER_ARGS[@]}
"

# Environment variables for TransformerEngine
export NVTE_APPLY_QK_LAYER_SCALING=0
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1

echo "Starting Qwen3-VL training with randomly initialized model..."
echo "Checkpoint path: ${CHECKPOINT_PATH}"
echo "Tensorboard logs: ${TENSORBOARD_LOGS_PATH}"
echo ""

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_qwen3_vl.py ${OPTIONS}
