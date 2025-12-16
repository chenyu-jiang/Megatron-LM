#!/bin/bash
# Qwen3-VL inference test script with image support
# Supports both text-only and multimodal (image + text) inference

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=1
MASTER_ADDR=localhost
MASTER_PORT=6001
NUM_NODES=1

# Model configuration (same as training script)
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

# MoE configuration
MOE_ARGS=(
    --num-experts 16
    --moe-router-topk 8
    --moe-ffn-hidden-size 768
    --moe-grouped-gemm
    --moe-token-dispatcher-type alltoall
)

# Vision model configuration
VISION_MODEL_ARGS=(
    --encoder-num-layers 2
    --img-h 1024
    --img-w 1024
    --patch-dim 16
    --disable-vision-class-token
)

# Model parallelism
MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --sequence-parallel
)

# Tokenizer and data
DATA_ARGS=(
    --split 98,2,0
    --no-build-tokenizer
)

# Inference-specific arguments
# For text-only inference, leave --image-paths empty
# For multimodal inference, provide both --prompts and --image-paths
INFERENCE_ARGS=(
    --num-tokens-to-generate 256
    --inference-max-seq-length 131072
    --max-batch-size 1
    --temperature 1.0
    --top-k 1
    --top-p 0.0
    --prompts "What is this image about?"
    # Uncomment and set image paths for multimodal inference:
    --image-paths "/nfs/cyjiang/workdir/test_image/demo.jpeg"
)

# Other arguments
OTHER_ARGS=(
    --transformer-impl transformer_engine
    --use-flash-attn
    --apply-layernorm-1p
    --bf16
    --num-workers 0
    --seed 1234
)

# Checkpoint path - SET THIS TO YOUR TRAINED MODEL CHECKPOINT
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"./checkpoints/qwen3_vl_2layer_test"}

echo "Starting Qwen3-VL inference with image support..."
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo ""

torchrun \
    --nproc_per_node=$GPUS_PER_NODE \
    --nnodes=$NUM_NODES \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    qwen3_vl_inference.py \
    --load ${CHECKPOINT_PATH} \
    "${TEXT_MODEL_ARGS[@]}" \
    "${MOE_ARGS[@]}" \
    "${VISION_MODEL_ARGS[@]}" \
    "${MODEL_PARALLEL_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${INFERENCE_ARGS[@]}" \
    "${OTHER_ARGS[@]}"
