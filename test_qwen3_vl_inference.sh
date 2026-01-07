#!/bin/bash
# Qwen3-VL inference test script with image support
# Supports both text-only and multimodal (image + text) inference

export CUDA_VISIBLE_DEVICES=2,3
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HF_HOME=/nfs/hf_cache

export HF_MODEL_NAME=Qwen/Qwen3-VL-30B-A3B-Instruct
export HF_MAPPING_FILE=/nfs/cyjiang/workdir/Megatron-LM/hf_checkpoint_mapping_qwen3vl.json

export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
export NCCL_ALGO=Ring
export CUBLAS_WORKSPACE_CONFIG=:4096:8

GPUS_PER_NODE=2
MASTER_ADDR=localhost
MASTER_PORT=6001
NUM_NODES=1

# Model configuration (same as training script)
TEXT_MODEL_ARGS=(
    --decoder-num-layers 48
    --vocab-size 151936
    --hidden-size 2048
    --num-attention-heads 32
    --kv-channels 128
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
    --num-experts 128
    --moe-router-topk 8
    --moe-ffn-hidden-size 768
    --moe-grouped-gemm
    --moe-token-dispatcher-type alltoall
)

# Vision model configuration
VISION_MODEL_ARGS=(
    --encoder-num-layers 27
    --img-h 1024
    --img-w 1024
    --patch-dim 16
    --disable-vision-class-token
)

# Model parallelism
MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 2
    --pipeline-model-parallel-size 1
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
    --num-tokens-to-generate 8
    --inference-max-seq-length 131072
    --max-batch-size 1
    --temperature 1.0
    --deterministic-mode
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
if [[ -n "${HF_MODEL_NAME}" ]]; then
    echo "HF model: ${HF_MODEL_NAME} (mapping: ${HF_MAPPING_FILE:-default})"
else
    echo "Checkpoint: ${CHECKPOINT_PATH}"
fi
echo ""

BASE_CMD=(
    torchrun \
    --nproc_per_node=$GPUS_PER_NODE \
    --nnodes=$NUM_NODES \
    --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        qwen3_vl_inference.py \
)

# Choose checkpoint source: HF or Megatron --load
if [[ -n "${HF_MODEL_NAME}" ]]; then
    BASE_CMD+=( --hf-model-name "${HF_MODEL_NAME}" )
    if [[ -n "${HF_MAPPING_FILE}" ]]; then
        BASE_CMD+=( --hf-mapping-file "${HF_MAPPING_FILE}" )
    fi
else
    BASE_CMD+=( --load "${CHECKPOINT_PATH}" )
fi

BASE_CMD+=(
    "${TEXT_MODEL_ARGS[@]}" \
    "${MOE_ARGS[@]}" \
    "${VISION_MODEL_ARGS[@]}" \
    "${MODEL_PARALLEL_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${INFERENCE_ARGS[@]}" \
    "${OTHER_ARGS[@]}"
)

"${BASE_CMD[@]}"