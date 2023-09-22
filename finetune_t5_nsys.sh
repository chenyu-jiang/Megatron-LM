
#!/bin/bash

GPUS_PER_NODE=4
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=18230
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DATA_PATH=/root/Megatron-LM/datasets/cleaned_supervised_proportional_inputs_document
TARGETS_DATA_PATH=/root/Megatron-LM/datasets/cleaned_supervised_proportional_targets_document
CHECKPOINT_PATH=/root/Megatron-LM/checkpoints

export DYNAPIPE_DEBUG=INFO
export DYNAPIPE_LOGGING_DEBUG_DIR=/root/Megatron-LM/dynapipe_debug
export NCCL_DEBUG=WARN
# export PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT --use_env"

# nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas  --sampling-period 250000 -c cudaProfilerApi --capture-range-end stop-shutdown --samples-per-backtrace 1 --cudabacktrace all:1000 -o t5_11b_6l_dynapipe_linear_gbs16384_sample -f true python -m torch.distributed.launch $DISTRIBUTED_ARGS \
nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s none -c cudaProfilerApi --capture-range-end stop-shutdown -o t5_11b_16l_dynapipe_linear_gbs65536 -f true python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_t5.py \
       --tensor-model-parallel-size 1 \
       --pipeline-model-parallel-size 4 \
       --encoder-num-layers 8 \
       --decoder-num-layers 8 \
       --hidden-size 1024 \
       --num-attention-heads 128 \
       --kv-channels 128 \
       --ffn-hidden-size 65536 \
       --encoder-seq-length 1024 \
       --decoder-seq-length 1024 \
       --micro-batch-size 8 \
       --global-batch-size 128 \
       --max-position-embeddings 8192 \
       --no-async-tensor-model-parallel-allreduce \
       --no-scatter-gather-tensors-in-pipeline \
       --train-iters 500 \
       --train-epochs 1 \
       --lr-decay-iters 100 \
       --data-path $DATA_PATH \
       --targets-data-path $TARGETS_DATA_PATH \
       --vocab-file /root/t5-base-vocab.txt \
       --data-impl mmap \
       --split 949,50,1 \
       --lr 0.0001 \
       --min-lr 0.00001 \
       --lr-decay-style linear \
       --lr-warmup-fraction .01 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --log-interval 20 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 5 \
       --fp16  \
       --vocab-extra-ids 100 \
       --num-workers 2 \
       --pipeline-model-parallel-split-rank 2 \
       --dataloader-type ordered \
       --recompute-method uniform \
       --use-dynapipe \
       --dynapipe-cost-model /root/Megatron-LM/t5_11b_cm.pkl \
       --dynapipe-device-to-node 0:0,1:0,2:0,3:0 \
       --dynapipe-device-memory-limit 28000 \
       --dynapipe-intra-node-bw 4800 \
       --dynapipe-inter-node-bw 100 \
       --dynapipe-layer-to-device 0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3 \
       --dynamic-batchsize \
       --tokens-per-global-batch 65536 \
       --dynapipe-prefetch-planner-num-workers 64 \
       --profile-with-nsys \
       --nsys-profile-warmup 20 \
       --nsys-profile-steps 20 \
       --dynapipe-reserve-all-memory \
       --dynapipe-custom-allocator \
       2>&1 | tee log_t5_11b_16l_dynapipe_linear_gbs65536_nsys.txt
