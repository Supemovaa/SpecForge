#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels
export SPECFORGE_DATA_NUM_PROC=32

NUM_GPUS=${1:-8}

ATTENTION_BACKEND=${2:-flex_attention}
BATCH_SIZE=4
ACCUMULATION_STEPS=1

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_dflash.py \
    --target-model-path /home/huggingface/Qwen/Qwen3-4B \
    --draft-config-path $ROOT_DIR/configs/qwen3-4b-dflash.json \
    --train-data-path $ROOT_DIR/data/rewritten_dataset/nemotron_math_train.jsonl \
    --output-dir $ROOT_DIR/outputs/qwen3-4b-nemotron \
    --num-epochs 6 \
    --batch-size $BATCH_SIZE \
    --accumulation-steps $ACCUMULATION_STEPS \
    --learning-rate 6e-4 \
    --warmup-ratio 0.04 \
    --max-grad-norm 1.0 \
    --max-length 3072 \
    --chat-template qwen3-instruct \
    --attention-backend $ATTENTION_BACKEND \
    --loss-decay-gamma 7.0 \
    --log-interval $((50 * $ACCUMULATION_STEPS)) \
    --save-interval $((1000 * $ACCUMULATION_STEPS)) \
    --report-to wandb \
    --wandb-project specforge-qwen3-4b-dflash \
    --target-model-backend sglang \
    --block-size 16 \
    --num-anchors 128 \
    --wandb-name qwen3-4b-dflash-nemotron