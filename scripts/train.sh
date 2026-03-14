#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Full distributed training on the 3×H200 server.
#
# Environment variables (override defaults by setting them before running):
#   DEPTH       — transformer depth      (default: 24  → ~350M params)
#   SEQ_LEN     — sequence length        (default: 2048)
#   BATCH_SIZE  — per-GPU batch size     (default: 32)
#   MAX_STEPS   — total training steps   (default: 50000)
#   NPROC       — number of GPUs to use  (default: 3, all H200s)
#   DATASET     — "text" or "synthetic"  (default: text)
#   DATA_PATH   — path to .bin file      (default: data/train.bin)
#   RESUME      — checkpoint to resume   (default: none)
#
# Usage:
#   bash scripts/train.sh
#   DEPTH=12 MAX_STEPS=10000 bash scripts/train.sh
#   RESUME=checkpoints/ckpt_0010000.pt bash scripts/train.sh
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Defaults
DEPTH="${DEPTH:-24}"
SEQ_LEN="${SEQ_LEN:-2048}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_STEPS="${MAX_STEPS:-50000}"
NPROC="${NPROC:-3}"
DATASET="${DATASET:-text}"
DATA_PATH="${DATA_PATH:-data/train.bin}"
RESUME="${RESUME:-}"

echo "============================================================"
echo "  liteGPT distributed training"
echo "  GPUs:       $NPROC × H200"
echo "  depth:      $DEPTH"
echo "  seq_len:    $SEQ_LEN"
echo "  batch/gpu:  $BATCH_SIZE  (total: $((BATCH_SIZE * NPROC)))"
echo "  max_steps:  $MAX_STEPS"
echo "  dataset:    $DATASET"
if [[ -n "$RESUME" ]]; then
echo "  resume:     $RESUME"
fi
echo "============================================================"

EXTRA_ARGS=""
if [[ -n "$RESUME" ]]; then
    EXTRA_ARGS="--resume $RESUME"
fi

if [[ "$DATASET" == "text" ]]; then
    DATA_ARG="--data_path $DATA_PATH"
else
    DATA_ARG=""
fi

torchrun \
    --nproc_per_node="$NPROC" \
    --master_port=29500 \
    -m litegpt.train \
        --depth       "$DEPTH"      \
        --seq_len     "$SEQ_LEN"    \
        --batch_size  "$BATCH_SIZE" \
        --max_steps   "$MAX_STEPS"  \
        --dataset     "$DATASET"    \
        $DATA_ARG                   \
        --log_every   100           \
        --save_every  5000          \
        --checkpoint_dir checkpoints \
        $EXTRA_ARGS
