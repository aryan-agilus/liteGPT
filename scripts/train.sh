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
NPROC="${NPROC:-3}"
DATASET="${DATASET:-text}"
DATA_PATH="${DATA_PATH:-data/pretrain.bin}"
MAX_STEPS="${MAX_STEPS:-50000}"
RESUME="${RESUME:-}"

# Memory config — tuned for 3×H200 (140 GB each) with depth=24
# batch=8 + grad_accum=4 → eff_batch = 8*4*3 = 96 seqs × 2048 tok = ~196K tok/step
# With --grad_ckpt: activation memory ~8x lower, safely fits with batch=16
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
GRAD_CKPT="${GRAD_CKPT:-true}"   # set to "" to disable

eff_batch=$((BATCH_SIZE * GRAD_ACCUM * NPROC))

echo "============================================================"
echo "  liteGPT distributed training"
echo "  GPUs:        $NPROC × H200"
echo "  depth:       $DEPTH"
echo "  seq_len:     $SEQ_LEN"
echo "  batch/gpu:   $BATCH_SIZE  grad_accum=$GRAD_ACCUM"
echo "  eff_batch:   $eff_batch sequences / step"
echo "  max_steps:   $MAX_STEPS"
echo "  dataset:     $DATASET"
echo "  grad_ckpt:   $GRAD_CKPT"
if [[ -n "$RESUME" ]]; then
echo "  resume:      $RESUME"
fi
echo "============================================================"

EXTRA_ARGS=""
[[ -n "$RESUME"    ]] && EXTRA_ARGS="$EXTRA_ARGS --resume $RESUME"
[[ -n "$GRAD_CKPT" ]] && EXTRA_ARGS="$EXTRA_ARGS --grad_ckpt"

DATA_ARG=""
[[ "$DATASET" == "text" ]] && DATA_ARG="--data_path $DATA_PATH"

torchrun \
    --nproc_per_node="$NPROC" \
    --master_port=29500 \
    -m litegpt.train \
        --depth       "$DEPTH"      \
        --seq_len     "$SEQ_LEN"    \
        --batch_size  "$BATCH_SIZE" \
        --grad_accum  "$GRAD_ACCUM" \
        --max_steps   "$MAX_STEPS"  \
        --dataset     "$DATASET"    \
        $DATA_ARG                   \
        --log_every   50            \
        --save_every  5000          \
        --checkpoint_dir checkpoints \
        $EXTRA_ARGS
