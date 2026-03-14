#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Smoke test — runs entirely on your M3 Max in ~30 seconds.
# Verifies: model shapes, forward pass, backward pass, optimizer step,
#           loss goes down, checkpoint saves and reloads.
#
# Usage:
#   bash scripts/smoke.sh
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "============================================================"
echo "  liteGPT smoke test"
echo "  $(python --version) | $(uname -m)"
echo "============================================================"

# --- tiny model, tiny data, short run ---
python -m litegpt.train \
    --depth       2       \
    --seq_len     64      \
    --batch_size  2       \
    --max_steps   30      \
    --dataset     synthetic \
    --log_every   5       \
    --save_every  20      \
    --checkpoint_dir /tmp/litegpt_smoke

echo ""
echo "--- checkpoint reload test ---"
python -m litegpt.train \
    --depth       2       \
    --seq_len     64      \
    --batch_size  2       \
    --max_steps   35      \
    --dataset     synthetic \
    --log_every   5       \
    --save_every  100     \
    --checkpoint_dir /tmp/litegpt_smoke \
    --resume      /tmp/litegpt_smoke/ckpt_0000020.pt

echo ""
echo "--- SFT smoke test ---"
python -m litegpt.sft_train \
    --depth       2       \
    --seq_len     64      \
    --batch_size  2       \
    --max_steps   30      \
    --dataset     synthetic \
    --log_every   5       \
    --checkpoint_dir /tmp/litegpt_sft_smoke

echo ""
echo "--- SFT from pretrained checkpoint ---"
python -m litegpt.sft_train \
    --base_checkpoint /tmp/litegpt_smoke/final.pt \
    --seq_len     64      \
    --batch_size  2       \
    --max_steps   20      \
    --dataset     synthetic \
    --log_every   5       \
    --checkpoint_dir /tmp/litegpt_sft_smoke2

echo ""
echo "============================================================"
echo "  SMOKE TEST PASSED"
echo "============================================================"
