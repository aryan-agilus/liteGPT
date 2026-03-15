"""
Training entry point for liteGPT.

Single-device (local smoke test):
    python -m litegpt.train --depth 2 --seq_len 64 --batch_size 2 \
        --max_steps 20 --dataset synthetic

Distributed (3×H200 on remote server):
    torchrun --nproc_per_node=3 -m litegpt.train \
        --depth 24 --seq_len 2048 --batch_size 32 \
        --max_steps 50000 --dataset text --data_path data/train.bin

Device/dtype are auto-detected:
    CUDA  → bfloat16 + autocast
    MPS   → float32  (no autocast; MPS bfloat16 support is incomplete)
    CPU   → float32
"""

import os
import math
import time
import argparse
import contextlib

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from litegpt.model import GPT, GPTConfig
from litegpt.optimizer import MuonAdamW, partition_params
from litegpt.dataset import get_dataloader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cosine_lr(
    step: int,
    max_steps: int,
    max_lr: float,
    min_lr_ratio: float = 0.1,
    warmup_ratio: float = 0.05,
) -> float:
    """Linear warmup → cosine decay."""
    warmup_steps = max(1, int(max_steps * warmup_ratio))
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return max_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)


def setup_distributed() -> tuple[int, int, bool]:
    """
    Detect torchrun environment and initialize process group.
    Returns (local_rank, world_size, is_ddp).
    """
    if "LOCAL_RANK" not in os.environ:
        return 0, 1, False

    local_rank  = int(os.environ["LOCAL_RANK"])
    world_size  = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return local_rank, world_size, True


def find_latest_checkpoint(checkpoint_dir: str, glob_pattern: str = "ckpt_*.pt") -> str | None:
    """Return the path to the most recent step checkpoint in checkpoint_dir, or None."""
    import glob as _glob
    pattern = os.path.join(checkpoint_dir, glob_pattern)
    candidates = sorted(_glob.glob(pattern))  # lexicographic = step order
    return candidates[-1] if candidates else None


def set_lr(optimizer: MuonAdamW, muon_lr: float, adamw_lr: float):
    for group in optimizer.param_groups:
        if group.get("use_muon", False):
            group["lr"] = muon_lr
        else:
            group["adamw_lr"] = adamw_lr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train a liteGPT model")

    # Model
    parser.add_argument("--depth",      type=int,   default=4,
                        help="Transformer depth. All other dims auto-scale.")

    # Data
    parser.add_argument("--dataset",    type=str,   default="synthetic",
                        choices=["synthetic", "text"])
    parser.add_argument("--data_path",  type=str,   default=None,
                        help="Path to pre-tokenized .bin file (for --dataset text)")
    parser.add_argument("--seq_len",    type=int,   default=256)

    # Training
    parser.add_argument("--batch_size", type=int,   default=4,
                        help="Per-device batch size")
    parser.add_argument("--max_steps",  type=int,   default=1000)
    parser.add_argument("--lr",         type=float, default=0.01,
                        help="Muon learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip",  type=float, default=1.0)
    parser.add_argument("--grad_accum", type=int,   default=1,
                        help="Gradient accumulation steps. "
                             "Effective batch = batch_size × grad_accum × world_size.")
    parser.add_argument("--grad_ckpt",  action="store_true",
                        help="Enable gradient checkpointing (~8x less activation memory, "
                             "~33% more compute). Recommended for depth>=12 on single GPU.")
    parser.add_argument("--compile",    action="store_true",
                        help="torch.compile the model for ~20-30%% extra throughput "
                             "(first step is slow due to compilation).")

    # Device / precision
    parser.add_argument("--device",     type=str,   default=None,
                        help="Override device (cuda/mps/cpu)")
    parser.add_argument("--dtype",      type=str,   default=None,
                        help="Override dtype (float32/bfloat16)")

    # Checkpointing / logging
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--resume",     type=str,   default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--save_every", type=int,   default=1000)
    parser.add_argument("--log_every",  type=int,   default=10)

    args = parser.parse_args()

    # Auto-resume: if no --resume given, look for the latest checkpoint
    if args.resume is None:
        args.resume = find_latest_checkpoint(args.checkpoint_dir)

    # ------------------------------------------------------------------
    # Distributed init
    # ------------------------------------------------------------------
    local_rank, world_size, is_ddp = setup_distributed()
    is_master = local_rank == 0

    # ------------------------------------------------------------------
    # Device & dtype
    # ------------------------------------------------------------------
    if args.device:
        device = torch.device(args.device)
    elif is_ddp:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = get_device()

    if args.dtype:
        dtype = getattr(torch, args.dtype)
    elif device.type == "cuda":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    # autocast only on CUDA — MPS float32 needs no cast, CPU too slow for bf16
    if device.type == "cuda" and dtype != torch.float32:
        autocast_ctx = torch.autocast(device_type="cuda", dtype=dtype)
    else:
        autocast_ctx = contextlib.nullcontext()

    eff_batch = args.batch_size * args.grad_accum * world_size

    if is_master:
        print(f"\n{'='*60}")
        print(f"  device={device}  dtype={dtype}  world_size={world_size}")
        print(f"  depth={args.depth}  seq_len={args.seq_len}  batch={args.batch_size}")
        print(f"  grad_accum={args.grad_accum}  eff_batch={eff_batch}  grad_ckpt={args.grad_ckpt}")
        print(f"  compile={args.compile}  max_steps={args.max_steps}  dataset={args.dataset}")
        print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    config = GPTConfig(depth=args.depth, seq_len=args.seq_len)
    model  = GPT(config).to(device).to(dtype)
    model.gradient_checkpointing = args.grad_ckpt
    model.setup_rope()

    if is_master:
        print(model)
        print()

    start_step = 0
    resume_ckpt: dict | None = None

    if args.resume:
        _ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(_ckpt["model"])
        start_step  = _ckpt.get("step", 0)
        resume_ckpt = _ckpt
        if is_master:
            print(f"Resumed from {args.resume} at step {start_step}")

    # torch.compile — fuses kernels, removes Python overhead, ~20-30% faster.
    # Must happen before DDP so DDP wraps the compiled model.
    if args.compile:
        if is_master:
            print("Compiling model with torch.compile (first step will be slow)...")
        model = torch.compile(model)

    if is_ddp:
        model = DDP(model, device_ids=[local_rank])

    raw_model = model.module if is_ddp else model
    # torch.compile wraps the model under ._orig_mod — unwrap so state_dict
    # keys are plain (e.g. "embed.weight") not "_orig_mod.embed.weight"
    if hasattr(raw_model, "_orig_mod"):
        raw_model = raw_model._orig_mod

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    muon_params, adamw_params = partition_params(raw_model)

    # AdamW lr is scaled by model_dim relative to GPT-2's 768 baseline
    base_adamw_lr = args.lr * (config.model_dim / 768) ** -0.5

    optimizer = MuonAdamW(
        muon_params=muon_params,
        adamw_params=adamw_params,
        lr=args.lr,
        adamw_lr=base_adamw_lr,
        weight_decay=args.weight_decay,
    )

    if resume_ckpt is not None and "optimizer" in resume_ckpt:
        optimizer.load_state_dict(resume_ckpt["optimizer"])

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    loader = get_dataloader(
        dataset_type=args.dataset,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        rank=local_rank,
        world_size=world_size,
        data_path=args.data_path,
    )
    data_iter = iter(loader)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    t0 = time.perf_counter()
    tokens_since_log = 0

    for step in range(start_step, args.max_steps):
        model.train()

        # LR schedule
        muon_lr  = cosine_lr(step, args.max_steps, args.lr)
        adamw_lr = muon_lr * (config.model_dim / 768) ** -0.5
        set_lr(optimizer, muon_lr, adamw_lr)

        # ── Gradient accumulation loop ────────────────────────────────
        # Accumulate gradients over grad_accum micro-steps before one
        # optimizer update. For DDP, suppress the all-reduce on every
        # micro-step except the last (no_sync saves ~2x communication).
        optimizer.zero_grad(set_to_none=True)
        # Accumulate loss as a CUDA tensor — avoids GPU→CPU sync on every micro-step.
        # We only call .item() once at log time, which is a single sync per log_every steps.
        loss_accum = torch.zeros(1, device=device)

        for micro in range(args.grad_accum):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                x, y = next(data_iter)

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # Only sync gradients across ranks on the final micro-step
            is_last_micro = micro == args.grad_accum - 1
            sync_ctx = (
                contextlib.nullcontext()
                if (not is_ddp or is_last_micro)
                else model.no_sync()
            )

            with sync_ctx, autocast_ctx:
                _, loss = model(x, y)
                loss = loss / args.grad_accum

            loss.backward()
            loss_accum += loss.detach()

        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        tokens_since_log += args.batch_size * args.grad_accum * args.seq_len * world_size

        # Logging
        if is_master and step % args.log_every == 0:
            elapsed = time.perf_counter() - t0
            tok_per_s = tokens_since_log / elapsed
            print(
                f"step {step:6d}/{args.max_steps} | "
                f"loss {loss_accum.item():.4f} | "
                f"lr {muon_lr:.2e} | "
                f"{tok_per_s:,.0f} tok/s"
            )
            t0 = time.perf_counter()
            tokens_since_log = 0

        # Checkpointing
        if is_master and step > 0 and step % args.save_every == 0:
            path = os.path.join(args.checkpoint_dir, f"ckpt_{step:07d}.pt")
            torch.save(
                {
                    "step":      step,
                    "config":    config,
                    "model":     raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                path,
            )
            print(f"  → saved {path}")

    # Final checkpoint
    if is_master:
        path = os.path.join(args.checkpoint_dir, "final.pt")
        torch.save(
            {
                "step":   args.max_steps,
                "config": config,
                "model":  raw_model.state_dict(),
            },
            path,
        )
        print(f"\nTraining complete. Saved: {path}")

    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
