"""
Supervised Fine-Tuning (SFT) for liteGPT.

SFT differs from pretraining in exactly one way:
  → Loss is computed ONLY on assistant tokens.
    User tokens, BOS, and padding are masked out.

This teaches the model to respond, not to repeat the prompt.

Usage — smoke test (local M3 Max, no checkpoint needed):
    python -m litegpt.sft_train --dataset synthetic --depth 2 \
        --seq_len 64 --batch_size 2 --max_steps 30

Usage — SFT from a pretrained checkpoint (H200 server):
    torchrun --nproc_per_node=3 -m litegpt.sft_train \
        --base_checkpoint checkpoints/pretrain/final.pt \
        --dataset bin \
        --tokens_path data/sft/sft_tokens.bin \
        --mask_path   data/sft/sft_mask.bin \
        --seq_len 2048 --batch_size 16 --max_steps 5000

Usage — SFT from HuggingFace dataset on-the-fly:
    python -m litegpt.sft_train \
        --base_checkpoint checkpoints/pretrain/final.pt \
        --dataset hf --hf_slug HuggingFaceTB/smoltalk \
        --max_examples 50000 --max_steps 5000

Checkpoints are saved to --checkpoint_dir (default: checkpoints/sft/).
"""

import os
import time
import argparse
import contextlib

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from litegpt.model import GPT, GPTConfig
from litegpt.optimizer import MuonAdamW, partition_params
from litegpt.sft_dataset import get_sft_dataloader
from litegpt.train import get_device, cosine_lr, setup_distributed, set_lr, find_latest_checkpoint


# ---------------------------------------------------------------------------
# Masked cross-entropy loss
# ---------------------------------------------------------------------------

def masked_cross_entropy(
    logits: torch.Tensor,   # (B, T, V)
    targets: torch.Tensor,  # (B, T)
    mask: torch.Tensor,     # (B, T)  float, 1.0 = compute loss
) -> torch.Tensor:
    """
    Cross-entropy loss averaged over masked positions only.
    Returns 0 (not nan) if no position is masked in this batch.
    """
    B, T, V = logits.shape
    loss_per_token = nn.functional.cross_entropy(
        logits.reshape(B * T, V),
        targets.reshape(B * T),
        reduction="none",
    )                                          # (B*T,)
    mask_flat = mask.reshape(B * T)
    n_active  = mask_flat.sum().clamp(min=1)
    return (loss_per_token * mask_flat).sum() / n_active


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SFT fine-tuning for liteGPT")

    # Model — only used when training from scratch (no base checkpoint)
    parser.add_argument("--depth",     type=int,   default=4)

    # Checkpoint
    parser.add_argument("--base_checkpoint", type=str, default=None,
                        help="Pretrained checkpoint to fine-tune from")
    parser.add_argument("--resume",          type=str, default=None,
                        help="SFT checkpoint to resume mid-run")
    parser.add_argument("--checkpoint_dir",  type=str, default="checkpoints/sft")
    parser.add_argument("--save_every",      type=int, default=500)

    # Data
    parser.add_argument("--dataset",      required=False, default="synthetic",
                        choices=["synthetic", "bin", "hf"],
                        help="synthetic=smoke test | bin=preprocessed | hf=HuggingFace live")
    parser.add_argument("--tokens_path",  type=str, default=None,
                        help="Path to sft_tokens.bin  (--dataset bin)")
    parser.add_argument("--mask_path",    type=str, default=None,
                        help="Path to sft_mask.bin    (--dataset bin)")
    parser.add_argument("--hf_slug",      type=str, default="HuggingFaceTB/smoltalk",
                        help="HuggingFace dataset slug (--dataset hf)")
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Cap examples (--dataset hf)")
    parser.add_argument("--seq_len",      type=int, default=2048)

    # Training
    parser.add_argument("--batch_size",   type=int,   default=4)
    parser.add_argument("--max_steps",    type=int,   default=2000)
    parser.add_argument("--lr",           type=float, default=1e-4,
                        help="Muon LR — lower than pretrain since weights are already good")
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--grad_clip",    type=float, default=1.0)
    parser.add_argument("--grad_accum",   type=int,   default=1,
                        help="Gradient accumulation steps.")
    parser.add_argument("--grad_ckpt",    action="store_true",
                        help="Enable gradient checkpointing.")
    parser.add_argument("--compile",      action="store_true",
                        help="torch.compile for ~20-30%% extra throughput.")
    parser.add_argument("--log_every",    type=int,   default=10)

    # Device
    parser.add_argument("--device",       type=str, default=None)
    parser.add_argument("--dtype",        type=str, default=None)

    args = parser.parse_args()

    # Auto-resume: if no --resume given, look for the latest SFT checkpoint
    if args.resume is None:
        args.resume = find_latest_checkpoint(args.checkpoint_dir, glob_pattern="sft_*.pt")

    # ------------------------------------------------------------------
    # Distributed
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

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=dtype)
        if device.type == "cuda" and dtype != torch.float32
        else contextlib.nullcontext()
    )

    eff_batch = args.batch_size * args.grad_accum * world_size

    if is_master:
        print(f"\n{'='*60}")
        print(f"  SFT fine-tuning")
        print(f"  device={device}  dtype={dtype}  world_size={world_size}")
        print(f"  dataset={args.dataset}  seq_len={args.seq_len}  batch={args.batch_size}")
        print(f"  grad_accum={args.grad_accum}  eff_batch={eff_batch}  grad_ckpt={args.grad_ckpt}")
        print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Model — load from base checkpoint or init fresh
    # ------------------------------------------------------------------
    start_step   = 0
    resume_ckpt: dict | None = None

    if args.base_checkpoint:
        if is_master:
            print(f"Loading base checkpoint: {args.base_checkpoint}")
        base = torch.load(args.base_checkpoint, map_location=device, weights_only=False)
        config = base["config"]
        model  = GPT(config).to(device).to(dtype)
        model.gradient_checkpointing = args.grad_ckpt
        model.load_state_dict(base["model"])
        model.setup_rope()
    else:
        # No base checkpoint — train SFT from a random init (useful for smoke tests)
        config = GPTConfig(depth=args.depth, seq_len=args.seq_len)
        model  = GPT(config).to(device).to(dtype)
        model.gradient_checkpointing = args.grad_ckpt
        model.setup_rope()

    if args.resume:
        _r = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(_r["model"])
        start_step  = _r.get("step", 0)
        resume_ckpt = _r
        if is_master:
            print(f"Resumed SFT from {args.resume} at step {start_step}")

    if is_master:
        print(model)
        print()

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
    # Optimizer — lower LR than pretrain (weights are already meaningful)
    # ------------------------------------------------------------------
    muon_params, adamw_params = partition_params(raw_model)
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
    loader = get_sft_dataloader(
        dataset_type=args.dataset,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        rank=local_rank,
        world_size=world_size,
        tokens_path=args.tokens_path,
        mask_path=args.mask_path,
        hf_slug=args.hf_slug,
        max_examples=args.max_examples,
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

        muon_lr  = cosine_lr(step, args.max_steps, args.lr)
        adamw_lr = muon_lr * (config.model_dim / 768) ** -0.5
        set_lr(optimizer, muon_lr, adamw_lr)

        optimizer.zero_grad(set_to_none=True)
        loss_accum = torch.zeros(1, device=device)
        non_blocking = device.type == "cuda"
        last_mask = None
        skip_step = False

        for micro in range(args.grad_accum):
            try:
                x, y, mask = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                x, y, mask = next(data_iter)

            x    = x.to(device, non_blocking=non_blocking)
            y    = y.to(device, non_blocking=non_blocking)
            mask = mask.to(device, non_blocking=non_blocking)
            last_mask = mask

            is_last_micro = micro == args.grad_accum - 1
            sync_ctx = (
                contextlib.nullcontext()
                if (not is_ddp or is_last_micro)
                else model.no_sync()
            )

            with sync_ctx, autocast_ctx:
                logits, _ = model(x)
                loss = masked_cross_entropy(logits, y, mask) / args.grad_accum

            if not loss.isfinite():
                print(f"  [warn] step {step} micro {micro}: non-finite loss, skipping step")
                optimizer.zero_grad(set_to_none=True)
                skip_step = True
                break

            loss.backward()
            loss_accum += loss.detach()

        if skip_step:
            continue

        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        tokens_since_log += args.batch_size * args.grad_accum * args.seq_len * world_size

        if is_master and step % args.log_every == 0:
            elapsed = time.perf_counter() - t0
            tok_per_s = tokens_since_log / elapsed
            active = int(last_mask.sum().item()) if (last_mask is not None and last_mask.isfinite().all()) else -1
            print(
                f"step {step:6d}/{args.max_steps} | "
                f"loss {loss_accum.item():.4f} | "
                f"lr {muon_lr:.2e} | "
                f"active_tok {active} | "
                f"{tok_per_s:,.0f} tok/s"
            )
            t0 = time.perf_counter()
            tokens_since_log = 0

        if is_master and step > 0 and step % args.save_every == 0:
            path = os.path.join(args.checkpoint_dir, f"sft_{step:07d}.pt")
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

    if is_master:
        path = os.path.join(args.checkpoint_dir, "sft_final.pt")
        torch.save(
            {
                "step":   args.max_steps,
                "config": config,
                "model":  raw_model.state_dict(),
            },
            path,
        )
        print(f"\nSFT complete. Saved: {path}")

    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
