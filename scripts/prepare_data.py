"""
Download and tokenize datasets for liteGPT training.

Pretrain (plain text → flat token binary):
    python scripts/prepare_data.py --mode pretrain \
        --dataset tinystories --out data/pretrain.bin

SFT (chat conversations → token + mask binaries):
    python scripts/prepare_data.py --mode sft \
        --dataset smoltalk --out data/sft

    Writes two files:
        data/sft_tokens.bin  — uint16 token ids
        data/sft_mask.bin    — uint8  loss mask (1 = train on this token)

Supported datasets:
    pretrain: tinystories, text (provide --text_file path/to/file.txt)
    sft:      smoltalk, openhermes (HuggingFace hub slugs also accepted)
"""

import argparse
import os
import sys
import numpy as np

# ---- lazy imports so the script fails fast with a clear message ----
def require(pkg: str):
    try:
        __import__(pkg)
    except ImportError:
        print(f"[error] '{pkg}' not installed. Run: uv pip install {pkg}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_uint16_bin(path: str, tokens: list[int]):
    arr = np.array(tokens, dtype=np.uint16)
    arr.tofile(path)
    mb = arr.nbytes / 1e6
    print(f"  wrote {len(arr):,} tokens  ({mb:.1f} MB) → {path}")


def write_uint8_bin(path: str, mask: list[int]):
    arr = np.array(mask, dtype=np.uint8)
    arr.tofile(path)
    print(f"  wrote mask ({arr.sum():,} trainable tokens) → {path}")


# ---------------------------------------------------------------------------
# Pretrain tokenization
# ---------------------------------------------------------------------------

def prepare_pretrain(args):
    require("datasets")
    from datasets import load_dataset
    from litegpt.tokenizer import Tokenizer

    tok = Tokenizer()
    all_tokens: list[int] = []

    if args.dataset == "tinystories":
        print("Downloading roneneldan/TinyStories …")
        ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
        for i, ex in enumerate(ds):
            if args.max_examples and i >= args.max_examples:
                break
            text = ex["text"].strip()
            if text:
                all_tokens.append(tok.bos_id)
                all_tokens.extend(tok.encode(text))
            if (i + 1) % 10_000 == 0:
                print(f"  {i+1:,} stories | {len(all_tokens):,} tokens")

    elif args.dataset == "text":
        if not args.text_file:
            print("[error] --text_file required for --dataset text")
            sys.exit(1)
        print(f"Tokenizing {args.text_file} …")
        with open(args.text_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    all_tokens.append(tok.bos_id)
                    all_tokens.extend(tok.encode(line))

    else:
        # Treat as HuggingFace dataset slug; assumes it has a "text" column
        print(f"Downloading {args.dataset} …")
        ds = load_dataset(args.dataset, split="train", streaming=True)
        for i, ex in enumerate(ds):
            if args.max_examples and i >= args.max_examples:
                break
            text = ex.get("text", ex.get("content", "")).strip()
            if text:
                all_tokens.append(tok.bos_id)
                all_tokens.extend(tok.encode(text))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    write_uint16_bin(args.out, all_tokens)


# ---------------------------------------------------------------------------
# SFT tokenization
# ---------------------------------------------------------------------------

SMOLTALK_SLUG = "HuggingFaceTB/smoltalk"
OPENHERMES_SLUG = "teknium/OpenHermes-2.5"


def tokenize_conversation(
    messages: list[dict],
    tok,
    seq_len: int,
) -> tuple[list[int], list[int]]:
    """
    Returns (tokens, mask) for a single conversation, truncated to seq_len+1.
    mask[i] = 1 means token[i] is an assistant token → include in loss.

    Chat format:
        <|bos|>
        <|user_start|> ... <|user_end|>
        <|assistant_start|> ... <|assistant_end|>
        ...
    """
    tokens: list[int] = [tok.bos_id]
    mask:   list[int] = [0]           # BOS not trained

    for msg in messages:
        role    = msg.get("role", "")
        content = msg.get("content", "").strip()

        if role == "user" or role == "system":
            # system prompt treated as user turn (we don't have a system token)
            toks = [tok.usr_s] + tok.encode(content) + [tok.usr_e]
            tokens.extend(toks)
            mask.extend([0] * len(toks))   # user tokens: no loss

        elif role == "assistant":
            prefix = [tok.asst_s]
            body   = tok.encode(content)
            suffix = [tok.asst_e]

            tokens.extend(prefix)
            mask.extend([0] * len(prefix))          # <|assistant_start|>: no loss

            tokens.extend(body)
            mask.extend([1] * len(body))            # response body: train here

            tokens.extend(suffix)
            mask.extend([1] * len(suffix))          # <|assistant_end|>: train here

    # Truncate to seq_len+1 (we'll produce x=[:seq_len], y=[1:seq_len+1])
    tokens = tokens[: seq_len + 1]
    mask   = mask[: seq_len + 1]

    # Right-pad if short (pad token = 0; padded positions have mask=0)
    pad_len = (seq_len + 1) - len(tokens)
    tokens += [0] * pad_len
    mask   += [0] * pad_len

    return tokens, mask


def prepare_sft(args):
    require("datasets")
    from datasets import load_dataset
    from litegpt.tokenizer import Tokenizer

    tok = Tokenizer()

    if args.dataset == "smoltalk":
        slug  = SMOLTALK_SLUG
        field = "messages"
    elif args.dataset == "openhermes":
        slug  = OPENHERMES_SLUG
        field = "conversations"
    else:
        slug  = args.dataset
        field = "messages"

    print(f"Downloading {slug} …")
    ds = load_dataset(slug, split="train", streaming=True)

    all_tokens: list[int] = []
    all_masks:  list[int] = []
    skipped = 0
    n = 0

    for ex in ds:
        if args.max_examples and n >= args.max_examples:
            break

        raw_msgs = ex.get(field, [])

        # Normalise OpenHermes format: {"from": "human"/"gpt", "value": ...}
        if raw_msgs and "from" in raw_msgs[0]:
            msgs = []
            for m in raw_msgs:
                role = "user" if m["from"] in ("human", "system") else "assistant"
                msgs.append({"role": role, "content": m.get("value", "")})
        else:
            msgs = raw_msgs

        if not msgs:
            skipped += 1
            continue

        # Skip conversations with no assistant turn (nothing to train on)
        if not any(m.get("role") == "assistant" for m in msgs):
            skipped += 1
            continue

        tokens, mask = tokenize_conversation(msgs, tok, seq_len=args.seq_len)
        all_tokens.extend(tokens)
        all_masks.extend(mask)
        n += 1

        if n % 10_000 == 0:
            print(f"  {n:,} conversations | {len(all_tokens):,} tokens")

    print(f"\n  processed {n:,} conversations (skipped {skipped:,})")

    os.makedirs(args.out, exist_ok=True)
    tok_path  = os.path.join(args.out, "sft_tokens.bin")
    mask_path = os.path.join(args.out, "sft_mask.bin")
    write_uint16_bin(tok_path, all_tokens)
    write_uint8_bin(mask_path, all_masks)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prepare training data for liteGPT")

    parser.add_argument("--mode",         required=True,  choices=["pretrain", "sft"])
    parser.add_argument("--dataset",      default=None,
                        help="Dataset name. pretrain: tinystories|text. sft: smoltalk|openhermes")
    parser.add_argument("--out",          required=True,
                        help="Output path: file for pretrain, directory for sft")
    parser.add_argument("--seq_len",      type=int, default=2048,
                        help="Max sequence length (sft mode only)")
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Cap number of examples (useful for quick tests)")
    parser.add_argument("--text_file",    type=str, default=None,
                        help="Path to plain text file (--dataset text only)")

    args = parser.parse_args()

    # Defaults per mode
    if args.dataset is None:
        args.dataset = "tinystories" if args.mode == "pretrain" else "smoltalk"

    print(f"\n[prepare_data] mode={args.mode}  dataset={args.dataset}\n")

    if args.mode == "pretrain":
        prepare_pretrain(args)
    else:
        prepare_sft(args)

    print("\nDone.")


if __name__ == "__main__":
    main()
