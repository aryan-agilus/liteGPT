"""
SFT datasets for liteGPT.

Key difference from pretraining: every sample carries a loss mask.
  mask[i] = 1  → this token position contributes to the loss
  mask[i] = 0  → skip (user tokens, BOS, padding)

Two backends:

  SFTBinDataset   — loads pre-processed .bin files from scripts/prepare_data.py
                    Best for production (memory-mapped, fast).

  SFTHFDataset    — tokenizes HuggingFace conversations on-the-fly.
                    Best for quick experiments without preprocessing.

  SyntheticSFTDataset — fake assistant-only sequences for smoke testing.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from litegpt.tokenizer import Tokenizer, VOCAB_SIZE


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class SyntheticSFTDataset(Dataset):
    """
    Fake SFT data for smoke tests.
    Generates chat-shaped token sequences with a valid assistant mask.
    """

    def __init__(self, seq_len: int = 256, num_samples: int = 256):
        self.seq_len    = seq_len
        self.num_samples = num_samples
        tok = Tokenizer()
        self.usr_s  = tok.usr_s
        self.usr_e  = tok.usr_e
        self.asst_s = tok.asst_s
        self.asst_e = tok.asst_e
        self.bos    = tok.bos_id

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        g = torch.Generator()
        g.manual_seed(idx)

        # Build a tiny fake conversation that fits in seq_len
        # [bos, usr_s, ...user..., usr_e, asst_s, ...assistant..., asst_e]
        user_len = self.seq_len // 4
        asst_len = self.seq_len // 2

        user_toks = torch.randint(100, VOCAB_SIZE - 10, (user_len,), generator=g).tolist()
        asst_toks = torch.randint(100, VOCAB_SIZE - 10, (asst_len,), generator=g).tolist()

        tokens = (
            [self.bos, self.usr_s]
            + user_toks
            + [self.usr_e, self.asst_s]
            + asst_toks
            + [self.asst_e]
        )
        mask = (
            [0, 0]                    # bos, usr_s
            + [0] * user_len          # user body
            + [0, 0]                  # usr_e, asst_s
            + [1] * asst_len          # assistant body ← train here
            + [1]                     # asst_e
        )

        # Truncate / pad to seq_len + 1
        total = self.seq_len + 1
        tokens = tokens[:total]
        mask   = mask[:total]
        pad    = total - len(tokens)
        tokens += [0] * pad
        mask   += [0] * pad

        t = torch.tensor(tokens, dtype=torch.long)
        m = torch.tensor(mask,   dtype=torch.float32)
        return t[:-1], t[1:], m[1:]   # x, y, mask (mask aligns with y)


class SFTBinDataset(Dataset):
    """
    Memory-mapped dataset over pre-processed binary files.

    Expects:
        tokens_path   — uint16 flat array of token ids
        mask_path     — uint8  flat array of loss mask bits

    Created by:  python scripts/prepare_data.py --mode sft --out data/sft
    The files contain padded, fixed-length chunks of (seq_len + 1) tokens each.
    """

    def __init__(self, tokens_path: str, mask_path: str, seq_len: int = 2048):
        self.seq_len = seq_len
        self.stride  = seq_len + 1

        # memmap: OS pages in only what's accessed — all 3 DDP processes share
        # the same physical pages via the page cache instead of each loading a
        # full copy into RAM.  Typical saving: ~9 GB → ~0 GB RSS per process.
        self.tokens = np.memmap(tokens_path, dtype=np.uint16, mode="r")
        self.mask   = np.memmap(mask_path,   dtype=np.uint8,  mode="r")

        assert len(self.tokens) == len(self.mask), (
            f"Token/mask length mismatch: {len(self.tokens)} vs {len(self.mask)}"
        )

        self.n = len(self.tokens) // self.stride

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        start  = idx * self.stride
        # Copy the slice out of the mmap — keeps the window small in RAM
        tokens = self.tokens[start : start + self.stride].astype(np.int64)
        mask   = self.mask  [start : start + self.stride].astype(np.float32)
        t = torch.from_numpy(tokens)
        m = torch.from_numpy(mask)
        return t[:-1], t[1:], m[1:]   # x, y, mask


class SFTHFDataset(Dataset):
    """
    On-the-fly SFT tokenization from a HuggingFace dataset.
    Requires `datasets` package.

    Good for: quick experiments, small datasets, avoiding preprocessing.
    For large-scale training use SFTBinDataset (pre-processed).
    """

    def __init__(
        self,
        hf_slug: str,
        split:   str = "train",
        seq_len: int = 2048,
        max_examples: int | None = None,
        messages_field: str = "messages",
    ):
        from datasets import load_dataset

        self.seq_len = seq_len
        self.tok     = Tokenizer()

        ds = load_dataset(hf_slug, split=split)
        if max_examples:
            ds = ds.select(range(min(max_examples, len(ds))))

        self.examples: list[list[dict]] = []
        field = messages_field

        for ex in ds:
            msgs = ex.get(field, [])
            # Normalise OpenHermes-style {"from": ..., "value": ...}
            if msgs and "from" in msgs[0]:
                msgs = [
                    {
                        "role": "user" if m["from"] in ("human", "system") else "assistant",
                        "content": m.get("value", ""),
                    }
                    for m in msgs
                ]
            if any(m.get("role") == "assistant" for m in msgs):
                self.examples.append(msgs)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from scripts.prepare_data import tokenize_conversation
        tokens, mask = tokenize_conversation(self.examples[idx], self.tok, self.seq_len)
        t = torch.tensor(tokens, dtype=torch.long)
        m = torch.tensor(mask,   dtype=torch.float32)
        return t[:-1], t[1:], m[1:]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_sft_dataloader(
    dataset_type:   str,
    seq_len:        int,
    batch_size:     int,
    rank:           int = 0,
    world_size:     int = 1,
    # SFTBinDataset args
    tokens_path:    str | None = None,
    mask_path:      str | None = None,
    # SFTHFDataset args
    hf_slug:        str | None = None,
    max_examples:   int | None = None,
) -> DataLoader:
    """
    Returns a DataLoader that yields (x, y, mask) triples.

    dataset_type:
        "synthetic"  — no deps, instant, for smoke tests
        "bin"        — pre-processed binary files (production)
        "hf"         — on-the-fly from HuggingFace (quick experiments)
    """
    if dataset_type == "synthetic":
        n = max(256, batch_size * world_size * 20)
        dataset = SyntheticSFTDataset(seq_len=seq_len, num_samples=n)

    elif dataset_type == "bin":
        if not tokens_path or not mask_path:
            raise ValueError("tokens_path and mask_path required for dataset_type='bin'")
        dataset = SFTBinDataset(tokens_path=tokens_path, mask_path=mask_path, seq_len=seq_len)

    elif dataset_type == "hf":
        if not hf_slug:
            raise ValueError("hf_slug required for dataset_type='hf'")
        dataset = SFTHFDataset(
            hf_slug=hf_slug, seq_len=seq_len, max_examples=max_examples
        )

    else:
        raise ValueError(f"Unknown sft dataset_type={dataset_type!r}")

    sampler = None
    shuffle  = True
    if world_size > 1:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
        )
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        pin_memory=torch.cuda.is_available(),
        num_workers=0,
        drop_last=True,
    )
