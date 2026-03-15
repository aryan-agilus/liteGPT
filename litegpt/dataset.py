"""
Data loading for liteGPT training.

Two modes:
  synthetic  — random token ids; instant, zero deps; use for smoke tests
  text       — pre-tokenized binary file (.bin of uint16 token ids)
               produced by scripts/tokenize_text.py

For distributed training, pass rank and world_size to get_dataloader —
it wraps the dataset in a DistributedSampler automatically.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from litegpt.tokenizer import VOCAB_SIZE


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class SyntheticDataset(Dataset):
    """
    Random (x, y) token pairs.  y = x shifted left by 1.
    Purpose: verify model, optimizer, and training loop shapes before
    touching real data.  Should converge to ~log(vocab_size) loss quickly.
    """

    def __init__(
        self,
        seq_len: int = 256,
        num_samples: int = 256,
        vocab_size: int = VOCAB_SIZE,
    ):
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.vocab_size = vocab_size

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Deterministic per index so DDP workers don't duplicate work
        g = torch.Generator()
        g.manual_seed(idx)
        tokens = torch.randint(0, self.vocab_size, (self.seq_len + 1,), generator=g)
        return tokens[:-1], tokens[1:]


class TokenBinDataset(Dataset):
    """
    Memory-mapped dataset over a flat uint16 binary token file.
    Produces non-overlapping (x, y) windows of length seq_len.

    Create with:  scripts/tokenize_text.py --out data/train.bin
    """

    def __init__(self, path: str, seq_len: int = 2048):
        # memmap: shared page cache across DDP processes — no per-process RAM copy
        self.data    = np.memmap(path, dtype=np.uint16, mode="r")
        self.seq_len = seq_len
        self.n       = (len(self.data) - 1) // seq_len

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_len
        chunk = torch.from_numpy(
            self.data[start : start + self.seq_len + 1].astype(np.int64)
        )
        return chunk[:-1], chunk[1:]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_dataloader(
    dataset_type: str,
    seq_len: int,
    batch_size: int,
    rank: int = 0,
    world_size: int = 1,
    data_path: str | None = None,
) -> DataLoader:
    """
    Returns a DataLoader ready for training.

    Args:
        dataset_type:  "synthetic" | "text"
        seq_len:       sequence length (tokens)
        batch_size:    per-device batch size
        rank:          DDP rank (0 for single-GPU / CPU)
        world_size:    total number of processes
        data_path:     required when dataset_type == "text"
    """
    if dataset_type == "synthetic":
        # Make enough samples to fill at least 20 steps per epoch
        n_samples = max(256, batch_size * world_size * 20)
        dataset = SyntheticDataset(seq_len=seq_len, num_samples=n_samples)

    elif dataset_type == "text":
        if data_path is None:
            raise ValueError("data_path required for dataset_type='text'")
        dataset = TokenBinDataset(path=data_path, seq_len=seq_len)

    else:
        raise ValueError(
            f"Unknown dataset_type={dataset_type!r}. "
            "Choose 'synthetic' (smoke test) or 'text' (pre-tokenized bin)."
        )

    sampler = None
    shuffle = True
    if world_size > 1:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
        )
        shuffle = False   # DistributedSampler handles shuffling

    pin_memory = torch.cuda.is_available()

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        pin_memory=pin_memory,
        num_workers=0,   # 0 = safe for MPS; increase for large CUDA training
        drop_last=True,
    )
