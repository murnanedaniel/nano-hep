"""HEP4M COCOA/Tokenized → flat autoregressive sequences for nano-hep.

v0 scope:
  - inputs: single modality (default 'track'), q0 only (coarsest codebook)
  - outputs: single modality (default 'truthpart'), q0 only
  - sequence layout (PAD on the right):
        <MOD:track_start> q0_track[0..N_trk-1]
        <MOD:truthpart_start> q0_truth[0..N_truth-1]
        <EOS> <PAD>…

The dataset reads the memmap files at TOKENIZED_ROOT/{split}/{mod}_{data,offsets,meta}.npy
directly — no hep4m import required for v0. Each element per modality is a fixed
row of shape (n_codebooks + n_pos_codebooks,) int16; we pull column 0 (the q0 id).

Loss mask is 1 only for *output* positions — shifted by one to predict the
next output token from the previous one (standard AR loss on y = x[1:]).

Usage
-----
    ds = HEPDataset(
        tokenized_root="/global/cfs/cdirs/m4958/data/COCOA/Tokenized",
        split="train",
        block_size=256,
        input_modality="track",
        output_modality="truthpart",
    )
    x, y, loss_mask = ds[0]  # all shape (block_size,)

`x` is the input token stream; `y = x` shifted by one (same length, last = PAD);
`loss_mask` marks positions where `y` should be predicted by the model.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .vocab import Vocab, DEFAULT_MODALITY_VOCAB


@dataclass
class ModalityMemmap:
    """Lazy memmap view of one modality in one split."""
    name: str
    n_codebooks: int
    n_pos_codebooks: int
    n_events: int
    offsets: np.memmap     # (n_events + 1,) uint64, element-count cumulative
    data: np.memmap        # (total_elements, n_codebooks + n_pos_codebooks) int16

    @classmethod
    def load(cls, root: str, split: str, modality: str) -> "ModalityMemmap":
        base = Path(root) / split
        meta = np.load(base / f"{modality}_meta.npz", allow_pickle=True)
        n_events = int(meta["n_events"])
        n_cb = int(meta["n_codebooks"])
        n_pos_cb = int(meta["n_pos_codebooks"]) if "n_pos_codebooks" in meta.files else 0
        width = n_cb + n_pos_cb
        # offsets: uint64, size n_events+1
        offsets = np.memmap(base / f"{modality}_offsets.npy", dtype=np.uint64, mode="r",
                            shape=(n_events + 1,))
        total_elems = int(offsets[-1])
        # data: int16, shape (total_elems, width) — stored row-major
        data = np.memmap(base / f"{modality}_data.npy", dtype=np.int16, mode="r",
                         shape=(total_elems, width))
        return cls(name=modality, n_codebooks=n_cb, n_pos_codebooks=n_pos_cb,
                   n_events=n_events, offsets=offsets, data=data)

    def event_q0(self, idx: int) -> np.ndarray:
        """Return the q0 (first codebook) token IDs for event `idx`, shape (n_elems,) int64."""
        a = int(self.offsets[idx]); b = int(self.offsets[idx + 1])
        if b == a:
            return np.empty((0,), dtype=np.int64)
        col0 = self.data[a:b, 0].astype(np.int64)
        return col0


class HEPDataset(Dataset):
    """Autoregressive dataset over (inputs → output) sequences.

    inputs is a list of modality names fed in order; output is a single modality.

    Returns (x, y, loss_mask) shape (block_size - 1,). Loss mask = 1 for positions
    where y should be counted (output-region targets only).
    """

    DEFAULT_ROOT = "/global/cfs/cdirs/m4958/data/COCOA/Tokenized"

    def __init__(
        self,
        tokenized_root: str = DEFAULT_ROOT,
        split: str = "train",
        input_modalities: list = ["track"],
        output_modality: str = "truthpart",
        block_size: int = 256,
        max_events: int = -1,
        vocab: Vocab | None = None,
    ):
        self.split = split
        self.input_modalities = list(input_modalities)
        self.output_modality = output_modality
        self.block_size = block_size

        self.inp_mms = {m: ModalityMemmap.load(tokenized_root, split, m) for m in self.input_modalities}
        self.out_mm = ModalityMemmap.load(tokenized_root, split, output_modality)
        n_events_all = [self.out_mm.n_events] + [mm.n_events for mm in self.inp_mms.values()]
        assert len(set(n_events_all)) == 1, f"n_events mismatch across modalities: {n_events_all}"

        self.n_events_total = self.out_mm.n_events
        self.n_events = min(self.n_events_total, max_events) if max_events > 0 else self.n_events_total

        if vocab is None:
            all_mods = self.input_modalities + [output_modality]
            vocab = Vocab.build(all_mods, {m: DEFAULT_MODALITY_VOCAB[m] for m in all_mods})
        self.vocab = vocab

    def __len__(self) -> int:
        return self.n_events

    # --- sequence building ---------------------------------------------------

    def build_sequence(self, idx: int) -> np.ndarray:
        """Returns a (block_size,) int64 array — PAD on the right."""
        parts = []
        for m in self.input_modalities:
            q0 = self.inp_mms[m].event_q0(idx)
            V = self.vocab.vocab_sizes[m]
            q0 = np.clip(q0, 0, V - 1)
            parts.append(np.array([self.vocab.mod_start[m]], dtype=np.int64))
            parts.append(q0 + self.vocab.offsets[m])
        # Output block
        out_q0 = self.out_mm.event_q0(idx)
        V_out = self.vocab.vocab_sizes[self.output_modality]
        out_q0 = np.clip(out_q0, 0, V_out - 1)
        parts.append(np.array([self.vocab.mod_start[self.output_modality]], dtype=np.int64))
        parts.append(out_q0 + self.vocab.offsets[self.output_modality])
        parts.append(np.array([self.vocab.eos], dtype=np.int64))

        seq = np.concatenate(parts)
        if len(seq) > self.block_size:
            seq = seq[: self.block_size]
        else:
            pad_len = self.block_size - len(seq)
            seq = np.concatenate([seq, np.full(pad_len, self.vocab.pad, dtype=np.int64)])
        return seq

    def build_loss_mask(self, seq: np.ndarray) -> np.ndarray:
        """Loss mask for y = seq shifted left. 1 iff y[i] is in the output region
        (from output's MOD_START through EOS, inclusive)."""
        out_start_tok = self.vocab.mod_start[self.output_modality]
        pad_tok = self.vocab.pad
        eos_tok = self.vocab.eos
        where_out_start = np.where(seq == out_start_tok)[0]
        loss = np.zeros(self.block_size, dtype=np.int64)
        if len(where_out_start) == 0:
            return loss
        out_start_pos = int(where_out_start[0])
        where_eos = np.where(seq == eos_tok)[0]
        if len(where_eos) == 0:
            last_nonpad = int(np.where(seq != pad_tok)[0][-1])
            y_end = last_nonpad
        else:
            y_end = int(where_eos[0])
        y_start = out_start_pos
        loss[y_start:y_end] = 1
        if loss[-1] == 1:
            loss[-1] = 0
        return loss

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq = self.build_sequence(idx)
        loss_mask = self.build_loss_mask(seq)
        x = torch.from_numpy(seq[:-1]).long()
        y = torch.from_numpy(seq[1:]).long()
        m = torch.from_numpy(loss_mask[:-1]).long()
        return x, y, m

    # --- metric helpers (used by train_hep.py val) ---------------------------

    def output_region_slice(self, seq: np.ndarray):
        """Return (output_start_pos, eos_pos) for seq. eos_pos = block_size if none."""
        out_start = int(np.where(seq == self.vocab.mod_start[self.output_modality])[0][0])
        eos_hits = np.where(seq == self.vocab.eos)[0]
        eos_pos = int(eos_hits[0]) if len(eos_hits) else int(self.block_size - 1)
        return out_start, eos_pos

    def true_cardinality(self, idx: int) -> int:
        """Number of real output-modality tokens (elements) for event idx."""
        a = int(self.out_mm.offsets[idx]); b = int(self.out_mm.offsets[idx + 1])
        return b - a
