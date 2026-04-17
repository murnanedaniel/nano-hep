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
    """Autoregressive dataset over (input_modality → output_modality) sequences.

    Returns (x, y, loss_mask) with `x` / `y` / `loss_mask` shape (block_size,).
    - x: token IDs
    - y = next-token labels (x shifted left by 1, last=PAD)
    - loss_mask = 1 where `y` should be counted (output region only, excluding PAD)
    """

    DEFAULT_ROOT = "/global/cfs/cdirs/m4958/data/COCOA/Tokenized"

    def __init__(
        self,
        tokenized_root: str = DEFAULT_ROOT,
        split: str = "train",
        input_modality: str = "track",
        output_modality: str = "truthpart",
        block_size: int = 256,
        max_events: int = -1,
        vocab: Vocab | None = None,
    ):
        self.split = split
        self.input_modality = input_modality
        self.output_modality = output_modality
        self.block_size = block_size

        self.inp_mm = ModalityMemmap.load(tokenized_root, split, input_modality)
        self.out_mm = ModalityMemmap.load(tokenized_root, split, output_modality)
        assert self.inp_mm.n_events == self.out_mm.n_events, \
            f"mismatch n_events: {self.inp_mm.n_events} vs {self.out_mm.n_events}"

        self.n_events_total = self.inp_mm.n_events
        self.n_events = min(self.n_events_total, max_events) if max_events > 0 else self.n_events_total

        if vocab is None:
            vocab = Vocab.build([input_modality, output_modality],
                                {input_modality: DEFAULT_MODALITY_VOCAB[input_modality],
                                 output_modality: DEFAULT_MODALITY_VOCAB[output_modality]})
        self.vocab = vocab

    def __len__(self) -> int:
        return self.n_events

    # --- sequence building ---------------------------------------------------

    def build_sequence(self, idx: int) -> np.ndarray:
        """Returns a (block_size,) int64 array — PAD on the right."""
        inp_q0 = self.inp_mm.event_q0(idx)
        out_q0 = self.out_mm.event_q0(idx)
        # Clip token IDs to valid range (some saved tokens can be out-of-band sentinels)
        V_in = self.vocab.vocab_sizes[self.input_modality]
        V_out = self.vocab.vocab_sizes[self.output_modality]
        inp_q0 = np.clip(inp_q0, 0, V_in - 1)
        out_q0 = np.clip(out_q0, 0, V_out - 1)
        inp_global = inp_q0 + self.vocab.offsets[self.input_modality]
        out_global = out_q0 + self.vocab.offsets[self.output_modality]
        # Layout: MOD_start[in], inp_global..., MOD_start[out], out_global..., EOS, PAD...
        parts = [
            np.array([self.vocab.mod_start[self.input_modality]], dtype=np.int64),
            inp_global,
            np.array([self.vocab.mod_start[self.output_modality]], dtype=np.int64),
            out_global,
            np.array([self.vocab.eos], dtype=np.int64),
        ]
        seq = np.concatenate(parts)
        if len(seq) > self.block_size:
            # truncate — output region may be partially cut; keep as much as possible
            seq = seq[: self.block_size]
        else:
            pad_len = self.block_size - len(seq)
            seq = np.concatenate([seq, np.full(pad_len, self.vocab.pad, dtype=np.int64)])
        return seq

    def build_loss_mask(self, seq: np.ndarray) -> np.ndarray:
        """Loss mask for y = seq shifted left. A position i contributes loss iff
        y[i] = seq[i+1] is a *real output token or EOS* (not input, not PAD).
        """
        # Find the MOD_start of the OUTPUT modality
        out_start_tok = self.vocab.mod_start[self.output_modality]
        pad_tok = self.vocab.pad
        where_out_start = np.where(seq == out_start_tok)[0]
        loss = np.zeros(self.block_size, dtype=np.int64)
        if len(where_out_start) == 0:
            return loss
        out_start_pos = int(where_out_start[0])
        # y[i] = seq[i+1]; we want y[i] ∈ output-region.
        # Output tokens occupy positions [out_start_pos+1, eos_pos] in seq (inclusive of EOS).
        eos_tok = self.vocab.eos
        where_eos = np.where(seq == eos_tok)[0]
        if len(where_eos) == 0:
            # truncated before EOS; count all positions up to last non-pad
            last_nonpad = int(np.where(seq != pad_tok)[0][-1])
            y_end = last_nonpad  # y[i] = seq[i+1]; can compute y up to i = block_size-2
        else:
            y_end = int(where_eos[0])
        # positions i for which y[i] is a target: out_start_pos <= i < y_end
        # (so y[i] = seq[i+1] covers from out_start_pos+1 to y_end)
        y_start = out_start_pos
        loss[y_start:y_end] = 1
        # sanity: cap at block_size - 1 (no loss at last position since there's no y)
        if loss[-1] == 1:
            loss[-1] = 0
        return loss

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq = self.build_sequence(idx)
        loss_mask = self.build_loss_mask(seq)
        x = torch.from_numpy(seq[:-1]).long()   # (block_size - 1,)
        y = torch.from_numpy(seq[1:]).long()    # (block_size - 1,)
        m = torch.from_numpy(loss_mask[:-1]).long()  # aligned with x → y prediction
        return x, y, m
