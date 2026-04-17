"""Union-vocabulary layout for the nano-hep autoregressive model.

Token IDs are disjoint across modalities and across codebooks. v0 uses only
codebook 0 ("q0", the coarsest in HEP4M's residual VQ) for simplicity; later
versions can add q1/q2 by extending the per-modality range.

Layout (v0, single codebook per modality):

  range(0, V_mod0)                                 → modality 0, q0 tokens
  range(V_mod0, V_mod0 + V_mod1)                   → modality 1, q0 tokens
  range(V_mod0 + V_mod1, V_mod0 + V_mod1 + V_mod2) → modality 2, q0 tokens
  ...
  <special tokens>                                  → end of vocab

Special tokens:
  MOD_START[mod] : one per modality, marks the start of that modality's block
                   in a sequence (and is predicted by the model as boundary)
  EOS            : end of the output sequence
  PAD            : ignored by loss (loss_mask = 0) and attention (attn_mask = 0)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class Vocab:
    """Immutable vocab description. Construct via Vocab.build()."""

    modalities: List[str]
    vocab_sizes: Dict[str, int]  # per-modality q0 vocab (typically 1024)
    offsets: Dict[str, int]       # first global id for this modality's q0
    mod_start: Dict[str, int]     # global id of the <MOD:{mod}_start> token
    eos: int
    pad: int
    total: int                    # total vocab size

    @staticmethod
    def build(modalities: List[str], vocab_sizes: Dict[str, int]) -> "Vocab":
        """`modalities` is the ordered list of all modalities that can appear in a sequence
        (inputs first, then outputs). Token IDs are assigned disjointly; a MOD_START
        special is created per modality."""
        assert all(m in vocab_sizes for m in modalities), f"missing vocab_sizes for some of {modalities}"
        # dedupe while preserving order
        seen = set(); mods_ord = []
        for m in modalities:
            if m not in seen:
                seen.add(m); mods_ord.append(m)
        offsets: Dict[str, int] = {}
        cursor = 0
        for m in mods_ord:
            offsets[m] = cursor
            cursor += vocab_sizes[m]
        # specials: one MOD_START per modality, then EOS, then PAD
        mod_start = {m: cursor + i for i, m in enumerate(mods_ord)}
        cursor += len(mods_ord)
        eos = cursor; cursor += 1
        pad = cursor; cursor += 1
        return Vocab(
            modalities=list(mods_ord),
            vocab_sizes={m: vocab_sizes[m] for m in mods_ord},
            offsets=dict(offsets),
            mod_start=dict(mod_start),
            eos=eos,
            pad=pad,
            total=cursor,
        )

    # --- encoders ------------------------------------------------------------

    def encode_modality_tokens(self, mod: str, local_ids):
        """local q0 token IDs (int, 0..V_mod) → global token IDs."""
        import numpy as np
        local = np.asarray(local_ids, dtype=np.int64)
        return local + self.offsets[mod]

    # --- decoders ------------------------------------------------------------

    def decode_global(self, global_ids):
        """Inverse of encode: global IDs → (modality_name or None, local_id).
        Special tokens return (None, -1 for EOS, -2 for PAD, -3 for MOD_START).
        """
        import numpy as np
        ids = np.asarray(global_ids, dtype=np.int64)
        out_mod = np.full(ids.shape, "", dtype=object)
        out_local = np.full(ids.shape, -999, dtype=np.int64)
        for m in self.modalities:
            lo = self.offsets[m]; hi = lo + self.vocab_sizes[m]
            in_range = (ids >= lo) & (ids < hi)
            out_mod[in_range] = m
            out_local[in_range] = ids[in_range] - lo
        # specials
        for m, tid in self.mod_start.items():
            mask = ids == tid
            out_mod[mask] = f"<MOD:{m}_start>"; out_local[mask] = -3
        out_mod[ids == self.eos] = "<EOS>"; out_local[ids == self.eos] = -1
        out_mod[ids == self.pad] = "<PAD>"; out_local[ids == self.pad] = -2
        return out_mod, out_local

    def __repr__(self) -> str:
        ranges = [f"{m}:[{self.offsets[m]},{self.offsets[m]+self.vocab_sizes[m]})" for m in self.modalities]
        return (f"Vocab(total={self.total}, modalities=[{', '.join(ranges)}], "
                f"mod_start={self.mod_start}, eos={self.eos}, pad={self.pad})")


# Default COCOA/Tokenized vocab: codebook size 1024 per modality for the 4 main pflow modalities.
DEFAULT_MODALITY_VOCAB: Dict[str, int] = {
    "track": 1024,
    "topo": 1024,
    "truthpart": 1024,
    "truthjet": 1024,
}
