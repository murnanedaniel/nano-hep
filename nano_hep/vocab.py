"""Union-vocabulary layout for nano-hep autoregressive model, with multi-quantizer support.

Per-modality vocab block = num_quantizers * codebook_size. Each particle/track
element is represented by a sequence of `num_quantizers` consecutive tokens:

  <MOD:track_start>
  trk_elem0_q0  trk_elem0_q1  trk_elem0_q2
  trk_elem1_q0  trk_elem1_q1  trk_elem1_q2
  ...
  <MOD:truthpart_start>
  tru_elem0_q0  tru_elem0_q1  tru_elem0_q2
  ...
  <EOS>

Global token IDs layout:
  offsets[mod] + 0*codebook_size          → q0 of modality `mod`
  offsets[mod] + 1*codebook_size          → q1 of modality `mod`
  offsets[mod] + 2*codebook_size          → q2 of modality `mod`
  ...
  offsets[mod] + num_q * codebook_size    → start of next modality

The decoder emits triples (for num_q=3). After AR sampling, group every
num_q consecutive tokens into an element, subtract offset, split into
per-quantizer local IDs, and pass to HEP4M's VQ-VAE `indices_to_zq`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class Vocab:
    """Immutable vocab description. Construct via Vocab.build()."""

    modalities: List[str]
    codebook_sizes: Dict[str, int]      # per-modality codebook size (e.g. 1024)
    num_quantizers: Dict[str, int]      # per-modality num_q (e.g. 3)
    offsets: Dict[str, int]             # first global id for modality (q0 start)
    mod_start: Dict[str, int]           # global id of <MOD:{mod}_start>
    eos: int
    pad: int
    total: int

    # Derived
    def modality_block_size(self, mod: str) -> int:
        """Number of global-id slots reserved for this modality (all quantizers)."""
        return self.num_quantizers[mod] * self.codebook_sizes[mod]

    @staticmethod
    def build(
        modalities: List[str],
        codebook_sizes: Dict[str, int],
        num_quantizers: Dict[str, int],
    ) -> "Vocab":
        """`modalities` ordered list (inputs first, then outputs). Token IDs
        are assigned disjointly; a MOD_START special is added per modality."""
        seen = set(); mods_ord = []
        for m in modalities:
            if m not in seen:
                seen.add(m); mods_ord.append(m)
        assert all(m in codebook_sizes and m in num_quantizers for m in mods_ord)

        offsets: Dict[str, int] = {}
        cursor = 0
        for m in mods_ord:
            offsets[m] = cursor
            cursor += num_quantizers[m] * codebook_sizes[m]
        mod_start = {m: cursor + i for i, m in enumerate(mods_ord)}
        cursor += len(mods_ord)
        eos = cursor; cursor += 1
        pad = cursor; cursor += 1
        return Vocab(
            modalities=list(mods_ord),
            codebook_sizes={m: codebook_sizes[m] for m in mods_ord},
            num_quantizers={m: num_quantizers[m] for m in mods_ord},
            offsets=dict(offsets),
            mod_start=dict(mod_start),
            eos=eos,
            pad=pad,
            total=cursor,
        )

    # --- encoders ------------------------------------------------------------

    def encode_element_triples(self, mod: str, codes: "np.ndarray") -> "np.ndarray":
        """Encode `codes` of shape (N_elems, num_q_mod) → flat (N_elems * num_q_mod,)
        global token IDs in interleaved order (el0_q0, el0_q1, ..., el1_q0, ...)."""
        import numpy as np
        codes = np.asarray(codes, dtype=np.int64)
        assert codes.ndim == 2 and codes.shape[1] == self.num_quantizers[mod], \
            f"expected (N, {self.num_quantizers[mod]}), got {codes.shape}"
        cb = self.codebook_sizes[mod]
        # per-quantizer offset within modality block
        q_offs = np.arange(self.num_quantizers[mod], dtype=np.int64) * cb  # (num_q,)
        shifted = codes + q_offs[None, :]   # (N, num_q)
        flat = shifted.reshape(-1)          # (N * num_q,)
        return flat + self.offsets[mod]

    # --- decoders ------------------------------------------------------------

    def decode_modality_triples(self, mod: str, global_ids: "np.ndarray") -> "np.ndarray":
        """Inverse of encode_element_triples: flat global IDs → (N_elems, num_q)
        local codebook IDs. Expects `global_ids` to be a multiple of num_q long
        and all within modality's range; invalid IDs produce -1."""
        import numpy as np
        ids = np.asarray(global_ids, dtype=np.int64) - self.offsets[mod]
        cb = self.codebook_sizes[mod]
        num_q = self.num_quantizers[mod]
        block = num_q * cb
        # keep only ids within modality block
        valid = (ids >= 0) & (ids < block)
        # group into triples
        n_keep = valid.sum()
        n_elems = n_keep // num_q
        kept = ids[valid][: n_elems * num_q].reshape(n_elems, num_q)
        # each column i should have values in [i*cb, (i+1)*cb); subtract that
        q_offs = np.arange(num_q, dtype=np.int64) * cb
        locals_ = kept - q_offs[None, :]
        return locals_

    def decode_global(self, global_ids):
        """Map each global id to (modality_name_or_special, q_index, local_id).
        Special tokens return q_index=-1, local_id as a sentinel."""
        import numpy as np
        ids = np.asarray(global_ids, dtype=np.int64)
        out_mod = np.full(ids.shape, "", dtype=object)
        out_q = np.full(ids.shape, -1, dtype=np.int64)
        out_local = np.full(ids.shape, -999, dtype=np.int64)
        for m in self.modalities:
            lo = self.offsets[m]
            hi = lo + self.num_quantizers[m] * self.codebook_sizes[m]
            in_range = (ids >= lo) & (ids < hi)
            local_block = ids[in_range] - lo
            out_mod[in_range] = m
            out_q[in_range] = local_block // self.codebook_sizes[m]
            out_local[in_range] = local_block % self.codebook_sizes[m]
        for m, tid in self.mod_start.items():
            mask = ids == tid
            out_mod[mask] = f"<MOD:{m}_start>"
        out_mod[ids == self.eos] = "<EOS>"
        out_mod[ids == self.pad] = "<PAD>"
        return out_mod, out_q, out_local

    def __repr__(self) -> str:
        ranges = [f"{m}:[{self.offsets[m]},{self.offsets[m] + self.num_quantizers[m]*self.codebook_sizes[m]})x{self.num_quantizers[m]}q"
                  for m in self.modalities]
        return (f"Vocab(total={self.total}, modalities=[{', '.join(ranges)}], "
                f"mod_start={self.mod_start}, eos={self.eos}, pad={self.pad})")


# Defaults match the COCOA VQ-VAE checkpoints at
# /global/cfs/cdirs/m4958/data/COCOA/Checkpoints/vqx{track,topo,trpart,trjet}_.../config_m.yml
DEFAULT_MODALITY_CODEBOOK_SIZE: Dict[str, int] = {
    "track": 256,
    "topo": 256,
    "truthpart": 128,
    "truthjet": 128,
}
DEFAULT_MODALITY_NUM_QUANTIZERS: Dict[str, int] = {
    "track": 3, "topo": 3, "truthpart": 3, "truthjet": 1,
}
