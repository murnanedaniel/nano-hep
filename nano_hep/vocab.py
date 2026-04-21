"""Union-vocabulary layout for nano-hep autoregressive model, multi-quantizer
content + global-position (gpos) tokens.

Per-modality vocab = two sub-blocks:
  1. content block:  num_q * codebook_size   tokens (VQ-VAE codebook IDs)
  2. pos block:      num_q_pos * pos_codebook_size  tokens (PosTokenizer IDs)

Per element the AR stream emits (num_q + num_q_pos) tokens, content first:

  <MOD:track_start>
  trk0_c0 trk0_c1 trk0_c2  trk0_p0 trk0_p1 trk0_p2
  trk1_c0 trk1_c1 trk1_c2  trk1_p0 trk1_p1 trk1_p2
  ...
  <MOD:truthpart_start>
  tru0_c0 tru0_c1 tru0_c2  tru0_p0 tru0_p1 tru0_p2
  ...
  <EOS>

Global token IDs layout (per modality m, content-first):
  offsets[m]       + q*codebook_size        → content q ∈ [0, num_q)
  pos_offsets[m]   + q*pos_codebook_size    → pos     q ∈ [0, num_q_pos)

Then mod_starts, eos, pad. See Vocab.build for cursor walk.

Design notes:
  - PosTokenizer in HEP4M is universal (same 3x1024 codebook for every
    modality's eta/cosphi/sinphi). We still allocate separate pos_offsets[m]
    per modality so the single GPT head learns to route pos predictions
    inside each modality's sub-vocab; the model is allowed to share weights
    via weight tying but token IDs remain disjoint.
  - Decoder strict mode: decode_element_triples_with_pos expects exactly
    (num_q + num_q_pos)-token groups. The AR parser is responsible for
    discarding incomplete/out-of-range trailing groups before calling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class Vocab:
    """Immutable vocab description. Construct via Vocab.build()."""

    modalities: List[str]
    codebook_sizes: Dict[str, int]          # per-modality content codebook size
    num_quantizers: Dict[str, int]          # per-modality content num_q
    pos_codebook_sizes: Dict[str, int]      # per-modality pos codebook size
    num_q_pos: Dict[str, int]               # per-modality pos num_q
    offsets: Dict[str, int]                 # first global id of content block
    pos_offsets: Dict[str, int]             # first global id of pos block
    mod_start: Dict[str, int]               # global id of <MOD:{mod}_start>
    eos: int
    pad: int
    total: int

    # Derived
    def content_block_size(self, mod: str) -> int:
        """Number of global-id slots reserved for this modality's CONTENT quantizers."""
        return self.num_quantizers[mod] * self.codebook_sizes[mod]

    def pos_block_size(self, mod: str) -> int:
        """Number of global-id slots reserved for this modality's POS quantizers."""
        return self.num_q_pos[mod] * self.pos_codebook_sizes[mod]

    def modality_block_size(self, mod: str) -> int:
        """Deprecated alias for content_block_size (kept for backwards compat)."""
        return self.content_block_size(mod)

    def element_token_width(self, mod: str) -> int:
        """Tokens emitted per element for this modality (content + pos)."""
        return self.num_quantizers[mod] + self.num_q_pos[mod]

    @staticmethod
    def build(
        modalities: List[str],
        codebook_sizes: Dict[str, int],
        num_quantizers: Dict[str, int],
        pos_codebook_sizes: Dict[str, int] = None,
        num_q_pos: Dict[str, int] = None,
    ) -> "Vocab":
        """`modalities` ordered list (inputs first, then outputs).

        Defaults:
            pos_codebook_sizes = {m: DEFAULT_POS_CODEBOOK_SIZE for m in mods}
            num_q_pos          = {m: DEFAULT_NUM_Q_POS         for m in mods}

        Cursor walk (per modality): content block, then pos block. After all
        modality blocks, mod_start specials, then eos, pad.
        """
        seen = set(); mods_ord = []
        for m in modalities:
            if m not in seen:
                seen.add(m); mods_ord.append(m)
        assert all(m in codebook_sizes and m in num_quantizers for m in mods_ord)

        if pos_codebook_sizes is None:
            pos_codebook_sizes = {m: DEFAULT_POS_CODEBOOK_SIZE for m in mods_ord}
        if num_q_pos is None:
            num_q_pos = {m: DEFAULT_NUM_Q_POS for m in mods_ord}
        assert all(m in pos_codebook_sizes and m in num_q_pos for m in mods_ord), \
            "pos_codebook_sizes and num_q_pos must cover all modalities"

        offsets: Dict[str, int] = {}
        pos_offsets: Dict[str, int] = {}
        cursor = 0
        for m in mods_ord:
            offsets[m] = cursor
            cursor += num_quantizers[m] * codebook_sizes[m]
            pos_offsets[m] = cursor
            cursor += num_q_pos[m] * pos_codebook_sizes[m]
        mod_start = {m: cursor + i for i, m in enumerate(mods_ord)}
        cursor += len(mods_ord)
        eos = cursor; cursor += 1
        pad = cursor; cursor += 1
        return Vocab(
            modalities=list(mods_ord),
            codebook_sizes={m: codebook_sizes[m] for m in mods_ord},
            num_quantizers={m: num_quantizers[m] for m in mods_ord},
            pos_codebook_sizes={m: pos_codebook_sizes[m] for m in mods_ord},
            num_q_pos={m: num_q_pos[m] for m in mods_ord},
            offsets=dict(offsets),
            pos_offsets=dict(pos_offsets),
            mod_start=dict(mod_start),
            eos=eos,
            pad=pad,
            total=cursor,
        )

    # --- encoders ------------------------------------------------------------

    def encode_element_triples(self, mod: str, codes: "np.ndarray") -> "np.ndarray":
        """Content-only encode (legacy): shape (N_elems, num_q_mod) → flat
        (N_elems * num_q_mod,) global token IDs. Used by debug tooling and
        tests that don't involve pos tokens."""
        import numpy as np
        codes = np.asarray(codes, dtype=np.int64)
        assert codes.ndim == 2 and codes.shape[1] == self.num_quantizers[mod], \
            f"expected (N, {self.num_quantizers[mod]}), got {codes.shape}"
        cb = self.codebook_sizes[mod]
        q_offs = np.arange(self.num_quantizers[mod], dtype=np.int64) * cb  # (num_q,)
        shifted = codes + q_offs[None, :]   # (N, num_q)
        flat = shifted.reshape(-1)          # (N * num_q,)
        return flat + self.offsets[mod]

    def encode_element_triples_with_pos(
        self,
        mod: str,
        content_codes: "np.ndarray",
        pos_codes: "np.ndarray",
    ) -> "np.ndarray":
        """Encode (content + pos) per element, content-first.

        Inputs:
          content_codes: (N, num_q_content) local codebook IDs
          pos_codes:     (N, num_q_pos)     local pos codebook IDs
        Output:
          flat (N * (num_q_content + num_q_pos),) global token IDs,
          interleaved as [el0_c0..c{qc-1} el0_p0..p{qp-1} el1_c0..]
        """
        import numpy as np
        c = np.asarray(content_codes, dtype=np.int64)
        p = np.asarray(pos_codes, dtype=np.int64)
        nq_c = self.num_quantizers[mod]
        nq_p = self.num_q_pos[mod]
        cb_c = self.codebook_sizes[mod]
        cb_p = self.pos_codebook_sizes[mod]
        assert c.ndim == 2 and c.shape[1] == nq_c, \
            f"content expected (N, {nq_c}), got {c.shape}"
        assert p.ndim == 2 and p.shape[1] == nq_p, \
            f"pos expected (N, {nq_p}), got {p.shape}"
        assert c.shape[0] == p.shape[0], \
            f"content N={c.shape[0]} mismatches pos N={p.shape[0]}"
        q_offs_c = np.arange(nq_c, dtype=np.int64) * cb_c
        q_offs_p = np.arange(nq_p, dtype=np.int64) * cb_p
        c_shifted = c + q_offs_c[None, :] + self.offsets[mod]         # (N, nq_c)
        p_shifted = p + q_offs_p[None, :] + self.pos_offsets[mod]     # (N, nq_p)
        # concat along axis=1 so per-element we have [c0..c{nq_c-1}, p0..p{nq_p-1}]
        per_elem = np.concatenate([c_shifted, p_shifted], axis=1)      # (N, nq_c+nq_p)
        return per_elem.reshape(-1)

    # --- decoders ------------------------------------------------------------

    def decode_modality_triples(self, mod: str, global_ids: "np.ndarray") -> "np.ndarray":
        """Content-only decode (legacy): flat global IDs → (N_elems, num_q)
        local codebook IDs. Drops out-of-range tokens and truncates to make
        length a multiple of num_q. Used only by debug tooling; the new AR
        parser handles content+pos groups via decode_element_triples_with_pos."""
        import numpy as np
        ids = np.asarray(global_ids, dtype=np.int64) - self.offsets[mod]
        cb = self.codebook_sizes[mod]
        num_q = self.num_quantizers[mod]
        block = num_q * cb
        valid = (ids >= 0) & (ids < block)
        n_keep = int(valid.sum())
        n_elems = n_keep // num_q
        kept = ids[valid][: n_elems * num_q].reshape(n_elems, num_q)
        q_offs = np.arange(num_q, dtype=np.int64) * cb
        locals_ = kept - q_offs[None, :]
        return locals_

    def decode_element_triples_with_pos(
        self, mod: str, global_ids: "np.ndarray"
    ) -> Tuple["np.ndarray", "np.ndarray"]:
        """Strict inverse of encode_element_triples_with_pos.

        Expects `global_ids` length N * (num_q_content + num_q_pos), with
        per-element layout [c0..c{qc-1}, p0..p{qp-1}] and each token in its
        expected range. Violations raise AssertionError — AR parser must
        trim to valid groups before calling.

        Returns (content_local (N, num_q_content), pos_local (N, num_q_pos))
        of local codebook IDs."""
        import numpy as np
        ids = np.asarray(global_ids, dtype=np.int64)
        nq_c = self.num_quantizers[mod]
        nq_p = self.num_q_pos[mod]
        group = nq_c + nq_p
        assert ids.size % group == 0, \
            f"global_ids length {ids.size} not a multiple of group={group}"
        n = ids.size // group
        grouped = ids.reshape(n, group)           # (N, nq_c + nq_p)
        c_block = grouped[:, :nq_c]               # (N, nq_c)
        p_block = grouped[:, nq_c:]               # (N, nq_p)
        cb_c = self.codebook_sizes[mod]
        cb_p = self.pos_codebook_sizes[mod]
        off_c = self.offsets[mod]
        off_p = self.pos_offsets[mod]
        content_lo = off_c; content_hi = off_c + nq_c * cb_c
        pos_lo     = off_p; pos_hi     = off_p + nq_p * cb_p
        assert ((c_block >= content_lo) & (c_block < content_hi)).all(), \
            f"content ids out of [{content_lo},{content_hi}) for mod={mod}"
        assert ((p_block >= pos_lo) & (p_block < pos_hi)).all(), \
            f"pos ids out of [{pos_lo},{pos_hi}) for mod={mod}"
        q_offs_c = np.arange(nq_c, dtype=np.int64) * cb_c
        q_offs_p = np.arange(nq_p, dtype=np.int64) * cb_p
        c_local = c_block - off_c - q_offs_c[None, :]
        p_local = p_block - off_p - q_offs_p[None, :]
        return c_local, p_local

    def decode_global(self, global_ids):
        """Map each global id to (modality_name_or_special, q_index, local_id).
        Special tokens return q_index=-1, local_id as a sentinel."""
        import numpy as np
        ids = np.asarray(global_ids, dtype=np.int64)
        out_mod = np.full(ids.shape, "", dtype=object)
        out_q = np.full(ids.shape, -1, dtype=np.int64)
        out_local = np.full(ids.shape, -999, dtype=np.int64)
        for m in self.modalities:
            # content block
            lo = self.offsets[m]
            hi = lo + self.num_quantizers[m] * self.codebook_sizes[m]
            in_range = (ids >= lo) & (ids < hi)
            local_block = ids[in_range] - lo
            out_mod[in_range] = m
            out_q[in_range] = local_block // self.codebook_sizes[m]
            out_local[in_range] = local_block % self.codebook_sizes[m]
            # pos block
            plo = self.pos_offsets[m]
            phi = plo + self.num_q_pos[m] * self.pos_codebook_sizes[m]
            in_range_p = (ids >= plo) & (ids < phi)
            local_block_p = ids[in_range_p] - plo
            out_mod[in_range_p] = f"{m}_pos"
            out_q[in_range_p] = local_block_p // self.pos_codebook_sizes[m]
            out_local[in_range_p] = local_block_p % self.pos_codebook_sizes[m]
        for m, tid in self.mod_start.items():
            mask = ids == tid
            out_mod[mask] = f"<MOD:{m}_start>"
        out_mod[ids == self.eos] = "<EOS>"
        out_mod[ids == self.pad] = "<PAD>"
        return out_mod, out_q, out_local

    def __repr__(self) -> str:
        parts = []
        for m in self.modalities:
            c_lo = self.offsets[m]
            c_hi = c_lo + self.num_quantizers[m] * self.codebook_sizes[m]
            p_lo = self.pos_offsets[m]
            p_hi = p_lo + self.num_q_pos[m] * self.pos_codebook_sizes[m]
            parts.append(
                f"{m}:content[{c_lo},{c_hi})x{self.num_quantizers[m]}q "
                f"pos[{p_lo},{p_hi})x{self.num_q_pos[m]}q"
            )
        return (f"Vocab(total={self.total}, modalities=[{', '.join(parts)}], "
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
# PosTokenizer (HEP4M/hep4m/models/pos_tokenizer.py) is universal across modalities
# — 3 codebooks × 1024 bins each for (eta, cos_phi, sin_phi). We allocate a
# per-modality pos block in vocab (so the GPT head learns modality-conditioned
# pos prediction) but the underlying codebook semantics are shared.
DEFAULT_POS_CODEBOOK_SIZE: int = 1024
DEFAULT_NUM_Q_POS: int = 3
