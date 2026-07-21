"""Batched autoregressive decode for nano-hep.

Drop-in companion to ``notebooks/token_temperature_scan.py:run_one_T``. The
per-event reference loop is sequential (~165 ms/event on a dedicated A100)
because each event runs its own sequence of forward passes; this module
groups events sharing a prefix length into one batched ``model(cur)`` call
per AR step, which gives a ~1.7× wall-clock speedup at N=256 with no
correctness drift at T=1 argmax.

Strategy choice: **prefix-length bucketing** beats right-padded full-batch.

A right-padded full-batch decode (every event padded to the longest prefix
in the chunk) has to pay O(T_max²) attention per row — and with COCOA single-
jet val showing prefix lengths from 9 to 261 the wasted compute on PAD
positions dominates any kernel-launch savings from the larger batch. Measured
on a dedicated A100-PCIE-40GB at N=256 (T=1 argmax, free mode):

    PER-EVENT          : 42.40 s  (165.6 ms/event)
    BUCKETED (per len) : 24.81 s  ( 96.9 ms/event)  speedup 1.7x
    FULL-BATCH B=32    : 47.70 s  (186.3 ms/event)  speedup 0.9x  (regression)
    FULL-BATCH B=64    : 54.04 s  (211.1 ms/event)  speedup 0.8x
    FULL-BATCH B=128   : 62.95 s  (245.9 ms/event)  speedup 0.7x
    FULL-BATCH B=256   : 72.71 s  (284.0 ms/event)  speedup 0.6x

Within a bucket all prefixes have the same length, so no padding is needed
and ``model(cur)`` returns the right-position logits for every row in the
inference fast-path. The function ``batched_ar_decode_full`` is retained for
distributions where prefix lengths cluster tightly (rare here) and as a
correctness-equivalent reference.

Why right-padding works under causal attention: a query at position ``i``
attends only to keys at positions ``[0, i]``. The transformer block produces
hidden states at every position. We pull logits per row at
``prefix_len[i] - 1``; positions ≥ prefix_len contain pad tokens but are
never read for sampling — they only ever serve as keys for future positions
within the SAME row, which we equally never read until that row's write
cursor advances past them.

Why we don't add an attention mask: nanoGPT's flash-attention path
(``F.scaled_dot_product_attention(..., is_causal=True)``) is the fast path;
adding a custom ``attn_mask`` would force a slower fallback. With strict
right-padding + per-row gather, the math is correct without any mask.

Why we run the model body manually rather than calling ``model(idx)``:
``GPT.forward`` returns logits at ONLY the last position when ``targets``
is None (an inference optimization). For batched right-padded decode we
need logits at per-row positions, not just the last one. The body
(``wte + wpe + drop + h_blocks + ln_f``) is a stable surface across
nanoGPT's nano_hep training runs, so calling it directly is safe.

Modes (``nhead_mode``):
  - free   : plain EOS termination. Default.
  - floor  : N-head asserts a cardinality floor (mask EOS until 6*N_hat
             tokens emitted, then allow EOS). Requires GPTWithNHead.
  - force  : exactly 6*N_hat tokens emitted, EOS suppressed throughout.
             Requires GPTWithNHead.
"""
from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch  # noqa  (defaultdict used by run_one_T_batched bucketing)


# ---------------------------------------------------------------------------
# Manual body call: (B, T) tokens → hidden states (B, T, n_embd)
# ---------------------------------------------------------------------------
def _gpt_body(model, idx: torch.Tensor) -> torch.Tensor:
    """Run nanoGPT's transformer body, returning hidden states at every
    position. Mirrors the inference path of ``GPT.forward`` minus the
    last-position lm_head shortcut."""
    device = idx.device
    b, t = idx.size()
    assert t <= model.config.block_size, (
        f"seq len {t} > block_size {model.config.block_size}"
    )
    pos = torch.arange(0, t, dtype=torch.long, device=device)
    tok_emb = model.transformer.wte(idx)
    pos_emb = model.transformer.wpe(pos)
    x = model.transformer.drop(tok_emb + pos_emb)
    for block in model.transformer.h:
        x, _ = block(x)
    x = model.transformer.ln_f(x)            # (B, T, n_embd)
    return x


# ---------------------------------------------------------------------------
# Full-batch right-padded decode
# ---------------------------------------------------------------------------
@torch.no_grad()
def batched_ar_decode_full(
    model,
    vocab,
    out_mod: str,
    prefixes: List[np.ndarray],     # length B; each (L_i,) int64
    *,
    T: float,
    max_new_tokens: int,
    device: torch.device | str,
    argmax_at_T1: bool = True,
    nhead_mode: str = "free",
    n_hat: Optional[torch.Tensor] = None,    # (B,) long; required for floor/force
):
    """Decode B events in one batched pass.

    All events are right-padded to the longest prefix. Per AR step, logits
    are gathered per row at each event's current write position, sampled,
    written into its row, and the per-row write cursor advances.

    Returns
    -------
    emitted_per_event: list of np.ndarray length B; each row is the
        emitted-token sequence for that event (length up to
        ``max_new_tokens``; truncated where the row terminated). EOS / pad
        not included in the returned arrays.
    """
    assert nhead_mode in ("free", "floor", "force")
    if isinstance(device, str):
        device = torch.device(device)
    B = len(prefixes)
    if B == 0:
        return []

    prefix_lens = [len(p) for p in prefixes]
    max_pre = max(prefix_lens)

    # Token vocab edges for slot-aware mask
    nq_c = vocab.num_quantizers[out_mod]
    nq_p = vocab.num_q_pos[out_mod]
    GROUP = nq_c + nq_p
    V_c = vocab.codebook_sizes[out_mod]
    V_p = vocab.pos_codebook_sizes[out_mod]
    content_lo = vocab.offsets[out_mod]
    pos_lo = vocab.pos_offsets[out_mod]
    eos_id = vocab.eos
    pad_id = vocab.pad

    use_nhead = nhead_mode != "free"
    if use_nhead:
        assert n_hat is not None and n_hat.shape == (B,), (
            "n_hat (B,) is required for floor/force"
        )
        n_hat_tokens = (GROUP * n_hat).to(device)
        V = vocab.total
        slot_allow = torch.zeros(GROUP, V, dtype=torch.bool, device=device)
        for s in range(nq_c):
            lo = content_lo + s * V_c
            slot_allow[s, lo: lo + V_c] = True
        for s in range(nq_p):
            lo = pos_lo + s * V_p
            slot_allow[nq_c + s, lo: lo + V_p] = True
    else:
        n_hat_tokens = None
        slot_allow = None

    # Build the (B, T) tensor: prefixes right-padded with PAD up to max_pre
    # plus space for max_new_tokens of decoding.
    T_total = min(max_pre + max_new_tokens, model.config.block_size)
    max_new_tokens = T_total - max_pre
    if max_new_tokens <= 0:
        return [np.zeros(0, dtype=np.int64) for _ in range(B)]

    cur = torch.full((B, T_total), pad_id, dtype=torch.long, device=device)
    for i, p in enumerate(prefixes):
        cur[i, : len(p)] = torch.from_numpy(p).to(device)

    write_pos = torch.tensor(prefix_lens, dtype=torch.long, device=device)  # (B,)
    done = torch.zeros(B, dtype=torch.bool, device=device)
    n_emitted = torch.zeros(B, dtype=torch.long, device=device)
    use_argmax = argmax_at_T1 and float(T) == 1.0

    arange_B = torch.arange(B, device=device)

    # Track emitted history per-row by reading back from cur after the loop.
    emit_starts = torch.tensor(prefix_lens, dtype=torch.long, device=device)
    final_write_pos = emit_starts.clone()  # rows that never decoded keep this

    for step in range(max_new_tokens):
        # In "force" mode, terminate row when its quota is reached BEFORE sampling.
        if use_nhead and nhead_mode == "force":
            done = done | (n_emitted >= n_hat_tokens)
            if bool(done.all().item()):
                break

        # Active rows; all rows must run forward together because GPT has no
        # per-row attention mask. Done rows stay frozen (next_tok forced to PAD).
        if bool(done.all().item()):
            break

        # Forward up to current max write_pos (any column beyond is unused this step).
        T_used = int(write_pos.max().item())
        if T_used <= 0:
            break
        h = _gpt_body(model, cur[:, :T_used])         # (B, T_used, n_embd)
        # Per-row hidden state at write_pos - 1
        gather_idx = (write_pos - 1).clamp(min=0)
        h_at = h[arange_B, gather_idx]                 # (B, n_embd)
        last = model.lm_head(h_at)                     # (B, V)

        # Slot-aware mask for floor/force
        if use_nhead:
            slot_in_group = (n_emitted % GROUP).clamp_min(0)
            allow = slot_allow[slot_in_group]                # (B, V)
            if nhead_mode == "force":
                restrict = torch.ones(B, dtype=torch.bool, device=device)
            else:
                restrict = n_emitted < n_hat_tokens
            mask = restrict.unsqueeze(-1) & (~allow)         # (B, V)
            last = last.masked_fill(mask, float("-inf"))

        if use_argmax:
            next_tok = last.argmax(dim=-1)                   # (B,)
        else:
            probs = torch.softmax(last / float(T), dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1).squeeze(-1)

        # Force PAD on already-done rows.
        next_tok = torch.where(done, torch.full_like(next_tok, pad_id), next_tok)

        # Update done after this emission.
        if nhead_mode == "force":
            new_done = (n_emitted + 1) >= n_hat_tokens
            done = done | new_done
        else:
            done = done | (next_tok == eos_id)

        # Write next_tok at each row's write_pos (rows that haven't yet hit
        # their column budget). Rows already done overwrite a PAD slot with
        # PAD — harmless.
        write_idx = write_pos.clamp(max=T_total - 1)
        cur[arange_B, write_idx] = next_tok
        # Advance write_pos for rows that emitted a real (non-pad) token,
        # otherwise leave it.
        is_real = (next_tok != pad_id)
        n_emitted = n_emitted + is_real.long()
        write_pos = write_pos + is_real.long()
        final_write_pos = torch.where(is_real, write_pos, final_write_pos)

        if bool(done.all().item()):
            break

    cur_np = cur.detach().cpu().numpy()
    starts = emit_starts.detach().cpu().numpy()
    final_ends = final_write_pos.detach().cpu().numpy()
    out: List[np.ndarray] = []
    for i in range(B):
        out.append(cur_np[i, starts[i]: final_ends[i]].astype(np.int64))
    return out


# ---------------------------------------------------------------------------
# KV-cached bucketed decode (the real speed win)
# ---------------------------------------------------------------------------
@torch.no_grad()
def batched_ar_decode_bucket_kv(
    model, vocab, out_mod: str, prefixes: torch.Tensor, *,
    T: float, max_new_tokens: int,
    argmax_at_T1: bool = True, nhead_mode: str = "free",
    n_hat: Optional[torch.Tensor] = None,
):
    """KV-cached, bucketed AR decode. Drop-in for ``batched_ar_decode_bucket``
    but reuses cached (k, v) per layer across AR steps.

    Each step's attention drops from O(t² · d · L) to O(t · d · L), which on
    nano_hep_89M (n_layer=12, n_embd=768, t≈100) is the difference between
    ~100 ms/event and ~2 ms/event. Combined with bucketed batching, total
    expected throughput jump is ~50-100× over per-event reference.

    Requires the vendored model.py + model_nhead.py (which thread ``past_kv``
    through ``CausalSelfAttention`` / ``Block`` / ``GPT.forward``). Backwards
    incompatible with stock nanoGPT.
    """
    assert nhead_mode in ("free", "floor", "force")
    device = prefixes.device
    B, T_prefix = prefixes.shape

    nq_c = vocab.num_quantizers[out_mod]
    nq_p = vocab.num_q_pos[out_mod]
    GROUP = nq_c + nq_p
    V_c = vocab.codebook_sizes[out_mod]
    V_p = vocab.pos_codebook_sizes[out_mod]
    content_lo = vocab.offsets[out_mod]
    pos_lo = vocab.pos_offsets[out_mod]

    use_nhead = nhead_mode != "free"
    if use_nhead:
        assert n_hat is not None and n_hat.shape == (B,)
        n_hat_tokens = (GROUP * n_hat).to(device)
        V = vocab.total
        slot_allow = torch.zeros(GROUP, V, dtype=torch.bool, device=device)
        for s in range(nq_c):
            lo = content_lo + s * V_c
            slot_allow[s, lo: lo + V_c] = True
        for s in range(nq_p):
            lo = pos_lo + s * V_p
            slot_allow[nq_c + s, lo: lo + V_p] = True
    else:
        n_hat_tokens = None
        slot_allow = None

    block_size = model.config.block_size
    max_new_tokens = min(max_new_tokens, block_size - T_prefix)
    if max_new_tokens <= 0:
        return (
            torch.empty((B, 0), dtype=torch.long, device=device),
            torch.zeros(B, dtype=torch.long, device=device),
        )

    pad_id = vocab.pad
    eos_id = vocab.eos
    use_argmax = argmax_at_T1 and float(T) == 1.0

    done = torch.zeros(B, dtype=torch.bool, device=device)
    n_emitted = torch.zeros(B, dtype=torch.long, device=device)
    emitted_steps: List[torch.Tensor] = []

    # ----- Step 1: prime cache with entire prefix -----
    logits, _, past_kv_list = model(prefixes, return_kv_list=True)
    last = logits[:, -1, :]   # (B, V) — logits predicting the FIRST emitted token

    def _sample(last_logits: torch.Tensor) -> torch.Tensor:
        if use_nhead:
            slot_in_group = (n_emitted % GROUP).clamp_min(0)
            allow = slot_allow[slot_in_group]
            if nhead_mode == "force":
                restrict = torch.ones(B, dtype=torch.bool, device=device)
            else:
                restrict = n_emitted < n_hat_tokens
            mask = restrict.unsqueeze(-1) & (~allow)
            ll = last_logits.masked_fill(mask, float("-inf"))
        else:
            ll = last_logits
        if use_argmax:
            return ll.argmax(dim=-1)
        probs = torch.softmax(ll / float(T), dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    for step in range(max_new_tokens):
        if use_nhead and nhead_mode == "force":
            done = done | (n_emitted >= n_hat_tokens)
            if bool(done.all().item()):
                break

        next_tok = _sample(last)
        next_tok = torch.where(done, torch.full_like(next_tok, pad_id), next_tok)

        if nhead_mode == "force":
            done = done | ((n_emitted + 1) >= n_hat_tokens)
        else:
            done = done | (next_tok == eos_id)

        emitted_steps.append(next_tok)
        is_real = (next_tok != pad_id)
        n_emitted = n_emitted + is_real.long()
        if bool(done.all().item()):
            break
        # Feed next_tok as the next position; cache grows by one.
        # T_past after this call will be T_prefix + step + 1.
        if T_prefix + step + 2 > block_size:
            break
        logits, _, past_kv_list = model(
            next_tok.unsqueeze(1), past_kv_list=past_kv_list, return_kv_list=True,
        )
        last = logits[:, -1, :]

    if not emitted_steps:
        return (
            torch.empty((B, 0), dtype=torch.long, device=device),
            torch.zeros(B, dtype=torch.long, device=device),
        )
    return torch.stack(emitted_steps, dim=1), n_emitted


# ---------------------------------------------------------------------------
# Strict 6-token group parser (per row) — same logic as ar_decode_tempered
# ---------------------------------------------------------------------------
def parse_emitted_groups(
    emitted_row: np.ndarray,
    vocab,
    out_mod: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Parse one row of emitted token IDs into (content_codes, pos_codes).

    Stops at the first invalid group, EOS, or pad token. Mirrors the strict
    parser from ``nano-hep/notebooks/token_temperature_scan.py:136-151``.
    """
    nq_c = vocab.num_quantizers[out_mod]
    nq_p = vocab.num_q_pos[out_mod]
    GROUP = nq_c + nq_p
    V_c = vocab.codebook_sizes[out_mod]
    V_p = vocab.pos_codebook_sizes[out_mod]
    content_lo = vocab.offsets[out_mod]
    content_hi = content_lo + nq_c * V_c
    pos_lo = vocab.pos_offsets[out_mod]
    pos_hi = pos_lo + nq_p * V_p
    eos = vocab.eos
    pad = vocab.pad

    e = emitted_row
    stop = len(e)
    for i, t in enumerate(e):
        if t == eos or t == pad:
            stop = i
            break
    e = e[:stop]

    pc_rows: List[np.ndarray] = []
    pp_rows: List[np.ndarray] = []
    j = 0
    while j + GROUP <= len(e):
        group = e[j:j + GROUP]
        content_ok = all(content_lo <= group[k] < content_hi for k in range(nq_c))
        pos_ok = all(pos_lo <= group[k] < pos_hi for k in range(nq_c, GROUP))
        if not (content_ok and pos_ok):
            break
        c_local, p_local = vocab.decode_element_triples_with_pos(
            out_mod, np.asarray(group, dtype=np.int64)
        )
        pc_rows.append(c_local[0])
        pp_rows.append(p_local[0])
        j += GROUP
    if pc_rows:
        return np.stack(pc_rows, 0), np.stack(pp_rows, 0)
    return (
        np.zeros((0, nq_c), dtype=np.int64),
        np.zeros((0, nq_p), dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# Top-level: run_one_T_batched (drop-in for nano_tts.run_one_T)
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_one_T_batched(
    model, vocab, val_ds, T, n_events, device, pflow, out_dir, *,
    argmax_at_T1=True, seed=42, nhead_mode="free",
    batch_size=128,
    save_per_event=False, variant_tag=None,
    use_kv_cache: bool = True,
):
    """Batched analog of ``token_temperature_scan.run_one_T``.

    Buckets events by exact prefix length and decodes each bucket as one
    batched AR pass. With ``use_kv_cache=True`` (default) each AR step is
    O(t·d·L) instead of O(t²·d·L); on nano_hep_89M this is the difference
    between ~100 ms/event and ~2 ms/event. Returns the same schema (scalar
    metrics + underscore-prefixed per-event arrays) as ``run_one_T`` so
    notebook code consuming the result works unchanged.

    ``batch_size`` is the per-bucket cap; bigger lets more events share a
    single ``model(cur)`` call.
    """
    decode_fn = batched_ar_decode_bucket_kv if use_kv_cache else batched_ar_decode_bucket
    torch.manual_seed(seed)
    np.random.seed(seed)

    out_mod = val_ds.output_modality
    nq_c = vocab.num_quantizers[out_mod]
    nq_p = vocab.num_q_pos[out_mod]
    n_cb_true = val_ds.out_mm.n_codebooks
    n_pos_true = val_ds.out_mm.n_pos_codebooks

    # ----- Build per-event prefixes + truth arrays -----
    prefix_arrays: List[np.ndarray] = []
    true_codes_list: List[np.ndarray] = []
    true_pos_list: List[np.ndarray] = []
    for i in range(n_events):
        seq = val_ds.build_sequence(i)
        out_start_pos, _ = val_ds.output_region_slice(seq)
        prefix_arrays.append(np.asarray(seq[: out_start_pos + 1], dtype=np.int64))
        a = int(val_ds.out_mm.offsets[i]); b = int(val_ds.out_mm.offsets[i + 1])
        tc = np.asarray(val_ds.out_mm.data[a:b, :n_cb_true], dtype=np.int64)
        tp = np.asarray(val_ds.out_mm.data[a:b, n_cb_true:n_cb_true + n_pos_true], dtype=np.int64)
        true_codes_list.append(tc)
        true_pos_list.append(tp)

    # ----- Bucket by exact prefix length, decode each bucket -----
    by_len: Dict[int, List[int]] = defaultdict(list)
    for i, p in enumerate(prefix_arrays):
        by_len[len(p)].append(i)
    pred_codes_list: List[np.ndarray] = [None] * n_events  # type: ignore[list-item]
    pred_pos_list:   List[np.ndarray] = [None] * n_events  # type: ignore[list-item]
    use_nhead_assist = (nhead_mode != "free") and hasattr(model, "n_head_predict")

    t0 = time.time()
    bucket_lens = sorted(by_len.keys())
    for L in bucket_lens:
        idxs_in_bucket = by_len[L]
        for chunk_start in range(0, len(idxs_in_bucket), batch_size):
            chunk = idxs_in_bucket[chunk_start: chunk_start + batch_size]
            prefixes = torch.stack(
                [torch.from_numpy(prefix_arrays[i]) for i in chunk], dim=0,
            ).to(device)
            n_hat = None
            if use_nhead_assist:
                pos_t = torch.full((len(chunk),), L - 1, dtype=torch.long, device=device)
                n_logits = model.n_head_predict(prefixes, pos_t)
                n_hat = n_logits.argmax(dim=-1).long()
            max_new = min(val_ds.block_size - L, model.config.block_size - L)
            if max_new <= 0:
                for ev in chunk:
                    pred_codes_list[ev] = np.zeros((0, nq_c), dtype=np.int64)
                    pred_pos_list[ev]   = np.zeros((0, nq_p), dtype=np.int64)
                continue
            emitted, _ = decode_fn(
                model, vocab, out_mod, prefixes,
                T=T, max_new_tokens=max_new,
                argmax_at_T1=argmax_at_T1, nhead_mode=nhead_mode, n_hat=n_hat,
            )
            em_np = emitted.detach().cpu().numpy()
            for row, ev in enumerate(chunk):
                pc, pp = parse_emitted_groups(em_np[row], vocab, out_mod)
                pred_codes_list[ev] = pc
                pred_pos_list[ev] = pp
    label = "KV-cached bucketed" if use_kv_cache else "bucketed"
    print(
        f"  [T={T}] {label} AR decode {n_events} events "
        f"({len(bucket_lens)} prefix-length buckets) in "
        f"{time.time() - t0:.1f}s",
        flush=True,
    )

    # ----- Pack truth + pred into (B, max_N, num_q) for PflowMetrics -----
    max_N = 1
    for pc in pred_codes_list:
        if pc.shape[0] > max_N:
            max_N = pc.shape[0]
    for tc in true_codes_list:
        if tc.shape[0] > max_N:
            max_N = tc.shape[0]

    B = n_events
    pred_codes = np.zeros((B, max_N, n_cb_true), dtype=np.int64)
    pred_pos = np.zeros((B, max_N, n_pos_true), dtype=np.int64)
    pred_mask = np.zeros((B, max_N), dtype=bool)
    true_codes = np.zeros((B, max_N, n_cb_true), dtype=np.int64)
    true_pos = np.zeros((B, max_N, n_pos_true), dtype=np.int64)
    true_mask = np.zeros((B, max_N), dtype=bool)
    n_pred_arr = np.zeros(B, dtype=np.int64)
    n_true_arr = np.zeros(B, dtype=np.int64)
    for i in range(B):
        nt = true_codes_list[i].shape[0]
        true_codes[i, :nt] = true_codes_list[i]
        true_pos[i, :nt] = true_pos_list[i]
        true_mask[i, :nt] = True
        n_true_arr[i] = nt
        np_ = pred_codes_list[i].shape[0]
        if np_ > 0:
            pred_codes[i, :np_, :nq_c] = pred_codes_list[i]
            pred_pos[i, :np_, :nq_p] = pred_pos_list[i]
            pred_mask[i, :np_] = True
        n_pred_arr[i] = np_

    outdir_T = Path(out_dir) / f"T_{T:.2f}_batched"
    outdir_T.mkdir(parents=True, exist_ok=True)
    metrics = pflow.compute_metrics(
        torch.from_numpy(pred_codes), torch.from_numpy(pred_pos), torch.from_numpy(pred_mask),
        torch.from_numpy(true_codes), torch.from_numpy(true_pos), torch.from_numpy(true_mask),
        outdir=str(outdir_T), ind_threshold=0.5,
    )
    res = n_pred_arr - n_true_arr
    card_acc = float((n_pred_arr == n_true_arr).mean())
    card_mae = float(np.abs(res).mean())
    card_bias = float(res.mean())
    card_std = float(res.std()) if B > 1 else 0.0

    eq_mask = (n_pred_arr == n_true_arr)
    if eq_mask.any():
        correct_c, total_c = 0, 0
        correct_p, total_p = 0, 0
        for i in np.where(eq_mask)[0]:
            n = int(n_true_arr[i])
            if n == 0:
                continue
            correct_c += int((pred_codes[i, :n, :nq_c] == true_codes[i, :n, :nq_c]).sum())
            total_c += n * nq_c
            correct_p += int((pred_pos[i, :n, :nq_p] == true_pos[i, :n, :nq_p]).sum())
            total_p += n * nq_p
        ar_content_acc = float(correct_c / total_c) if total_c else float("nan")
        ar_pos_acc = float(correct_p / total_p) if total_p else float("nan")
    else:
        ar_content_acc = float("nan")
        ar_pos_acc = float("nan")
    ar_token_acc_coverage = float(eq_mask.mean())

    out: Dict[str, Any] = {
        "T": float(T),
        "n_events": n_events,
        "cardinality_acc": card_acc,
        "cardinality_mae": card_mae,
        "cardinality_bias": card_bias,
        "cardinality_std": card_std,
        "n_pred_mean": float(n_pred_arr.mean()),
        "n_pred_std": float(n_pred_arr.std()) if B > 1 else 0.0,
        "n_true_mean": float(n_true_arr.mean()),
        "n_true_std": float(n_true_arr.std()) if B > 1 else 0.0,
        "ar_content_token_acc": ar_content_acc,
        "ar_pos_token_acc": ar_pos_acc,
        "ar_token_acc_coverage": ar_token_acc_coverage,
    }
    for k in (
        "mean_jet_pt_response", "median_jet_pt_response",
        "std_jet_pt_response", "iqr_jet_pt_response",
        "mean_reco_cardinality_at_threshold", "n_events",
    ):
        if k in metrics:
            out[f"pflow_{k}"] = float(metrics[k])
    out["_n_pred_arr"] = n_pred_arr.tolist()
    out["_n_true_arr"] = n_true_arr.tolist()
    jt = metrics.get("_jet_table")
    if jt is not None:
        out["_jet_table"] = {k: (v.tolist() if hasattr(v, "tolist") else list(v))
                             for k, v in jt.items()}
    return out


# ---------------------------------------------------------------------------
# Per-bucket batched decode (the fast path for variable prefix lengths)
# ---------------------------------------------------------------------------
@torch.no_grad()
def batched_ar_decode_bucket(
    model, vocab, out_mod: str, prefixes: torch.Tensor, *,
    T: float, max_new_tokens: int,
    argmax_at_T1: bool = True, nhead_mode: str = "free",
    n_hat: Optional[torch.Tensor] = None,
):
    """Decode B events whose prefixes ALL have the same length.

    ``prefixes`` is a (B, T_prefix) tensor — same prefix length per row.
    Calls ``model(cur)`` directly which uses the inference fast-path
    (logits at last position only); since all rows share their write
    position, this returns the right logits for every row without padding.

    Returns
    -------
    emitted: (B, T_emit) long tensor — tokens emitted AFTER the prefix.
        Once a row terminates (EOS in free mode, quota in force mode),
        subsequent slots hold ``vocab.pad``.
    n_emitted: (B,) long tensor — count of real (non-pad) tokens per row.
    """
    assert nhead_mode in ("free", "floor", "force")
    device = prefixes.device
    B, T_prefix = prefixes.shape

    nq_c = vocab.num_quantizers[out_mod]
    nq_p = vocab.num_q_pos[out_mod]
    GROUP = nq_c + nq_p
    V_c = vocab.codebook_sizes[out_mod]
    V_p = vocab.pos_codebook_sizes[out_mod]
    content_lo = vocab.offsets[out_mod]
    pos_lo = vocab.pos_offsets[out_mod]

    use_nhead = nhead_mode != "free"
    if use_nhead:
        assert n_hat is not None and n_hat.shape == (B,), "n_hat must be (B,) for floor/force"
        n_hat_tokens = (GROUP * n_hat).to(device)
        V = vocab.total
        slot_allow = torch.zeros(GROUP, V, dtype=torch.bool, device=device)
        for s in range(nq_c):
            lo = content_lo + s * V_c
            slot_allow[s, lo: lo + V_c] = True
        for s in range(nq_p):
            lo = pos_lo + s * V_p
            slot_allow[nq_c + s, lo: lo + V_p] = True
    else:
        n_hat_tokens = None
        slot_allow = None

    block_size = model.config.block_size
    cur = prefixes.clone()
    done = torch.zeros(B, dtype=torch.bool, device=device)
    n_emitted = torch.zeros(B, dtype=torch.long, device=device)
    use_argmax = argmax_at_T1 and float(T) == 1.0
    pad_id = vocab.pad
    eos_id = vocab.eos

    emitted_steps: List[torch.Tensor] = []
    for step in range(max_new_tokens):
        if cur.shape[1] >= block_size:
            break
        if use_nhead and nhead_mode == "force":
            done = done | (n_emitted >= n_hat_tokens)
            if bool(done.all().item()):
                break

        logits, _ = model(cur)              # (B, 1, V) — fast inference path
        last = logits[:, -1, :]             # (B, V)

        if use_nhead:
            slot_in_group = (n_emitted % GROUP).clamp_min(0)
            allow = slot_allow[slot_in_group]                 # (B, V)
            if nhead_mode == "force":
                restrict = torch.ones(B, dtype=torch.bool, device=device)
            else:
                restrict = n_emitted < n_hat_tokens
            mask = restrict.unsqueeze(-1) & (~allow)
            last = last.masked_fill(mask, float("-inf"))

        if use_argmax:
            next_tok = last.argmax(dim=-1)
        else:
            probs = torch.softmax(last / float(T), dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1).squeeze(-1)

        next_tok = torch.where(done, torch.full_like(next_tok, pad_id), next_tok)
        if nhead_mode == "force":
            done = done | ((n_emitted + 1) >= n_hat_tokens)
        else:
            done = done | (next_tok == eos_id)

        emitted_steps.append(next_tok)
        is_real = (next_tok != pad_id)
        n_emitted = n_emitted + is_real.long()
        cur = torch.cat([cur, next_tok.unsqueeze(1)], dim=1)
        if bool(done.all().item()):
            break

    if not emitted_steps:
        return (
            torch.empty((B, 0), dtype=torch.long, device=device),
            torch.zeros(B, dtype=torch.long, device=device),
        )
    return torch.stack(emitted_steps, dim=1), n_emitted
