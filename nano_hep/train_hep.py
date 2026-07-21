"""Trimmed nanoGPT training loop for HEP tokenized data.

Uses upstream model.GPT as-is. Loss is masked to output-region tokens via
targets=-1 (nanoGPT's forward uses ignore_index=-1 in F.cross_entropy).

Usage
-----
    python -m nano_hep.train_hep --config configs/nano_hep_smoke.yml

Minimal; DDP later. Single GPU for v0.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

# ensure parent dir (where upstream model.py lives) is on path
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))

from model import GPT, GPTConfig  # noqa: E402  (upstream nanoGPT)
from .data import HEPDataset      # noqa: E402
from .model_nhead import GPTWithNHead  # noqa: E402
from .vocab import Vocab, DEFAULT_MODALITY_CODEBOOK_SIZE, DEFAULT_MODALITY_NUM_QUANTIZERS  # noqa: E402


# ---- DDP setup ------------------------------------------------------------
def _ddp_init():
    """If launched via torchrun / slurm, init process group. Return
    (is_ddp, rank, local_rank, world_size)."""
    ws = int(os.environ.get("WORLD_SIZE", "1"))
    if ws <= 1:
        return False, 0, 0, 1
    import torch.distributed as dist
    # torchrun sets RANK, LOCAL_RANK, WORLD_SIZE. For slurm srun we map SLURM_* → torch env.
    if "SLURM_PROCID" in os.environ and "RANK" not in os.environ:
        os.environ["RANK"] = os.environ["SLURM_PROCID"]
        os.environ["LOCAL_RANK"] = os.environ.get("SLURM_LOCALID", "0")
        # torchrun usually provides MASTER_ADDR/PORT; Slurm setups need one set externally
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    # Long timeout tolerates rank-0-only validation blocks (~minutes for AR decode)
    from datetime import timedelta
    dist.init_process_group(backend="nccl", init_method="env://",
                            timeout=timedelta(minutes=120))
    return True, rank, local_rank, ws


def _is_main(rank: int) -> bool:
    return rank == 0


def _apply_loss_mask(y: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    """Return targets with ignored positions set to -1 (nanoGPT's ignore_index).
    y, loss_mask are both (B, T) long."""
    return torch.where(loss_mask.bool(), y, torch.full_like(y, -1))


def _get_lr(step: int, cfg: Dict[str, Any]) -> float:
    """Linear warmup → cosine decay."""
    warm = cfg["lr"]["warmup_steps"]
    max_steps = cfg["training"]["max_steps"]
    peak = cfg["lr"]["peak"]
    floor = cfg["lr"]["floor"]
    if step < warm:
        return peak * (step + 1) / max(warm, 1)
    if step >= max_steps:
        return floor
    progress = (step - warm) / max(max_steps - warm, 1)
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def estimate_val_loss(model, val_loader, device, vocab, max_batches: int = 10) -> Dict[str, float]:
    """Teacher-forced metrics: val_loss, token_accuracy on output positions.
    If the batch is a 5-tuple (n-head enabled), also compute N-head accuracy."""
    model.eval()
    losses = []
    total_tokens = 0
    n_correct = 0
    n_total = 0
    n_head_correct = 0
    n_head_total = 0
    n_head_abs_err = 0
    for i, batch in enumerate(val_loader):
        if i >= max_batches: break
        has_nhead = len(batch) == 5
        if has_nhead:
            x, y, m, n_head_pos, n_targets = batch
            n_head_pos = n_head_pos.to(device, non_blocking=True)
            n_targets = n_targets.to(device, non_blocking=True)
        else:
            x, y, m = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        m = m.to(device, non_blocking=True)
        y_mask = _apply_loss_mask(y, m)
        if has_nhead and hasattr(model, "n_head_mlp"):
            out = model(x, targets=y_mask, n_head_pos=n_head_pos, n_targets=n_targets)
            logits, loss = out[0], out[1]
            # compute n_head acc
            h_at = None  # direct n_logits not returned in this path — re-run head
            n_logits = model.n_head_predict(x, n_head_pos)
            n_pred = n_logits.argmax(dim=-1)
            n_head_correct += (n_pred == n_targets.clamp(0, n_logits.shape[-1] - 1)).sum().item()
            n_head_total += n_targets.shape[0]
            n_head_abs_err += (n_pred - n_targets.clamp(0, n_logits.shape[-1] - 1)).abs().sum().item()
        else:
            logits, loss = model(x, targets=y_mask)
        if loss is not None and not torch.isnan(loss):
            losses.append(loss.item())
            total_tokens += m.sum().item()
        pred = logits.argmax(dim=-1)  # (B, T)
        mask_bool = m.bool()
        n_correct += ((pred == y) & mask_bool).sum().item()
        n_total += mask_bool.sum().item()
    model.train()
    out = {
        "val_loss": float(np.mean(losses)) if losses else float("nan"),
        "val_tokens": total_tokens,
        "val_token_accuracy": float(n_correct / max(n_total, 1)),
    }
    if n_head_total > 0:
        out["val_nhead_acc"] = float(n_head_correct / n_head_total)
        out["val_nhead_mae"] = float(n_head_abs_err / n_head_total)
    return out


@torch.no_grad()
def autoregressive_eval_with_pflow(
    model, val_ds, device, vocab,
    n_events: int = 64, max_new_tokens: int = 64,
    pflow=None,  # optional PflowMetrics instance
    nhead_mode: str = "free",   # "free" | "floor" | "force"
) -> Dict[str, Any]:
    """Like autoregressive_eval, but also collects per-event (content, pos)
    code pairs so pflow metrics can be computed via VQ-VAE + PosTokenizer
    decode.

    AR stream emits 6 tokens per output element (content triple + pos triple,
    content-first). We greedy-generate to EOS or block_size, then parse the
    output region into strict 6-token groups. Any group with an out-of-range
    token aborts the parser and discards the tail — so a poorly-calibrated
    model simply yields fewer complete elements (honest behavior, no
    truth-pos oracle).

    `nhead_mode` (requires model to have `n_head_mlp`):
      - "free"   : plain EOS termination (default, backwards-compat).
      - "floor"  : mask EOS until we've emitted >= GROUP*N_hat tokens, then
                   allow EOS. Guarantees n_pred >= N_hat; an honest lower
                   floor from the N-head.
      - "force"  : mask EOS always, stop after exactly GROUP*N_hat tokens.
                   Locks n_pred to the N-head's argmax.
    For floor/force, N_hat is computed once per event from the N-head logits
    at the last prefix position (the output-modality MOD_START token).

    If `pflow` is provided, returns the full pflow dict merged into `out`.
    """
    # Call the standard AR eval first for token-level metrics.
    out = autoregressive_eval(model, val_ds, device, vocab,
                              n_events=n_events, max_new_tokens=max_new_tokens)
    if pflow is None:
        return out

    out_mod = val_ds.output_modality
    nq_c    = vocab.num_quantizers[out_mod]
    nq_p    = vocab.num_q_pos[out_mod]
    GROUP   = nq_c + nq_p
    V_c     = vocab.codebook_sizes[out_mod]
    V_p     = vocab.pos_codebook_sizes[out_mod]
    content_lo = vocab.offsets[out_mod];     content_hi = content_lo + nq_c * V_c
    pos_lo     = vocab.pos_offsets[out_mod]; pos_hi     = pos_lo     + nq_p * V_p

    # Re-run AR decode to collect per-event emitted tokens.
    model.eval()
    B = min(n_events, len(val_ds))
    max_N = 0
    true_codes_list, true_pos_list = [], []
    pred_codes_list, pred_pos_list = [], []
    n_cb_true = val_ds.out_mm.n_codebooks
    n_pos_true = val_ds.out_mm.n_pos_codebooks
    assert nhead_mode in ("free", "floor", "force"), f"bad nhead_mode={nhead_mode!r}"
    use_nhead_assist = nhead_mode != "free" and hasattr(model, "n_head_mlp")
    for i in range(B):
        # --- truth (content + pos) ---
        a = int(val_ds.out_mm.offsets[i]); b = int(val_ds.out_mm.offsets[i + 1])
        tc = np.asarray(val_ds.out_mm.data[a:b, :n_cb_true], dtype=np.int64)
        tp = np.asarray(val_ds.out_mm.data[a:b, n_cb_true:n_cb_true + n_pos_true], dtype=np.int64)
        true_codes_list.append(tc)
        true_pos_list.append(tp)

        # --- AR decode ---
        seq = val_ds.build_sequence(i)
        out_start_pos, _ = val_ds.output_region_slice(seq)
        prefix_len = out_start_pos + 1
        prefix = torch.from_numpy(seq[:prefix_len]).long().unsqueeze(0).to(device)
        cur = prefix
        max_len = min(val_ds.block_size, prefix_len + max_new_tokens)
        emitted = []

        # If nhead-assisted, ask the N-head for N_hat at the [MOD_START_out] position
        # (== out_start_pos == prefix_len - 1). One extra forward-through-backbone call
        # per event; cost dwarfed by the AR loop that follows.
        n_hat_tokens = None
        if use_nhead_assist:
            pos_t = torch.tensor([prefix.shape[1] - 1], dtype=torch.long, device=device)
            n_logits = model.n_head_predict(prefix, pos_t)          # (1, max_n+1)
            n_hat = int(n_logits.argmax(dim=-1).item())
            n_hat_tokens = GROUP * n_hat

        while cur.shape[1] < max_len:
            # Early-stop if "force" mode has emitted its full quota.
            if nhead_mode == "force" and len(emitted) >= n_hat_tokens:
                break
            logits, _ = model(cur)
            last = logits[0, -1]

            # Slot-aware masking for floor/force:
            # - "force": always restrict logits to the valid per-slot range AND mask EOS.
            #   This guarantees every emitted token is parseable → strict group parser
            #   accepts 6*N_hat tokens as N_hat valid groups → card = N_hat.
            # - "floor": apply the same restriction only while we're below 6*N_hat; above
            #   it, fall back to vanilla AR (any token, EOS allowed).
            # Otherwise (free mode) the logits are used as-is.
            slot_in_group = len(emitted) % GROUP
            restrict = (nhead_mode == "force") or \
                       (nhead_mode == "floor" and len(emitted) < n_hat_tokens)
            if restrict:
                if slot_in_group < nq_c:
                    lo = content_lo + slot_in_group * V_c
                    hi = lo + V_c
                else:
                    q = slot_in_group - nq_c
                    lo = pos_lo + q * V_p
                    hi = lo + V_p
                masked = torch.full_like(last, float("-inf"))
                masked[lo:hi] = last[lo:hi]
                last = masked
            next_tok = int(last.argmax().item())
            if next_tok == vocab.eos:
                break
            emitted.append(next_tok)
            cur = torch.cat([cur, torch.tensor([[next_tok]], device=device, dtype=torch.long)], dim=1)

        # --- strict group parse: first nq_c tokens in content range, next nq_p in pos range ---
        pc_rows = []
        pp_rows = []
        j = 0
        while j + GROUP <= len(emitted):
            content_ok = all(content_lo <= emitted[j + k] < content_hi for k in range(nq_c))
            pos_ok     = all(pos_lo     <= emitted[j + k] < pos_hi     for k in range(nq_c, GROUP))
            if not (content_ok and pos_ok):
                break  # abandon tail on any range violation
            group_ids = np.array(emitted[j : j + GROUP], dtype=np.int64)
            c_local, p_local = vocab.decode_element_triples_with_pos(out_mod, group_ids)
            pc_rows.append(c_local[0])
            pp_rows.append(p_local[0])
            j += GROUP
        if pc_rows:
            pc = np.stack(pc_rows, axis=0)  # (n, nq_c)
            pp = np.stack(pp_rows, axis=0)  # (n, nq_p)
        else:
            pc = np.zeros((0, nq_c), dtype=np.int64)
            pp = np.zeros((0, nq_p), dtype=np.int64)
        pred_codes_list.append(pc)
        pred_pos_list.append(pp)
        max_N = max(max_N, pc.shape[0], tc.shape[0])

    max_N = max(max_N, 1)
    B = len(true_codes_list)
    # Pad to (B, max_N, n_cb_true) for content and (B, max_N, n_pos_true) for pos.
    pred_codes = np.zeros((B, max_N, n_cb_true), dtype=np.int64)
    pred_pos   = np.zeros((B, max_N, n_pos_true), dtype=np.int64)
    pred_mask  = np.zeros((B, max_N), dtype=bool)
    true_codes = np.zeros((B, max_N, n_cb_true), dtype=np.int64)
    true_pos   = np.zeros((B, max_N, n_pos_true), dtype=np.int64)
    true_mask  = np.zeros((B, max_N), dtype=bool)
    for i in range(B):
        nt = true_codes_list[i].shape[0]
        if nt > 0:
            true_codes[i, :nt] = true_codes_list[i]
            true_pos  [i, :nt] = true_pos_list[i]
            true_mask [i, :nt] = True
        np_ = pred_codes_list[i].shape[0]
        if np_ > 0:
            n_qc = min(pred_codes_list[i].shape[1], n_cb_true)
            n_qp = min(pred_pos_list[i].shape[1],   n_pos_true)
            pred_codes[i, :np_, :n_qc] = pred_codes_list[i][:, :n_qc]
            pred_pos  [i, :np_, :n_qp] = pred_pos_list[i][:, :n_qp]
            pred_mask [i, :np_] = True

    # Tempdir outdir so pflow_report.run_report_from_arrays writes jet_response.png,
    # marginals.png, cardinality.png — caller uploads them as wandb.Image (mirrors
    # HEP4M's pflow_eval logging).
    import tempfile as _tempfile
    _outdir = _tempfile.mkdtemp(prefix="nanohep_pflow_eval_")
    pf_metrics = pflow.compute_metrics(
        torch.from_numpy(pred_codes), torch.from_numpy(pred_pos), torch.from_numpy(pred_mask),
        torch.from_numpy(true_codes), torch.from_numpy(true_pos), torch.from_numpy(true_mask),
        outdir=_outdir, ind_threshold=0.5,
    )
    model.train()
    # Preserve the raw histograms + jet table + outdir for caller (wandb upload).
    out["_pflow_outdir"] = _outdir
    out["_pflow_histograms"] = pf_metrics.get("_histograms", {})
    out["_pflow_jet_table"] = pf_metrics.get("_jet_table", {})
    for k, v in pf_metrics.items():
        if not str(k).startswith("_"):
            out[f"pflow_{k}"] = v
    return out


@torch.no_grad()
def autoregressive_eval(
    model,
    val_ds,
    device,
    vocab,
    n_events: int = 64,
    max_new_tokens: int = 64,
) -> Dict[str, Any]:
    """Run AR generation on a small val subset. For each event:
      1. Feed the prefix up through MOD_START[output] and generate tokens greedily.
      2. Stop at EOS or max_new_tokens.
      3. Cardinality = (count of output-modality tokens emitted before EOS) //
         GROUP, where GROUP = num_q_content + num_q_pos = 6. A token is
         considered "output-modality" if it falls in either the content range
         or the pos range for the output modality.

    Reports:
      ar_cardinality_acc  — fraction of events where pred cardinality == true.
      ar_cardinality_mae  — mean |pred − true|.
      ar_n_pred_mean/std  — cardinality-pred stats (#elements = tokens // 6).
      ar_n_true_mean/std  — cardinality-true stats.
      ar_eos_position_error — mean |predicted EOS pos − true EOS pos|.
      ar_token_accuracy   — accuracy of generated tokens vs teacher tokens,
                            when compared position by position up to min(pred, true).
      ar_scatter_npy      — (n_events,2) array (n_true, n_pred).
    """
    model.eval()
    out_mod = val_ds.output_modality
    nq_c    = vocab.num_quantizers[out_mod]
    nq_p    = vocab.num_q_pos[out_mod]
    GROUP   = nq_c + nq_p                       # tokens per element (content + pos)
    V_c     = vocab.codebook_sizes[out_mod]
    V_p     = vocab.pos_codebook_sizes[out_mod]
    content_lo = vocab.offsets[out_mod];     content_hi = content_lo + nq_c * V_c
    pos_lo     = vocab.pos_offsets[out_mod]; pos_hi     = pos_lo     + nq_p * V_p
    eos_id = vocab.eos
    pad_id = vocab.pad
    mod_start_out = vocab.mod_start[out_mod]

    n_true_list, n_pred_list, eos_pos_err, ar_n_correct, ar_n_total = [], [], [], 0, 0

    for i in range(min(n_events, len(val_ds))):
        seq = val_ds.build_sequence(i)  # (block_size,) int64
        out_start_pos, true_eos_pos = val_ds.output_region_slice(seq)
        n_true_tokens = true_eos_pos - (out_start_pos + 1)  # tokens between MOD_START[out] and EOS
        n_true = n_true_tokens // GROUP   # number of elements (particles); 6 tokens each
        # Prefix: everything up through MOD_START[out] (inclusive)
        prefix_len = out_start_pos + 1
        prefix = torch.from_numpy(seq[:prefix_len]).long().unsqueeze(0).to(device)

        # Autoregressive generation (greedy argmax; matches HEP4M pflow_eval top_k=1).
        cur = prefix
        n_pred_out_tokens = 0  # any token in content or pos range of out_mod
        emitted = []
        max_len = min(val_ds.block_size, prefix_len + max_new_tokens)
        ended = False
        while cur.shape[1] < max_len:
            logits, _ = model(cur)  # (B=1, 1, V)
            next_tok = int(logits[0, -1].argmax().item())
            emitted.append(next_tok)
            cur = torch.cat([cur, torch.tensor([[next_tok]], device=device, dtype=torch.long)], dim=1)
            if next_tok == eos_id:
                ended = True
                break
            # Count output-modality tokens (content OR pos — both belong to the output element)
            if (content_lo <= next_tok < content_hi) or (pos_lo <= next_tok < pos_hi):
                n_pred_out_tokens += 1
        n_pred = n_pred_out_tokens // GROUP  # complete 6-token groups = elements

        # Predicted EOS position = len(prefix) + len(emitted) - 1 (inclusive of the EOS if ended)
        if ended:
            pred_eos_pos = prefix_len + len(emitted) - 1
        else:
            pred_eos_pos = prefix_len + len(emitted)  # ran out without EOS
        eos_pos_err.append(abs(pred_eos_pos - true_eos_pos))

        # Teacher tokens for accuracy (between out_start_pos+1 and true_eos_pos)
        teacher = seq[out_start_pos + 1 : true_eos_pos]  # includes neither MOD_START[out] nor EOS
        # Predicted output tokens (emitted excluding EOS if present)
        if ended:
            pred_tokens = emitted[:-1]
        else:
            pred_tokens = emitted
        k = min(len(pred_tokens), len(teacher))
        if k > 0:
            t = np.asarray(teacher[:k])
            p = np.asarray(pred_tokens[:k])
            ar_n_correct += int((t == p).sum())
            ar_n_total += k

        n_true_list.append(int(n_true))
        n_pred_list.append(int(n_pred))

    n_true_arr = np.asarray(n_true_list)
    n_pred_arr = np.asarray(n_pred_list)
    resid = n_pred_arr - n_true_arr
    out = {
        "ar_cardinality_acc": float((n_pred_arr == n_true_arr).mean()),
        "ar_cardinality_mae": float(np.abs(resid).mean()),
        "ar_n_pred_mean": float(n_pred_arr.mean()),
        "ar_n_pred_std": float(n_pred_arr.std()),
        "ar_n_true_mean": float(n_true_arr.mean()),
        "ar_n_true_std": float(n_true_arr.std()),
        "ar_eos_position_error": float(np.mean(eos_pos_err)),
        "ar_token_accuracy": float(ar_n_correct / max(ar_n_total, 1)),
        "_n_true_arr": n_true_arr,  # for wandb plots (popped before logging scalars)
        "_n_pred_arr": n_pred_arr,
    }
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out_dir", default=None, help="override cfg.training.out_dir")
    ap.add_argument("--resume_from", default=None,
                    help="path to last.ckpt (or an out_dir to auto-resolve); when set, "
                         "restores model + optimizer + step + wandb run_id so a SLURM-"
                         "timeout resume continues the same wandb run cleanly.")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.out_dir is not None:
        cfg["training"]["out_dir"] = args.out_dir
    out_dir = Path(cfg["training"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    is_ddp, rank, local_rank, world_size = _ddp_init()
    main_rank = _is_main(rank)
    torch.manual_seed(cfg["training"].get("seed", 42) + rank)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_type = "cuda" if device == "cuda" else "cpu"
    dtype = torch.bfloat16 if cfg["training"].get("bf16", True) and device == "cuda" else torch.float32
    if is_ddp and main_rank:
        print(f"[DDP] world_size={world_size} rank={rank} local_rank={local_rank}", flush=True)

    # ---- data -----------------------------------------------------------------
    # Accept either `input_modality` (single, legacy) or `input_modalities` (list, new).
    if "input_modalities" in cfg["data"]:
        in_mods = list(cfg["data"]["input_modalities"])
    else:
        in_mods = [cfg["data"]["input_modality"]]
    out_mod = cfg["data"]["output_modality"]
    all_mods = in_mods + [out_mod]
    # num_quantizers can be scalar (applied to all mods) or dict {mod: int}
    num_q_cfg = cfg["data"].get("num_quantizers", 1)
    if isinstance(num_q_cfg, int):
        num_q_map = {m: num_q_cfg for m in all_mods}
    else:
        num_q_map = dict(num_q_cfg)
    v = Vocab.build(all_mods,
                    {m: DEFAULT_MODALITY_CODEBOOK_SIZE[m] for m in all_mods},
                    num_q_map)
    if main_rank:
        print(f"Vocab: {v}")

    nhead_cfg = cfg.get("n_head", {"enabled": False})
    use_nhead = bool(nhead_cfg.get("enabled", False))
    train_ds = HEPDataset(
        tokenized_root=cfg["data"]["tokenized_root"],
        split="train",
        input_modalities=in_mods, output_modality=out_mod,
        block_size=cfg["data"]["block_size"],
        max_events=cfg["data"].get("max_train_events", -1),
        vocab=v,
        return_n_head_signal=use_nhead,
    )
    val_ds = HEPDataset(
        tokenized_root=cfg["data"]["tokenized_root"],
        split="val",
        input_modalities=in_mods, output_modality=out_mod,
        block_size=cfg["data"]["block_size"],
        max_events=cfg["data"].get("max_val_events", 512),
        vocab=v,
        return_n_head_signal=use_nhead,
    )
    if main_rank:
        print(f"train events: {len(train_ds):,}  val events: {len(val_ds):,}")

    train_sampler = None
    if is_ddp:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size,
                                           rank=rank, shuffle=True, drop_last=True)
    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"],
                              shuffle=(train_sampler is None),
                              sampler=train_sampler,
                              num_workers=cfg["training"].get("num_workers", 2),
                              pin_memory=True, drop_last=True)
    # val runs only on rank 0 to keep eval logic simple (AR decode is not DDP-safe)
    val_loader = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"],
                            shuffle=False, num_workers=cfg["training"].get("num_workers", 2),
                            pin_memory=True, drop_last=False)

    # ---- model ----------------------------------------------------------------
    gpt_cfg = GPTConfig(
        block_size=cfg["data"]["block_size"] - 1,  # x/y are shifted by 1, so sequence length to model is block_size-1
        vocab_size=v.total,
        n_layer=cfg["model"]["n_layer"],
        n_head=cfg["model"]["n_head"],
        n_embd=cfg["model"]["n_embd"],
        dropout=cfg["model"].get("dropout", 0.0),
        bias=cfg["model"].get("bias", False),
    )
    if use_nhead:
        max_n = int(nhead_cfg.get("max_n", 20))
        model = GPTWithNHead(gpt_cfg, max_n=max_n).to(device)
        if main_rank:
            print(f"[n_head] enabled (max_n={max_n}, loss_weight={nhead_cfg.get('loss_weight', 0.1)})")
    else:
        model = GPT(gpt_cfg).to(device)
    if device == "cuda" and cfg["training"].get("compile", False):
        if main_rank: print("torch.compile...")
        model = torch.compile(model)

    # Optimizer configured on the raw model (needs param groups that reference modules)
    raw_model = model
    optimizer = raw_model.configure_optimizers(
        weight_decay=cfg["optim"]["weight_decay"],
        learning_rate=cfg["lr"]["peak"],
        betas=(cfg["optim"]["beta1"], cfg["optim"]["beta2"]),
        device_type=device_type,
    )

    if is_ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False)
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16))

    # ---- resume from checkpoint (before wandb.init so we can reuse run_id) --
    resume_state = None
    resume_wandb_id = None
    if args.resume_from:
        resume_path = Path(args.resume_from)
        if resume_path.is_dir():
            resume_path = resume_path / "last.ckpt"
        if resume_path.exists():
            if main_rank:
                print(f"[resume] loading {resume_path}", flush=True)
            resume_state = torch.load(resume_path, map_location="cpu", weights_only=False)
            # Model state — tolerate DDP-wrapped vs raw
            msd = resume_state["model"]
            if any(k.startswith("module.") for k in msd.keys()):
                msd = {k.replace("module.", "", 1) if k.startswith("module.") else k: v
                       for k, v in msd.items()}
            ckpt_has_nhead = any(k.startswith("n_head_mlp") for k in msd.keys())
            arch_expand = use_nhead and not ckpt_has_nhead
            load_info = raw_model.load_state_dict(msd, strict=not arch_expand)
            if arch_expand and main_rank:
                missing = [k for k in getattr(load_info, "missing_keys", []) if k.startswith("n_head_mlp")]
                print(f"[resume] architectural expansion: base GPT loaded, "
                      f"{len(missing)} N-head params freshly initialized", flush=True)
            if "optimizer" in resume_state:
                saved_groups = len(resume_state["optimizer"].get("param_groups", []))
                if use_nhead:
                    # Always use base-only + add-nhead schema for N-head runs so the
                    # param_group layout matches the schema that was saved.
                    # (arch_expand=True saves fresh nhead state; arch_expand=False
                    #  loads existing nhead state from the saved 3rd/4th groups.)
                    if main_rank:
                        print(f"[resume] rebuilding optimizer under N-head schema "
                              f"(arch_expand={arch_expand}, saved_groups={saved_groups})",
                              flush=True)
                    optimizer = raw_model.configure_optimizers_base_only(
                        weight_decay=cfg["optim"]["weight_decay"],
                        learning_rate=cfg["lr"]["peak"],
                        betas=(cfg["optim"]["beta1"], cfg["optim"]["beta2"]),
                        device_type=device_type,
                    )
                    if arch_expand:
                        # Old ckpt has base-only optimizer (2 groups); load, then append nhead.
                        optimizer.load_state_dict(resume_state["optimizer"])
                        raw_model.add_n_head_param_groups(
                            optimizer, weight_decay=cfg["optim"]["weight_decay"],
                        )
                    else:
                        # Old ckpt already has nhead groups appended — add_n_head first so
                        # both sides have the same group count before loading state.
                        raw_model.add_n_head_param_groups(
                            optimizer, weight_decay=cfg["optim"]["weight_decay"],
                        )
                        optimizer.load_state_dict(resume_state["optimizer"])
                else:
                    optimizer.load_state_dict(resume_state["optimizer"])
            if "scaler" in resume_state and resume_state["scaler"] is not None:
                try:
                    scaler.load_state_dict(resume_state["scaler"])
                except Exception:
                    pass
            resume_wandb_id = resume_state.get("wandb_run_id")
        else:
            if main_rank:
                print(f"[resume] no checkpoint at {resume_path}; starting fresh", flush=True)

    # ---- pflow metrics (optional; only on rank 0) ----------------------------
    pflow = None
    pflow_cfg = cfg.get("pflow_metrics", {"enabled": False})
    if pflow_cfg.get("enabled", False) and main_rank:
        from .pflow_metrics import PflowMetrics
        pflow = PflowMetrics(
            modality_dict_path=pflow_cfg["modality_dict_path"],
            output_modality=out_mod,
            device=device,
        )
        print(f"PflowMetrics enabled for output modality '{out_mod}'.")

    # ---- wandb (rank 0 only) -------------------------------------------------
    use_wandb = cfg["training"].get("wandb", True) and main_rank
    if use_wandb:
        import wandb
        wandb_kwargs = dict(
            project=cfg["training"].get("wandb_project", "nano-hep"),
            name=cfg["training"].get("run_name", out_dir.name),
            config=cfg,
            dir=str(out_dir),
        )
        # On resume, reuse the same run id + allow wandb to append to the existing
        # cloud run so step counters stay continuous across SLURM-timeout restarts.
        if resume_wandb_id:
            wandb_kwargs["id"] = resume_wandb_id
            wandb_kwargs["resume"] = "allow"
        wandb.init(**wandb_kwargs)
        # Make trainer step the default x-axis so all charts are continuous across
        # resumes (wandb's per-run _step resets to 0 each init; our "step" metric
        # is the monotonic global step we log in train/ + val/).
        try:
            wandb.define_metric("step")
            wandb.define_metric("*", step_metric="step")
        except Exception as _e:
            print(f"[wandb] define_metric failed (non-fatal): {_e}", flush=True)

    # ---- train loop -----------------------------------------------------------
    max_steps = cfg["training"]["max_steps"]
    log_every = cfg["training"].get("log_every", 50)
    val_every = cfg["training"].get("val_every", 500)
    ckpt_every = cfg["training"].get("ckpt_every", 2000)
    grad_clip = cfg["training"].get("grad_clip", 1.0)

    step = 0
    epoch = 0
    best_val = float("inf")
    if resume_state is not None:
        step = int(resume_state.get("step", 0))
        epoch = int(resume_state.get("epoch", 0))
        best_val = float(resume_state.get("best_val", best_val))
        if main_rank:
            print(f"[resume] continuing from step={step} epoch={epoch} best_val={best_val:.4f}",
                  flush=True)
    t_last = time.time()
    model.train()
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)
    data_iter = iter(train_loader)
    n_head_loss_weight = float(nhead_cfg.get("loss_weight", 0.1)) if use_nhead else 0.0
    while step < max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            epoch += 1
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            data_iter = iter(train_loader)
            batch = next(data_iter)

        if use_nhead and len(batch) == 5:
            x, y, m, n_head_pos, n_targets = batch
            n_head_pos = n_head_pos.to(device, non_blocking=True)
            n_targets = n_targets.to(device, non_blocking=True)
        else:
            x, y, m = batch
            n_head_pos = None
            n_targets = None
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        m = m.to(device, non_blocking=True)
        y_mask = _apply_loss_mask(y, m)

        lr = _get_lr(step, cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        with torch.autocast(device_type=device_type, dtype=dtype, enabled=(device == "cuda")):
            if n_head_pos is not None:
                out = model(x, targets=y_mask, n_head_pos=n_head_pos, n_targets=n_targets)
                if len(out) == 3:
                    _, tok_loss, n_loss = out
                    loss = tok_loss + n_head_loss_weight * n_loss
                else:
                    _, loss = out
                    tok_loss = loss
                    n_loss = None
            else:
                _, loss = model(x, targets=y_mask)
                tok_loss = loss
                n_loss = None
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer); scaler.update()

        step += 1

        if step % log_every == 0 and main_rank:
            now = time.time()
            tps = (log_every * cfg["training"]["batch_size"] * world_size * (cfg["data"]["block_size"] - 1)) / (now - t_last)
            n_loss_tokens = m.sum().item()
            extra = ""
            if n_loss is not None:
                extra = f"  tok_loss {tok_loss.item():.4f}  n_loss {n_loss.item():.4f}"
            msg = f"step {step}/{max_steps}  loss {loss.item():.4f}{extra}  lr {lr:.2e}  loss_tokens/batch {n_loss_tokens}  tok/s {tps:,.0f}"
            print(msg, flush=True)
            if use_wandb:
                log = {"train/loss": loss.item(), "train/lr": lr, "train/tok_per_s": tps,
                       "train/loss_tokens_per_batch": n_loss_tokens, "step": step}
                if n_loss is not None:
                    log["train/tok_loss"] = tok_loss.item()
                    log["train/n_head_loss"] = n_loss.item()
                wandb.log(log)
            t_last = now

        if (step % val_every == 0 or step == max_steps) and main_rank:
            # Run val on raw (un-DDP'd) model
            _val_model = raw_model
            val = estimate_val_loss(_val_model, val_loader, device, v, max_batches=10)
            ar_n_events = cfg["training"].get("ar_eval_events", 64)
            ar = autoregressive_eval_with_pflow(
                _val_model, val_ds, device, v,
                n_events=ar_n_events,
                max_new_tokens=val_ds.block_size,
                pflow=pflow,
            )
            # If the N-head is live, also run two N-head-assisted evals so we
            # can see how the head propagates to jet-level metrics:
            #   floor : mask EOS (+ slot-range) until 6*N_hat tokens have been
            #           emitted, then allow EOS. Guarantees n_pred >= N_hat, but
            #           the model may choose to continue past N_hat.
            #   force : mask EOS (+ slot-range) for the entire decode, stop
            #           exactly at 6*N_hat. Locks n_pred = N_hat.
            ar_floor = None
            ar_nhead = None
            if use_nhead and hasattr(raw_model, "n_head_mlp"):
                ar_floor = autoregressive_eval_with_pflow(
                    _val_model, val_ds, device, v,
                    n_events=ar_n_events,
                    max_new_tokens=val_ds.block_size,
                    pflow=pflow,
                    nhead_mode="floor",
                )
                ar_nhead = autoregressive_eval_with_pflow(
                    _val_model, val_ds, device, v,
                    n_events=ar_n_events,
                    max_new_tokens=val_ds.block_size,
                    pflow=pflow,
                    nhead_mode="force",
                )
            nhead_msg = ""
            if "val_nhead_acc" in val:
                nhead_msg = f"  nhead_acc {val['val_nhead_acc']:.3f}  nhead_mae {val['val_nhead_mae']:.2f}"
            floor_msg = ""
            if ar_floor is not None:
                floor_msg = (f"  floor_card_acc {ar_floor['ar_cardinality_acc']:.3f}"
                             f"  floor_median {ar_floor.get('pflow_median_jet_pt_response', float('nan')):.3f}"
                             f"  floor_iqr {ar_floor.get('pflow_iqr_jet_pt_response', float('nan')):.3f}")
            forceN_msg = ""
            if ar_nhead is not None:
                forceN_msg = (f"  forceN_card_acc {ar_nhead['ar_cardinality_acc']:.3f}"
                              f"  forceN_median {ar_nhead.get('pflow_median_jet_pt_response', float('nan')):.3f}"
                              f"  forceN_iqr {ar_nhead.get('pflow_iqr_jet_pt_response', float('nan')):.3f}")
            forceN_msg = floor_msg + forceN_msg
            print(f"  val_loss {val['val_loss']:.4f}  tok_acc {val['val_token_accuracy']:.3f}{nhead_msg}  "
                  f"ar_card_acc {ar['ar_cardinality_acc']:.3f}  ar_card_mae {ar['ar_cardinality_mae']:.2f}  "
                  f"ar_tok_acc {ar['ar_token_accuracy']:.3f}  ar_n_pred {ar['ar_n_pred_mean']:.1f}±{ar['ar_n_pred_std']:.1f}  "
                  f"(n_true {ar['ar_n_true_mean']:.1f}±{ar['ar_n_true_std']:.1f}){forceN_msg}",
                  flush=True)
            if use_wandb:
                log_dict = {
                    "val/loss": val["val_loss"],
                    "val/tokens_seen": val["val_tokens"],
                    "val/token_accuracy": val["val_token_accuracy"],
                    "step": step,
                }
                if "val_nhead_acc" in val:
                    log_dict["val/nhead_acc"] = val["val_nhead_acc"]
                    log_dict["val/nhead_mae"] = val["val_nhead_mae"]
                for k, vv in ar.items():
                    if k.startswith("_"):
                        continue
                    # pflow_* keys use underscore-prefix to match HEP4M's
                    # val_pflow_* naming — enables direct side-by-side wandb
                    # comparison across the two projects.
                    if k.startswith("pflow_"):
                        log_dict[f"val_{k}"] = vv
                    else:
                        log_dict[f"val/{k}"] = vv
                # Apples-to-apples aliases matching HEP4M's val/cardinality_* keys.
                # HEP4M has no "ar_" equivalent; share unprefixed names for cross-run plots.
                if "ar_cardinality_acc" in ar:
                    log_dict["val/cardinality_acc"]  = ar["ar_cardinality_acc"]
                    log_dict["val/cardinality_mae"]  = ar["ar_cardinality_mae"]
                    log_dict["val/n_pred_mean"]      = ar["ar_n_pred_mean"]
                    log_dict["val/n_pred_std"]       = ar["ar_n_pred_std"]
                    log_dict["val/n_true_mean"]      = ar["ar_n_true_mean"]
                    log_dict["val/n_true_std"]       = ar["ar_n_true_std"]
                    # bias + std of residual (nano-hep didn't have these; add for parity)
                    _nt = ar.get("_n_true_arr"); _np_arr = ar.get("_n_pred_arr")
                    if _nt is not None and _np_arr is not None and len(_nt) > 1:
                        _r = _np_arr - _nt
                        log_dict["val/cardinality_bias"] = float(_r.mean())
                        log_dict["val/cardinality_std"]  = float(_r.std())
                # Floor-N (EOS masked only BELOW 6*N_hat) + Force-N (EOS masked
                # throughout, stops at 6*N_hat). Suffixed `_floorN` / `_forceN`
                # so wandb panels can plot the three modes alongside free-AR.
                def _log_mode(src, suffix):
                    if src is None:
                        return
                    log_dict[f"val/cardinality_acc_{suffix}"] = src["ar_cardinality_acc"]
                    log_dict[f"val/cardinality_mae_{suffix}"] = src["ar_cardinality_mae"]
                    log_dict[f"val/n_pred_mean_{suffix}"]     = src["ar_n_pred_mean"]
                    for k in ("pflow_median_jet_pt_response",
                              "pflow_iqr_jet_pt_response",
                              "pflow_mean_jet_pt_response",
                              "pflow_std_jet_pt_response",
                              "pflow_mean_reco_cardinality_at_threshold"):
                        if k in src:
                            log_dict[f"val_{k}_{suffix}"] = src[k]
                _log_mode(ar_floor, "floorN")
                _log_mode(ar_nhead, "forceN")
                # Cardinality scatter plot (wandb Image) every val
                try:
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
                    n_true = ar["_n_true_arr"]; n_pred = ar["_n_pred_arr"]
                    mx = max(n_true.max(), n_pred.max(), 1) + 1
                    ax1.scatter(n_true, n_pred, alpha=0.3, s=10)
                    ax1.plot([0, mx], [0, mx], "k--", lw=0.5)
                    ax1.set_xlabel("n_true"); ax1.set_ylabel("n_pred")
                    ax1.set_title(f"cardinality: pred vs true  (step {step})")
                    bins = np.arange(0, mx + 1) - 0.5
                    ax2.hist(n_true, bins=bins, alpha=0.5, label="true", density=True)
                    ax2.hist(n_pred, bins=bins, alpha=0.5, label="pred", density=True)
                    ax2.set_xlabel("cardinality"); ax2.legend(fontsize=8)
                    ax2.set_title(f"cardinality hist  acc={ar['ar_cardinality_acc']:.2f} mae={ar['ar_cardinality_mae']:.1f}")
                    fig.tight_layout()
                    log_dict["val/cardinality_plot"] = wandb.Image(fig)
                    plt.close(fig)
                except Exception as e:
                    print(f"  (skipped cardinality plot: {e})")

                # pflow jet-level plots (jet_response / marginals / cardinality) +
                # residual histograms + jet table — mirrors HEP4M's pflow_eval
                # upload path so both wandb projects show identical media.
                _pflow_outdir     = ar.pop("_pflow_outdir", None)
                _pflow_histograms = ar.pop("_pflow_histograms", {}) or {}
                _pflow_jet_table  = ar.pop("_pflow_jet_table", {}) or {}
                if _pflow_outdir:
                    try:
                        from PIL import Image as _PILImage
                        import os as _os2
                        # Keys match HEP4M's hep4m_lightning.py pflow_eval upload.
                        for _name, _fname in [
                            ("val_pflow_jet_response",  "jet_response.png"),
                            ("val_pflow_marginals",     "marginals.png"),
                            ("val_pflow_cardinality",   "cardinality.png"),
                        ]:
                            _path = _os2.path.join(_pflow_outdir, _fname)
                            if _os2.path.isfile(_path):
                                log_dict[_name] = wandb.Image(_PILImage.open(_path).copy())
                    except Exception as e:
                        print(f"  (skipped pflow plot upload: {e})")
                # Residual / marginal histograms (interactive in wandb UI).
                for _k, _pair in _pflow_histograms.items():
                    try:
                        counts, edges = _pair
                        log_dict[f"val_pflow_hist_{_k}"] = wandb.Histogram(
                            np_histogram=(counts, edges))
                    except Exception as e:
                        print(f"  (skipped hist {_k}: {e})")
                # Jet-level table.
                if _pflow_jet_table:
                    try:
                        cols = list(_pflow_jet_table.keys())
                        log_dict["val_pflow_jets"] = wandb.Table(
                            columns=cols,
                            data=list(zip(*(_pflow_jet_table[c].tolist() for c in cols))),
                        )
                    except Exception as e:
                        print(f"  (skipped jet table: {e})")

                wandb.log(log_dict)

                # Clean up tempdir
                if _pflow_outdir:
                    try:
                        import shutil as _shutil
                        _shutil.rmtree(_pflow_outdir, ignore_errors=True)
                    except Exception:
                        pass
            # Always save last.ckpt (with optimizer + run_id) so chain resume works.
            if main_rank:
                _model_sd = (raw_model._orig_mod if hasattr(raw_model, "_orig_mod")
                             else raw_model).state_dict()
                _vocab = {"modalities": v.modalities, "codebook_sizes": v.codebook_sizes,
                          "num_quantizers": v.num_quantizers,
                          "pos_codebook_sizes": v.pos_codebook_sizes,
                          "num_q_pos": v.num_q_pos,
                          "offsets": v.offsets, "pos_offsets": v.pos_offsets,
                          "mod_start": v.mod_start,
                          "eos": v.eos, "pad": v.pad, "total": v.total}
                _run_id = None
                if use_wandb:
                    try:
                        _run_id = wandb.run.id
                    except Exception:
                        pass
                _full_state = {
                    "model": _model_sd,
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict() if scaler is not None else None,
                    "config": cfg,
                    "gpt_config": gpt_cfg.__dict__,
                    "vocab": _vocab,
                    "step": step, "epoch": epoch, "best_val": best_val,
                    "val_loss": val["val_loss"],
                    "wandb_run_id": _run_id,
                }
                torch.save(_full_state, out_dir / "last.ckpt")

                if val["val_loss"] < best_val:
                    best_val = val["val_loss"]
                    # best.ckpt: minimal (model + config + vocab), matches previous format
                    torch.save({
                        "model": _model_sd,
                        "config": cfg, "gpt_config": gpt_cfg.__dict__, "vocab": _vocab,
                        "step": step, "val_loss": val["val_loss"],
                    }, out_dir / "best.ckpt")
                    print(f"  saved best.ckpt (val_loss={val['val_loss']:.4f})", flush=True)

        if step % ckpt_every == 0 and main_rank:
            torch.save({"model": (raw_model._orig_mod if hasattr(raw_model, "_orig_mod") else raw_model).state_dict(),
                        "step": step, "gpt_config": gpt_cfg.__dict__},
                       out_dir / f"step{step:06d}.ckpt")

    if main_rank:
        print("done.")
    if is_ddp:
        import torch.distributed as dist
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
