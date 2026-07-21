"""nano-hep token-temperature scan.

Loads a trained gpos-AR checkpoint, runs autoregressive inference with
temperature-controlled sampling on a fixed val subset, and reports
jet-level metrics per T. Mirrors the HEP4M scan notebook's output schema
so the two CSVs can be overlaid in a comparison plot.

Usage (under an interactive GPU alloc):
    srun --jobid=<JOBID> -N1 -n1 --gpus-per-task=1 \
        /pscratch/sd/d/danieltm/envs/hep4m2/bin/python \
        notebooks/token_temperature_scan.py \
        [--ckpt runs/.../last.ckpt] [--n_events 256] [--out_dir results/]

Design:
    - Reuses `nano_hep.pflow_metrics.PflowMetrics.compute_metrics` for jet
      metrics (same backend as HEP4M pflow_eval, so cross-model comparable).
    - Reuses strict 6-token group parser from
      `nano_hep.train_hep.autoregressive_eval_with_pflow` semantics (content
      range check + pos range check, abandon tail on violation).
    - Replaces argmax in the AR loop with tempered multinomial:
        probs = softmax(logits / T, dim=-1); next = multinomial(probs, 1).
      At T == 1.0 + argmax flag → exactly matches the existing deterministic
      baseline (sanity anchor).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Paths
REPO = Path("/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/nano-hep")
HEP4M_ROOT = "/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/HEP4M"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if HEP4M_ROOT not in sys.path:
    sys.path.insert(0, HEP4M_ROOT)

from nano_hep.vocab import Vocab, DEFAULT_MODALITY_CODEBOOK_SIZE
from nano_hep.data import HEPDataset
from model import GPT, GPTConfig  # upstream nanoGPT at repo root
from nano_hep.pflow_metrics import PflowMetrics


# ---------------------------------------------------------------------------
# AR decode with temperature (mirror of autoregressive_eval_with_pflow core)
# ---------------------------------------------------------------------------
@torch.no_grad()
def ar_decode_tempered(model, vocab, val_ds, idx, T, max_new_tokens, device, *,
                       argmax_at_T1=False, nhead_mode="free"):
    """Returns (pred_content_codes (N,num_q), pred_pos_codes (N,num_q_pos))
    via AR generation with softmax(logits / T) + multinomial sampling.

    If argmax_at_T1 is True AND T == 1.0, uses pure argmax (deterministic
    baseline anchor — matches autoregressive_eval_with_pflow exactly).

    `nhead_mode` (only meaningful when model is GPTWithNHead):
      - "free"   : plain EOS termination (default; backwards-compat).
      - "floor"  : mask EOS until >= 6*N_hat tokens emitted, then allow EOS.
      - "force"  : mask EOS always; stop after exactly 6*N_hat tokens.
    """
    out_mod = val_ds.output_modality
    nq_c  = vocab.num_quantizers[out_mod]
    nq_p  = vocab.num_q_pos[out_mod]
    GROUP = nq_c + nq_p
    V_c   = vocab.codebook_sizes[out_mod]
    V_p   = vocab.pos_codebook_sizes[out_mod]
    content_lo = vocab.offsets[out_mod];     content_hi = content_lo + nq_c * V_c
    pos_lo     = vocab.pos_offsets[out_mod]; pos_hi     = pos_lo     + nq_p * V_p

    seq = val_ds.build_sequence(idx)
    out_start_pos, _ = val_ds.output_region_slice(seq)
    prefix_len = out_start_pos + 1
    prefix = torch.from_numpy(seq[:prefix_len]).long().unsqueeze(0).to(device)
    cur = prefix
    max_len = min(val_ds.block_size, prefix_len + max_new_tokens)
    emitted = []
    use_argmax = argmax_at_T1 and float(T) == 1.0

    # N-head target (force/floor modes only, requires GPTWithNHead)
    assert nhead_mode in ("free", "floor", "force"), f"bad nhead_mode={nhead_mode!r}"
    use_nhead_assist = nhead_mode != "free" and hasattr(model, "n_head_mlp")
    n_hat_tokens = None
    if use_nhead_assist:
        pos_t = torch.tensor([prefix.shape[1] - 1], dtype=torch.long, device=device)
        n_logits = model.n_head_predict(prefix, pos_t)
        n_hat = int(n_logits.argmax(dim=-1).item())
        n_hat_tokens = GROUP * n_hat

    while cur.shape[1] < max_len:
        # "force" mode quota
        if nhead_mode == "force" and use_nhead_assist and len(emitted) >= n_hat_tokens:
            break
        logits, _ = model(cur)  # (1, 1, V)
        last = logits[0, -1]
        # Slot-aware masking under force / floor-below-quota:
        # restrict to the valid per-slot sub-vocab so the strict 6-token parser
        # always accepts emitted groups. "force" masks throughout; "floor" masks
        # only while we're below 6*N_hat tokens, then falls back to vanilla.
        if use_nhead_assist:
            restrict = (nhead_mode == "force") or \
                       (nhead_mode == "floor" and len(emitted) < n_hat_tokens)
            if restrict:
                slot_in_group = len(emitted) % GROUP
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
        if use_argmax:
            next_tok = int(last.argmax().item())
        else:
            probs = torch.softmax(last / float(T), dim=-1)
            next_tok = int(torch.multinomial(probs, 1).item())
        if next_tok == vocab.eos:
            break
        emitted.append(next_tok)
        cur = torch.cat(
            [cur, torch.tensor([[next_tok]], device=device, dtype=torch.long)], dim=1
        )

    # Strict 6-token group parse — same as train_hep.py
    pc_rows, pp_rows = [], []
    j = 0
    while j + GROUP <= len(emitted):
        content_ok = all(content_lo <= emitted[j + k] < content_hi for k in range(nq_c))
        pos_ok = all(pos_lo <= emitted[j + k] < pos_hi for k in range(nq_c, GROUP))
        if not (content_ok and pos_ok):
            break
        group_ids = np.array(emitted[j : j + GROUP], dtype=np.int64)
        c_local, p_local = vocab.decode_element_triples_with_pos(out_mod, group_ids)
        pc_rows.append(c_local[0])
        pp_rows.append(p_local[0])
        j += GROUP
    if pc_rows:
        return np.stack(pc_rows, 0), np.stack(pp_rows, 0)
    return np.zeros((0, nq_c), dtype=np.int64), np.zeros((0, nq_p), dtype=np.int64)


# ---------------------------------------------------------------------------
def load_model_from_ckpt(ckpt_path: Path, device: str = "cuda"):
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    v_state = state["vocab"]
    gpt_kwargs = state["gpt_config"]
    cfg = state["config"]
    # Vocab.build — pass pos fields if present, else rely on defaults
    num_q = v_state["num_quantizers"]
    num_q_pos = v_state.get("num_q_pos")
    pos_cb = v_state.get("pos_codebook_sizes")
    vocab = Vocab.build(
        v_state["modalities"], v_state["codebook_sizes"], num_q,
        pos_codebook_sizes=pos_cb, num_q_pos=num_q_pos,
    )
    gpt_cfg = GPTConfig(**gpt_kwargs)
    # Detect N-head checkpoint (added 2026-04-22). If any n_head_mlp.* key is
    # present in the saved state_dict, load as GPTWithNHead so the classifier
    # weights are restored; otherwise load as plain GPT.
    model_sd = state["model"]
    has_nhead = any(k.startswith("n_head_mlp") for k in model_sd.keys())
    if has_nhead:
        from nano_hep.model_nhead import GPTWithNHead
        nhead_cfg = cfg.get("n_head", {})
        max_n = int(nhead_cfg.get("max_n", 20))
        model = GPTWithNHead(gpt_cfg, max_n=max_n)
    else:
        model = GPT(gpt_cfg)
    model.load_state_dict(model_sd)
    model.to(device).eval()
    step = int(state.get("step", 0))
    return model, vocab, cfg, step


def build_val_ds(cfg, vocab, max_events):
    # Single-modality vs list
    in_mods = cfg["data"].get("input_modalities") or [cfg["data"].get("input_modality", "track")]
    ds = HEPDataset(
        tokenized_root=cfg["data"]["tokenized_root"],
        split="val",
        input_modalities=in_mods,
        output_modality=cfg["data"]["output_modality"],
        block_size=cfg["data"]["block_size"],
        max_events=max_events,
        vocab=vocab,
    )
    return ds


def run_one_T(model, vocab, val_ds, T, n_events, device, pflow, out_dir, *,
              argmax_at_T1=True, seed=42, nhead_mode="free",
              save_per_event=False, variant_tag=None):
    """Run AR decode on first n_events events with temperature T.
    Returns a dict of scalar metrics + attaches full arrays under _-prefixed keys.

    `nhead_mode`: "free" (EOS), "floor" (EOS-floor at N_head), or "force"
    (exactly 6*N_head, ignore EOS). The latter two require `model` to be a
    GPTWithNHead (will fall back to free otherwise).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    out_mod = val_ds.output_modality
    nq_c = vocab.num_quantizers[out_mod]
    nq_p = vocab.num_q_pos[out_mod]
    n_cb_true = val_ds.out_mm.n_codebooks
    n_pos_true = val_ds.out_mm.n_pos_codebooks

    pred_codes_list, pred_pos_list = [], []
    true_codes_list, true_pos_list = [], []
    max_N = 0
    t0 = time.time()
    for i in range(n_events):
        a = int(val_ds.out_mm.offsets[i]); b = int(val_ds.out_mm.offsets[i + 1])
        tc = np.asarray(val_ds.out_mm.data[a:b, :n_cb_true], dtype=np.int64)
        tp = np.asarray(val_ds.out_mm.data[a:b, n_cb_true:n_cb_true + n_pos_true], dtype=np.int64)
        true_codes_list.append(tc); true_pos_list.append(tp)

        pc, pp = ar_decode_tempered(
            model, vocab, val_ds, i, T, max_new_tokens=val_ds.block_size,
            device=device, argmax_at_T1=argmax_at_T1, nhead_mode=nhead_mode,
        )
        pred_codes_list.append(pc); pred_pos_list.append(pp)
        max_N = max(max_N, pc.shape[0], tc.shape[0])
    print(f"  [T={T}] AR decode {n_events} events in {time.time() - t0:.1f}s", flush=True)

    max_N = max(max_N, 1)
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

    # Jet-level metrics via PflowMetrics
    outdir_T = Path(out_dir) / f"T_{T:.2f}"
    outdir_T.mkdir(parents=True, exist_ok=True)
    metrics = pflow.compute_metrics(
        torch.from_numpy(pred_codes), torch.from_numpy(pred_pos), torch.from_numpy(pred_mask),
        torch.from_numpy(true_codes), torch.from_numpy(true_pos), torch.from_numpy(true_mask),
        outdir=str(outdir_T), ind_threshold=0.5,
    )
    # Cardinality scalars
    res = n_pred_arr - n_true_arr
    card_acc = float((n_pred_arr == n_true_arr).mean())
    card_mae = float(np.abs(res).mean())
    card_bias = float(res.mean())
    card_std = float(res.std()) if B > 1 else 0.0

    # Sampled-token accuracy: over events where N_pred == N_true, what fraction
    # of (event, slot, quantizer) token triples were sampled equal to truth?
    # This is T-dependent (we're comparing SAMPLED tokens, not the model's
    # distribution) so it's the right quantity for "are tokens wrong at this T".
    # Events with cardinality mismatch are excluded (alignment is ambiguous);
    # we also separately report the fraction of events used.
    eq_mask = (n_pred_arr == n_true_arr)
    if eq_mask.any():
        correct_c, total_c = 0, 0
        correct_p, total_p = 0, 0
        for i in np.where(eq_mask)[0]:
            n = int(n_true_arr[i])
            if n == 0:
                continue
            correct_c += int((pred_codes[i, :n, :nq_c] == true_codes[i, :n, :nq_c]).sum())
            total_c   += n * nq_c
            correct_p += int((pred_pos[i, :n, :nq_p] == true_pos[i, :n, :nq_p]).sum())
            total_p   += n * nq_p
        ar_content_acc = float(correct_c / total_c) if total_c else float("nan")
        ar_pos_acc     = float(correct_p / total_p) if total_p else float("nan")
    else:
        ar_content_acc = float("nan")
        ar_pos_acc     = float("nan")
    ar_token_acc_coverage = float(eq_mask.mean())

    # Pull the pflow scalars we care about
    out = {
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
        "ar_pos_token_acc":     ar_pos_acc,
        "ar_token_acc_coverage": ar_token_acc_coverage,
    }
    for k in (
        "mean_jet_pt_response", "median_jet_pt_response",
        "std_jet_pt_response", "iqr_jet_pt_response",
        "mean_reco_cardinality_at_threshold",
        "n_events",
    ):
        if k in metrics:
            out[f"pflow_{k}"] = float(metrics[k])
    # Store full per-event arrays for later violin plots
    out["_n_pred_arr"] = n_pred_arr.tolist()
    out["_n_true_arr"] = n_true_arr.tolist()
    # Per-event jet-level arrays (for response histogram, scatter, etc.)
    # `_jet_table` is produced by hep4m.pflow_report.run_report_from_arrays;
    # keys typically include 'jet_pt_truth', 'jet_pt_reco', 'jet_pt_response'.
    # Pass through untouched so callers can histogram them.
    jt = metrics.get("_jet_table")
    if jt is not None:
        out["_jet_table"] = {k: (v.tolist() if hasattr(v, "tolist") else list(v))
                             for k, v in jt.items()}

    # Per-event particle arrays for the HEP4M autoresearch OneMetric calibration.
    # Re-invokes PflowMetrics' decode path to extract jagged (pt, eta, phi)
    # per event for both pred and truth; same conventions as HEP4M's harness.
    if save_per_event:
        pred_dec = pflow.decode_tokens_to_features(
            torch.from_numpy(pred_codes), torch.from_numpy(pred_pos),
            torch.from_numpy(pred_mask),
        )
        true_dec = pflow.decode_tokens_to_features(
            torch.from_numpy(true_codes), torch.from_numpy(true_pos),
            torch.from_numpy(true_mask),
        )
        p_pt, p_eta, p_phi = pflow._x_to_pt_eta_phi(
            pred_dec["x_hat_cont"], pred_dec["x_gpos_hat"], torch.from_numpy(pred_mask),
        )
        t_pt, t_eta, t_phi = pflow._x_to_pt_eta_phi(
            true_dec["x_hat_cont"], true_dec["x_gpos_hat"], torch.from_numpy(true_mask),
        )
        tag = variant_tag or f"T{T:g}"
        per_event_path = Path(out_dir) / f"{tag}_per_event.npz"
        np.savez(
            per_event_path,
            pred_pt=np.array(p_pt, dtype=object),
            pred_eta=np.array(p_eta, dtype=object),
            pred_phi=np.array(p_phi, dtype=object),
            true_pt=np.array(t_pt, dtype=object),
            true_eta=np.array(t_eta, dtype=object),
            true_phi=np.array(t_phi, dtype=object),
        )
        out["_per_event_path"] = str(per_event_path)
        print(f"  [T={T}] per-event arrays saved -> {per_event_path.name}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(REPO / "runs/nano_hep_89M_ddp/last.ckpt"))
    ap.add_argument("--n_events", type=int, default=256)
    ap.add_argument("--T_grid", default="0.5,0.7,1.0,1.3,1.6,2.0,3.0",
                    help="comma-separated list of temperatures")
    ap.add_argument("--out_dir", default=str(REPO / "notebooks/results_T_scan"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip_argmax_anchor", action="store_true",
                    help="if set, T=1.0 uses multinomial like other T; by default T=1.0 uses argmax "
                         "for the deterministic baseline anchor")
    ap.add_argument("--save_per_event", action="store_true",
                    help="save per-event (pt, eta, phi) npz alongside each T for One Metric calibration")
    args = ap.parse_args()

    T_grid = [float(x) for x in args.T_grid.split(",")]
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = Path(args.ckpt)
    ckpt_mtime = datetime.fromtimestamp(ckpt_path.stat().st_mtime).isoformat()
    print(f"=== nano-hep token-temperature scan ===")
    print(f"ckpt: {ckpt_path}")
    print(f"ckpt mtime: {ckpt_mtime}")
    print(f"T grid: {T_grid}")
    print(f"n_events: {args.n_events}, seed: {args.seed}, device: {device}")
    print()

    model, vocab, cfg, step = load_model_from_ckpt(ckpt_path, device=device)
    print(f"loaded model at step={step}, vocab total={vocab.total}")
    print(f"gpt: n_layer={cfg['model']['n_layer']}, n_embd={cfg['model']['n_embd']}")

    val_ds = build_val_ds(cfg, vocab, max_events=args.n_events)
    print(f"val_ds: {len(val_ds)} events, block_size={val_ds.block_size}")

    # PflowMetrics needs the HEP4M modality_dict path
    pflow_cfg = cfg.get("pflow_metrics", {})
    md_path = pflow_cfg.get("modality_dict_path")
    if md_path is None:
        raise RuntimeError("config.pflow_metrics.modality_dict_path missing; cannot init PflowMetrics")
    pflow = PflowMetrics(
        modality_dict_path=md_path,
        output_modality=cfg["data"]["output_modality"],
        device=device,
    )
    print(f"PflowMetrics loaded from {md_path}")
    print()

    rows = []
    for T in T_grid:
        print(f"--- T = {T} ---")
        res = run_one_T(
            model, vocab, val_ds, T, args.n_events, device, pflow, out_dir,
            argmax_at_T1=(not args.skip_argmax_anchor), seed=args.seed,
            save_per_event=args.save_per_event,
            variant_tag=f"nanohep_T{T:g}",
        )
        for k in (
            "cardinality_acc", "cardinality_mae",
            "pflow_median_jet_pt_response", "pflow_iqr_jet_pt_response",
            "pflow_mean_reco_cardinality_at_threshold",
        ):
            if k in res:
                print(f"  {k}: {res[k]:.4f}")
        rows.append(res)
        # persist after each T so partial crash is recoverable
        df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])
        df.to_csv(out_dir / "nano_hep_T_scan.csv", index=False)
        with open(out_dir / "nano_hep_T_scan.json", "w") as f:
            json.dump(rows, f, indent=2)
        print()

    print("=== DONE ===")
    print(f"Results: {out_dir / 'nano_hep_T_scan.csv'}")


if __name__ == "__main__":
    main()
