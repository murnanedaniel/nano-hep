"""Oracle-N diagnostic for nano-hep AR model.

Runs AR decode with the cardinality FORCED to truth: exactly `6 * n_true[i]`
tokens per event, ignoring any EOS emission by masking EOS logit. Measures
jet metrics to determine whether cardinality is the sole bottleneck or
there's also slot-level content degradation.

Three modes compared at T=1.0 argmax (deterministic):
  - "free" (control): normal AR decode, strict 6-token group parse (matches
    in-chain val baseline).
  - "force_N_true" (oracle): decode exactly 6 * n_true tokens with EOS
    suppressed; no strict parse — all emitted tokens assumed valid, any
    out-of-range tokens fall back to argmax of valid range per quantizer.
  - "truncate_or_pad": take `free` pred_codes and (a) truncate if
    n_pred > n_true, (b) pad with repeats of last group if n_pred < n_true.
    This is a cheaper proxy for oracle-N that stays content-faithful to
    what the model actually would have produced.

The interesting comparison is free vs force_N_true — if force jumps jet
metrics to ~1.0 / tight iqr, cardinality is the whole bottleneck. If it
doesn't, content prediction is also weak.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path("/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/nano-hep")
HEP4M = "/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/HEP4M"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if HEP4M not in sys.path:
    sys.path.insert(0, HEP4M)

from nano_hep.vocab import Vocab
from nano_hep.data import HEPDataset
from model import GPT, GPTConfig
from nano_hep.pflow_metrics import PflowMetrics


@torch.no_grad()
def ar_decode_free(model, vocab, val_ds, idx, device):
    """Normal AR decode — stop at EOS or max_len, strict 6-group parse.
    Returns (pred_content (N,num_q_c), pred_pos (N,num_q_p))."""
    out_mod = val_ds.output_modality
    nq_c = vocab.num_quantizers[out_mod]
    nq_p = vocab.num_q_pos[out_mod]
    GROUP = nq_c + nq_p
    V_c = vocab.codebook_sizes[out_mod]
    V_p = vocab.pos_codebook_sizes[out_mod]
    content_lo = vocab.offsets[out_mod]; content_hi = content_lo + nq_c * V_c
    pos_lo = vocab.pos_offsets[out_mod]; pos_hi = pos_lo + nq_p * V_p

    seq = val_ds.build_sequence(idx)
    out_start_pos, _ = val_ds.output_region_slice(seq)
    prefix_len = out_start_pos + 1
    prefix = torch.from_numpy(seq[:prefix_len]).long().unsqueeze(0).to(device)
    cur = prefix
    max_len = val_ds.block_size
    emitted = []
    while cur.shape[1] < max_len:
        logits, _ = model(cur)
        next_tok = int(logits[0, -1].argmax().item())
        if next_tok == vocab.eos:
            break
        emitted.append(next_tok)
        cur = torch.cat([cur, torch.tensor([[next_tok]], device=device, dtype=torch.long)], dim=1)

    pc, pp = [], []
    j = 0
    while j + GROUP <= len(emitted):
        c_ok = all(content_lo <= emitted[j + k] < content_hi for k in range(nq_c))
        p_ok = all(pos_lo <= emitted[j + k] < pos_hi for k in range(nq_c, GROUP))
        if not (c_ok and p_ok):
            break
        g = np.array(emitted[j:j + GROUP], dtype=np.int64)
        cl, pl = vocab.decode_element_triples_with_pos(out_mod, g)
        pc.append(cl[0]); pp.append(pl[0])
        j += GROUP
    if pc:
        return np.stack(pc), np.stack(pp)
    return np.zeros((0, nq_c), dtype=np.int64), np.zeros((0, nq_p), dtype=np.int64)


@torch.no_grad()
def ar_decode_force_N(model, vocab, val_ds, idx, n_true, device):
    """Oracle: decode exactly 6 * n_true tokens with EOS masked.
    If the model produces an out-of-range token, fall back to argmax of the
    valid range for that position. Returns (pred_content, pred_pos)."""
    out_mod = val_ds.output_modality
    nq_c = vocab.num_quantizers[out_mod]
    nq_p = vocab.num_q_pos[out_mod]
    GROUP = nq_c + nq_p
    V_c = vocab.codebook_sizes[out_mod]
    V_p = vocab.pos_codebook_sizes[out_mod]
    content_lo = vocab.offsets[out_mod]; content_hi = content_lo + nq_c * V_c
    pos_lo = vocab.pos_offsets[out_mod]; pos_hi = pos_lo + nq_p * V_p

    seq = val_ds.build_sequence(idx)
    out_start_pos, _ = val_ds.output_region_slice(seq)
    prefix_len = out_start_pos + 1
    prefix = torch.from_numpy(seq[:prefix_len]).long().unsqueeze(0).to(device)
    cur = prefix
    target_tokens = int(6 * n_true)
    emitted = []

    # Per-position valid-token range for each slot within a group.
    # positions 0..nq_c-1 → content quantizer i; content_lo + i*V_c .. content_lo + (i+1)*V_c
    # positions nq_c..GROUP-1 → pos quantizer (i-nq_c); pos_lo + (i-nq_c)*V_p .. +V_p
    for step in range(target_tokens):
        if cur.shape[1] >= val_ds.block_size:
            break
        logits, _ = model(cur)
        row = logits[0, -1].clone()
        # Mask EOS so it can never be the argmax
        row[vocab.eos] = float("-inf")
        # Mask positions outside the valid range for this slot
        slot_in_group = step % GROUP
        if slot_in_group < nq_c:  # content quantizer slot_in_group
            lo = content_lo + slot_in_group * V_c
            hi = lo + V_c
        else:
            q = slot_in_group - nq_c
            lo = pos_lo + q * V_p
            hi = lo + V_p
        mask = torch.full_like(row, float("-inf"))
        mask[lo:hi] = row[lo:hi]
        next_tok = int(mask.argmax().item())
        emitted.append(next_tok)
        cur = torch.cat([cur, torch.tensor([[next_tok]], device=device, dtype=torch.long)], dim=1)

    # Parse emitted into groups (guaranteed valid now).
    pc, pp = [], []
    j = 0
    while j + GROUP <= len(emitted):
        g = np.array(emitted[j:j + GROUP], dtype=np.int64)
        cl, pl = vocab.decode_element_triples_with_pos(out_mod, g)
        pc.append(cl[0]); pp.append(pl[0])
        j += GROUP
    if pc:
        return np.stack(pc), np.stack(pp)
    return np.zeros((0, nq_c), dtype=np.int64), np.zeros((0, nq_p), dtype=np.int64)


def load_model(ckpt_path: Path, device: str):
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    v_state = state["vocab"]
    vocab = Vocab.build(
        v_state["modalities"], v_state["codebook_sizes"], v_state["num_quantizers"],
        pos_codebook_sizes=v_state.get("pos_codebook_sizes"),
        num_q_pos=v_state.get("num_q_pos"),
    )
    gpt = GPT(GPTConfig(**state["gpt_config"]))
    gpt.load_state_dict(state["model"])
    gpt.to(device).eval()
    return gpt, vocab, state["config"], int(state.get("step", 0))


def run_mode(model, vocab, val_ds, n_events, device, pflow, out_dir, mode: str):
    """mode in {'free', 'force_N_true'}."""
    out_mod = val_ds.output_modality
    nq_c = vocab.num_quantizers[out_mod]
    nq_p = vocab.num_q_pos[out_mod]
    n_cb_true = val_ds.out_mm.n_codebooks
    n_pos_true = val_ds.out_mm.n_pos_codebooks

    pcs, pps, tcs, tps = [], [], [], []
    n_pred, n_true = [], []
    max_N = 0
    t0 = time.time()
    for i in range(n_events):
        a = int(val_ds.out_mm.offsets[i]); b = int(val_ds.out_mm.offsets[i + 1])
        tc = np.asarray(val_ds.out_mm.data[a:b, :n_cb_true], dtype=np.int64)
        tp = np.asarray(val_ds.out_mm.data[a:b, n_cb_true:n_cb_true + n_pos_true], dtype=np.int64)
        tcs.append(tc); tps.append(tp); n_true.append(len(tc))

        if mode == "free":
            pc, pp = ar_decode_free(model, vocab, val_ds, i, device)
        elif mode == "force_N_true":
            pc, pp = ar_decode_force_N(model, vocab, val_ds, i, len(tc), device)
        else:
            raise ValueError(mode)
        pcs.append(pc); pps.append(pp); n_pred.append(len(pc))
        max_N = max(max_N, len(pc), len(tc))
    print(f"  [{mode}] decoded {n_events} events in {time.time()-t0:.1f}s", flush=True)

    B = n_events
    max_N = max(max_N, 1)
    pred_codes = np.zeros((B, max_N, n_cb_true), dtype=np.int64)
    pred_pos = np.zeros((B, max_N, n_pos_true), dtype=np.int64)
    pred_mask = np.zeros((B, max_N), dtype=bool)
    true_codes = np.zeros((B, max_N, n_cb_true), dtype=np.int64)
    true_pos = np.zeros((B, max_N, n_pos_true), dtype=np.int64)
    true_mask = np.zeros((B, max_N), dtype=bool)
    for i in range(B):
        nt = len(tcs[i])
        true_codes[i, :nt] = tcs[i]; true_pos[i, :nt] = tps[i]; true_mask[i, :nt] = True
        np_ = len(pcs[i])
        if np_ > 0:
            pred_codes[i, :np_, :nq_c] = pcs[i]
            pred_pos[i, :np_, :nq_p] = pps[i]
            pred_mask[i, :np_] = True

    outdir = Path(out_dir) / mode
    outdir.mkdir(parents=True, exist_ok=True)
    metrics = pflow.compute_metrics(
        torch.from_numpy(pred_codes), torch.from_numpy(pred_pos), torch.from_numpy(pred_mask),
        torch.from_numpy(true_codes), torch.from_numpy(true_pos), torch.from_numpy(true_mask),
        outdir=str(outdir), ind_threshold=0.5,
    )
    n_pred_arr = np.array(n_pred, dtype=np.int64)
    n_true_arr = np.array(n_true, dtype=np.int64)
    res = n_pred_arr - n_true_arr
    out = {
        "mode": mode, "n_events": n_events,
        "cardinality_acc": float((n_pred_arr == n_true_arr).mean()),
        "cardinality_mae": float(np.abs(res).mean()),
        "cardinality_bias": float(res.mean()),
        "n_pred_mean": float(n_pred_arr.mean()),
        "n_true_mean": float(n_true_arr.mean()),
    }
    for k in ("mean_jet_pt_response", "median_jet_pt_response",
              "std_jet_pt_response", "iqr_jet_pt_response",
              "mean_reco_cardinality_at_threshold"):
        if k in metrics:
            out[f"pflow_{k}"] = float(metrics[k])
    out["_n_pred_arr"] = n_pred_arr.tolist()
    out["_n_true_arr"] = n_true_arr.tolist()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(REPO / "runs/nano_hep_89M_ddp/last.ckpt"))
    ap.add_argument("--n_events", type=int, default=256)
    ap.add_argument("--out_dir", default=str(REPO / "notebooks/results_oracleN"))
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    print("=== nano-hep Oracle-N diagnostic ===")
    print(f"ckpt: {ckpt_path}  mtime: {datetime.fromtimestamp(ckpt_path.stat().st_mtime).isoformat()}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}  n_events: {args.n_events}")

    model, vocab, cfg, step = load_model(ckpt_path, device)
    print(f"model: step={step}  n_layer={cfg['model']['n_layer']}  n_embd={cfg['model']['n_embd']}")

    in_mods = cfg["data"].get("input_modalities") or [cfg["data"].get("input_modality", "track")]
    val_ds = HEPDataset(
        tokenized_root=cfg["data"]["tokenized_root"], split="val",
        input_modalities=in_mods, output_modality=cfg["data"]["output_modality"],
        block_size=cfg["data"]["block_size"], max_events=args.n_events, vocab=vocab,
    )
    pflow = PflowMetrics(
        modality_dict_path=cfg["pflow_metrics"]["modality_dict_path"],
        output_modality=cfg["data"]["output_modality"], device=device,
    )
    print()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for mode in ("free", "force_N_true"):
        print(f"--- {mode} ---")
        r = run_mode(model, vocab, val_ds, args.n_events, device, pflow, out_dir, mode)
        for k in ("cardinality_acc", "cardinality_mae", "cardinality_bias",
                  "pflow_median_jet_pt_response", "pflow_iqr_jet_pt_response",
                  "pflow_mean_reco_cardinality_at_threshold"):
            if k in r:
                print(f"  {k}: {r[k]:.4f}")
        results.append(r)

    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in results])
    df.to_csv(out_dir / "oracle_N_results.csv", index=False)
    with open(out_dir / "oracle_N_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n=== DONE ===")
    print(f"Results: {out_dir / 'oracle_N_results.csv'}")


if __name__ == "__main__":
    main()
