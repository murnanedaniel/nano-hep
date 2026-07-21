"""BDT for predicting n_truth from high-level event features.

Given the empirical finding that n_truth >= n_track in 100% of COCOA val
events (charged particles always have a track), the residual n_truth -
n_track is the "neutral particle count" and is what the AR model has to
infer from the topocluster pattern. This script trains a sklearn BDT to
see how far simple aggregate features can take us — gives us the
achievable-floor on cardinality accuracy for a non-tokenizing baseline.

Features (all derivable from tokenized data + VQ decode):
  - n_track, n_topo
  - Σ pt_track, Σ pt_topo, max pt_track, max pt_topo, mean/std pt_{track,topo}
  - Σ E, mean/std of η and φ (cos, sin) per modality
  - Top-1, top-2 pt values per modality
  - event-level HT = Σ pt_track + Σ pt_topo

Target: n_truth (integer, 0..~27)
Metrics: R², MAE, exact-N accuracy, |Δ|≤1 accuracy.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
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

from nano_hep.data import ModalityMemmap

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
def decode_modality_features(mod_name: str, split: str, n_events: int, device: str):
    """Decode tokens -> physical features via HEP4M VQ-VAE + PosTokenizer.
    Returns per-event lists: pt (jagged), eta (jagged), phi (jagged), also
    counts n_per_event.
    """
    from hep4m.models.vqvae import VQVAE
    from hep4m.models.pos_tokenizer import PosTokenizer
    from hep4m.utility.var_transformation import VarTransformation
    import yaml

    md_path = REPO.parent / "HEP4M" / "data" / "experiments" / "HEP4M-pflow-pilot" / "pflow_pilot_10M_nocard" / "modality_dict.yml"
    with open(md_path) as f:
        md = yaml.safe_load(f)
    entry = md[mod_name]
    with open(entry["config_path_v"]) as f:
        cfg_v = yaml.safe_load(f)
    with open(entry["config_path_m"]) as f:
        cfg_m = yaml.safe_load(f)
    cfg_m["_config_v"] = cfg_v
    vq = VQVAE(cfg_m)
    state = torch.load(entry["checkpoint_path"], map_location="cpu", weights_only=False)
    if "state_dict" in state:
        state = state["state_dict"]
    state = {k.replace("model.", "", 1) if k.startswith("model.") else k: v for k, v in state.items()}
    vq.load_state_dict(state, strict=False)
    vq.to(device).eval()
    for p in vq.parameters():
        p.requires_grad = False

    pos_tok = PosTokenizer().to(device)

    feat_names = cfg_v["features"][f"{mod_name}_feat0"][1]
    tdict = cfg_v.get("transformation_dict", {})
    # pt_log or pt, always first cont feat
    inv_pt = VarTransformation(tdict[feat_names[0]]) if feat_names[0] in tdict else None

    mm = ModalityMemmap.load("/global/cfs/cdirs/m4958/data/COCOA/Tokenized_89M", split, mod_name)

    N = min(mm.n_events, n_events)
    n_per_event = (mm.offsets[1:N+1] - mm.offsets[:N]).astype(np.int64)

    # Pad to max_N for batched decode
    max_N = int(n_per_event.max())
    if max_N == 0:
        return [np.array([], dtype=np.float32)] * 3, n_per_event
    codes = np.zeros((N, max_N, mm.n_codebooks), dtype=np.int64)
    pos_codes = np.zeros((N, max_N, mm.n_pos_codebooks), dtype=np.int64)
    mask = np.zeros((N, max_N), dtype=bool)
    for i in range(N):
        ne = int(n_per_event[i])
        if ne == 0: continue
        a = int(mm.offsets[i]); b = a + ne
        codes[i, :ne] = np.asarray(mm.data[a:b, :mm.n_codebooks], dtype=np.int64)
        pos_codes[i, :ne] = np.asarray(mm.data[a:b, mm.n_codebooks:mm.n_codebooks + mm.n_pos_codebooks], dtype=np.int64)
        mask[i, :ne] = True

    # Batch decode in chunks to avoid GPU OOM.
    chunk = 512
    pts = np.zeros((N, max_N), dtype=np.float32)
    etas = np.zeros((N, max_N), dtype=np.float32)
    cosphis = np.zeros((N, max_N), dtype=np.float32)
    sinphis = np.zeros((N, max_N), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            c = torch.from_numpy(codes[s:e]).to(device)
            pc = torch.from_numpy(pos_codes[s:e]).to(device)
            m = torch.from_numpy(mask[s:e]).to(device)
            z_q = vq.indices_to_zq(c, m)
            x_hat_cont, _ = vq.decode(z_q, x_mask=m)
            pt = x_hat_cont[..., 0].cpu().numpy()
            if inv_pt is not None:
                pt = inv_pt.inverse(pt)
            x_gpos = pos_tok.decode(pc).cpu().numpy()  # (B, N, 3): eta, cosphi, sinphi
            pts[s:e] = pt
            etas[s:e] = x_gpos[..., 0]
            cosphis[s:e] = x_gpos[..., 1]
            sinphis[s:e] = x_gpos[..., 2]

    # Return jagged lists per event
    pt_jag = [pts[i, :int(n_per_event[i])] for i in range(N)]
    eta_jag = [etas[i, :int(n_per_event[i])] for i in range(N)]
    phi_jag = [np.arctan2(sinphis[i, :int(n_per_event[i])], cosphis[i, :int(n_per_event[i])]) for i in range(N)]
    return (pt_jag, eta_jag, phi_jag), n_per_event


def build_features(track_feats, topo_feats, n_track, n_topo):
    track_pt, track_eta, track_phi = track_feats
    topo_pt, topo_eta, topo_phi = topo_feats
    N = len(n_track)

    def agg(jag, n):
        """Per-event: (sum, max, top2, top3, mean, std). Zero-fill for empty."""
        s = np.zeros(N); mx = np.zeros(N); t2 = np.zeros(N); t3 = np.zeros(N)
        mn = np.zeros(N); sd = np.zeros(N)
        for i in range(N):
            if n[i] == 0: continue
            v = jag[i]
            s[i] = v.sum()
            srt = np.sort(v)[::-1]
            mx[i] = srt[0]
            t2[i] = srt[1] if len(srt) > 1 else 0.0
            t3[i] = srt[2] if len(srt) > 2 else 0.0
            mn[i] = v.mean()
            sd[i] = v.std() if len(v) > 1 else 0.0
        return s, mx, t2, t3, mn, sd

    def agg_eta(jag, n):
        mn = np.zeros(N); sd = np.zeros(N); ptp = np.zeros(N)
        for i in range(N):
            if n[i] == 0: continue
            v = jag[i]
            mn[i] = v.mean()
            sd[i] = v.std() if len(v) > 1 else 0.0
            ptp[i] = v.max() - v.min() if len(v) > 1 else 0.0
        return mn, sd, ptp

    tr_pt_s, tr_pt_mx, tr_pt_t2, tr_pt_t3, tr_pt_mn, tr_pt_sd = agg(track_pt, n_track)
    tp_pt_s, tp_pt_mx, tp_pt_t2, tp_pt_t3, tp_pt_mn, tp_pt_sd = agg(topo_pt, n_topo)
    tr_eta_mn, tr_eta_sd, tr_eta_ptp = agg_eta(track_eta, n_track)
    tp_eta_mn, tp_eta_sd, tp_eta_ptp = agg_eta(topo_eta, n_topo)

    # HT and ratios
    ht = tr_pt_s + tp_pt_s
    pt_ratio = np.where(tp_pt_s > 0, tr_pt_s / (tp_pt_s + 1e-6), 0.0)

    features = {
        "n_track": n_track, "n_topo": n_topo,
        "n_tot": n_track + n_topo,
        "tr_pt_sum": tr_pt_s, "tr_pt_max": tr_pt_mx, "tr_pt_t2": tr_pt_t2, "tr_pt_t3": tr_pt_t3,
        "tr_pt_mean": tr_pt_mn, "tr_pt_std": tr_pt_sd,
        "tp_pt_sum": tp_pt_s, "tp_pt_max": tp_pt_mx, "tp_pt_t2": tp_pt_t2, "tp_pt_t3": tp_pt_t3,
        "tp_pt_mean": tp_pt_mn, "tp_pt_std": tp_pt_sd,
        "tr_eta_mean": tr_eta_mn, "tr_eta_std": tr_eta_sd, "tr_eta_range": tr_eta_ptp,
        "tp_eta_mean": tp_eta_mn, "tp_eta_std": tp_eta_sd, "tp_eta_range": tp_eta_ptp,
        "ht": ht, "tr_tp_pt_ratio": pt_ratio,
    }
    return pd.DataFrame(features)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_events", type=int, default=10000)
    ap.add_argument("--split", default="val")
    ap.add_argument("--out_dir", default=str(REPO / "notebooks/results_bdt_n"))
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Decoding {args.n_events} {args.split} events on {device}...")

    t0 = time.time()
    track_feats, n_track = decode_modality_features("track", args.split, args.n_events, device)
    print(f"  track decoded in {time.time()-t0:.1f}s; mean n_track = {n_track.mean():.2f}")
    t0 = time.time()
    topo_feats, n_topo = decode_modality_features("topo", args.split, args.n_events, device)
    print(f"  topo decoded in {time.time()-t0:.1f}s; mean n_topo = {n_topo.mean():.2f}")

    # Truth cardinality (the target)
    truth_mm = ModalityMemmap.load("/global/cfs/cdirs/m4958/data/COCOA/Tokenized_89M", args.split, "truthpart")
    N = min(len(n_track), len(n_topo), truth_mm.n_events)
    n_truth = (truth_mm.offsets[1:N+1] - truth_mm.offsets[:N]).astype(np.int64)
    # Align lengths
    n_track = n_track[:N]; n_topo = n_topo[:N]
    track_feats = tuple([lst[:N] for lst in track_feats])
    topo_feats = tuple([lst[:N] for lst in topo_feats])

    X = build_features(track_feats, topo_feats, n_track, n_topo)
    y = n_truth
    print(f"  Features: {list(X.columns)}")
    print(f"  X shape: {X.shape}, y: mean={y.mean():.2f} std={y.std():.2f}")

    # Train / test split
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"  train: {len(X_tr)}, test: {len(X_te)}")

    # === Baselines ===
    print()
    print("=== Baselines ===")
    # B1: constant = train mean (rounded)
    y_pred_const = np.full_like(y_te, int(round(y_tr.mean())))
    print(f"  constant-mean pred ({int(round(y_tr.mean()))}):  "
          f"R²={r2_score(y_te, y_pred_const):.4f} MAE={mean_absolute_error(y_te, y_pred_const):.3f} "
          f"exact={np.mean(y_pred_const==y_te):.4f} within1={np.mean(np.abs(y_pred_const-y_te)<=1):.4f}")
    # B2: n_track alone (copy from charged count)
    y_pred_track = X_te["n_track"].to_numpy().astype(np.int64)
    print(f"  n_track copy:                   "
          f"R²={r2_score(y_te, y_pred_track):.4f} MAE={mean_absolute_error(y_te, y_pred_track):.3f} "
          f"exact={np.mean(y_pred_track==y_te):.4f} within1={np.mean(np.abs(y_pred_track-y_te)<=1):.4f}")
    # B3: n_track + mean delta from train set
    mean_delta = int(round((y_tr - X_tr["n_track"]).mean()))
    y_pred_b3 = X_te["n_track"].to_numpy().astype(np.int64) + mean_delta
    print(f"  n_track + mean_delta ({mean_delta}):       "
          f"R²={r2_score(y_te, y_pred_b3):.4f} MAE={mean_absolute_error(y_te, y_pred_b3):.3f} "
          f"exact={np.mean(y_pred_b3==y_te):.4f} within1={np.mean(np.abs(y_pred_b3-y_te)<=1):.4f}")

    # === BDT ===
    print()
    print("=== GradientBoostingRegressor ===")
    t0 = time.time()
    bdt = GradientBoostingRegressor(n_estimators=500, max_depth=5, learning_rate=0.05,
                                    random_state=42, verbose=0)
    bdt.fit(X_tr, y_tr)
    y_pred = bdt.predict(X_te)
    y_pred_int = np.round(y_pred).clip(0, 50).astype(np.int64)
    print(f"  trained in {time.time()-t0:.1f}s")
    print(f"  BDT (continuous):              "
          f"R²={r2_score(y_te, y_pred):.4f} MAE={mean_absolute_error(y_te, y_pred):.3f}")
    print(f"  BDT (rounded to nearest int):  "
          f"R²={r2_score(y_te, y_pred_int):.4f} MAE={mean_absolute_error(y_te, y_pred_int):.3f} "
          f"exact={np.mean(y_pred_int==y_te):.4f} within1={np.mean(np.abs(y_pred_int-y_te)<=1):.4f}")

    print()
    print("=== Top feature importances ===")
    imp = pd.Series(bdt.feature_importances_, index=X.columns).sort_values(ascending=False)
    print(imp.head(12).to_string())

    # Save
    pd.DataFrame({"y_true": y_te, "y_pred_bdt": y_pred_int, "y_pred_nt": y_pred_track,
                  "n_track": X_te["n_track"].to_numpy()}).to_csv(out / "bdt_preds.csv", index=False)
    json.dump({
        "n_events": args.n_events,
        "feature_importances": imp.to_dict(),
        "bdt_exact_acc": float(np.mean(y_pred_int == y_te)),
        "bdt_within1_acc": float(np.mean(np.abs(y_pred_int - y_te) <= 1)),
        "bdt_mae": float(mean_absolute_error(y_te, y_pred_int)),
        "bdt_r2": float(r2_score(y_te, y_pred_int)),
        "n_track_copy_exact": float(np.mean(y_pred_track == y_te)),
        "n_track_copy_mae": float(mean_absolute_error(y_te, y_pred_track)),
    }, open(out / "bdt_summary.json", "w"), indent=2)
    print(f"\nSaved results → {out}")


if __name__ == "__main__":
    main()
