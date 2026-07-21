"""BDT ablations for n_truth prediction.

Builds on bdt_n_truth.py — decodes more data, tries more model variants, and
estimates the Bayes ceiling by binning events on (n_track, n_topo) and
measuring the within-bin spread of n_truth.

Ablations:
  A. GradientBoostingRegressor @ defaults (baseline, already run)
  B. HistGradientBoostingRegressor — 10-100x faster, often better
  C. HistGradientBoostingClassifier — directly optimizes exact-N via CE
  D. RandomForestRegressor — orthogonal ensemble, different inductive bias
  E. Deeper HGB (depth=10, leaves=255) with 2000 trees
  F. Feature-lite HGB: ONLY (n_track, n_topo, ht) to see how much signal is in
     counts vs energy distribution

Bayes ceiling:
  Bin events on (n_track, n_topo). For each populated bin, the within-bin
  variance of n_truth is the irreducible noise floor. Compute:
    - Bin-conditional mean (the Bayes predictor rounded to nearest int)
    - Bin-conditional exact-N accuracy = P(n_truth == round(E[n_truth | n_track, n_topo]))
    - Bin-conditional within-1 accuracy
  Average over test-set events.

Cache decoded features to disk on first run; reuse on subsequent.
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
if str(REPO) not in sys.path: sys.path.insert(0, str(REPO))
if HEP4M not in sys.path: sys.path.insert(0, HEP4M)

from nano_hep.data import ModalityMemmap
from sklearn.ensemble import (GradientBoostingRegressor, HistGradientBoostingRegressor,
                               HistGradientBoostingClassifier, RandomForestRegressor)
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(REPO / "notebooks"))
from bdt_n_truth import decode_modality_features, build_features  # noqa: E402


def decode_or_load(split: str, n_events: int, device: str, cache_dir: Path):
    cache = cache_dir / f"features_{split}_{n_events}.npz"
    if cache.exists():
        print(f"  loading cached features {cache}")
        z = np.load(cache, allow_pickle=True)
        return pd.DataFrame({k: z[k] for k in z.files})
    print(f"  decoding tracks ({n_events} events)...")
    t0 = time.time()
    track_feats, n_track = decode_modality_features("track", split, n_events, device)
    print(f"    done in {time.time()-t0:.1f}s")
    print(f"  decoding topos...")
    t0 = time.time()
    topo_feats, n_topo = decode_modality_features("topo", split, n_events, device)
    print(f"    done in {time.time()-t0:.1f}s")

    truth_mm = ModalityMemmap.load("/global/cfs/cdirs/m4958/data/COCOA/Tokenized_89M", split, "truthpart")
    N = min(len(n_track), len(n_topo), truth_mm.n_events)
    n_truth = (truth_mm.offsets[1:N+1] - truth_mm.offsets[:N]).astype(np.int64)
    n_track = n_track[:N]; n_topo = n_topo[:N]
    track_feats = tuple([lst[:N] for lst in track_feats])
    topo_feats = tuple([lst[:N] for lst in topo_feats])
    X = build_features(track_feats, topo_feats, n_track, n_topo)
    X["n_truth"] = n_truth
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **{c: X[c].to_numpy() for c in X.columns})
    print(f"  cached features → {cache}")
    return X


def metrics(y_te, y_pred_raw):
    y_int = np.round(y_pred_raw).clip(0, 100).astype(np.int64)
    return {
        "r2": float(r2_score(y_te, y_pred_raw)),
        "mae": float(mean_absolute_error(y_te, y_int)),
        "exact": float(np.mean(y_int == y_te)),
        "within1": float(np.mean(np.abs(y_int - y_te) <= 1)),
        "within2": float(np.mean(np.abs(y_int - y_te) <= 2)),
    }


def bayes_ceiling(X_tr: pd.DataFrame, X_te: pd.DataFrame):
    """Bin on (n_track, n_topo), predict bin-conditional mean (nearest int)."""
    tr = X_tr.copy()
    means = tr.groupby(["n_track", "n_topo"])["n_truth"].mean()
    modes = tr.groupby(["n_track", "n_topo"])["n_truth"].apply(lambda s: s.value_counts().idxmax())
    # Fall back to global mean for unseen bins
    global_mean = tr["n_truth"].mean()
    pred_mean = X_te.apply(
        lambda r: means.get((r["n_track"], r["n_topo"]), global_mean), axis=1
    ).to_numpy()
    pred_mode = X_te.apply(
        lambda r: modes.get((r["n_track"], r["n_topo"]), int(round(global_mean))), axis=1
    ).to_numpy()
    y_te = X_te["n_truth"].to_numpy()
    return {
        "bin_mean": metrics(y_te, pred_mean),
        "bin_mode": metrics(y_te, pred_mode.astype(np.float64)),
        "n_populated_bins": int(tr.groupby(["n_track", "n_topo"]).size().shape[0]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_events", type=int, default=100000)
    ap.add_argument("--split", default="val")
    ap.add_argument("--cache_dir", default=str(REPO / "notebooks/cache_bdt"))
    ap.add_argument("--out_dir", default=str(REPO / "notebooks/results_bdt_n_ablations"))
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    Xy = decode_or_load(args.split, args.n_events, device, Path(args.cache_dir))
    print(f"  Xy shape: {Xy.shape}, y: mean={Xy['n_truth'].mean():.2f} std={Xy['n_truth'].std():.2f}")

    feat_cols = [c for c in Xy.columns if c != "n_truth"]
    X = Xy[feat_cols]
    y = Xy["n_truth"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"  train {len(X_tr)}, test {len(X_te)}")

    results = {}

    print("\n== Baselines ==")
    results["constant"] = metrics(y_te, np.full(len(y_te), y_tr.mean()))
    print(f"  constant (μ={y_tr.mean():.2f}): {results['constant']}")
    results["n_track_copy"] = metrics(y_te, X_te["n_track"].to_numpy().astype(float))
    print(f"  n_track copy:               {results['n_track_copy']}")
    md = (y_tr - X_tr["n_track"]).mean()
    results["n_track_plus_delta"] = metrics(y_te, X_te["n_track"].to_numpy() + md)
    print(f"  n_track + {md:.2f}:          {results['n_track_plus_delta']}")

    print("\n== Bayes ceiling (bin on n_track, n_topo) ==")
    bce = bayes_ceiling(Xy.iloc[X_tr.index], Xy.iloc[X_te.index])
    results["bayes_bin_mean"] = bce["bin_mean"]
    results["bayes_bin_mode"] = bce["bin_mode"]
    print(f"  n_populated_bins: {bce['n_populated_bins']}")
    print(f"  bin-cond mean (regression-optimal): {bce['bin_mean']}")
    print(f"  bin-cond mode (classification-optimal): {bce['bin_mode']}")

    print("\n== Full-feature models ==")
    # A. GBR (reference from prev run, retrained on bigger data)
    t0 = time.time()
    gbr = GradientBoostingRegressor(n_estimators=500, max_depth=5, learning_rate=0.05, random_state=42)
    gbr.fit(X_tr, y_tr)
    results["GBR_500x5"] = metrics(y_te, gbr.predict(X_te))
    print(f"  GBR_500x5 ({time.time()-t0:.1f}s): {results['GBR_500x5']}")

    # B. HGB regressor defaults
    t0 = time.time()
    hgb = HistGradientBoostingRegressor(max_iter=500, max_depth=None, max_leaf_nodes=63,
                                         learning_rate=0.05, random_state=42)
    hgb.fit(X_tr, y_tr)
    results["HGB_500"] = metrics(y_te, hgb.predict(X_te))
    print(f"  HGB_500    ({time.time()-t0:.1f}s): {results['HGB_500']}")

    # C. HGB deep
    t0 = time.time()
    hgb_deep = HistGradientBoostingRegressor(max_iter=2000, max_leaf_nodes=255, learning_rate=0.03,
                                               min_samples_leaf=20, random_state=42, early_stopping=True)
    hgb_deep.fit(X_tr, y_tr)
    results["HGB_deep"] = metrics(y_te, hgb_deep.predict(X_te))
    print(f"  HGB_deep   ({time.time()-t0:.1f}s): {results['HGB_deep']}")

    # D. HGB Classifier — directly optimizes log-likelihood of integer N
    t0 = time.time()
    # Clip y to [0, 18] — rare classes above 18 cause stratified-split errors in early_stopping
    y_tr_cls = y_tr.clip(0, 18)
    y_te_cls = y_te.clip(0, 18)
    cls = HistGradientBoostingClassifier(max_iter=1000, max_leaf_nodes=63, learning_rate=0.05,
                                          min_samples_leaf=20, random_state=42,
                                          early_stopping=False)
    cls.fit(X_tr, y_tr_cls)
    # Use predicted class (argmax) for exact-N, predicted mean of P(N) for R²/MAE
    y_pred_cls_int = cls.predict(X_te).astype(np.int64)
    proba = cls.predict_proba(X_te)
    classes = cls.classes_
    y_pred_cls_mean = (proba * classes[None, :]).sum(axis=1)
    results["HGBClassifier_argmax"] = metrics(y_te, y_pred_cls_int.astype(float))
    results["HGBClassifier_mean"]   = metrics(y_te, y_pred_cls_mean)
    print(f"  HGBCls     ({time.time()-t0:.1f}s): argmax={results['HGBClassifier_argmax']}")
    print(f"                     mean={results['HGBClassifier_mean']}")

    # E. RandomForest for orthogonal inductive bias
    t0 = time.time()
    rf = RandomForestRegressor(n_estimators=500, max_depth=None, min_samples_leaf=5,
                                random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    results["RF_500"] = metrics(y_te, rf.predict(X_te))
    print(f"  RF_500     ({time.time()-t0:.1f}s): {results['RF_500']}")

    # F. Feature-lite: only (n_track, n_topo, ht)
    print("\n== Feature ablation (lite: n_track, n_topo, ht) ==")
    lite_cols = ["n_track", "n_topo", "ht"]
    hgb_lite = HistGradientBoostingRegressor(max_iter=500, max_leaf_nodes=63,
                                              learning_rate=0.05, random_state=42)
    hgb_lite.fit(X_tr[lite_cols], y_tr)
    results["HGB_lite_3feat"] = metrics(y_te, hgb_lite.predict(X_te[lite_cols]))
    print(f"  HGB_lite_3feat: {results['HGB_lite_3feat']}")

    # G. counts-only: just (n_track, n_topo)
    hgb_counts = HistGradientBoostingRegressor(max_iter=500, max_leaf_nodes=63,
                                                learning_rate=0.05, random_state=42)
    hgb_counts.fit(X_tr[["n_track", "n_topo"]], y_tr)
    results["HGB_counts_2feat"] = metrics(y_te, hgb_counts.predict(X_te[["n_track", "n_topo"]]))
    print(f"  HGB_counts_2feat (just 2 counts): {results['HGB_counts_2feat']}")

    # Best feature importances (from HGB_deep)
    print("\n== Feature importances (HGB_deep via permutation) ==")
    from sklearn.inspection import permutation_importance
    imp = permutation_importance(hgb_deep, X_te, y_te, n_repeats=3, random_state=42, n_jobs=-1, scoring="r2")
    imp_ser = pd.Series(imp.importances_mean, index=feat_cols).sort_values(ascending=False)
    print(imp_ser.head(12).to_string())

    # Summary + save
    print("\n== SUMMARY ==")
    rows = []
    for name, m in results.items():
        rows.append({"model": name, **m})
    df = pd.DataFrame(rows).sort_values("exact", ascending=False)
    print(df.to_string(index=False))
    df.to_csv(out / "ablation_results.csv", index=False)
    json.dump(results, open(out / "ablation_results.json", "w"), indent=2)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
