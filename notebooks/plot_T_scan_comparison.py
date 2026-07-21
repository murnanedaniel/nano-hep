"""Overlay nano-hep vs HEP4M token-temperature scan curves.

Reads the two CSVs produced by the per-model scans (identical column
schema) and plots jet-response + cardinality metrics vs T on shared axes.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.sort_values("T").reset_index(drop=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nano_csv", default="/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/nano-hep/notebooks/results_T_scan/nano_hep_T_scan.csv")
    ap.add_argument("--hep4m_csv", default="/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/HEP4M/notebooks/paper/results_T_scan/hep4m_T_scan.csv")
    ap.add_argument("--out", default="/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/nano-hep/notebooks/results_T_scan/comparison_T_scan.png")
    args = ap.parse_args()

    nano = load(Path(args.nano_csv))
    hep4m = load(Path(args.hep4m_csv))

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True)

    def plot_metric(ax, col, ylabel, title, truth_line=None):
        ax.plot(nano["T"], nano[col], "o-", color="#1f77b4", lw=2, ms=7, label="nano-hep (step 8000)")
        ax.plot(hep4m["T"], hep4m[col], "s-", color="#d62728", lw=2, ms=7, label="HEP4M (epoch 11)")
        if truth_line is not None:
            ax.axhline(truth_line, color="k", ls=":", lw=1, alpha=0.6, label=f"truth ({truth_line:.2f})")
        ax.axvline(1.0, color="grey", ls="--", lw=0.8, alpha=0.5)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    # Row 1: jet-level quality
    plot_metric(axes[0, 0], "pflow_median_jet_pt_response",
                "median jet pT response",
                "Jet pT response: median", truth_line=1.0)
    plot_metric(axes[0, 1], "pflow_iqr_jet_pt_response",
                "IQR(jet pT response)",
                "Jet pT response: IQR (lower=tighter)")
    n_true = float(nano["n_true_mean"].iloc[0])
    plot_metric(axes[0, 2], "pflow_mean_reco_cardinality_at_threshold",
                "mean N reco particles",
                "Reco cardinality @ threshold", truth_line=n_true)

    # Row 2: cardinality head quality
    plot_metric(axes[1, 0], "cardinality_acc",
                "P(n_pred == n_true)",
                "Cardinality accuracy")
    plot_metric(axes[1, 1], "cardinality_mae",
                "|n_pred - n_true|",
                "Cardinality MAE (lower=better)")
    plot_metric(axes[1, 2], "n_pred_mean",
                "mean n_pred",
                "Predicted N particles", truth_line=n_true)

    axes[0, 0].legend(loc="best", fontsize=9)
    for ax in axes[1, :]:
        ax.set_xlabel("Temperature T")
    fig.suptitle("Token temperature scan — nano-hep vs HEP4M (256 val events, k=-1)", fontsize=13)
    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"saved {out_path}")

    # Also emit a combined CSV with a 'model' column for anyone wanting to re-plot.
    nano["model"] = "nano-hep"
    hep4m["model"] = "HEP4M"
    combined = pd.concat([nano, hep4m], ignore_index=True)
    combined_csv = out_path.with_suffix(".combined.csv")
    combined.to_csv(combined_csv, index=False)
    print(f"saved {combined_csv}")


if __name__ == "__main__":
    main()
