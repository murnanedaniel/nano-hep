# Overnight apples-to-apples: HEP4M (parallel) vs nano-hep (AR) on 1M COCOA events

Both models trained from scratch on the same 1M-event slice of
`/global/cfs/cdirs/m4958/data/COCOA/Tokenized/` (track+topo → truthpart),
using the same VQ codebooks and the same `run_report_from_arrays` metric.

## Training summary

| | HEP4M (parallel) | nano-hep (AR, DDP) |
|---|---|---|
| architecture | Encoder + positional-query decoder | nanoGPT causal LM (upstream) |
| parameters | 232M | 86.4M |
| training events | 1,000,000 | 1,000,000 |
| hardware | 4 nodes × 4 A100 (16 GPUs) | 4 nodes × 4 A100 (16 GPUs, DDP) |
| effective batch | 256 × 16 = 4096 | 96 × 16 = 1536 |
| schedule | 60 epochs, schedulefree lr=1e-3 | 15000 steps, cosine lr peak 6e-4 |
| wall time | ~100 min | ~75 min |
| best val loss | 7.1207 (epoch 8) | 2.1201 (step ~5500) |
| final val loss | (overfit after ep 8, best ckpt used) | 3.33 (overfit; best.ckpt used) |

Both models overfit on 1M events within the first ~15% of the training budget,
so the numbers below use the best-val-loss checkpoint for each.

## Jet-level metrics (2000 val events, identical eval path)

| metric | HEP4M 1M (ep 8) | nano-hep 1M DDP (best) | nano-hep pilot (prior, 100k) | HEP4M pilot (prior, 10M ep 3) |
|---|---|---|---|---|
| mean_jet_pt_response | 3.810 | **0.978** | 0.957 | 0.510 |
| median_jet_pt_response | 1.173 | **0.950** | 0.943 | 0.282 |
| std_jet_pt_response | 12.491 | **0.455** | 0.462 | 2.116 |
| iqr_jet_pt_response | 0.687 | **0.209** | 0.237 | 0.364 |
| mean_reco_cardinality (true ≈ 6.4) | 13.77 (**2× over**) | **6.08** | 6.49 | 6.12 |

## Headline

**At equal data (1M events), equal compute class (16 GPUs),
equal codebook, equal metric — the AR model wins decisively on every
jet-kinematic metric.**

- Jet pT response is **centered near 1** for nano-hep (0.978 mean, 0.950 median)
  vs HEP4M's **3.8 mean / 1.17 median** — HEP4M systematically over-predicts
  jet energies by ~2× because it predicts ~2× too many particles.
- Spread is **20× tighter** for nano-hep (std 0.46 vs 12.49) —
  HEP4M has heavy-tailed outlier events that dominate the mean.
- Cardinality is **spot-on** for nano-hep (6.08 vs true 6.4) and **catastrophic**
  for HEP4M (13.77 — 2.15× the truth).

The cardinality blow-up is the root cause: HEP4M's indicator head (threshold
0.45) passes too many slots, each slot gets decoded into a particle, jet pT
sums up both real and phantom particles. AR's cardinality comes from the
length of the emitted token sequence before `<EOS>`, which the model learns
to place correctly.

## What changed from the pilot comparison

`docs/comparison_ar_vs_parallel.md` (prior) used:
- HEP4M: pretrained 10M × 3 epoch (epoch 3 ckpt, user asserts this is overfit/memorized).
- nano-hep: 100k events × 5000 steps, 26.8M params, single GPU.

This run:
- Both **1M events, 4×4 GPU, same codebook, same metric**.
- nano-hep scaled to **86.4M params** with DDP (16-way), batch 1536.
- HEP4M trained from scratch on twostep branch (232M params).

## Caveats

1. **gpos still substituted from truth** for both — neither model emits
   eta/phi tokens in v0, so the jet pT response reflects VQ codebook-content
   quality, not full kinematic reconstruction. Both sides get the same
   gpos assistance.
2. **HEP4M converged at epoch 8** on 1M, the rest is overfit. 1M was probably
   too small for 232M params. A 10M re-run on 1M would be sample-limited.
3. **nano-hep converged at step ~5500** (val_loss 2.12 → 2.38 by step 10500).
   Effective batch 1536 × 5500 = ~8.4M tokens seen, ~1 epoch-equivalent.
4. Cardinality is a function of the indicator threshold (0.45 for HEP4M,
   element-count-based for nano-hep) — both funnel through the same
   `run_report_from_arrays` so the `mean_reco_cardinality_at_threshold`
   is directly comparable.

## Reproducing

```bash
# HEP4M 1M eval:
/pscratch/sd/d/danieltm/envs/hep4m2/bin/python scripts/eval_hep4m_reference.py \
  --run_dir /global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/HEP4M/data/experiments/HEP4M-twostep-overnight/twostep_1M_overnightxxx0t5zdax3f \
  --ckpt epoch=8-val_total_loss=7.1207.ckpt \
  --reduce_ds_val 2000 --out_json runs/eval_overnight/hep4m_1M_overnight.json

# nano-hep DDP 1M eval:
/pscratch/sd/d/danieltm/envs/hep4m2/bin/python scripts/eval_nanohep_ckpt.py \
  --ckpt runs/overnight_1M_ddp/best.ckpt --n_val 2000 \
  --out_json runs/eval_overnight/nanohep_1M_overnight.json
```

Both produce JSON with identical metric keys (output of
`hep4m.performance.pflow_report.run_report_from_arrays`). Same val events,
same detokenize path, same metric function.
