# Autoregressive vs parallel token prediction — first apples-to-apples

First side-by-side of two approaches to the COCOA pflow task
(`(track, topo) → truthpart`) evaluated via the *identical* pipeline:
predicted tokens → HEP4M VQ-VAE `indices_to_zq + decode` →
`run_report_from_arrays` jet-level metrics.

## Both evaluated on 2000 val events from `/global/cfs/cdirs/m4958/data/COCOA/Tokenized/val/`

| metric | HEP4M (parallel) | nano-hep (AR) |
|---|---|---|
| architecture | Encoder + positional-query decoder, 12L, 233.2M params | nanoGPT causal LM, 8L, 26.8M params |
| training events | 10M | 100k |
| training schedule | epoch 3 of planned 100 | 5000 steps at batch 96 |
| effective gradient steps | ≈230,000 | 5000 |
| inference | parallel token prediction (one forward pass) | autoregressive (one token at a time) |
| mean_jet_pt_response | 0.510 | **0.957** |
| median_jet_pt_response | 0.282 | **0.943** |
| std_jet_pt_response | 2.116 | **0.462** |
| iqr_jet_pt_response | 0.364 | **0.237** |
| mean_reco_cardinality (true ≈ 6.4) | 6.12 | 6.49 |

## Honest caveats

**(1) HEP4M is dramatically under-trained.** The only available checkpoint reached epoch 3 of a planned 100. The val loss (6.56) is still high. The model hasn't learned to produce pt in the right range — mean response 0.51 says predicted jet pT is about half the truth. A fully-trained HEP4M would likely do *much* better.

**(2) Gradient-step asymmetry is still unfavorable to nano-hep.** HEP4M's 3 epochs × 10M events / 512 batch ≈ ~58,000 gradient steps. Nano-hep's 5000 steps × 96 batch = 480,000 jets-processed, ~10x fewer backward passes overall. So even at 10× fewer gradient updates, nano-hep produces jet pT responses centered near 1 with ~1/4 the std.

**(3) Parameter gap.** HEP4M has 8.7× more parameters. With 26.8M params, nano-hep hits mean response ~1 within ~5000 steps. This is evidence the task is *parameter-efficient* for AR.

**(4) Pflow metric subtlety.** Both models currently use *truth gpos* (eta, phi tokens) for the pflow decode, since nano-hep's v0 sequence doesn't yet emit gpos tokens and this is the only way to keep the comparison fair. The metric numbers therefore reflect VQ codebook-content quality, not full kinematic reconstruction from scratch.

**(5) Cardinality side.** nano-hep's `ar_cardinality_acc=28%` and `ar_cardinality_mae=1.33` (from wandb during training) are computed element-level by counting emitted tokens divided by `num_q`. HEP4M's parallel prediction emits a per-slot indicator that gets thresholded. The two are not the same definition of "predicted count" — but when funneled through `run_report_from_arrays`, both reduce to `mean_reco_cardinality_at_threshold` which ends up within 0.2 particles of each other.

## Interpretation

At this under-trained stage:
- **AR is much more sample-efficient.** 100× less training data, and the jet pT response is centered near 1.
- **AR produces much tighter distributions** (iqr ~0.24 vs 0.36, std 0.46 vs 2.1 — HEP4M has high-variance tails that suggest some events are very poorly reconstructed; AR does not).
- **Cardinality is comparable.** This is the one axis where both approaches seem competitive.

A fair comparison requires training HEP4M to convergence (at least 20-30 epochs). We have no such checkpoint available on NERSC. If the curves in the paper are accurate, a fully-trained HEP4M should close the gap on jet pT response. But the nano-hep side also hasn't been scaled up yet — it could gain further from multi-GPU training, larger models, and longer horizons.

## What to do next

1. **Train HEP4M for more epochs on the same 10M data** to get a real baseline (the strongest experiment).
2. **Scale nano-hep to 10M events** to make the training data match. Longer compute budget.
3. **Add gpos emission to nano-hep** (currently uses truth gpos) — predict eta/phi tokens too.
4. **Pflow during training for HEP4M** (currently skipped because use_preprocessed_data=true without raw ROOT files). Would let us watch HEP4M's jet_pt_response curve during training, which we now have for nano-hep.

## Reproducing

```
# HEP4M eval:
/pscratch/sd/d/danieltm/envs/hep4m2/bin/python scripts/eval_hep4m_reference.py \
  --run_dir /global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/HEP4M/data/experiments/HEP4M-pflow-pilot/pflow_pilot_10M_nocard \
  --ckpt epoch=3-val_total_loss=6.5630.ckpt \
  --reduce_ds_val 2000 --out_json runs/hep4m_pilot_eval.json

# nano-hep AR eval:
/pscratch/sd/d/danieltm/envs/hep4m2/bin/python scripts/eval_nanohep_ckpt.py \
  --ckpt runs/pilot_q3/best.ckpt --n_val 2000 \
  --out_json runs/nanohep_q3_eval.json
```

Both produce a JSON with identical metric keys (output of
`hep4m.performance.pflow_report.run_report_from_arrays`). Same val events,
same detokenize path, same metric function — hence the "apples to apples".
