# nano-hep

Minimal autoregressive LM baseline for the HEP4M multi-modal particle-flow task.
Fork of karpathy/nanoGPT with a thin integration layer that feeds HEP4M's
tokenized COCOA data (`/global/cfs/cdirs/m4958/data/COCOA/Tokenized/`) as a flat
next-token-prediction sequence.

## Why

HEP4M is a bespoke encoder/decoder with per-modality positional queries,
indicator heads, empty-token classes, etc. Before iterating on that machinery,
we want a **control**: can a plain prefix-LM on the same tokens match HEP4M?

This repo is that control experiment.

## Layout

Upstream nanoGPT files are **unmodified** at the repo root (`model.py`, `train.py`,
`configurator.py`, `sample.py`, `bench.py`, `data/`, `config/`, `transformer_sizing.ipynb`,
`scaling_laws.ipynb`). Everything new is under `nano_hep/`:

```
nano_hep/
├── vocab.py        # union-vocab layout (per-modality offsets + specials)
├── data.py         # memmap reader → flat AR sequences
├── data_test.py    # CPU unit tests
├── train_hep.py    # trimmed train loop; uses upstream model.GPT
└── eval_hep.py     # (v1) detokenize predictions via HEP4M VQ-VAE
configs/
├── nano_hep_smoke.yml  # 10k events / 500 steps / ~10 min
└── nano_hep_pilot.yml  # 100k events / 5k steps / ~1h
scripts/
├── smoke.sh
└── pilot_1gpu.sh
```

## Data format (v0)

Per event, the model sees one flat sequence of integer token IDs:

```
  <MOD:track_start> q0_track[0] q0_track[1] ... q0_track[N_trk-1]
  <MOD:truthpart_start> q0_truth[0] ... q0_truth[N_truth-1]
  <EOS> <PAD>…
```

- Per-modality token IDs live in disjoint vocab ranges (see `vocab.py`).
- Only q0 (the coarsest of HEP4M's 3 VQ quantizers) is used in v0.
- Position tokens (eta/phi gpos quantizers) are **not used yet**.
- Loss is masked to the output region only (input positions get `target = -1`).

## Running (v0)

Tests (CPU, no allocation needed):

```
cd /global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/nano-hep
python -m nano_hep.data_test
```

Smoke (requires 1 GPU, ~10 min):

```
bash scripts/smoke.sh                    # standalone, needs GPU visible
bash scripts/smoke.sh 51683435           # on an existing slurm alloc
```

Pilot (100k events, 5k steps, ~1h on 1 GPU):

```
bash scripts/pilot_1gpu.sh               # or pass JOBID
```

Outputs `runs/{smoke,pilot}/best.ckpt` + wandb logging to project `nano-hep`.

## HEP4M reuse

For v0 we only need `TokenDatasetNumpy`'s on-disk format (memmap `_data.npy`,
`_offsets.npy`, `_meta.npz`). We read the memmaps directly — **no HEP4M
imports required**. Later (for eval-time detokenization):

- `hep4m.models.vqvae.VQVAE.indices_to_zq + decode` — needs
  `vector_quantize_pytorch` (not currently installed in the `influencer` env).
- `hep4m.performance.pflow_report.run_report_from_arrays` — the reuse that
  lets us compare against HEP4M on the same metric surface.

## Scope (v0)

- Single input modality (`track`) → single output modality (`truthpart`).
- Only q0 codebook per modality.
- Greedy decoding at inference (no eval script yet).
- Single-GPU training only.
- Pure causal mask (nanoGPT default). Prefix-LM mask is v1.

## Next

1. (smoke + pilot validate the data pipeline and scale).
2. Add `eval_hep.py`: autoregressive decode, detokenize via VQ-VAE, run
   `pflow_report.run_report_from_arrays` for jet-level metrics.
3. Compare val_loss + pflow_jet_response to HEP4M's published pilot numbers.
4. If the plain baseline matches, simplify HEP4M. If it doesn't, identify
   the specific architectural piece that matters.
