"""Evaluate HEP4M's pretrained pflow checkpoint using the SAME VQ-VAE decode +
run_report_from_arrays path that nano-hep uses.

Protocol:
  1. Load HEP4MLightning from the pretrained ckpt.
  2. Build HEP4M's DataModule, manually call setup("fit") (HEP4M's setup only
     handles stage=='fit' / None).
  3. Loop over val batches, call `lm.model.fast_forward(batch)` (parallel
     prediction, same as HEP4M training).
  4. Argmax the token_logits → predicted truthpart tokens.
     For gpos_tokens, argmax if the model emits them; otherwise reuse truth gpos.
  5. Feed (pred_codes, true_codes, ind_mask, pos_codes) into nano-hep's
     PflowMetrics which detokenizes via HEP4M's frozen VQ-VAE and calls
     `run_report_from_arrays`.

This yields a jet-level metric dict directly comparable to nano-hep's
val/pflow_* scalars.

Run in hep4m2 env.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
import yaml

HEP4M_ROOT = "/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/HEP4M"
NANOHEP_ROOT = "/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/nano-hep"
sys.path.insert(0, HEP4M_ROOT)
sys.path.insert(0, NANOHEP_ROOT)

from hep4m.lightnings.hep4m_lightning import HEP4MLightning, HEP4MDataModule
from nano_hep.pflow_metrics import PflowMetrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--reduce_ds_val", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--out_json", default="runs/hep4m_pilot_eval.json")
    ap.add_argument("--use_true_gpos", action="store_true", default=True,
                    help="If the model doesn't emit gpos_tokens, substitute truth gpos (apples-to-apples with nano-hep which also doesn't yet emit gpos).")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = Path(args.run_dir)
    ckpt_path = run_dir / "checkpoints" / args.ckpt
    assert ckpt_path.exists(), ckpt_path

    with open(run_dir / "config_t.yml") as f: cfg_t = yaml.safe_load(f)
    with open(run_dir / "config_m.yml") as f: cfg_m = yaml.safe_load(f)
    with open(run_dir / "modality_dict.yml") as f: mod_dict = yaml.safe_load(f)

    # Trim for a fast eval pass
    cfg_t["reduce_ds_val"] = args.reduce_ds_val
    cfg_t["batchsize_val"] = args.batch_size
    cfg_t["num_devices"] = 1
    cfg_t.setdefault("num_nodes", 1)
    cfg_t["persistent_workers"] = False
    cfg_t["num_workers"] = 2

    print(f"Loading HEP4M Lightning from {ckpt_path}...")
    # HEP4MLightning kwargs vary by branch: twostep-smoke takes `comet_logger`,
    # logger-factory branches take `wandb_logger`. Try in that order.
    try:
        lm = HEP4MLightning(
            config_m=cfg_m, modality_dict=mod_dict, config_t=cfg_t,
            comet_logger=None, device=device,
        )
    except TypeError:
        lm = HEP4MLightning(
            config_m=cfg_m, modality_dict=mod_dict, config_t=cfg_t,
            wandb_logger=None, device=device,
        )
    lm.set_training_plan("fresh")

    # Load weights (skip Trainer — we just want forward passes)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = state.get("state_dict", state)
    missing, unexpected = lm.load_state_dict(sd, strict=False)
    print(f"ckpt loaded. missing={len(missing)} unexpected={len(unexpected)}")
    lm.to(device).eval()

    dm = HEP4MDataModule(
        config_t=cfg_t, config_v_dict=lm.config_v_dict, modality_dict=mod_dict,
    )
    # HEP4M's setup only handles stage 'fit' or None — pass None
    dm.setup(stage=None)
    # HEP4MDataModule expects self.trainer (checks .world_size) — attach a tiny stub
    class _FakeTrainer:
        world_size = 1
    dm.trainer = _FakeTrainer()
    val_loader = dm.val_dataloader()
    print(f"val loader: {len(val_loader.dataset)} events, batch_size={args.batch_size}")

    # PflowMetrics from nano-hep: same detokenize + run_report_from_arrays as nano-hep.
    pf = PflowMetrics(
        modality_dict_path=str(run_dir / "modality_dict.yml"),
        output_modality="truthpart", device=device,
    )

    # Accumulate per-event (truth, reco) across all val batches, then call one
    # big run_report_from_arrays with the full set — avoids per-batch noise.
    all_pred_codes, all_pred_pos = [], []
    all_true_codes, all_true_pos = [], []
    all_pred_mask, all_true_mask = [], []
    n_seen = 0

    def _to(d):
        if isinstance(d, dict): return {k: _to(v) for k, v in d.items()}
        if isinstance(d, torch.Tensor): return d.to(device)
        return d

    with torch.no_grad():
        for bi, batch in enumerate(val_loader):
            batch = _to(batch)
            # fast_forward with sample_tokens=False → argmax in token_predictor
            # returns a dict with per-modality predictions inside `pred_tokens_dict`
            # (see hep4m.py:334-388)
            try:
                # twostep API: fast_forward(input_dict, output_modalities, q_mask_dict,
                #                          get_logits=True, use_truth_cardinality=..., target_dict=...)
                pred = lm.model.fast_forward(
                    input_dict=batch["input"],
                    output_modalities=["truthpart"],
                    q_mask_dict=batch.get("q_mask_dict"),
                    get_logits=True,
                    use_truth_cardinality=True,
                    target_dict=batch["target"],
                )
            except TypeError:
                # fallback for older single-arg API
                pred = lm.model.fast_forward(batch, get_logits=True)
            except Exception as e:
                print(f"batch {bi}: fast_forward failed: {e}"); continue

            # lm.model.fast_forward returns the raw predicted_token_dict after mode step 6
            # Structure: {modality: {'token_logits': (B,N,num_q,V), 'indicator_logits': (B,N),
            #                        maybe 'gpos_token_logits': (B,N,num_q_pos,V_pos)}}
            if "truthpart" not in pred:
                print(f"batch {bi}: no truthpart in pred; keys={list(pred.keys())}"); continue
            tp = pred["truthpart"]
            token_logits = tp["token_logits"]              # (B,N,num_q,V)
            indicator_logits = tp.get("indicator_logits")   # (B,N)
            tokens = token_logits.argmax(dim=-1)             # (B,N,num_q)
            # Threshold indicator → mask
            if indicator_logits is not None:
                pred_mask = (torch.sigmoid(indicator_logits) > 0.45)
            else:
                pred_mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)

            # gpos: argmax if emitted, else use truth pos
            if "gpos_token_logits" in tp and tp["gpos_token_logits"] is not None:
                pred_pos = tp["gpos_token_logits"].argmax(dim=-1)
            else:
                # Truth pos_tokens from batch['target']['truthpart']['pos_tokens']
                tgt = batch["target"]["truthpart"]
                pred_pos = tgt["pos_tokens"]

            # Truth (from target dict)
            tgt = batch["target"]["truthpart"]
            true_codes = tgt["tokens"]      # (B,N,num_q)
            true_pos = tgt.get("pos_tokens", pred_pos)
            # Truth mask: q_mask (True where real) if present, else indicators
            true_mask = tgt.get("q_mask")
            if true_mask is None:
                true_mask = tgt.get("indicators")
            if true_mask is None:
                true_mask = torch.ones(true_codes.shape[:2], dtype=torch.bool, device=true_codes.device)
            true_mask = true_mask.bool()

            all_pred_codes.append(tokens.cpu())
            all_pred_pos.append(pred_pos.cpu())
            all_pred_mask.append(pred_mask.cpu())
            all_true_codes.append(true_codes.cpu())
            all_true_pos.append(true_pos.cpu())
            all_true_mask.append(true_mask.cpu())
            n_seen += tokens.shape[0]
            if n_seen >= args.reduce_ds_val:
                break

    # Pad all collected tensors to a common N dim, then concat batch-wise.
    def _pad_stack(tensors, fill=0):
        max_N = max(t.shape[1] for t in tensors)
        out = []
        for t in tensors:
            if t.shape[1] < max_N:
                pad_shape = list(t.shape); pad_shape[1] = max_N - t.shape[1]
                pad = torch.full(pad_shape, fill, dtype=t.dtype)
                t = torch.cat([t, pad], dim=1)
            out.append(t)
        return torch.cat(out, dim=0)

    pred_codes = _pad_stack(all_pred_codes)
    pred_pos = _pad_stack(all_pred_pos)
    pred_mask = _pad_stack(all_pred_mask, fill=False)
    true_codes = _pad_stack(all_true_codes)
    true_pos = _pad_stack(all_true_pos)
    true_mask = _pad_stack(all_true_mask, fill=False)

    print(f"collected B={pred_codes.shape[0]} events, max_N={pred_codes.shape[1]}")

    # outdir=None skips the plot-emit path (incompatible plot_jets API between
    # twostep-smoke and the pflow_report we imported). Scalar metrics still computed.
    metrics = pf.compute_metrics(
        pred_codes, pred_pos, pred_mask,
        true_codes, true_pos, true_mask,
        outdir=None,
        ind_threshold=0.45,
    )
    clean = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
             for k, v in metrics.items() if not str(k).startswith("_")}
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(clean, f, indent=2)
    print("\n=== HEP4M pflow 10M-train eval (parallel forward → VQ-VAE → pflow_report) ===")
    for k, v in sorted(clean.items()):
        print(f"  {k}: {v}")
    print(f"\nsaved → {args.out_json}")


if __name__ == "__main__":
    main()
