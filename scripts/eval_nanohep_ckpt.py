"""Evaluate a nano-hep AR checkpoint on val with the FIXED PflowMetrics.

Mirrors the training-time eval but callable on a saved checkpoint, so we can
re-score runs that trained under the old (buggy) inverse-transform pflow code.

Run in hep4m2 env.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
import yaml

HEP4M_ROOT = "/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/HEP4M"
NANOHEP_ROOT = "/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/nano-hep"
sys.path.insert(0, HEP4M_ROOT); sys.path.insert(0, NANOHEP_ROOT)

from model import GPT, GPTConfig  # upstream nanoGPT
from nano_hep.data import HEPDataset
from nano_hep.vocab import Vocab
from nano_hep.pflow_metrics import PflowMetrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n_val", type=int, default=2000)
    ap.add_argument("--max_new_tokens", type=int, default=80)
    ap.add_argument("--modality_dict", default="/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/HEP4M/data/experiments/HEP4M-pflow-pilot/pflow_pilot_10M_nocard/modality_dict.yml")
    ap.add_argument("--out_json", default="runs/nanohep_eval.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    gpt_kwargs = state["gpt_config"]
    v_state = state["vocab"]
    cfg = state["config"]

    # Some old ckpts didn't save num_quantizers; derive from training config.
    num_q_cfg = cfg["data"].get("num_quantizers", 1)
    if isinstance(num_q_cfg, int):
        num_q_map = {m: num_q_cfg for m in v_state["modalities"]}
    else:
        num_q_map = dict(num_q_cfg)
    vocab = Vocab.build(
        v_state["modalities"],
        v_state["codebook_sizes"],
        num_q_map,
    )
    gpt_cfg = GPTConfig(**gpt_kwargs)
    model = GPT(gpt_cfg)
    model.load_state_dict(state["model"])
    model.to(device).eval()

    # Build val dataset matching the training config
    ds = HEPDataset(
        tokenized_root=cfg["data"]["tokenized_root"],
        split="val",
        input_modalities=cfg["data"].get("input_modalities", [cfg["data"].get("input_modality", "track")]),
        output_modality=cfg["data"]["output_modality"],
        block_size=cfg["data"]["block_size"],
        max_events=args.n_val,
        vocab=vocab,
        num_quantizers=cfg["data"].get("num_quantizers", 1),
    )
    print(f"val: {len(ds)} events, block_size={ds.block_size}")

    out_mod = ds.output_modality
    num_q_out = vocab.num_quantizers[out_mod]
    out_offset = vocab.offsets[out_mod]
    V_out = vocab.codebook_sizes[out_mod]
    eos_id = vocab.eos

    # AR decode on each val event
    pred_codes_list, pred_mask_list = [], []
    true_codes_list, true_mask_list, true_pos_list = [], [], []
    n_cb = ds.out_mm.n_codebooks
    n_pos = ds.out_mm.n_pos_codebooks

    with torch.no_grad():
        for i in range(len(ds)):
            # truth codes + pos (all codebooks)
            a = int(ds.out_mm.offsets[i]); b = int(ds.out_mm.offsets[i + 1])
            tc = np.asarray(ds.out_mm.data[a:b, :n_cb], dtype=np.int64)
            tp = np.asarray(ds.out_mm.data[a:b, n_cb:n_cb + n_pos], dtype=np.int64)
            true_codes_list.append(tc)
            true_pos_list.append(tp)
            true_mask_list.append(tc.shape[0])

            # AR generate prediction
            seq = ds.build_sequence(i)
            out_start_pos, _ = ds.output_region_slice(seq)
            prefix_len = out_start_pos + 1
            cur = torch.from_numpy(seq[:prefix_len]).long().unsqueeze(0).to(device)
            emitted = []
            while cur.shape[1] < min(ds.block_size, prefix_len + args.max_new_tokens):
                logits, _ = model(cur)
                nx = int(logits[0, -1].argmax().item())
                if nx == eos_id:
                    break
                emitted.append(nx)
                cur = torch.cat([cur, torch.tensor([[nx]], device=device, dtype=torch.long)], dim=1)
            out_tokens = [t for t in emitted if out_offset <= t < out_offset + num_q_out * V_out]
            n_elems = len(out_tokens) // num_q_out
            if n_elems == 0:
                pc = np.zeros((0, num_q_out), dtype=np.int64)
            else:
                pc = vocab.decode_modality_triples(out_mod, np.array(out_tokens[: n_elems * num_q_out]))
            pc = np.clip(pc, 0, V_out - 1)
            pred_codes_list.append(pc)
            pred_mask_list.append(n_elems)
            if (i + 1) % 200 == 0:
                print(f"  eval {i+1}/{len(ds)}")

    # Pad all to common max_N
    max_N = max(max(true_mask_list), max(pred_mask_list), 1)
    B = len(true_codes_list)
    pred_codes = np.zeros((B, max_N, num_q_out), dtype=np.int64)
    pred_mask = np.zeros((B, max_N), dtype=bool)
    true_codes = np.zeros((B, max_N, n_cb), dtype=np.int64)
    true_pos = np.zeros((B, max_N, n_pos), dtype=np.int64)
    true_mask = np.zeros((B, max_N), dtype=bool)
    for i in range(B):
        nt = true_mask_list[i]
        if nt > 0:
            true_codes[i, :nt] = true_codes_list[i]
            true_pos[i, :nt] = true_pos_list[i]
            true_mask[i, :nt] = True
        npr = pred_mask_list[i]
        if npr > 0:
            nq_fill = min(pred_codes_list[i].shape[1], n_cb)
            pred_codes[i, :npr, :nq_fill] = pred_codes_list[i][:, :nq_fill]
            pred_mask[i, :npr] = True
    # Model doesn't emit gpos; substitute truth gpos
    pred_pos = true_pos
    # Pad pred_codes to n_cb if needed
    if pred_codes.shape[2] < n_cb:
        extra = np.zeros((B, max_N, n_cb - pred_codes.shape[2]), dtype=np.int64)
        pred_codes = np.concatenate([pred_codes, extra], axis=2)

    print(f"\nRunning PflowMetrics (fixed inverse transform)...")
    pf = PflowMetrics(
        modality_dict_path=args.modality_dict,
        output_modality=out_mod, device=device,
    )
    metrics = pf.compute_metrics(
        torch.from_numpy(pred_codes), torch.from_numpy(pred_pos), torch.from_numpy(pred_mask),
        torch.from_numpy(true_codes), torch.from_numpy(true_pos), torch.from_numpy(true_mask),
        outdir=None,  # skip plotting (plot_jets API mismatch on twostep-smoke branch)
        ind_threshold=0.5,
    )
    clean = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
             for k, v in metrics.items() if not str(k).startswith("_")}
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(clean, f, indent=2)
    print(f"\n=== nano-hep AR eval (ckpt={Path(args.ckpt).name}) ===")
    for k, v in sorted(clean.items()):
        print(f"  {k}: {v}")
    print(f"\nsaved → {args.out_json}")


if __name__ == "__main__":
    main()
