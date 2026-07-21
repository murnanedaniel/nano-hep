"""Memory-probe for nano-hep training.

Instantiates the full training stack (Vocab, HEPDataset, GPT, AdamW) on a
single GPU with the production config and runs 3 forward+backward+step
iterations at each candidate batch size. Reports peak allocated + reserved
memory per batch. Use the largest batch whose peak_reserved / device_total
stays below some safety threshold (≈0.85) — then subtract ~10% headroom
for the DDP reduction bucket overhead you don't see in single-rank.
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

REPO = Path("/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/nano-hep")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from nano_hep.vocab import Vocab, DEFAULT_MODALITY_CODEBOOK_SIZE
from nano_hep.data import HEPDataset
from model import GPT, GPTConfig


def build(cfg: dict, batch_size: int, device: str):
    in_mods = cfg["data"]["input_modalities"]
    out_mod = cfg["data"]["output_modality"]
    all_mods = list(in_mods) + [out_mod]
    num_q_cfg = cfg["data"]["num_quantizers"]
    num_q_map = {m: num_q_cfg for m in all_mods} if isinstance(num_q_cfg, int) else dict(num_q_cfg)
    vocab = Vocab.build(
        all_mods, {m: DEFAULT_MODALITY_CODEBOOK_SIZE[m] for m in all_mods}, num_q_map
    )
    train_ds = HEPDataset(
        tokenized_root=cfg["data"]["tokenized_root"],
        split="val",  # val is smaller, probe loads faster
        input_modalities=in_mods, output_modality=out_mod,
        block_size=cfg["data"]["block_size"],
        max_events=max(batch_size * 8, 1024),
        vocab=vocab,
    )
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False,
                        num_workers=0, pin_memory=True, drop_last=True)
    gpt_cfg = GPTConfig(
        block_size=cfg["data"]["block_size"] - 1,
        vocab_size=vocab.total,
        n_layer=cfg["model"]["n_layer"],
        n_head=cfg["model"]["n_head"],
        n_embd=cfg["model"]["n_embd"],
        dropout=cfg["model"].get("dropout", 0.0),
        bias=cfg["model"].get("bias", False),
    )
    model = GPT(gpt_cfg).to(device)
    opt = model.configure_optimizers(
        weight_decay=cfg["optim"]["weight_decay"],
        learning_rate=cfg["lr"]["peak"],
        betas=(cfg["optim"]["beta1"], cfg["optim"]["beta2"]),
        device_type="cuda",
    )
    return model, opt, loader


def probe_one(cfg: dict, batch_size: int, n_steps: int, device: str) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model, opt, loader = build(cfg, batch_size, device)
    model.train()
    dtype = torch.bfloat16 if cfg["training"].get("bf16", True) else torch.float32

    it = iter(loader)
    step = 0
    while step < n_steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        # HEPDataset returns (x, y, loss_mask) — same as train_hep.py:577
        x = batch[0].to(device, non_blocking=True)
        y = batch[1].to(device, non_blocking=True)
        m = batch[2].to(device, non_blocking=True)
        y_mask = torch.where(m.bool(), y, torch.full_like(y, -1))
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(dtype=dtype, enabled=(dtype == torch.bfloat16)):
            _, loss = model(x, targets=y_mask)
        loss.backward()
        opt.step()
        step += 1

    torch.cuda.synchronize()
    peak_alloc = torch.cuda.max_memory_allocated() / 1024**3
    peak_reserved = torch.cuda.max_memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3

    # Clean up before next probe
    del model, opt, loader, it
    torch.cuda.empty_cache()
    return {
        "batch_size": batch_size,
        "peak_alloc_GiB": peak_alloc,
        "peak_reserved_GiB": peak_reserved,
        "total_GiB": total,
        "reserved_pct": 100 * peak_reserved / total,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs/nano_hep_89M_ddp.yml"))
    ap.add_argument("--batch_sizes", default="96,128,160,192,224,256,288,320")
    ap.add_argument("--n_steps", type=int, default=3)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    sizes = [int(s) for s in args.batch_sizes.split(",")]
    device = "cuda"
    print(f"Probing batch sizes on {torch.cuda.get_device_name(0)}: {sizes}")
    print(f"n_steps per probe: {args.n_steps}, block_size: {cfg['data']['block_size']}")
    print()
    print(f"{'bs':>5} {'alloc_GiB':>10} {'resv_GiB':>10} {'resv_%':>7}  status")
    results = []
    for bs in sizes:
        try:
            r = probe_one(cfg, bs, args.n_steps, device)
            results.append(r)
            print(f"{bs:>5} {r['peak_alloc_GiB']:>10.2f} {r['peak_reserved_GiB']:>10.2f} "
                  f"{r['reserved_pct']:>6.1f}%  OK")
        except torch.cuda.OutOfMemoryError as e:
            print(f"{bs:>5}    OOM ({str(e)[:60]})")
            results.append({"batch_size": bs, "oom": True})
            torch.cuda.empty_cache()
            break
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"{bs:>5}    OOM ({str(e)[:60]})")
                results.append({"batch_size": bs, "oom": True})
                torch.cuda.empty_cache()
                break
            raise

    print()
    ok = [r for r in results if not r.get("oom")]
    if ok:
        # Use 85% of device memory as the ceiling. Then apply 10% DDP safety margin
        # → pick the largest bs whose reserved_pct <= 75%.
        safe = [r for r in ok if r["reserved_pct"] <= 75]
        pick = safe[-1] if safe else ok[0]
        print(f"Recommended bs (single-GPU probe, <75% reserved → room for DDP overhead): {pick['batch_size']}")
        print(f"  at that bs: peak_reserved = {pick['peak_reserved_GiB']:.1f} GiB "
              f"({pick['reserved_pct']:.1f}% of {pick['total_GiB']:.1f} GiB)")


if __name__ == "__main__":
    main()
