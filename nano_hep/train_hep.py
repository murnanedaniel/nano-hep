"""Trimmed nanoGPT training loop for HEP tokenized data.

Uses upstream model.GPT as-is. Loss is masked to output-region tokens via
targets=-1 (nanoGPT's forward uses ignore_index=-1 in F.cross_entropy).

Usage
-----
    python -m nano_hep.train_hep --config configs/nano_hep_smoke.yml

Minimal; DDP later. Single GPU for v0.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

# ensure parent dir (where upstream model.py lives) is on path
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))

from model import GPT, GPTConfig  # noqa: E402  (upstream nanoGPT)
from .data import HEPDataset      # noqa: E402
from .vocab import Vocab, DEFAULT_MODALITY_VOCAB  # noqa: E402


def _apply_loss_mask(y: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    """Return targets with ignored positions set to -1 (nanoGPT's ignore_index).
    y, loss_mask are both (B, T) long."""
    return torch.where(loss_mask.bool(), y, torch.full_like(y, -1))


def _get_lr(step: int, cfg: Dict[str, Any]) -> float:
    """Linear warmup → cosine decay."""
    warm = cfg["lr"]["warmup_steps"]
    max_steps = cfg["training"]["max_steps"]
    peak = cfg["lr"]["peak"]
    floor = cfg["lr"]["floor"]
    if step < warm:
        return peak * (step + 1) / max(warm, 1)
    if step >= max_steps:
        return floor
    progress = (step - warm) / max(max_steps - warm, 1)
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def estimate_val_loss(model, val_loader, device, max_batches: int = 10) -> Dict[str, float]:
    model.eval()
    losses = []
    total_tokens = 0
    for i, (x, y, m) in enumerate(val_loader):
        if i >= max_batches: break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        m = m.to(device, non_blocking=True)
        y_mask = _apply_loss_mask(y, m)
        _, loss = model(x, targets=y_mask)
        if loss is not None and not torch.isnan(loss):
            losses.append(loss.item())
            total_tokens += m.sum().item()
    model.train()
    return {"val_loss": float(np.mean(losses)) if losses else float("nan"),
            "val_tokens": total_tokens}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out_dir", default=None, help="override cfg.training.out_dir")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.out_dir is not None:
        cfg["training"]["out_dir"] = args.out_dir
    out_dir = Path(cfg["training"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg["training"].get("seed", 42))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_type = "cuda" if device == "cuda" else "cpu"
    dtype = torch.bfloat16 if cfg["training"].get("bf16", True) and device == "cuda" else torch.float32

    # ---- data -----------------------------------------------------------------
    in_mod = cfg["data"]["input_modality"]
    out_mod = cfg["data"]["output_modality"]
    v = Vocab.build([in_mod, out_mod],
                    {in_mod: DEFAULT_MODALITY_VOCAB[in_mod],
                     out_mod: DEFAULT_MODALITY_VOCAB[out_mod]})
    print(f"Vocab: {v}")

    train_ds = HEPDataset(
        tokenized_root=cfg["data"]["tokenized_root"],
        split="train",
        input_modality=in_mod, output_modality=out_mod,
        block_size=cfg["data"]["block_size"],
        max_events=cfg["data"].get("max_train_events", -1),
        vocab=v,
    )
    val_ds = HEPDataset(
        tokenized_root=cfg["data"]["tokenized_root"],
        split="val",
        input_modality=in_mod, output_modality=out_mod,
        block_size=cfg["data"]["block_size"],
        max_events=cfg["data"].get("max_val_events", 512),
        vocab=v,
    )
    print(f"train events: {len(train_ds):,}  val events: {len(val_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"],
                              shuffle=True, num_workers=cfg["training"].get("num_workers", 2),
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"],
                            shuffle=False, num_workers=cfg["training"].get("num_workers", 2),
                            pin_memory=True, drop_last=False)

    # ---- model ----------------------------------------------------------------
    gpt_cfg = GPTConfig(
        block_size=cfg["data"]["block_size"] - 1,  # x/y are shifted by 1, so sequence length to model is block_size-1
        vocab_size=v.total,
        n_layer=cfg["model"]["n_layer"],
        n_head=cfg["model"]["n_head"],
        n_embd=cfg["model"]["n_embd"],
        dropout=cfg["model"].get("dropout", 0.0),
        bias=cfg["model"].get("bias", False),
    )
    model = GPT(gpt_cfg).to(device)
    if device == "cuda" and cfg["training"].get("compile", False):
        print("torch.compile...")
        model = torch.compile(model)

    optimizer = model.configure_optimizers(
        weight_decay=cfg["optim"]["weight_decay"],
        learning_rate=cfg["lr"]["peak"],
        betas=(cfg["optim"]["beta1"], cfg["optim"]["beta2"]),
        device_type=device_type,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16))

    # ---- wandb ---------------------------------------------------------------
    use_wandb = cfg["training"].get("wandb", True)
    if use_wandb:
        import wandb
        wandb.init(project=cfg["training"].get("wandb_project", "nano-hep"),
                   name=cfg["training"].get("run_name", out_dir.name),
                   config=cfg, dir=str(out_dir))

    # ---- train loop -----------------------------------------------------------
    max_steps = cfg["training"]["max_steps"]
    log_every = cfg["training"].get("log_every", 50)
    val_every = cfg["training"].get("val_every", 500)
    ckpt_every = cfg["training"].get("ckpt_every", 2000)
    grad_clip = cfg["training"].get("grad_clip", 1.0)

    step = 0
    t_last = time.time()
    best_val = float("inf")
    model.train()
    data_iter = iter(train_loader)
    while step < max_steps:
        try:
            x, y, m = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y, m = next(data_iter)

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        m = m.to(device, non_blocking=True)
        y_mask = _apply_loss_mask(y, m)

        lr = _get_lr(step, cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        with torch.autocast(device_type=device_type, dtype=dtype, enabled=(device == "cuda")):
            _, loss = model(x, targets=y_mask)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer); scaler.update()

        step += 1

        if step % log_every == 0:
            now = time.time()
            tps = (log_every * cfg["training"]["batch_size"] * (cfg["data"]["block_size"] - 1)) / (now - t_last)
            n_loss_tokens = m.sum().item()
            msg = f"step {step}/{max_steps}  loss {loss.item():.4f}  lr {lr:.2e}  loss_tokens/batch {n_loss_tokens}  tok/s {tps:,.0f}"
            print(msg, flush=True)
            if use_wandb:
                wandb.log({"train/loss": loss.item(), "train/lr": lr, "train/tok_per_s": tps,
                           "train/loss_tokens_per_batch": n_loss_tokens, "step": step})
            t_last = now

        if step % val_every == 0 or step == max_steps:
            val = estimate_val_loss(model, val_loader, device, max_batches=10)
            print(f"  val_loss {val['val_loss']:.4f}  val_tokens {val['val_tokens']}", flush=True)
            if use_wandb:
                wandb.log({"val/loss": val["val_loss"], "val/tokens_seen": val["val_tokens"], "step": step})
            if val["val_loss"] < best_val:
                best_val = val["val_loss"]
                torch.save({
                    "model": (model._orig_mod if hasattr(model, "_orig_mod") else model).state_dict(),
                    "config": cfg,
                    "gpt_config": gpt_cfg.__dict__,
                    "vocab": {"modalities": v.modalities, "vocab_sizes": v.vocab_sizes,
                              "offsets": v.offsets, "mod_start": v.mod_start,
                              "eos": v.eos, "pad": v.pad, "total": v.total},
                    "step": step, "val_loss": val["val_loss"],
                }, out_dir / "best.ckpt")
                print(f"  saved best.ckpt (val_loss={val['val_loss']:.4f})", flush=True)

        if step % ckpt_every == 0:
            torch.save({"model": (model._orig_mod if hasattr(model, "_orig_mod") else model).state_dict(),
                        "step": step, "gpt_config": gpt_cfg.__dict__},
                       out_dir / f"step{step:06d}.ckpt")

    print("done.")


if __name__ == "__main__":
    main()
