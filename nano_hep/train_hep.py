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
from .vocab import Vocab, DEFAULT_MODALITY_CODEBOOK_SIZE, DEFAULT_MODALITY_NUM_QUANTIZERS  # noqa: E402


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
def estimate_val_loss(model, val_loader, device, vocab, max_batches: int = 10) -> Dict[str, float]:
    """Teacher-forced metrics: val_loss, token_accuracy on output positions."""
    model.eval()
    losses = []
    total_tokens = 0
    n_correct = 0
    n_total = 0
    for i, (x, y, m) in enumerate(val_loader):
        if i >= max_batches: break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        m = m.to(device, non_blocking=True)
        y_mask = _apply_loss_mask(y, m)
        logits, loss = model(x, targets=y_mask)
        if loss is not None and not torch.isnan(loss):
            losses.append(loss.item())
            total_tokens += m.sum().item()
        # token accuracy on loss-mask positions
        pred = logits.argmax(dim=-1)  # (B, T)
        mask_bool = m.bool()
        n_correct += ((pred == y) & mask_bool).sum().item()
        n_total += mask_bool.sum().item()
    model.train()
    return {
        "val_loss": float(np.mean(losses)) if losses else float("nan"),
        "val_tokens": total_tokens,
        "val_token_accuracy": float(n_correct / max(n_total, 1)),
    }


@torch.no_grad()
def autoregressive_eval_with_pflow(
    model, val_ds, device, vocab,
    n_events: int = 64, max_new_tokens: int = 64,
    pflow=None,  # optional PflowMetrics instance
) -> Dict[str, Any]:
    """Like autoregressive_eval, but also collects per-event code triples
    + positional tokens so pflow metrics can be computed via VQ-VAE decode.

    If `pflow` is provided, returns the full pflow dict merged into the output.
    Otherwise returns the same dict as autoregressive_eval.
    """
    # Call the standard AR eval first for token-level metrics.
    out = autoregressive_eval(model, val_ds, device, vocab,
                              n_events=n_events, max_new_tokens=max_new_tokens)
    if pflow is None:
        return out

    # For pflow, we need to:
    #  - gather predicted output-modality token triples per event from model AR decoding
    #  - gather true output-modality tokens AND pos tokens
    # Since AR decode doesn't emit positional tokens in our current sequence layout
    # (we only include VQ codebook tokens, not gpos), we substitute true-gpos for pred-gpos.
    # This isolates the VQ-codebook quality vs the positional prediction (which the model
    # doesn't yet emit).
    import awkward as ak
    out_mod = val_ds.output_modality
    num_q_out = vocab.num_quantizers[out_mod]
    out_offset = vocab.offsets[out_mod]
    V_out = vocab.codebook_sizes[out_mod]

    # Re-run AR decode to collect per-event emitted tokens (small overhead for n_events<=128).
    model.eval()
    B = min(n_events, len(val_ds))
    # Find max number of output elements across this subset (needed for padding)
    max_N = 0
    true_codes_list, true_pos_list = [], []
    pred_codes_list, pred_mask_list = [], []
    n_cb_true = val_ds.out_mm.n_codebooks
    n_pos_true = val_ds.out_mm.n_pos_codebooks
    for i in range(B):
        # Real truth codes (all n_cb) + pos codes
        a = int(val_ds.out_mm.offsets[i]); b = int(val_ds.out_mm.offsets[i + 1])
        tc = np.asarray(val_ds.out_mm.data[a:b, :n_cb_true], dtype=np.int64)
        tp = np.asarray(val_ds.out_mm.data[a:b, n_cb_true:n_cb_true + n_pos_true], dtype=np.int64)
        true_codes_list.append(tc)
        true_pos_list.append(tp)
        # Run AR decode for pred tokens
        seq = val_ds.build_sequence(i)
        out_start_pos, _ = val_ds.output_region_slice(seq)
        prefix_len = out_start_pos + 1
        prefix = torch.from_numpy(seq[:prefix_len]).long().unsqueeze(0).to(device)
        cur = prefix
        max_len = min(val_ds.block_size, prefix_len + max_new_tokens)
        emitted = []
        while cur.shape[1] < max_len:
            logits, _ = model(cur)
            next_tok = int(logits[0, -1].argmax().item())
            if next_tok == vocab.eos:
                break
            emitted.append(next_tok)
            cur = torch.cat([cur, torch.tensor([[next_tok]], device=device, dtype=torch.long)], dim=1)
        # Group emitted tokens into triples within the output modality range; drop partial trailing
        out_tokens = [t for t in emitted if out_offset <= t < out_offset + num_q_out * V_out]
        n_elems = len(out_tokens) // num_q_out
        if n_elems == 0:
            pc = np.zeros((0, num_q_out), dtype=np.int64)
        else:
            pc = vocab.decode_modality_triples(out_mod, np.array(out_tokens[: n_elems * num_q_out]))
        # Clamp to codebook_size (just in case)
        pc = np.clip(pc, 0, V_out - 1)
        pred_codes_list.append(pc)
        pred_mask_list.append(n_elems)
        max_N = max(max_N, n_elems, tc.shape[0])

    max_N = max(max_N, 1)
    B = len(true_codes_list)
    # pad everything to (B, max_N, ...)
    pred_codes = np.zeros((B, max_N, num_q_out), dtype=np.int64)
    pred_mask = np.zeros((B, max_N), dtype=bool)
    true_codes = np.zeros((B, max_N, n_cb_true), dtype=np.int64)
    true_pos = np.zeros((B, max_N, n_pos_true), dtype=np.int64)
    true_mask = np.zeros((B, max_N), dtype=bool)
    for i in range(B):
        nt = true_codes_list[i].shape[0]
        if nt > 0:
            true_codes[i, :nt] = true_codes_list[i]
            true_pos[i, :nt] = true_pos_list[i]
            true_mask[i, :nt] = True
        np_ = pred_codes_list[i].shape[0]
        if np_ > 0:
            # Pad/truncate pred codes to match VQ-VAE's expected quantizer count
            n_q_fill = min(pred_codes_list[i].shape[1], n_cb_true)
            pred_codes[i, :np_, :n_q_fill] = pred_codes_list[i][:, :n_q_fill]
            pred_mask[i, :np_] = True

    # Use TRUE pos codes for both — model doesn't yet emit gpos tokens. This gives
    # a pflow number that reflects VQ codebook quality with "oracle" pos info.
    # Once we extend to emit gpos tokens too, swap in pred-pos.
    pred_pos = true_pos

    # Pad pred_codes to full n_cb_true (missing quantizers get code 0, which is benign for VQ-VAE).
    if pred_codes.shape[2] < n_cb_true:
        extra = np.zeros((B, max_N, n_cb_true - pred_codes.shape[2]), dtype=np.int64)
        pred_codes = np.concatenate([pred_codes, extra], axis=2)

    pf_metrics = pflow.compute_metrics(
        torch.from_numpy(pred_codes), torch.from_numpy(pred_pos), torch.from_numpy(pred_mask),
        torch.from_numpy(true_codes), torch.from_numpy(true_pos), torch.from_numpy(true_mask),
        outdir=None, ind_threshold=0.5,
    )
    model.train()
    for k, v in pf_metrics.items():
        if not str(k).startswith("_"):
            out[f"pflow_{k}"] = v
    return out


@torch.no_grad()
def autoregressive_eval(
    model,
    val_ds,
    device,
    vocab,
    n_events: int = 64,
    max_new_tokens: int = 64,
) -> Dict[str, Any]:
    """Run AR generation on a small val subset. For each event:
      1. Feed the prefix up through MOD_START[output] and generate tokens greedily.
      2. Stop at EOS or max_new_tokens.
      3. Compute predicted cardinality = count of output-modality tokens emitted
         before EOS.

    Reports:
      ar_cardinality_acc  — fraction of events where pred cardinality == true.
      ar_cardinality_mae  — mean |pred − true|.
      ar_n_pred_mean/std  — cardinality-pred stats.
      ar_n_true_mean/std  — cardinality-true stats.
      ar_eos_position_error — mean |predicted EOS pos − true EOS pos|.
      ar_token_accuracy   — accuracy of generated tokens vs teacher tokens,
                            when compared position by position up to min(pred, true).
      ar_scatter_npy      — (n_events,2) array (n_true, n_pred).
    """
    model.eval()
    out_mod = val_ds.output_modality
    out_offset = vocab.offsets[out_mod]
    num_q_out = vocab.num_quantizers[out_mod]
    V_out = vocab.codebook_sizes[out_mod]
    out_block_size = num_q_out * V_out  # total width of modality block
    out_token_range = (out_offset, out_offset + out_block_size)
    eos_id = vocab.eos
    pad_id = vocab.pad
    mod_start_out = vocab.mod_start[out_mod]

    n_true_list, n_pred_list, eos_pos_err, ar_n_correct, ar_n_total = [], [], [], 0, 0

    for i in range(min(n_events, len(val_ds))):
        seq = val_ds.build_sequence(i)  # (block_size,) int64
        out_start_pos, true_eos_pos = val_ds.output_region_slice(seq)
        n_true_tokens = true_eos_pos - (out_start_pos + 1)  # tokens between MOD_START[out] and EOS
        n_true = n_true_tokens // num_q_out   # number of elements (particles)
        # Prefix: everything up through MOD_START[out] (inclusive)
        prefix_len = out_start_pos + 1
        prefix = torch.from_numpy(seq[:prefix_len]).long().unsqueeze(0).to(device)

        # Autoregressive generation
        cur = prefix
        n_pred_tokens = 0
        emitted = []
        max_len = min(val_ds.block_size, prefix_len + max_new_tokens)
        ended = False
        while cur.shape[1] < max_len:
            logits, _ = model(cur)  # (B=1, 1, V)
            next_tok = int(logits[0, -1].argmax().item())
            emitted.append(next_tok)
            cur = torch.cat([cur, torch.tensor([[next_tok]], device=device, dtype=torch.long)], dim=1)
            if next_tok == eos_id:
                ended = True
                break
            # Count output-modality tokens
            if out_token_range[0] <= next_tok < out_token_range[1]:
                n_pred_tokens += 1
        n_pred = n_pred_tokens // num_q_out  # elements

        # Predicted EOS position = len(prefix) + len(emitted) - 1 (inclusive of the EOS if ended)
        if ended:
            pred_eos_pos = prefix_len + len(emitted) - 1
        else:
            pred_eos_pos = prefix_len + len(emitted)  # ran out without EOS
        eos_pos_err.append(abs(pred_eos_pos - true_eos_pos))

        # Teacher tokens for accuracy (between out_start_pos+1 and true_eos_pos)
        teacher = seq[out_start_pos + 1 : true_eos_pos]  # includes neither MOD_START[out] nor EOS
        # Predicted output tokens (emitted excluding EOS if present)
        if ended:
            pred_tokens = emitted[:-1]
        else:
            pred_tokens = emitted
        k = min(len(pred_tokens), len(teacher))
        if k > 0:
            t = np.asarray(teacher[:k])
            p = np.asarray(pred_tokens[:k])
            ar_n_correct += int((t == p).sum())
            ar_n_total += k

        n_true_list.append(int(n_true))
        n_pred_list.append(int(n_pred))

    n_true_arr = np.asarray(n_true_list)
    n_pred_arr = np.asarray(n_pred_list)
    resid = n_pred_arr - n_true_arr
    out = {
        "ar_cardinality_acc": float((n_pred_arr == n_true_arr).mean()),
        "ar_cardinality_mae": float(np.abs(resid).mean()),
        "ar_n_pred_mean": float(n_pred_arr.mean()),
        "ar_n_pred_std": float(n_pred_arr.std()),
        "ar_n_true_mean": float(n_true_arr.mean()),
        "ar_n_true_std": float(n_true_arr.std()),
        "ar_eos_position_error": float(np.mean(eos_pos_err)),
        "ar_token_accuracy": float(ar_n_correct / max(ar_n_total, 1)),
        "_n_true_arr": n_true_arr,  # for wandb plots (popped before logging scalars)
        "_n_pred_arr": n_pred_arr,
    }
    model.train()
    return out


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
    # Accept either `input_modality` (single, legacy) or `input_modalities` (list, new).
    if "input_modalities" in cfg["data"]:
        in_mods = list(cfg["data"]["input_modalities"])
    else:
        in_mods = [cfg["data"]["input_modality"]]
    out_mod = cfg["data"]["output_modality"]
    all_mods = in_mods + [out_mod]
    # num_quantizers can be scalar (applied to all mods) or dict {mod: int}
    num_q_cfg = cfg["data"].get("num_quantizers", 1)
    if isinstance(num_q_cfg, int):
        num_q_map = {m: num_q_cfg for m in all_mods}
    else:
        num_q_map = dict(num_q_cfg)
    v = Vocab.build(all_mods,
                    {m: DEFAULT_MODALITY_CODEBOOK_SIZE[m] for m in all_mods},
                    num_q_map)
    print(f"Vocab: {v}")

    train_ds = HEPDataset(
        tokenized_root=cfg["data"]["tokenized_root"],
        split="train",
        input_modalities=in_mods, output_modality=out_mod,
        block_size=cfg["data"]["block_size"],
        max_events=cfg["data"].get("max_train_events", -1),
        vocab=v,
    )
    val_ds = HEPDataset(
        tokenized_root=cfg["data"]["tokenized_root"],
        split="val",
        input_modalities=in_mods, output_modality=out_mod,
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

    # ---- pflow metrics (optional) --------------------------------------------
    pflow = None
    pflow_cfg = cfg.get("pflow_metrics", {"enabled": False})
    if pflow_cfg.get("enabled", False):
        from .pflow_metrics import PflowMetrics
        pflow = PflowMetrics(
            modality_dict_path=pflow_cfg["modality_dict_path"],
            output_modality=out_mod,
            device=device,
        )
        print(f"PflowMetrics enabled for output modality '{out_mod}'.")

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
            val = estimate_val_loss(model, val_loader, device, v, max_batches=10)
            ar_n_events = cfg["training"].get("ar_eval_events", 64)
            ar = autoregressive_eval_with_pflow(
                model, val_ds, device, v,
                n_events=ar_n_events,
                max_new_tokens=val_ds.block_size,
                pflow=pflow,
            )
            print(f"  val_loss {val['val_loss']:.4f}  tok_acc {val['val_token_accuracy']:.3f}  "
                  f"ar_card_acc {ar['ar_cardinality_acc']:.3f}  ar_card_mae {ar['ar_cardinality_mae']:.2f}  "
                  f"ar_tok_acc {ar['ar_token_accuracy']:.3f}  ar_n_pred {ar['ar_n_pred_mean']:.1f}±{ar['ar_n_pred_std']:.1f}  "
                  f"(n_true {ar['ar_n_true_mean']:.1f}±{ar['ar_n_true_std']:.1f})",
                  flush=True)
            if use_wandb:
                log_dict = {
                    "val/loss": val["val_loss"],
                    "val/tokens_seen": val["val_tokens"],
                    "val/token_accuracy": val["val_token_accuracy"],
                    "step": step,
                }
                for k, vv in ar.items():
                    if not k.startswith("_"):
                        log_dict[f"val/{k}"] = vv
                # Cardinality scatter plot (wandb Image) every val
                try:
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
                    n_true = ar["_n_true_arr"]; n_pred = ar["_n_pred_arr"]
                    mx = max(n_true.max(), n_pred.max(), 1) + 1
                    ax1.scatter(n_true, n_pred, alpha=0.3, s=10)
                    ax1.plot([0, mx], [0, mx], "k--", lw=0.5)
                    ax1.set_xlabel("n_true"); ax1.set_ylabel("n_pred")
                    ax1.set_title(f"cardinality: pred vs true  (step {step})")
                    bins = np.arange(0, mx + 1) - 0.5
                    ax2.hist(n_true, bins=bins, alpha=0.5, label="true", density=True)
                    ax2.hist(n_pred, bins=bins, alpha=0.5, label="pred", density=True)
                    ax2.set_xlabel("cardinality"); ax2.legend(fontsize=8)
                    ax2.set_title(f"cardinality hist  acc={ar['ar_cardinality_acc']:.2f} mae={ar['ar_cardinality_mae']:.1f}")
                    fig.tight_layout()
                    log_dict["val/cardinality_plot"] = wandb.Image(fig)
                    plt.close(fig)
                except Exception as e:
                    print(f"  (skipped cardinality plot: {e})")
                wandb.log(log_dict)
            if val["val_loss"] < best_val:
                best_val = val["val_loss"]
                torch.save({
                    "model": (model._orig_mod if hasattr(model, "_orig_mod") else model).state_dict(),
                    "config": cfg,
                    "gpt_config": gpt_cfg.__dict__,
                    "vocab": {"modalities": v.modalities, "codebook_sizes": v.codebook_sizes,
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
