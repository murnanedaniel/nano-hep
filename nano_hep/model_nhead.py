"""GPT with an auxiliary N-head for supervised cardinality prediction.

The N-head is an MLP that reads the transformer hidden state at the
output-modality MOD_START token (the position right after all inputs have
been consumed, before any output token is generated) and classifies the
true output cardinality.

Training objective:
    total_loss = token_loss + n_head.loss_weight * n_head_loss
    n_head_loss = CE(n_head_mlp(h[n_head_pos]), clamp(n_true, 0, max_n))

At inference time the N-head is optional — it can be used to:
  - Mask EOS before n_head_argmax tokens have been emitted (hard cardinality
    floor), OR
  - Force exactly 6 * n_head_argmax tokens to be emitted (Pix2Seq-style
    length-first decoding).

Both inference uses are implemented as eval-time flags in the training
script, not here — this module only owns the architectural surgery +
combined loss.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from model import GPT, GPTConfig  # upstream nanoGPT  # noqa: E402


class GPTWithNHead(GPT):
    """GPT extended with an auxiliary N-classification head.

    Args:
        config: GPTConfig (same as nanoGPT)
        max_n: highest N value the head can emit. Targets are clamped to
            [0, max_n]. Default 20 (COCOA truth N has max ≈ 27, but 20
            covers >99.9%).
    """

    def __init__(self, config: GPTConfig, max_n: int = 20):
        super().__init__(config)
        self.max_n = int(max_n)
        self.n_head_mlp = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd, bias=config.bias),
            nn.GELU(),
            nn.Linear(config.n_embd, self.max_n + 1, bias=config.bias),
        )
        # Init the new head (parent __init__ ran apply(_init_weights) before
        # we registered the head, so we init here explicitly).
        for m in self.n_head_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        n_head_pos: torch.Tensor | None = None,
        n_targets: torch.Tensor | None = None,
        past_kv_list=None,
        return_kv_list: bool = False,
    ):
        """Returns:
          (logits, token_loss, n_loss) if n_head_pos is not None
          (logits, token_loss) otherwise (matches upstream GPT signature)
          (logits, token_loss, new_kv_list) if return_kv_list (n_head not supported with KV cache)
        When `targets` is None (inference), also returns only logits at the
        last position (same as upstream). In that inference path `n_loss` is
        always None; the caller should invoke `n_head_predict` explicitly.
        """
        device = idx.device
        b, t = idx.size()

        if past_kv_list is not None:
            T_past = past_kv_list[0][0].shape[2]
            assert t == 1, "KV-cached forward only supports t=1 new tokens per call"
            assert n_head_pos is None and n_targets is None and targets is None, (
                "KV-cached path is inference-only; n_head/targets must be None"
            )
        else:
            T_past = 0
        assert T_past + t <= self.config.block_size, (
            f"seq len {T_past + t} > block_size {self.config.block_size}"
        )
        pos = torch.arange(T_past, T_past + t, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        new_kv_list = [] if return_kv_list else None
        for li, block in enumerate(self.transformer.h):
            past_kv = past_kv_list[li] if past_kv_list is not None else None
            x, new_kv = block(x, past_kv=past_kv, return_kv=return_kv_list)
            if return_kv_list:
                new_kv_list.append(new_kv)
        x = self.transformer.ln_f(x)  # (B, T, n_embd)

        if targets is None:
            logits = self.lm_head(x[:, [-1], :])
            if return_kv_list:
                return logits, None, new_kv_list
            if n_head_pos is not None:
                # Inference-time N-head evaluation (1-position gather).
                B = b
                h_at = x[torch.arange(B, device=device), n_head_pos]
                n_logits = self.n_head_mlp(h_at)
                return logits, None, n_logits
            return logits, None

        logits = self.lm_head(x)
        token_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
        )

        if n_head_pos is None or n_targets is None:
            return logits, token_loss

        B = b
        h_at = x[torch.arange(B, device=device), n_head_pos]  # (B, n_embd)
        n_logits = self.n_head_mlp(h_at)  # (B, max_n + 1)
        n_targets_c = n_targets.clamp(0, self.max_n).long()
        n_loss = F.cross_entropy(n_logits, n_targets_c)
        return logits, token_loss, n_loss

    @torch.no_grad()
    def n_head_predict(self, idx: torch.Tensor, n_head_pos: torch.Tensor):
        """Return per-sample N-logits (B, max_n+1). For inference-time use."""
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x, _ = block(x)
        x = self.transformer.ln_f(x)
        h_at = x[torch.arange(b, device=device), n_head_pos]
        return self.n_head_mlp(h_at)

    # ------------------------------------------------------------------
    # Optimizer construction helpers (for clean resume from base-GPT ckpt)
    # ------------------------------------------------------------------
    def configure_optimizers_base_only(self, weight_decay, learning_rate, betas, device_type):
        """Build an AdamW optimizer whose param groups contain ONLY the
        base-GPT parameters (i.e. everything except `n_head_mlp.*`).

        Intended for use at resume time when the old checkpoint's optimizer
        state matches only the base params. After `optimizer.load_state_dict(
        old_state)` succeeds, call `add_n_head_param_groups(optimizer, wd)`
        to register the N-head params as fresh-state groups.
        """
        import inspect
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        base = {pn: p for pn, p in param_dict.items() if not pn.startswith("n_head_mlp")}
        decay_params = [p for n, p in base.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in base.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra = dict(fused=True) if use_fused else {}
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra)
        print(f"[GPTWithNHead] base-only optimizer: "
              f"{len(decay_params)} decay / {len(nodecay_params)} nodecay params")
        return optimizer

    def add_n_head_param_groups(self, optimizer: torch.optim.Optimizer, weight_decay: float):
        """After loading old optimizer state into a base-only optimizer, call
        this to register the N-head params with fresh state as NEW param
        groups (so existing state_dict positions are undisturbed)."""
        head = {pn: p for pn, p in self.named_parameters()
                if pn.startswith("n_head_mlp") and p.requires_grad}
        decay_params = [p for n, p in head.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in head.items() if p.dim() < 2]
        if decay_params:
            optimizer.add_param_group({"params": decay_params, "weight_decay": weight_decay})
        if nodecay_params:
            optimizer.add_param_group({"params": nodecay_params, "weight_decay": 0.0})
        print(f"[GPTWithNHead] added {len(decay_params)} decay + {len(nodecay_params)} nodecay "
              f"N-head params as fresh-state optimizer groups")
