"""Pflow metrics for nano-hep, matching HEP4M's eval surface.

Detokenizes output-modality token triples via the frozen HEP4M VQ-VAE,
builds (pt, eta, phi) awkward arrays, and calls
`hep4m.performance.pflow_report.run_report_from_arrays` to produce jet-level
metrics identical to HEP4M's training logs (jet_pt_response, cardinality
at threshold).

**Requires the `hep4m2` conda env** — imports vector_quantize_pytorch and HEP4M.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import yaml

# HEP4M imports
_HEP4M_ROOT = "/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/HEP4M"
if _HEP4M_ROOT not in sys.path:
    sys.path.insert(0, _HEP4M_ROOT)


def _load_vqvae(config_path_v: str, config_path_m: str, checkpoint_path: str, device: str):
    """Load a HEP4M VQ-VAE tokenizer with its pretrained weights, freeze, eval mode."""
    from hep4m.models.vqvae import VQVAE
    with open(config_path_v) as f:
        cfg_v = yaml.safe_load(f)
    with open(config_path_m) as f:
        cfg_m = yaml.safe_load(f)
    cfg_m["_config_v"] = cfg_v
    vq = VQVAE(cfg_m)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "state_dict" in state:
        state = state["state_dict"]
    # Strip "model." prefixes if present
    state = {k.replace("model.", "", 1) if k.startswith("model.") else k: v for k, v in state.items()}
    vq.load_state_dict(state, strict=False)
    vq.to(device).eval()
    for p in vq.parameters():
        p.requires_grad = False
    return vq


class PflowMetrics:
    """Callable that, given (pred_codes, true_codes) for a batch of events,
    decodes each via HEP4M VQ-VAE and produces pflow metrics + plots."""

    def __init__(self, modality_dict_path: str, output_modality: str, device: str):
        with open(modality_dict_path) as f:
            self.mod_dict = yaml.safe_load(f)
        self.output_modality = output_modality
        self.device = device
        entry = self.mod_dict[output_modality]
        self.vq = _load_vqvae(
            entry["config_path_v"], entry["config_path_m"],
            entry["checkpoint_path"], device,
        )

        # Read pos_tokenizer if available (for eta/phi inverse).
        from hep4m.models.pos_tokenizer import PosTokenizer
        self.pos_tok = PosTokenizer().to(device)

        # Read the VQ-VAE config_v to know which features it emits (order of x_hat cont vars).
        with open(entry["config_path_v"]) as f:
            cfg_v = yaml.safe_load(f)
        key = f"{output_modality}_feat0"
        self.feat_names = cfg_v["features"][key][1]
        self.gpos_names = cfg_v["features"][key][3] if len(cfg_v["features"][key]) > 3 else []

        # Build proper inverse-transform objects per continuous feature.
        # feat_names e.g. ['truthpart_pt', 'truthpart_e'].
        from hep4m.utility.var_transformation import VarTransformation
        tdict = cfg_v.get("transformation_dict", {})
        self._inverters = []
        for fn in self.feat_names:
            if fn in tdict:
                self._inverters.append(VarTransformation(tdict[fn]))
            else:
                self._inverters.append(None)

    @torch.no_grad()
    def decode_tokens_to_features(self, codes: torch.Tensor, pos_codes: torch.Tensor,
                                   mask: torch.Tensor) -> Dict[str, Any]:
        """codes: (B, N, num_q) long; pos_codes: (B, N, 3) long; mask: (B, N) bool (True=real)."""
        codes = codes.to(self.device)
        # VQVAE.indices_to_zq wants (B, N, num_q) or (B, N) depending on config
        z_q = self.vq.indices_to_zq(codes, mask.to(self.device))
        x_hat_cont, _ = self.vq.decode(z_q, x_mask=mask.to(self.device))
        # pos decode
        if pos_codes is not None and self.pos_tok is not None:
            pos_codes = pos_codes.to(self.device)
            x_gpos_hat = self.pos_tok.decode(pos_codes)  # (B, N, 3) eta, cosphi, sinphi
        else:
            x_gpos_hat = None
        return {"x_hat_cont": x_hat_cont.cpu(), "x_gpos_hat": x_gpos_hat.cpu() if x_gpos_hat is not None else None}

    def _x_to_pt_eta_phi(self, x_hat_cont: torch.Tensor, x_gpos_hat: torch.Tensor,
                          mask: torch.Tensor) -> Tuple[list, list, list]:
        """Convert decoded (continuous features + gpos) into jagged (pt, eta, phi) per event.
        Assumes feat_names[0] is pt (first continuous var is typically pt/pt_log). Attempts
        to inverse-log if the feature name contains 'log'."""
        B, N, _ = x_hat_cont.shape
        pt_list, eta_list, phi_list = [], [], []
        m = mask.bool().cpu().numpy()
        x_cont = x_hat_cont.numpy()
        x_gpos = x_gpos_hat.numpy() if x_gpos_hat is not None else None

        pt_idx = 0  # first cont feature is pt-like
        inv = self._inverters[pt_idx] if pt_idx < len(self._inverters) else None

        for b in range(B):
            sel = m[b]
            pt = x_cont[b, sel, pt_idx]
            if inv is not None:
                pt = inv.inverse(pt)
            if x_gpos is not None and x_gpos.shape[-1] == 3:
                eta = x_gpos[b, sel, 0]
                cosphi = x_gpos[b, sel, 1]
                sinphi = x_gpos[b, sel, 2]
                phi = np.arctan2(sinphi, cosphi)
            else:
                eta = np.zeros_like(pt); phi = np.zeros_like(pt)
            pt_list.append(pt); eta_list.append(eta); phi_list.append(phi)
        return pt_list, eta_list, phi_list

    def compute_metrics(
        self,
        pred_codes: torch.Tensor, pred_gpos: torch.Tensor, pred_mask: torch.Tensor,
        true_codes: torch.Tensor, true_gpos: torch.Tensor, true_mask: torch.Tensor,
        outdir: "str | None" = None, ind_threshold: float = 0.45,
    ) -> Dict[str, Any]:
        """Decode both (pred, true), build awkward arrays, call run_report_from_arrays."""
        import awkward as ak
        from hep4m.performance.pflow_report import run_report_from_arrays

        reco = self.decode_tokens_to_features(pred_codes, pred_gpos, pred_mask)
        truth = self.decode_tokens_to_features(true_codes, true_gpos, true_mask)

        reco_pt, reco_eta, reco_phi = self._x_to_pt_eta_phi(reco["x_hat_cont"], reco["x_gpos_hat"], pred_mask)
        truth_pt, truth_eta, truth_phi = self._x_to_pt_eta_phi(truth["x_hat_cont"], truth["x_gpos_hat"], true_mask)

        reco_dict = {
            "pt": ak.Array(reco_pt),
            "eta": ak.Array(reco_eta),
            "phi": ak.Array(reco_phi),
        }
        truth_dict = {
            "pt": ak.Array(truth_pt),
            "eta": ak.Array(truth_eta),
            "phi": ak.Array(truth_phi),
        }

        outdir = Path(outdir) if outdir else None
        if outdir is not None:
            outdir.mkdir(parents=True, exist_ok=True)
        metrics = run_report_from_arrays(
            truth=truth_dict, reco=reco_dict, reco_ind=None,
            outdir=str(outdir) if outdir else None,
            output_modality=self.output_modality,
            ind_threshold=ind_threshold,
            n_event_displays=0,
        )
        return metrics
