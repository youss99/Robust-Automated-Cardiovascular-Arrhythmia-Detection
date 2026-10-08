"""
ML inference helpers for the LTST pipeline.

Loads the PatchECG vanilla_vit regression checkpoint
(ltstdb_two_lead_regression_supervised_no_tta) and converts a 2-lead WFDB
record into per-10s-segment ST-deviation predictions, ready to be consumed
by run_fsm_on_record.

The checkpoint expects:
  * 2 leads in order [II, V2], in mV (raw, not pre-normalized)
  * 100 Hz, exactly 1000 samples per segment
RevIN normalization (BN mode) is applied internally per batch.

LTSTDB ships at 250 Hz; we resample each 2500-sample window to 1000 samples
with scipy.signal.resample_poly (matches prepare_regression_datasets.py).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
from scipy.signal import resample_poly

# --- make the PatchECG repo importable -------------------------------------
_DEFAULT_PATCHECG_ROOT = "/home/student/GIT/Robust-Automated-Cardiovascular-Arrhythmia-Detection"
PATCHECG_ROOT = Path(os.environ.get("PATCHECG_REPO_ROOT", _DEFAULT_PATCHECG_ROOT))
if str(PATCHECG_ROOT) not in sys.path:
    sys.path.insert(0, str(PATCHECG_ROOT))

from src.core.constants.definitions import DataAugmentation, Mode  # noqa: E402
from src.core.models.layers.revin import RevIN                      # noqa: E402
from src.core.models.vit_1d import ViT                               # noqa: E402
from src.core.support.patch_mask import create_patch                 # noqa: E402

# ---------------------------------------------------------------------------
# Constants matched to the trained checkpoint
# ---------------------------------------------------------------------------
DEFAULT_CKPT = (
    PATCHECG_ROOT
    / "results/fine-tune/saved_models/patient-segment-regression"
    / "ltstdb_two_lead_regression_supervised_no_tta"
    / "ltstdb_two_lead_regression_supervised_no_tta_best.pt"
)

MODEL_FS = 100                                       # Hz the checkpoint was trained at
SEGMENT_SECONDS = 10
SEGMENT_SAMPLES_AT_MODEL = MODEL_FS * SEGMENT_SECONDS  # 1000
PATCH_LEN = 50
N_LEADS = 2


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_patchecg_model(
    ckpt_path: Union[str, Path] = DEFAULT_CKPT,
    device: Union[str, torch.device] = "cpu",
    lead_mode: str = "two_lead",
) -> Tuple[ViT, RevIN, torch.device]:
    """Build the ViT, load the trained weights, and instantiate the matching RevIN.

    lead_mode:
      "two_lead"  - 2 input leads, 2 outputs (default; existing behaviour).
                    seq_len = SEGMENT_SAMPLES_AT_MODEL * 2 = 2000, num_classes = 2.
      "lead_II"   - 1 input lead (II), 1 output.
                    seq_len = SEGMENT_SAMPLES_AT_MODEL = 1000, num_classes = 1.
      "lead_V2"   - 1 input lead (V2), 1 output. Same architecture as lead_II.
    """
    if lead_mode == "two_lead":
        seq_len = SEGMENT_SAMPLES_AT_MODEL * N_LEADS
        num_classes = N_LEADS
        revin_n = N_LEADS
    elif lead_mode in ("lead_II", "lead_V2"):
        seq_len = SEGMENT_SAMPLES_AT_MODEL
        num_classes = 1
        revin_n = 1
    else:
        raise ValueError(f"unknown lead_mode={lead_mode!r}; expected two_lead | lead_II | lead_V2")

    device = torch.device(device)
    model = ViT(
        seq_len=seq_len,
        patch_size=PATCH_LEN,
        stride=PATCH_LEN,
        num_classes=num_classes,
        dim=128,
        depth=6,
        heads=8,
        mlp_dim=512,
        mode=Mode.REGRESSION,
        channels=1,
        dim_head=128 // 8,
        dropout=0.0,
        emb_dropout=0.0,
    ).to(device).eval()

    state = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    revin = RevIN(num_features=revin_n, affine=False, norm_stats="BN").to(device)
    return model, revin, device


# ---------------------------------------------------------------------------
# Resampling helper
# ---------------------------------------------------------------------------
def _resample_to_model_fs(seg_at_fs: np.ndarray, fs: int) -> np.ndarray:
    """
    seg_at_fs: (n_segments, segment_samples_at_fs, n_leads)
    Returns:    (n_segments, n_leads, SEGMENT_SAMPLES_AT_MODEL)
    """
    if fs == MODEL_FS:
        out = seg_at_fs
    else:
        out = resample_poly(seg_at_fs, up=MODEL_FS, down=fs, axis=1)
        if out.shape[1] > SEGMENT_SAMPLES_AT_MODEL:
            out = out[:, :SEGMENT_SAMPLES_AT_MODEL, :]
        elif out.shape[1] < SEGMENT_SAMPLES_AT_MODEL:
            pad = SEGMENT_SAMPLES_AT_MODEL - out.shape[1]
            out = np.pad(out, ((0, 0), (0, pad), (0, 0)))
    return np.transpose(out, (0, 2, 1)).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Per-record batched inference
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_st_per_segment(
    signal_2lead: np.ndarray,
    fs: int,
    model: ViT,
    revin: RevIN,
    device: torch.device,
    lead_order: Tuple[str, str] = ("II", "V2"),
    batch_size: int = 64,
    lead_mode: str = "two_lead",
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Run the model over all non-overlapping 10s windows of `signal_2lead`.
    Trailing samples that don't fill a full 10s window are dropped.

    Args:
        signal_2lead: (N, 2) ndarray in mV. Column order MUST match `lead_order`.
                      For LTSTDB read with wfdb.rdrecord(...).p_signal[:, [0, 1]]
                      this is (II, V2).
        fs:           sampling rate of `signal_2lead` (e.g. 250 for LTSTDB).
        model, revin, device: from `load_patchecg_model`.
        lead_order:   names assigned to columns 0 and 1 (used as dict keys).
        batch_size:   number of segments per forward pass.

    Returns:
        samp_arr  : (n_segments,) int64 — END sample index of each window in
                    the ORIGINAL fs domain (i.e. (i+1)*seg_samples_at_fs - 1).
        st_by_lead: dict mapping lead name -> (n_segments,) float32.
                    Segments containing non-finite input values yield NaN.
    """
    if signal_2lead.ndim != 2 or signal_2lead.shape[1] != N_LEADS:
        raise ValueError(f"signal_2lead must be shape (N, 2); got {signal_2lead.shape}")
    if len(lead_order) != N_LEADS:
        raise ValueError(f"lead_order must have {N_LEADS} entries; got {lead_order}")

    seg_samples_at_fs = int(fs * SEGMENT_SECONDS)
    n_total = signal_2lead.shape[0]
    n_segments = n_total // seg_samples_at_fs

    if n_segments == 0:
        return (
            np.empty(0, dtype=np.int64),
            {lead_order[0]: np.empty(0, dtype=np.float32),
             lead_order[1]: np.empty(0, dtype=np.float32)},
        )

    used = signal_2lead[: n_segments * seg_samples_at_fs]
    segs = used.reshape(n_segments, seg_samples_at_fs, N_LEADS).astype(np.float32, copy=False)

    # Mark segments with any non-finite value so we can NaN their predictions.
    finite_per_seg = np.isfinite(segs).all(axis=(1, 2))
    if not finite_per_seg.all():
        segs = np.where(np.isfinite(segs), segs, 0.0).astype(np.float32, copy=False)

    segs_model = _resample_to_model_fs(segs, fs)  # (n_segments, 2, 1000)

    # Select which leads to feed the model based on lead_mode.
    if lead_mode == "two_lead":
        model_in = segs_model                   # (n_segments, 2, 1000)
        n_out_leads = N_LEADS
    elif lead_mode == "lead_II":
        model_in = segs_model[:, 0:1, :]        # (n_segments, 1, 1000) — II only
        n_out_leads = 1
    elif lead_mode == "lead_V2":
        model_in = segs_model[:, 1:2, :]        # (n_segments, 1, 1000) — V2 only
        n_out_leads = 1
    else:
        raise ValueError(f"unknown lead_mode={lead_mode!r}")

    out = np.empty((n_segments, n_out_leads), dtype=np.float32)
    for start in range(0, n_segments, batch_size):
        chunk = model_in[start : start + batch_size]
        xb = torch.from_numpy(chunk).to(device)
        xb = revin(xb, "norm")
        xb_patch = create_patch(xb, patch_len=PATCH_LEN, stride=PATCH_LEN)
        pred = model(
            xb_patch,
            mask=None,
            inverted_mask=None,
            indices=None,
            augmentation=DataAugmentation.none,
        )
        pr = pred.detach().cpu().numpy()
        if pr.ndim == 1:
            pr = pr[:, None]
        out[start : start + chunk.shape[0]] = pr

    if not finite_per_seg.all():
        out[~finite_per_seg] = np.nan

    samp_arr = (np.arange(1, n_segments + 1) * seg_samples_at_fs - 1).astype(np.int64)

    if lead_mode == "two_lead":
        return samp_arr, {lead_order[0]: out[:, 0], lead_order[1]: out[:, 1]}
    elif lead_mode == "lead_II":
        return samp_arr, {lead_order[0]: out[:, 0]}  # V2 absent
    else:  # lead_V2
        return samp_arr, {lead_order[1]: out[:, 0]}  # II absent


# ---------------------------------------------------------------------------
# Tiny self-test / smoke
# ---------------------------------------------------------------------------
def _smoke_test() -> None:
    """python ml_inference.py --smoke  -> runs 30s of random input through the model."""
    model, revin, device = load_patchecg_model()
    fs = 250
    sig = np.random.default_rng(0).standard_normal((30 * fs, 2)).astype(np.float32) * 0.3
    samp_arr, st_by_lead = predict_st_per_segment(sig, fs, model, revin, device)
    print(f"n_segments = {len(samp_arr)}")
    print(f"samp_arr   = {samp_arr}")
    print(f"II preds   = {st_by_lead['II']}")
    print(f"V2 preds   = {st_by_lead['V2']}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="run a smoke test on random input")
    if parser.parse_args().smoke:
        _smoke_test()
