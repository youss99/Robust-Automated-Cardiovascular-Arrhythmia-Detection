"""
Mini demo: build a synthetic 10s, 2-lead (II, V2) ECG at 100 Hz,
run the trained ltstdb_two_lead_regression model, and print the
predicted regression value for each lead.

Run from the repo root:
    cd /home/student/GIT/Robust-Automated-Cardiovascular-Arrhythmia-Detection
    python run_inference_demo.py
"""

import numpy as np
import torch

from src.core.constants.definitions import DataAugmentation, Mode
from src.core.models.layers.revin import RevIN
from src.core.models.vit_1d import ViT
from src.core.support.patch_mask import create_patch


CKPT = (
    "results/fine-tune/saved_models/patient-segment-regression/"
    "ltstdb_two_lead_regression_supervised_no_tta/"
    "ltstdb_two_lead_regression_supervised_no_tta_best.pt"
)
FS = 100
DURATION_S = 10
N_SAMPLES = FS * DURATION_S  # 1000


def synth_ecg(fs: int = FS, duration_s: int = DURATION_S, hr_bpm: float = 72.0,
              lead_amp: float = 1.0, seed: int = 0) -> np.ndarray:
    """
    Cheap synthetic 2-lead ECG (lead II, lead V2) for sanity-checking the pipeline.
    Not physiologically accurate -- just produces P-QRS-T-shaped beats in mV-ish range.
    Returns array of shape (2, fs * duration_s), float32.
    """
    rng = np.random.default_rng(seed)
    n = fs * duration_s
    t = np.arange(n) / fs
    rr = 60.0 / hr_bpm
    beat_times = np.arange(rr / 2, duration_s, rr)

    def gauss(center, width, amp):
        return amp * np.exp(-0.5 * ((t - center) / width) ** 2)

    def beat(center, p, q, r, s, tw):
        # widths in seconds, amplitudes in (arbitrary) mV
        return (
            gauss(center - 0.16, 0.025, p)
            + gauss(center - 0.02, 0.010, q)
            + gauss(center,         0.010, r)
            + gauss(center + 0.02, 0.010, s)
            + gauss(center + 0.30, 0.060, tw)
        )

    # lead II:  prominent upright R, small Q/S, upright T
    lead_ii = np.zeros(n, dtype=np.float32)
    for c in beat_times:
        lead_ii += beat(c, p=0.10, q=-0.10, r=1.10 * lead_amp, s=-0.20, tw=0.25)

    # lead V2: small/biphasic R, deep S, taller upright T
    lead_v2 = np.zeros(n, dtype=np.float32)
    for c in beat_times:
        lead_v2 += beat(c, p=0.05, q=-0.05, r=0.40 * lead_amp, s=-0.90, tw=0.45)

    # tiny baseline wander + noise so RevIN sees non-degenerate stats
    wander = 0.05 * np.sin(2 * np.pi * 0.3 * t)
    lead_ii = lead_ii + wander + rng.normal(0, 0.01, size=n).astype(np.float32)
    lead_v2 = lead_v2 + wander + rng.normal(0, 0.01, size=n).astype(np.float32)

    return np.stack([lead_ii, lead_v2], axis=0).astype(np.float32)


def build_model(device: torch.device) -> ViT:
    model = ViT(
        seq_len=N_SAMPLES * 2,   # leads concatenated as patches: 1000 * 2 = 2000
        patch_size=50,
        stride=50,
        num_classes=2,            # one regression target per lead
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
    state = torch.load(CKPT, map_location=device, weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    return model


@torch.no_grad()
def predict(model: ViT, revin: RevIN, xb_mv: torch.Tensor) -> torch.Tensor:
    """xb_mv: (B, 2, 1000) float32 in mV, lead order [II, V2]. Returns (B, 2)."""
    assert xb_mv.ndim == 3 and xb_mv.shape[1] == 2 and xb_mv.shape[2] == N_SAMPLES, (
        f"Expected (B, 2, {N_SAMPLES}), got {tuple(xb_mv.shape)}"
    )
    xb = revin(xb_mv, "norm")                                  # (B, 2, 1000)
    xb_patch = create_patch(xb, patch_len=50, stride=50)        # (B, 2, 20, 50)
    pred = model(
        xb_patch,
        mask=None,
        inverted_mask=None,
        indices=None,
        augmentation=DataAugmentation.none,
    )
    return pred                                                 # (B, 2)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = build_model(device)
    revin = RevIN(num_features=2, affine=False, norm_stats="BN").to(device)

    sig = synth_ecg(seed=0)                       # (2, 1000)
    print(f"Input signal shape: {sig.shape}")
    print(f"  lead II (mV)  min/mean/max: {sig[0].min():+.3f} / {sig[0].mean():+.3f} / {sig[0].max():+.3f}")
    print(f"  lead V2 (mV)  min/mean/max: {sig[1].min():+.3f} / {sig[1].mean():+.3f} / {sig[1].max():+.3f}")

    xb = torch.from_numpy(sig).unsqueeze(0).to(device)  # (1, 2, 1000)
    pred = predict(model, revin, xb).cpu().numpy()[0]

    print("\nPredicted regression targets:")
    print(f"  lead II (target_lead0_mV) -> {pred[0]:+.6f}")
    print(f"  lead V2 (target_lead1_mV) -> {pred[1]:+.6f}")


if __name__ == "__main__":
    main()
