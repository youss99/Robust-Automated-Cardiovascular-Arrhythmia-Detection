"""
Central-beat classifier = RevIN (BN or instance) -> 50-sample patches -> PatchECG ViT (thesis Small) -> 3-class head.

Shared by training, evaluation and patient inference so normalization is applied identically everywhere.
"""
import numpy as np
import pandas as pd
import torch
from torch import nn

from src.core.constants.definitions import DataAugmentation, Mode
from src.core.models.layers.revin import RevIN
from src.core.models.vit_1d import ViT
from src.incart_beat import preprocessing as pp

ARCH = dict(patch_len=50, stride=50, d_model=128, n_layers=6, n_heads=8, d_ff=512, dropout=0.0)
REVIN_EPS = 1e-5  # RevIN default, used by both normalization modes


class BeatClassifier(nn.Module):
    def __init__(self, norm_stats, n_classes=len(pp.CLASS_NAMES)):
        super().__init__()
        self.norm_stats = norm_stats
        # affine=False as in RevInSupport used by pretraining/finetuning (no learnable RevIN params)
        self.revin = RevIN(num_features=1, eps=REVIN_EPS, affine=False, norm_stats=norm_stats)
        self.vit = ViT(seq_len=pp.WINDOW_LEN, patch_size=ARCH['patch_len'], stride=ARCH['stride'],
                       num_classes=n_classes, dim=ARCH['d_model'], depth=ARCH['n_layers'], heads=ARCH['n_heads'],
                       mlp_dim=ARCH['d_ff'], mode=Mode.CLASSIFICATION, channels=1,
                       dim_head=ARCH['d_model'] // ARCH['n_heads'], dropout=ARCH['dropout'],
                       emb_dropout=ARCH['dropout'])

    def forward(self, x):
        """x: [bs x WINDOW_LEN] microvolts at TARGET_FS -> logits [bs x n_classes]."""
        xb = self.revin(x.unsqueeze(1), 'norm')                    # [bs x 1 x 1000]
        # Same as patch_mask.create_patch for seq_len % patch_len == 0 (not imported: it pulls in matplotlib
        # and the whole learner_utils model zoo, which is slow to import from /project).
        xb = xb.unfold(dimension=-1, size=ARCH['patch_len'], step=ARCH['stride'])  # [bs x 1 x 20 x 50]
        return self.vit(xb, mask=None, inverted_mask=None, indices=None, augmentation=DataAugmentation.none)


def load_pretrained_backbone(model, ckpt_path):
    """Copy every non-head ViT weight from a PatchECG pretrain checkpoint; fail loudly on any mismatch."""
    src = torch.load(ckpt_path, map_location='cpu', weights_only=True)['model']
    dst = model.vit.state_dict()
    backbone = [k for k in dst if 'head' not in k]
    missing = [k for k in backbone if k not in src or src[k].shape != dst[k].shape]
    if missing:
        raise RuntimeError(f'Backbone keys missing or mismatched in {ckpt_path}: {missing}')
    model.vit.load_state_dict({**dst, **{k: src[k] for k in backbone}})
    return backbone


def load_beat_classifier(ckpt_path, device='cpu'):
    """Load a trained classifier plus its saved metadata (preprocessing config, normalization, classes)."""
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    meta = ck['meta']
    if meta['preprocessing'] != pp.preprocessing_config(meta['preprocessing']['lead']):
        raise RuntimeError('Checkpoint was trained with a different preprocessing definition than this code')
    model = BeatClassifier(norm_stats=meta['norm_stats'])
    model.load_state_dict(ck['model'])
    return model.to(device).eval(), meta


@torch.no_grad()
def predict_beats(model, signal, fs, units, beat_samples, device='cpu', batch_size=128):
    """
    Patient inference: classify each beat in `beat_samples` (indices at `fs`) using the shared preprocessing.
    `signal` is one physical lead (lead II) in `units` (e.g. 'mV' for the Homer CSVs). Beats without a full
    10 s window are returned with NaN probabilities. With norm_stats='BN' results depend on `batch_size` and on
    which beats share a batch; 'instance' models do not.
    """
    x = pp.prepare_lead(signal, fs, units)
    centres = pp.map_samples(beat_samples, fs)
    windows, valid = pp.extract_windows(x, centres)
    probs = np.full((len(centres), len(pp.CLASS_NAMES)), np.nan, dtype=np.float32)
    out = []
    for i in range(0, len(windows), batch_size):
        xb = torch.from_numpy(windows[i:i + batch_size]).to(device)
        out.append(torch.softmax(model(xb), dim=-1).cpu().numpy())
    if out:
        probs[valid] = np.concatenate(out)
    df = pd.DataFrame(probs, columns=[f'p_{c}' for c in pp.CLASS_NAMES])
    df.insert(0, 'sample', np.asarray(beat_samples))
    df.insert(1, 'full_window', valid)
    df['pred'] = [pp.CLASS_NAMES[k] if v else None for k, v in zip(np.nan_to_num(probs).argmax(1), valid)]
    return df
