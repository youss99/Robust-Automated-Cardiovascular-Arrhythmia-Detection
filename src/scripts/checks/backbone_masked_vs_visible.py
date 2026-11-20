#!/usr/bin/env python3
"""Was a pre-trained backbone trained to reproduce VISIBLE patches?

The intended pre-training loss scores only the masked patches, so a backbone trained with it is never asked to output
the patches it can see. Before the November 2026 fix, single-lead pre-training also scored visible patches
(README_pretraining_loss_fix_2026-11.md). A backbone trained that way reproduces visible patches well: its error on
visible patches is clearly lower than its error on masked patches. This script measures both errors for one backbone,
with the same steps as a pre-training validation pass (normalisation, random masks, model).

Usage, from the repo root (a GPU is used when there is one):
    .venv/bin/python src/scripts/checks/backbone_masked_vs_visible.py \\
        --backbone results/pre-train/saved_models/unified/<name>/<name>_best.pt \\
        --h5 <dataset folder>/UnifiedDataset.h5 --leads II

--h5 is a 12-lead file in the repo's layout: dataset "tracings", shape [records, 12, samples], leads in the order
I, II, III, AVR, AVL, AVF, V1..V6. Windows are taken evenly from the whole file, and windows where a selected lead is
flat are skipped. With --split_csv (a modified_dataset.csv whose rows match the H5 rows and which has the
strat_fold_unannotated column) only that split's validation windows are used.
"""
import argparse
import sys
import warnings
from pathlib import Path

import h5py
import numpy as np
import torch

warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.core.constants.definitions import DataAugmentation, Mode, Model  # noqa: E402
from src.core.datasets.ecg_interface import EcgInterface  # noqa: E402
from src.core.support.patch_mask import PatchMaskSupport  # noqa: E402
from src.core.support.transforms import RevInSupport  # noqa: E402
from src.core.utils.learner_utils import get_model  # noqa: E402

# Reading of the result: the ratio visible-patch error / masked-patch error. In the measurements of the README
# (Section 3), every single-lead backbone trained before the fix came out below 0.6 and every backbone trained with
# the intended loss above 1.0. A backbone that has hardly trained has both errors near 1 and cannot be read.
COPIES_VISIBLE_BELOW, MASKED_ONLY_ABOVE, UNTRAINED_ABOVE = 0.6, 0.9, 0.9


def measure(backbone, h5_path, leads, model_name='vanilla_vit', per_lead=False, revin_mode='BN', patch_len=50,
            n_windows=3000, batch_size=128, seed=0, n_layers=6, d_model=128, n_heads=8, d_ff=512, device=None,
            split_csv=None):
    """Return (masked-patch MSE, visible-patch MSE, windows used, windows skipped)."""
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    channels = [EcgInterface.ecg_dict[lead] for lead in leads]
    tracings = h5py.File(h5_path, 'r')['tracings']
    n_records, _, seq = tracings.shape
    pool = np.arange(n_records)
    if split_csv:
        import pandas as pd
        fold = pd.read_csv(split_csv)['strat_fold_unannotated'].to_numpy()
        assert len(fold) == n_records, f'{split_csv} has {len(fold)} rows but the H5 has {n_records}'
        pool = np.flatnonzero(fold == 3)                                    # Split.VALIDATION
    rows = pool[np.unique(np.linspace(0, len(pool) - 1, min(n_windows, len(pool))).astype(int))]

    args = argparse.Namespace(model=Model[model_name], head_type=Mode.PRETRAIN, n_classes=None, class_token=None,
                              patch_len=patch_len, stride=patch_len, d_model=d_model, n_layers=n_layers,
                              n_heads=n_heads, d_ff=d_ff, dropout=0.0, head_dropout=0.0, shared_embedding=True,
                              linear_only=False)
    aug = DataAugmentation.per_lead_aug if per_lead else DataAugmentation.none
    model = get_model(c_in=len(leads) if per_lead else 1, seq_len=seq if per_lead else seq * len(leads), args=args,
                      global_rank=1)
    model.load_state_dict(torch.load(backbone, map_location='cpu', weights_only=False)['model'], strict=True)
    model = model.to(device).eval()

    np.random.seed(seed); torch.manual_seed(seed)
    sums = {'masked': 0.0, 'visible': 0.0}; counts = {'masked': 0, 'visible': 0}
    used = skipped = 0
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            raw = torch.tensor(tracings[rows[start:start + batch_size]][:, channels, :], dtype=torch.float32)
            keep = raw.std(dim=-1).min(dim=1).values > 1e-6
            skipped += int((~keep).sum()); raw = raw[keep].to(device)
            if len(raw) < 2:
                continue
            used += len(raw)
            x = RevInSupport(len(leads), denorm=False, norm_stats=revin_mode).before_forward(raw)
            masker = PatchMaskSupport(patch_len=patch_len, stride=patch_len, mask_ratio=0.4, per_lead=per_lead)
            patches, target, mask, inverted = masker.before_forward(input=x, target=x.clone(), device=device)
            pred = model(patches, mask=mask, inverted_mask=inverted, indices=None, augmentation=aug).reshape(target.shape)
            masked = masker.binary_mask.sum(dim=-1) == 0                    # a patch is masked when its mask row is zero
            error = ((pred - target) ** 2).mean(dim=-1)
            for key, sel in (('masked', masked), ('visible', ~masked)):
                sums[key] += error[sel].sum().item(); counts[key] += int(sel.sum())
    return sums['masked'] / counts['masked'], sums['visible'] / counts['visible'], used, skipped


def reading(masked_mse, visible_mse):
    ratio = visible_mse / masked_mse
    if masked_mse > UNTRAINED_ABOVE and visible_mse > UNTRAINED_ABOVE:
        return ratio, 'inconclusive: both errors are near 1 (the error of predicting zero), so the backbone is barely trained'
    if ratio < COPIES_VISIBLE_BELOW:
        return ratio, 'reproduces visible patches: trained with the single-lead loss from before the fix'
    if ratio > MASKED_ONLY_ABOVE:
        return ratio, 'does not reproduce visible patches: consistent with the intended masked-patch loss'
    return ratio, 'inconclusive: between the two patterns'


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--backbone', required=True, help='pre-training checkpoint (..._best.pt)')
    p.add_argument('--h5', required=True, help='12-lead H5 file with a "tracings" dataset')
    p.add_argument('--leads', required=True, help='leads the backbone was pre-trained on, comma separated, e.g. II or II,V2')
    p.add_argument('--model', default='vanilla_vit', choices=['vanilla_vit', 'patch_tst'])
    p.add_argument('--per_lead', action='store_true', help='backbone pre-trained with --data_augmentation per_lead_aug')
    p.add_argument('--revin_mode', default='BN', choices=['BN', 'instance'])
    p.add_argument('--patch_len', type=int, default=50)
    p.add_argument('--n_layers', type=int, default=6)
    p.add_argument('--d_model', type=int, default=128)
    p.add_argument('--n_heads', type=int, default=8)
    p.add_argument('--d_ff', type=int, default=512)
    p.add_argument('--n_windows', type=int, default=3000)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--split_csv', default=None, help='modified_dataset.csv matching the H5: use its validation rows only')
    a = p.parse_args()
    leads = [lead.strip() for lead in a.leads.split(',')]
    m, v, used, skipped = measure(a.backbone, a.h5, leads, a.model, a.per_lead, a.revin_mode, a.patch_len, a.n_windows,
                                  a.batch_size, a.seed, a.n_layers, a.d_model, a.n_heads, a.d_ff, split_csv=a.split_csv)
    ratio, text = reading(m, v)
    print(f'backbone          : {a.backbone}')
    print(f'windows           : {used} used, {skipped} skipped (flat lead)')
    print(f'masked-patch MSE  : {m:.3f}')
    print(f'visible-patch MSE : {v:.3f}')
    print(f'visible / masked  : {ratio:.2f}')
    print(f'reading           : {text}')
