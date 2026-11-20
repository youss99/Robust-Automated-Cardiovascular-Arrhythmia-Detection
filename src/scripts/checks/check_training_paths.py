#!/usr/bin/env python3
"""Self-test of the pre-training and regression fine-tuning code paths.

Covers PatchECG (--model vanilla_vit) and PatchTST (--model patch_tst) with one lead, with two leads, and in the
12-lead layouts of the thesis, using the same classes the training scripts use. CPU only, synthetic data, nothing written outside a temporary folder, about
one minute. Run it from the repo root before submitting jobs:

    .venv/bin/python src/scripts/checks/check_training_paths.py

Exit code 0 only if every check passes. Lines marked INFO describe behaviour and are not checks.
Background and the expected result on the code before the fix: README_pretraining_loss_fix_2026-11.md.
"""
import argparse
import contextlib
import io
import os
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.core.constants.definitions import DataAugmentation, Mode, Model  # noqa: E402
from src.core.datasets.ecg_interface import ClassificationType  # noqa: E402
from src.core.models.optim_factory import prepare_optimizer  # noqa: E402
from src.core.models.patchTST import ClassToken  # noqa: E402
from src.core.support.patch_mask import FinetunePatchSupport, PatchMaskSupport  # noqa: E402
from src.core.support.tracking import TrainingTracker, compute_scores  # noqa: E402
from src.core.support.transforms import RevInSupport  # noqa: E402
from src.core.utils.learner_utils import get_model, transfer_weights  # noqa: E402

PATCH, SEQ, N_PATCH, MASKED_PER_LEAD = 50, 1000, 20, 8          # 10 s at 100 Hz, mask ratio 0.4
RESULTS = []                                                     # (status, section, name, detail)

# label, model, number of leads, per_lead (= --data_augmentation per_lead_aug: leads fed separately)
CONFIGS = [
    ('PatchECG, 1 lead', Model.vanilla_vit, 1, False),
    ('PatchECG, 2 leads in one sequence', Model.vanilla_vit, 2, False),
    ('PatchTST, 1 lead', Model.patch_tst, 1, False),
    ('PatchTST, 1 lead, per_lead_aug (the thesis flag for PatchTST)', Model.patch_tst, 1, True),
    ('PatchTST, 2 leads fed separately (per_lead_aug)', Model.patch_tst, 2, True),
    ('PatchTST, 2 leads in one sequence', Model.patch_tst, 2, False),
    # the two layouts the thesis itself trains (its Tables A.3 and A.5)
    ('PatchECG, 12 leads in one sequence (thesis layout)', Model.vanilla_vit, 12, False),
    ('PatchTST, 12 leads fed separately (thesis layout)', Model.patch_tst, 12, True),
]


def record(section, name, ok, detail=''):
    RESULTS.append(('PASS' if ok else 'FAIL', section, name, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f'  ({detail})' if detail else ''), flush=True)


def info(text):
    print(f'[INFO] {text}', flush=True)


def guarded(section, name, fn):
    """Run one check; an exception is a failure, with its text as the detail."""
    try:
        ok, detail = fn()
    except Exception as exc:  # noqa: BLE001
        ok, detail = False, f'raised {type(exc).__name__}: {str(exc)[:160]}'
    record(section, name, ok, detail)


def build(model, head, n_leads, per_lead, n_classes=None):
    """Build a model the way patchECG_pretrain.py and patchECG_finetune.py do."""
    aug = DataAugmentation.per_lead_aug if per_lead else DataAugmentation.none
    seq_len = SEQ if per_lead else SEQ * n_leads
    c_in = n_leads if per_lead else 1
    args = argparse.Namespace(
        model=model, head_type=head, n_classes=n_classes, patch_len=PATCH, stride=PATCH, d_model=128, n_layers=6,
        n_heads=8, d_ff=512, dropout=0.0, head_dropout=0.0, shared_embedding=True, linear_only=False,
        class_token=None if head is Mode.PRETRAIN else ClassToken.CLS_TOKEN,
        layer_decay=0.65, opt='adamw', weight_decay=0.05, lr=1e-3, opt_eps=1e-8, opt_betas=None, momentum=0.9,
        sam=False, finetuning=True)
    return get_model(c_in=c_in, seq_len=seq_len, args=args, global_rank=1), aug, args


def masked_patches(binary_mask):
    """Independent of the repo's mask handling: a patch is masked when its whole row of the binary mask is zero."""
    return binary_mask.sum(dim=-1) == 0                          # [bs x n_leads x num_patch]


# ---------------------------------------------------------------------------------------------- 1. pre-training loss
def check_loss():
    print('\n== 1. Pre-training loss: is it the mean squared error over the masked patches only? ==')
    torch.manual_seed(0); np.random.seed(0)                     # same masks and errors in every run of this script
    for n_leads in (1, 2, 12):
        for per_lead in (False, True):
            name = (f'masked-patch loss, {n_leads} lead{"s" if n_leads > 1 else ""}, '
                    f'{"per_lead_aug masks" if per_lead else "default masks"}')

            def run(n_leads=n_leads, per_lead=per_lead):
                bad, worst, shape = [], 0.0, None
                for bs in (128, 100, 26, 3, 2, 1):
                    pm = PatchMaskSupport(patch_len=PATCH, stride=PATCH, mask_ratio=0.4, per_lead=per_lead)
                    x = torch.randn(bs, n_leads, SEQ)
                    _, target, _, inverted = pm.before_forward(input=x, target=x.clone(), device='cpu')
                    masked = masked_patches(pm.binary_mask)
                    if bs == 128:
                        shape = tuple(inverted.shape)
                    if not bool((masked.sum(dim=(1, 2)) == MASKED_PER_LEAD * n_leads).all()):
                        bad.append(f'bs {bs}: masked patches per sample are not {MASKED_PER_LEAD * n_leads}')
                        continue
                    pred = torch.randn_like(target)
                    per_patch = ((pred - target) ** 2).mean(dim=-1)
                    cases = {'random error': (pred, per_patch[masked].mean().item())}
                    on_masked = target.clone(); on_masked[masked] += 1.0
                    cases['error only on masked patches'] = (on_masked, 1.0)
                    on_visible = target.clone(); on_visible[~masked] += 1.0
                    cases['error only on visible patches'] = (on_visible, 0.0)
                    one_sample = target.clone(); one_sample[0] += 1.0
                    cases['error only in sample 0'] = (one_sample, masked[0].sum().item() / masked.sum().item())
                    for case, (p, want) in cases.items():
                        got = pm.loss(p, target).item()
                        err = abs(got - want) / max(abs(want), 1.0)
                        worst = max(worst, err)
                        if err > 1e-4:
                            bad.append(f'bs {bs}, {case}: loss {got:.4f}, intended {want:.4f}')
                if bad:
                    return False, f'mask shape at bs 128: {shape}; ' + '; '.join(bad[:4]) + (' ...' if len(bad) > 4 else '')
                return True, f'batch sizes 128, 100, 26, 3, 2, 1; 4 cases each; worst relative error {worst:.1e}'
            guarded('loss', name, run)


# ---------------------------------------------------------------------------------------------- 2. model forward
def check_pretrain_forward():
    print('\n== 2. Pre-training forward pass: output shape, and masked patches must be invisible to the model ==')
    for label, model_enum, n_leads, per_lead in CONFIGS:
        def run(model_enum=model_enum, n_leads=n_leads, per_lead=per_lead):
            torch.manual_seed(1); np.random.seed(1)
            model, aug, _ = build(model_enum, Mode.PRETRAIN, n_leads, per_lead)
            model.eval()
            pm = PatchMaskSupport(patch_len=PATCH, stride=PATCH, mask_ratio=0.4, per_lead=per_lead)
            x = torch.randn(12, n_leads, SEQ)
            xp, target, mask, inverted = pm.before_forward(input=x, target=x.clone(), device='cpu')
            masked = masked_patches(pm.binary_mask)
            with torch.no_grad():
                out = model(xp, mask=mask, inverted_mask=inverted, indices=None, augmentation=aug).reshape(target.shape)
                hidden = xp.clone(); hidden[masked] += 5.0 * torch.randn_like(hidden[masked])
                out_hidden = model(hidden, mask=mask, inverted_mask=inverted, indices=None,
                                   augmentation=aug).reshape(target.shape)
                shown = xp.clone()
                first_visible = int((~masked[0, 0]).nonzero()[0])
                # change the patch's SHAPE: PatchECG normalises every patch (LayerNorm), so a constant offset or a
                # rescaling of a whole patch does not reach the model by design
                shown[0, 0, first_visible] += 5.0 * torch.randn(PATCH)
                out_shown = model(shown, mask=mask, inverted_mask=inverted, indices=None,
                                  augmentation=aug).reshape(target.shape)
            d_hidden = (out_hidden - out).abs().max().item()
            d_shown = (out_shown[0] - out[0]).abs().max().item()
            loss = pm.loss(out, target).item()
            ok = d_hidden < 1e-5 and d_shown > 1e-4 and np.isfinite(loss)
            return ok, (f'reconstruction {tuple(out.shape)}; changing masked patches moves the output by {d_hidden:.1e}, '
                        f'changing one visible patch by {d_shown:.1e}')
        guarded('pretrain forward', f'pre-training, {label}', run)


def check_regression_forward():
    print('\n== 3. Regression forward pass: output shape, and which leads each output reads ==')
    for label, model_enum, n_leads, per_lead in CONFIGS:
        def run(model_enum=model_enum, n_leads=n_leads, per_lead=per_lead):
            torch.manual_seed(2)
            model, aug, _ = build(model_enum, Mode.REGRESSION, n_leads, per_lead, n_classes=n_leads)
            model.eval()
            patcher = FinetunePatchSupport(patch_len=PATCH, stride=PATCH, per_lead=per_lead)

            def predict(batch):
                xp, _, _, _ = patcher.before_forward(input=batch, target=None, device='cpu')
                with torch.no_grad():
                    return model(xp, mask=None, inverted_mask=None, indices=None, augmentation=aug)
            x = torch.randn(8, n_leads, SEQ)
            y = predict(x)
            if tuple(y.shape) != (8, n_leads):
                return False, f'output {tuple(y.shape)}, expected {(8, n_leads)}'
            if n_leads == 1:
                return True, f'output {tuple(y.shape)}'
            x2 = x.clone(); x2[:, 1] += torch.randn(8, SEQ)                 # change the second lead only
            y2 = predict(x2)
            d_first, d_second = (y2[:, 0] - y[:, 0]).abs().max().item(), (y2[:, 1] - y[:, 1]).abs().max().item()
            if per_lead:
                ok = d_first < 1e-5 and d_second > 1e-4
                what = 'each output reads only its own lead'
            else:
                ok = d_first > 1e-5 and d_second > 1e-5
                what = 'both outputs read both leads'
            return ok, (f'output {tuple(y.shape)}; changing lead 2 moves output 1 by {d_first:.1e} and output 2 by '
                        f'{d_second:.1e} ({what})')
        guarded('regression forward', f'regression, {label}', run)


# ---------------------------------------------------------------------------------------------- 4. logged loss
def check_logging():
    print('\n== 4. Logged loss: is train_loss / valid_loss the mean loss per window? ==')
    batches = [(128, 0.5), (128, 0.7), (44, 0.9)]                           # (batch size, batch loss)
    want = sum(bs * loss for bs, loss in batches) / sum(bs for bs, _ in batches)
    cases = [('pre-training (masked-patch loss)', PatchMaskSupport(PATCH, PATCH, 0.4, False).loss, Mode.PRETRAIN,
              ClassificationType.PRETRAIN),
             ('regression fine-tuning (MSELoss)', torch.nn.MSELoss(reduction='mean'), Mode.REGRESSION,
              ClassificationType.REGRESSION)]
    for label, loss_func, mode, ctype in cases:
        def run(loss_func=loss_func, mode=mode, ctype=ctype):
            with tempfile.TemporaryDirectory() as tmp:
                tracker = TrainingTracker(loss_func=loss_func, mode=mode, number_of_classes=1,
                                          save_path=os.path.join(tmp, 'unused'), classification_type=ctype)
                tracker.before_fit(); tracker.reset()
                for bs, loss in batches:
                    tracker.accumulate((torch.zeros(bs, 1, 4), torch.zeros(bs, 1)), torch.tensor(loss), torch.zeros(bs, 1))
                values, _ = compute_scores(predictions=tracker.predictions, labels=tracker.labels, number_of_classes=1,
                                           metrics_by_class=False, batch_recorder=tracker.batch_recorder,
                                           classification_type=ctype, mode=mode)
            got = values['loss']
            return abs(got - want) < 1e-6, f'three batches with mean loss {want:.4f} per window are logged as {got:.4f}'
        guarded('logging', f'logged loss, {label}', run)


# ---------------------------------------------------------------------------------------------- 5. weight transfer
def check_transfer():
    print('\n== 5. Loading a pre-trained backbone into a regression model ==')
    cases = [  # label, model, backbone (leads, per_lead), regression model (leads, per_lead), must it load?
        ('PatchECG, 1-lead backbone -> 1-lead model', Model.vanilla_vit, (1, False), (1, False), True),
        ('PatchECG, 2-lead backbone -> 2-lead model', Model.vanilla_vit, (2, False), (2, False), True),
        ('PatchTST, 1-lead backbone -> 1-lead model', Model.patch_tst, (1, False), (1, False), True),
        ('PatchTST, per_lead_aug 2-lead backbone -> per_lead_aug 2-lead model', Model.patch_tst, (2, True), (2, True), True),
        ('PatchTST, per_lead_aug 2-lead backbone -> 1-lead model', Model.patch_tst, (2, True), (1, False), True),
        ('PatchTST, 2-leads-in-one-sequence backbone -> same model', Model.patch_tst, (2, False), (2, False), True),
        ('PatchECG, 2-lead backbone -> 1-lead model must be refused', Model.vanilla_vit, (2, False), (1, False), False),
        ('PatchTST, 2-leads-in-one-sequence backbone -> 1-lead model must be refused', Model.patch_tst, (2, False), (1, False), False),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        for i, (label, model_enum, src_cfg, dst_cfg, must_load) in enumerate(cases):
            def run(i=i, model_enum=model_enum, src_cfg=src_cfg, dst_cfg=dst_cfg, must_load=must_load):
                torch.manual_seed(10 + i)
                src, _, _ = build(model_enum, Mode.PRETRAIN, *src_cfg)
                for module in src.modules():                                 # make BatchNorm buffers non-default
                    if isinstance(module, torch.nn.BatchNorm1d):
                        module.running_mean.fill_(0.123)
                path = os.path.join(tmp, f'backbone_{i}.pt')
                torch.save({'model': src.state_dict()}, path)
                dst, _, _ = build(model_enum, Mode.REGRESSION, *dst_cfg, n_classes=dst_cfg[0])
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        dst = transfer_weights(path, dst)
                except ValueError as exc:
                    return (not must_load), f'refused: {str(exc)[:70]}...'
                if not must_load:
                    return False, 'loaded without an error although the number of positions differs'
                s, d = src.state_dict(), dst.state_dict()
                body = [k for k in d if 'head' not in k]
                same = [k for k in body if k in s and s[k].shape == d[k].shape and torch.equal(s[k], d[k])]
                return len(same) == len(body), f'{len(same)} of {len(body)} non-head tensors identical to the backbone'
            guarded('transfer', label, run)


# ---------------------------------------------------------------------------------------------- 6. layer-wise decay
def check_layer_decay():
    print('\n== 6. Layer-wise learning-rate decay of a pretrained fine-tune (--layer_decay 0.65) ==')
    for label, model_enum in (('PatchECG', Model.vanilla_vit), ('PatchTST', Model.patch_tst)):
        def run(model_enum=model_enum):
            model, _, args = build(model_enum, Mode.REGRESSION, 1, False, n_classes=1)
            optimizer = prepare_optimizer(model=model, args=args)
            name_of = {id(p): n for n, p in model.named_parameters()}
            seen, wrong = 0, []
            for group in optimizer.param_groups:
                for p in group['params']:
                    name = name_of[id(p)]; seen += 1
                    if 'head' in name:
                        want = 1.0
                    elif 'layers.' in name:
                        want = 0.65 ** (6 - int(name.split('layers.')[1].split('.')[0]))
                    else:                                                    # patch, position, CLS and mask embeddings
                        want = 0.65 ** 7
                    if abs(group['lr_scale'] - want) > 1e-9:
                        wrong.append(f"{name}: {group['lr_scale']:.4f} instead of {want:.4f}")
            # PatchTST keeps one fixed attention-scale constant per layer (requires_grad=False): not optimised by design
            trainable = sum(p.requires_grad for p in model.parameters())
            fixed = len(name_of) - trainable
            return (not wrong and seen == trainable), (
                f'all {seen} trainable parameter tensors ({fixed} fixed constants excluded); head 1.0, encoder layers '
                f'0.65^6..0.65^1, embeddings 0.65^7' if not wrong else '; '.join(wrong[:3]))
        guarded('layer decay', f'learning-rate scale per layer, {label}', run)


# ---------------------------------------------------------------------------------------------- 7. normalisation
def check_batch_dependence():
    print('\n== 7. Does a window\'s prediction depend on the other windows in its batch? ==')
    torch.manual_seed(3)
    x = torch.randn(64, 1, SEQ) * (0.5 + 2.0 * torch.rand(64, 1, 1))        # windows of different amplitude
    batch_a, batch_b = x[:32], torch.cat([x[:1], x[32:63]])                  # window 0 with two sets of batch mates
    for label, model_enum in (('PatchECG', Model.vanilla_vit), ('PatchTST', Model.patch_tst)):
        model, aug, _ = build(model_enum, Mode.REGRESSION, 1, False, n_classes=1)
        model.eval()
        for mode in ('BN', 'instance'):
            def difference(mode=mode, model=model, aug=aug):
                out = []
                for batch in (batch_a, batch_b):
                    xn = RevInSupport(1, denorm=False, norm_stats=mode).before_forward(batch)
                    xp, _, _, _ = FinetunePatchSupport(PATCH, PATCH, False).before_forward(input=xn, target=None, device='cpu')
                    with torch.no_grad():
                        out.append(model(xp, mask=None, inverted_mask=None, indices=None, augmentation=aug)[0, 0].item())
                return abs(out[0] - out[1])
            if model_enum is Model.patch_tst and mode == 'BN':
                try:
                    info(f'PatchTST, --revin_mode BN: the same window is predicted {difference():.1e} apart in two batches. '
                         f'Expected: PatchTST has no per-patch normalisation, so batch statistics reach the model. '
                         f'Use --revin_mode instance when predictions must not depend on the batch.')
                except Exception as exc:  # noqa: BLE001
                    info(f'PatchTST, --revin_mode BN: could not be evaluated ({type(exc).__name__})')
                continue

            def run(difference=difference):
                d = difference()
                return d < 1e-4, f'same window in two different batches: predictions differ by {d:.1e}'
            guarded('batch dependence', f'{label}, --revin_mode {mode}: prediction independent of batch mates', run)


if __name__ == '__main__':
    print(f'Code under test: {ROOT}')
    check_loss()
    check_pretrain_forward()
    check_regression_forward()
    check_logging()
    check_transfer()
    check_layer_decay()
    check_batch_dependence()
    failed = [r for r in RESULTS if r[0] == 'FAIL']
    print(f'\n{len(RESULTS) - len(failed)} of {len(RESULTS)} checks passed.')
    if failed:
        print('FAILED:')
        for _, _, name, detail in failed:
            print(f'  - {name}: {detail}')
        sys.exit(1)
    print('ALL CHECKS PASSED')
