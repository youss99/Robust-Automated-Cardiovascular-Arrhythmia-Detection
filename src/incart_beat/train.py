"""
INCART lead-II central-beat classification: three matched fine-tuning runs from patch_ecg_unified_ii_fixed.

  A: RevIN 'BN' (existing, batch-dependent)  + cross-entropy
  B: RevIN 'instance' (per-window)           + cross-entropy
  C: RevIN 'instance' (per-window)           + softmax focal loss (gamma=2, no class weights)
  D: as B, but trained on the 7 original INCART symbols (N, R, j, n, B, A, V); scored after summing to 3 classes

Everything else is shared: initial weights + head init (one saved init file), seed, per-epoch example order
(ordinary shuffle, independent of model RNG), batch size, AdamW + layer decay, learning rates, epoch budget.
Hyperparameters follow the thesis PatchECG Small fine-tuning recipe (Chatterjee, Table A.2); as in
Learner.fine_tune, full fine-tuning restarts from the best head-phase model and optimizer state.
Checkpoint selection: mean of PAC and PVC validation AUPRC, validation at natural class frequencies with a fixed
batching protocol (fixed seeded order, batch 128). The test split is never loaded unless --evaluate_test is given.

Train (GPU job):  python -m src.incart_beat.train --cache_dir ... --backbone ... --runs A,B,C
Test (later):     python -m src.incart_beat.train --cache_dir ... --runs B --evaluate_test
"""
import argparse
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

from src.incart_beat import preprocessing as pp
from src.incart_beat.model import ARCH, REVIN_EPS, BeatClassifier, load_pretrained_backbone

RUNS = {'A': {'norm_stats': 'BN', 'loss': 'ce'},
        'B': {'norm_stats': 'instance', 'loss': 'ce'},
        'C': {'norm_stats': 'instance', 'loss': 'focal'},
        'D': {'norm_stats': 'instance', 'loss': 'ce', 'labels': 'fine'}}
# Run D trains on the original INCART symbols of the included beats and is scored after summing the background
# sub-classes, so its evaluation windows, metrics and checkpoint selection are identical to the 3-class runs.
FINE_CLASSES = ['N', 'R', 'j', 'n', 'B', 'A', 'V']
assert set(FINE_CLASSES) == set(pp.LABEL_MAP)
COLLAPSE = np.zeros((len(FINE_CLASSES), len(pp.CLASS_NAMES)), dtype=np.float32)  # fine prob -> 3-class prob
for _i, _s in enumerate(FINE_CLASSES):
    COLLAPSE[_i, pp.LABEL_MAP[_s]] = 1.0
PAC, PVC = pp.CLASS_NAMES.index('PAC'), pp.CLASS_NAMES.index('PVC')
DEFAULT_SEED = 200  # seed of the original runs; other seeds get a _s<seed> suffix
PAC_NOTE = ('Most training PACs come from two patients (see prep_report.txt); results for PAC reflect very limited '
            'patient diversity, which focal loss does not address.')


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--cache_dir', required=True)
    p.add_argument('--lead', default=pp.LEAD, help='must match the cache built by prepare_incart --lead')
    p.add_argument('--backbone', default=None,
                   help='default: results/pre-train/saved_models/unified/patch_ecg_unified_<lead>_fixed/..._best.pt')
    p.add_argument('--out_root', default='results/fine-tune/saved_models/incart')
    p.add_argument('--runs', default='A,B,C')
    p.add_argument('--seed', type=int, default=200)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=0.05)
    p.add_argument('--layer_decay', type=float, default=0.65)
    # Thesis Table A.2 (PatchECG Small fine-tuning): 20 head + 70 full epochs, batch 128, lr 1e-3 constant,
    # AdamW eps 1e-8 betas (0.9, 0.999), dropout 0, layer decay 0.65, weight decay 0.05
    p.add_argument('--head_epochs', type=int, default=20)
    p.add_argument('--full_epochs', type=int, default=70)
    p.add_argument('--snapshot_every', type=int, default=5, help='epochs between resumable snapshots')
    p.add_argument('--focal_gamma', type=float, default=2.0)
    p.add_argument('--eval_batch_size', type=int, default=128)
    p.add_argument('--eval_order_seed', type=int, default=0)
    p.add_argument('--bd_per_class', type=int, default=128, help='batch-dependence windows per class')
    p.add_argument('--bd_lowvar', type=int, default=32, help='lowest window-std and lowest patch-std windows each')
    p.add_argument('--bd_seeds', default='1,2,3', help='companion-set seeds')
    p.add_argument('--bd_select_seed', type=int, default=DEFAULT_SEED,
                   help='fixed selection of batch-dependence windows, independent of the training seed')
    p.add_argument('--smoke', action='store_true', help='50 steps/epoch, 1+1 epochs, separate output dir')
    p.add_argument('--evaluate_test', action='store_true')
    a = p.parse_args()
    if a.backbone is None:
        name = f'patch_ecg_unified_{a.lead.lower()}_fixed'
        a.backbone = f'results/pre-train/saved_models/unified/{name}/{name}_best.pt'
    # file-name tag for lead-specific shared files; empty for lead II so its existing outputs stay valid
    a.tag = '' if a.lead == pp.LEAD else f'_{a.lead.lower()}'
    # run-folder suffix for extra seeds; empty for the original seed so its outputs stay valid
    a.sfx = '' if a.seed == DEFAULT_SEED else f'_s{a.seed}'
    if a.smoke:
        a.head_epochs, a.full_epochs, a.out_root = 1, 1, a.out_root + '_smoke'
    return a


def log(msg):
    print(msg, flush=True)


def save_atomic(obj, path):
    torch.save(obj, path + '.tmp')
    os.replace(path + '.tmp', path)


def state_hash(sd):
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


# ----------------------------------------------------------------------------------------------- data
class WindowSource:
    """Prepared signals on the GPU; windows are cut on the fly with the shared window definition."""

    def __init__(self, cache_dir, device):
        self.signals = torch.from_numpy(np.load(os.path.join(cache_dir, 'signals_uV_100hz.npy'))).to(device)
        recs = pd.read_csv(os.path.join(cache_dir, 'records.csv'))
        beats = pd.read_csv(os.path.join(cache_dir, 'beats.csv'), keep_default_na=False)
        beats = beats[beats.status == 'included'].reset_index(drop=True)
        beats['row'] = beats.record.map(dict(zip(recs.record, recs.row)))
        beats['start'], valid = pp.window_bounds(beats.sample_100hz.values, self.signals.shape[1])
        assert valid.all() and (beats.label >= 0).all()
        beats['fine_label'] = beats.symbol.map({s: i for i, s in enumerate(FINE_CLASSES)})
        assert beats.fine_label.notna().all()
        self.beats = beats
        self.device = device
        self.offsets = torch.arange(pp.WINDOW_LEN, device=device)

    def split(self, name):
        df = self.beats[self.beats.split == name].reset_index(drop=True)
        t = lambda c: torch.as_tensor(df[c].values, dtype=torch.long, device=self.device)
        return df, t('row'), t('start'), t('label')

    def fine_labels(self, df):
        return torch.as_tensor(df.fine_label.values.astype(np.int64), device=self.device)

    def gather(self, rows, starts):
        return self.signals[rows[:, None], starts[:, None] + self.offsets]


@torch.no_grad()
def predict(model, src, rows, starts, batch_size, order=None):
    """Softmax probabilities in the original index order; `order` fixes which windows share a batch."""
    model.eval()
    n = len(rows)
    order = torch.arange(n, device=rows.device) if order is None else order.to(rows.device)
    out = None
    for i in range(0, n, batch_size):
        idx = order[i:i + batch_size]
        p = torch.softmax(model(src.gather(rows[idx], starts[idx])), dim=-1)
        if out is None:
            out = torch.empty((n, p.shape[1]), device=rows.device)
        out[idx] = p
    return out.cpu().numpy()


def label_scheme(spec):
    """Classes the head is trained on, and a function mapping its probabilities to the 3 evaluation classes."""
    if spec.get('labels', 'coarse') == 'fine':
        return FINE_CLASSES, (lambda P: P @ COLLAPSE)
    return pp.CLASS_NAMES, (lambda P: P)


def eval_order(n, seed):
    return torch.randperm(n, generator=torch.Generator().manual_seed(seed))


# ----------------------------------------------------------------------------------------------- metrics
def class_metrics(y, probs):
    pred = probs.argmax(1)
    rows = []
    for k, name in enumerate(pp.CLASS_NAMES):
        pos = y == k
        tp, fp, fn = int((pos & (pred == k)).sum()), int((~pos & (pred == k)).sum()), int((pos & (pred != k)).sum())
        prec = tp / (tp + fp) if tp + fp else np.nan
        rec = tp / (tp + fn) if tp + fn else np.nan
        f1 = 2 * prec * rec / (prec + rec) if tp else (0.0 if tp + fp + fn else np.nan)
        both = 0 < pos.sum() < len(y)
        rows.append({'class': name, 'n_true': int(pos.sum()), 'n_pred': int((pred == k).sum()), 'precision': prec,
                     'recall': rec, 'f1': f1,
                     'auprc': average_precision_score(pos, probs[:, k]) if both else np.nan,
                     'auroc': roc_auc_score(pos, probs[:, k]) if both else np.nan,
                     'prevalence': float(pos.mean())})
    return pd.DataFrame(rows)


def summarize(y, probs):
    pc = class_metrics(y, probs).set_index('class')
    return {'sel_mean_auprc_pac_pvc': float(np.mean([pc.loc['PAC', 'auprc'], pc.loc['PVC', 'auprc']])),
            **{f'{c}_{m}': float(pc.loc[c, m]) for c in ('PAC', 'PVC') for m in ('auprc', 'auroc', 'precision',
                                                                                  'recall', 'f1')},
            'macro_f1': float(pc.f1.mean())}


def full_report(df, probs, out_dir, prefix, fine_probs=None, fine_names=None):
    y = df.label.values
    pc = class_metrics(y, probs)
    pc.to_csv(os.path.join(out_dir, f'{prefix}_per_class.csv'), index=False)
    cm = pd.DataFrame(confusion_matrix(y, probs.argmax(1), labels=range(len(pp.CLASS_NAMES))),
                      index=[f'true_{c}' for c in pp.CLASS_NAMES], columns=[f'pred_{c}' for c in pp.CLASS_NAMES])
    cm.to_csv(os.path.join(out_dir, f'{prefix}_confusion.csv'))
    per_patient = []
    for pid, g in df.groupby('patient'):
        m = class_metrics(g.label.values, probs[g.index.values])
        row = {'patient': pid, 'n_windows': len(g)}
        for _, r in m.iterrows():
            row.update({f"{r['class']}_{k}": r[k] for k in ('n_true', 'n_pred', 'precision', 'recall', 'f1', 'auprc')})
        per_patient.append(row)
    pd.DataFrame(per_patient).to_csv(os.path.join(out_dir, f'{prefix}_per_patient.csv'), index=False)
    preds = df[['beat_id', 'patient', 'record', 'ann_index', 'sample_native', 'sample_100hz', 'symbol', 'label']].copy()
    for k, c in enumerate(pp.CLASS_NAMES):
        preds[f'p_{c}'] = probs[:, k]
    preds['pred'] = probs.argmax(1)
    if fine_probs is not None:
        for k, c in enumerate(fine_names):
            preds[f'pfine_{c}'] = fine_probs[:, k]
    preds.to_csv(os.path.join(out_dir, f'{prefix}_predictions.csv'), index=False)
    summ = summarize(y, probs)
    with open(os.path.join(out_dir, f'{prefix}_metrics.json'), 'w') as f:
        json.dump({**summ, 'confusion_rows_true_cols_pred': cm.values.tolist(), 'note': PAC_NOTE}, f, indent=1)
    log(f'[{prefix}] per-class:\n{pc.to_string(index=False)}\nconfusion (rows true, cols pred):\n{cm.to_string()}')
    return summ


# ----------------------------------------------------------------------------------------------- batch dependence
def select_bd_windows(src, va, va_rows, va_starts, args):
    rng = np.random.default_rng(args.bd_select_seed)
    y = va.label.values
    sel = []
    for k in range(len(pp.CLASS_NAMES)):
        idx = np.flatnonzero(y == k)
        sel += list(rng.choice(idx, min(args.bd_per_class, len(idx)), replace=False))
    win_std, patch_std = [], []
    with torch.no_grad():
        for i in range(0, len(va_rows), 4096):
            w = src.gather(va_rows[i:i + 4096], va_starts[i:i + 4096])
            win_std.append(w.std(1).cpu())
            patch_std.append(w.reshape(len(w), -1, ARCH['patch_len']).std(2).min(1).values.cpu())
    win_std, patch_std = torch.cat(win_std).numpy(), torch.cat(patch_std).numpy()
    low = list(np.argsort(win_std)[:args.bd_lowvar]) + list(np.argsort(patch_std)[:args.bd_lowvar])
    sel = list(dict.fromkeys(sel + low))  # keep order, drop duplicates
    is_low = np.isin(sel, low)
    return np.asarray(sel), is_low, win_std, patch_std


@torch.no_grad()
def batch_dependence(model, src, va_rows, va_starts, sel, is_low, args, post=lambda P: P):
    model.eval()
    n_cls = len(pp.CLASS_NAMES)
    sel_t = torch.as_tensor(sel, device=va_rows.device)
    rows, starts = va_rows[sel_t], va_starts[sel_t]
    ref = post(predict(model, src, rows, starts, batch_size=1))
    conds = {'batch32_fixed_order': post(predict(model, src, rows, starts, 32)),
             'batch256_fixed_order': post(predict(model, src, rows, starts, 256))}
    pool = np.setdiff1d(np.arange(len(va_rows)), sel)
    for s in [int(v) for v in args.bd_seeds.split(',')]:
        rng = np.random.default_rng(1000 + s)
        out = np.empty((len(sel), n_cls), dtype=np.float32)
        for i in range(0, len(sel), 32):
            comp = torch.as_tensor(rng.choice(pool, 256 - len(sel[i:i + 32]), replace=False), device=va_rows.device)
            x = torch.cat([src.gather(rows[i:i + 32], starts[i:i + 32]), src.gather(va_rows[comp], va_starts[comp])])
            out[i:i + 32] = post(torch.softmax(model(x), dim=-1)[:len(sel[i:i + 32])].cpu().numpy())
        conds[f'batch256_32targets_224random_companions_seed{s}'] = out
    summary, per_window = {}, np.zeros(len(sel))
    for name, p in conds.items():
        d = np.abs(p - ref).max(1)
        changed = p.argmax(1) != ref.argmax(1)
        per_window = np.maximum(per_window, d)
        summary[name] = {'max_abs_prob_diff': float(d.max()), 'mean_abs_prob_diff': float(d.mean()),
                         'max_abs_prob_diff_lowvar': float(d[is_low].max()) if is_low.any() else None,
                         'n_pred_class_changes': int(changed.sum()),
                         'n_pred_class_changes_lowvar': int(changed[is_low].sum())}
    return {'reference': 'each window alone (batch size 1), eval mode', 'n_windows': int(len(sel)),
            'n_lowvar_windows': int(is_low.sum()), 'conditions': summary}, per_window


# ----------------------------------------------------------------------------------------------- training
def layer_id(name, n_ids):
    """Mirror of optim_factory.get_num_layer_for_vit (that module imports timm, slow to load from /project)."""
    if any(k in name for k in ('to_patch_embedding', 'pos_embedding', 'cls_token', 'mask_token')):
        return 0
    if 'layers' in name:
        return int(name.split('layers.')[1].split('.')[0]) + 1
    return n_ids - 1


def build_optimizer(model, args):
    """Same groups as optim_factory.prepare_optimizer + learner.update_lr_schedule: AdamW, biases without
    weight decay, lr x layer_decay**depth (head x1, top block x0.65, ..., patch embedding x0.65^7)."""
    scales = [args.layer_decay ** (ARCH['n_layers'] + 1 - i) for i in range(ARCH['n_layers'] + 2)]
    groups = {}
    for name, param in model.named_parameters():
        lid, no_decay = layer_id(name, len(scales)), name.endswith('.bias')
        g = groups.setdefault((lid, no_decay), {'params': [], 'weight_decay': 0. if no_decay else args.weight_decay,
                                                'lr_scale': scales[lid], 'lr': args.lr * scales[lid]})
        g['params'].append(param)
    return torch.optim.AdamW(list(groups.values()), lr=args.lr, eps=1e-8, betas=(0.9, 0.999))


def loss_fn(logits, y, kind, gamma):
    if kind == 'ce':
        return F.cross_entropy(logits, y)
    logpt = F.log_softmax(logits, dim=-1).gather(1, y[:, None]).squeeze(1)
    return (-(1 - logpt.exp()).pow(gamma) * logpt).mean()


def make_init_state(args, n_classes=len(pp.CLASS_NAMES)):
    cls = '' if n_classes == len(pp.CLASS_NAMES) else f'_{n_classes}cls'  # 3-class name unchanged
    path = os.path.join(args.out_root, '_shared', f'init{args.tag}_seed{args.seed}{cls}.pt')
    if not os.path.exists(path):
        torch.manual_seed(args.seed)
        # norm mode has no parameters; init is shared by all runs with the same head size
        model = BeatClassifier(norm_stats='BN', n_classes=n_classes)
        loaded = load_pretrained_backbone(model, args.backbone)
        log(f'Backbone: {len(loaded)} tensors loaded from {args.backbone}; fresh head '
            f'{[k for k in model.state_dict() if "head" in k]}')
        save_atomic({'model': model.state_dict(), 'backbone': args.backbone, 'seed': args.seed}, path)
    sd = torch.load(path, map_location='cpu', weights_only=False)['model']
    return sd, state_hash(sd)


def train_run(name, args, src, data, device, bd):
    spec = RUNS[name]
    train_classes, post = label_scheme(spec)
    fine = spec.get('labels', 'coarse') == 'fine'
    init_sd, init_hash = make_init_state(args, len(train_classes))
    run_dir = os.path.join(args.out_root, f'incart_{args.lead.lower()}_beat_{name}{args.sfx}')
    os.makedirs(run_dir, exist_ok=True)
    if os.path.exists(os.path.join(run_dir, 'DONE')):
        log(f'== Run {name}: DONE marker present, skipping')
        return run_dir
    tr, tr_rows, tr_starts, tr_y = data['train']
    va, va_rows, va_starts, va_y = data['val']
    tr_target = src.fine_labels(tr) if fine else tr_y
    total_epochs = args.head_epochs + args.full_epochs
    steps_per_epoch = len(tr) // args.batch_size  # drop_last
    if args.smoke:
        steps_per_epoch = min(steps_per_epoch, 50)
    va_order = eval_order(len(va), args.eval_order_seed)

    meta = {'run': name, **spec, 'focal_gamma': args.focal_gamma if spec['loss'] == 'focal' else None,
            'focal_class_weights': None, 'revin_eps': REVIN_EPS, 'revin_affine': False,
            'labels': spec.get('labels', 'coarse'), 'train_classes': list(train_classes),
            'preprocessing': pp.preprocessing_config(args.lead), 'class_names': pp.CLASS_NAMES, 'arch': ARCH,
            'backbone': args.backbone, 'init_hash': init_hash, 'seed': args.seed,
            'eval_protocol': f'validation in fixed order randperm(seed={args.eval_order_seed}), '
                             f'batch {args.eval_batch_size}, eval mode, natural class frequencies',
            'selection': 'max mean(PAC AUPRC, PVC AUPRC) on validation over all epochs',
            'train_config': {k: v for k, v in vars(args).items() if k not in ('runs', 'evaluate_test')},
            'steps_per_epoch': steps_per_epoch, 'note': PAC_NOTE}
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(meta, f, indent=1)

    model = BeatClassifier(norm_stats=spec['norm_stats'], n_classes=len(train_classes))
    model.load_state_dict(init_sd)
    model.to(device)
    opt = build_optimizer(model, args)
    log(f'== Run {name}: {spec}, init hash {init_hash}, {steps_per_epoch} steps/epoch x {total_epochs} epochs')

    bd_before = os.path.join(run_dir, 'batch_dependence_before.json')
    if not os.path.exists(bd_before):
        res, _ = batch_dependence(model, src, va_rows, va_starts, bd['sel'], bd['is_low'], args, post)
        with open(bd_before, 'w') as f:
            json.dump(res, f, indent=1)
        log(f'[{name}] batch dependence BEFORE training: {json.dumps(res["conditions"], indent=1)}')

    snap_path, best_path = os.path.join(run_dir, 'snapshot.pt'), os.path.join(run_dir, 'best.pt')
    start_epoch, best, history = 0, -np.inf, []
    if os.path.exists(snap_path):
        snap = torch.load(snap_path, map_location='cpu', weights_only=False)
        model.load_state_dict(snap['model'])
        opt.load_state_dict(snap['opt'])
        start_epoch, best, history = snap['epoch'] + 1, snap['best'], snap['history']
        log(f'[{name}] resumed from snapshot after epoch {snap["epoch"]} (best {best:.4f})')

    for epoch in range(start_epoch, total_epochs):
        phase = 'head' if epoch < args.head_epochs else 'full'
        if epoch == args.head_epochs > 0:
            # As Learner.fine_tune: reload the best head-phase model AND optimizer before full fine-tuning
            ck = torch.load(best_path, map_location='cpu', weights_only=False)
            model.load_state_dict(ck['model'])
            opt.load_state_dict(ck['opt'])
            log(f"[{name}] full fine-tuning starts from best head epoch {ck['meta']['epoch']} "
                f"(val sel {ck['meta']['val_selection_metric']:.4f})")
        for n_, p_ in model.named_parameters():
            p_.requires_grad = phase == 'full' or 'head' in n_
        model.train()
        perm = torch.randperm(len(tr), generator=torch.Generator().manual_seed(args.seed * 1000 + epoch)).to(device)
        t0, losses = time.time(), torch.zeros((), device=device)
        for s in range(steps_per_epoch):
            idx = perm[s * args.batch_size:(s + 1) * args.batch_size]
            loss = loss_fn(model(src.gather(tr_rows[idx], tr_starts[idx])), tr_target[idx], spec['loss'],
                           args.focal_gamma)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses += loss.detach()
        torch.cuda.synchronize()
        t_train = time.time() - t0
        probs = post(predict(model, src, va_rows, va_starts, args.eval_batch_size, va_order))
        summ = summarize(va.label.values, probs)
        history.append({'epoch': epoch, 'phase': phase, 'train_loss': float(losses) / steps_per_epoch,
                        'train_s': round(t_train, 1), 'val_s': round(time.time() - t0 - t_train, 1), **summ})
        h = history[-1]
        log(f"[{name}] ep {epoch:02d} {phase} loss {h['train_loss']:.4f} | val sel {h['sel_mean_auprc_pac_pvc']:.4f} "
            f"PAC AUPRC {h['PAC_auprc']:.4f} PVC AUPRC {h['PVC_auprc']:.4f} | {h['train_s']}s + {h['val_s']}s")
        if summ['sel_mean_auprc_pac_pvc'] > best:
            best = summ['sel_mean_auprc_pac_pvc']
            save_atomic({'model': model.state_dict(), 'opt': opt.state_dict(),
                         'meta': {**meta, 'epoch': epoch, 'phase': phase, 'val_selection_metric': best}}, best_path)
        if (epoch + 1) % args.snapshot_every == 0 or epoch + 1 in (args.head_epochs, total_epochs):
            save_atomic({'model': model.state_dict(), 'opt': opt.state_dict(), 'epoch': epoch, 'best': best,
                         'history': history}, snap_path)
        pd.DataFrame(history).to_csv(os.path.join(run_dir, 'history.csv'), index=False)

    # Final validation report + batch dependence with the selected checkpoint
    ck = torch.load(best_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ck['model'])
    log(f"[{name}] selected epoch {ck['meta']['epoch']} ({ck['meta']['phase']}), "
        f"val mean PAC/PVC AUPRC {ck['meta']['val_selection_metric']:.4f}")
    fine_probs = predict(model, src, va_rows, va_starts, args.eval_batch_size, va_order)
    full_report(va, post(fine_probs), run_dir, 'val', fine_probs if fine else None, train_classes)
    res, per_window = batch_dependence(model, src, va_rows, va_starts, bd['sel'], bd['is_low'], args, post)
    with open(os.path.join(run_dir, 'batch_dependence_after.json'), 'w') as f:
        json.dump(res, f, indent=1)
    bd['table'].assign(max_abs_prob_diff_after=per_window).to_csv(
        os.path.join(run_dir, 'batch_dependence_windows_after.csv'), index=False)
    log(f'[{name}] batch dependence AFTER training: {json.dumps(res["conditions"], indent=1)}')
    open(os.path.join(run_dir, 'DONE'), 'w').write(time.strftime('%Y-%m-%d %H:%M:%S\n'))
    return run_dir


def compare(args):
    """Table of every finished run for this lead and seed (not only the runs of this job, so a job training only
    some runs never drops the others from the table)."""
    rows = []
    for name in sorted(RUNS):
        d = os.path.join(args.out_root, f'incart_{args.lead.lower()}_beat_{name}{args.sfx}')
        if not os.path.exists(os.path.join(d, 'DONE')):
            continue
        cfg = json.load(open(os.path.join(d, 'config.json')))
        m = json.load(open(os.path.join(d, 'val_metrics.json')))
        ck_meta = torch.load(os.path.join(d, 'best.pt'), map_location='cpu', weights_only=False)['meta']
        row = {'run': name, 'norm': cfg['norm_stats'], 'loss': cfg['loss'], 'labels': cfg.get('labels', 'coarse'),
               'best_epoch': ck_meta['epoch'],
               **{k: v for k, v in m.items() if not isinstance(v, list) and k != 'note'}}
        for when in ('before', 'after'):
            bdj = json.load(open(os.path.join(d, f'batch_dependence_{when}.json')))['conditions']
            row[f'bd_{when}_max_abs_diff'] = max(v['max_abs_prob_diff'] for v in bdj.values())
            row[f'bd_{when}_class_changes'] = max(v['n_pred_class_changes'] for v in bdj.values())
        rows.append(row)
    if rows:
        os.makedirs(os.path.join(args.out_root, '_comparison'), exist_ok=True)
        df = pd.DataFrame(rows)
        path = os.path.join(args.out_root, '_comparison', f'val_comparison{args.tag}{args.sfx}.csv')
        df.to_csv(f'{path}.{os.getpid()}.tmp', index=False)
        os.replace(f'{path}.{os.getpid()}.tmp', path)
        log('\n== Validation comparison (test untouched)\n' + df.T.to_string() + f'\n\n{PAC_NOTE}')


def evaluate_test(args, src, device):
    for name in args.runs.split(','):
        run_dir = os.path.join(args.out_root, f'incart_{args.lead.lower()}_beat_{name}{args.sfx}')
        assert os.path.exists(os.path.join(run_dir, 'DONE')), f'run {name} is not finished'
        ck = torch.load(os.path.join(run_dir, 'best.pt'), map_location='cpu', weights_only=False)
        train_classes, post = label_scheme(ck['meta'])
        model = BeatClassifier(norm_stats=ck['meta']['norm_stats'], n_classes=len(train_classes))
        model.load_state_dict(ck['model'])
        model.to(device)
        te, te_rows, te_starts, _ = src.split('test')
        fine_probs = predict(model, src, te_rows, te_starts, args.eval_batch_size,
                             eval_order(len(te), args.eval_order_seed))
        log(f'== TEST run {name} (selected epoch {ck["meta"]["epoch"]})')
        fine = ck['meta'].get('labels', 'coarse') == 'fine'
        full_report(te, post(fine_probs), run_dir, 'test', fine_probs if fine else None, train_classes)


def main():
    args = get_args()
    torch.backends.cuda.matmul.allow_tf32 = False  # full fp32 so batch-dependence differences are interpretable
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert device.type == 'cuda', 'run this inside a GPU Slurm job'
    os.makedirs(os.path.join(args.out_root, '_shared'), exist_ok=True)
    src = WindowSource(args.cache_dir, device)
    if args.evaluate_test:
        return evaluate_test(args, src, device)

    data = {s: src.split(s) for s in ('train', 'val')}  # test is never loaded here
    for s, (df, *_r) in data.items():
        log(f'{s}: {len(df)} windows, {df.patient.nunique()} patients, class counts '
            f'{df.label.value_counts().sort_index().to_dict()}')
    va, va_rows, va_starts, _ = data['val']
    sel, is_low, win_std, patch_std = select_bd_windows(src, va, va_rows, va_starts, args)
    table = va.iloc[sel][['beat_id', 'patient', 'symbol', 'label']].assign(
        lowvar=is_low, window_std_uV=win_std[sel], min_patch_std_uV=patch_std[sel])
    table_path = os.path.join(args.out_root, '_shared', f'batch_dependence_windows{args.tag}.csv')
    table.to_csv(f'{table_path}.{os.getpid()}.tmp', index=False)  # concurrent seed jobs write identical content
    os.replace(f'{table_path}.{os.getpid()}.tmp', table_path)
    log(f'Batch-dependence set: {len(sel)} val windows ({is_low.sum()} low-variance; window std '
        f'{win_std[sel][is_low].min():.1f}-{win_std[sel][is_low].max():.1f} uV, min patch std from '
        f'{patch_std[sel][is_low].min():.2f} uV)')
    bd = {'sel': sel, 'is_low': is_low, 'table': table}

    names = args.runs.split(',')
    t0 = time.time()
    for name in names:
        train_run(name, args, src, data, device, bd)
        log(f'elapsed {time.time() - t0:.0f}s')
    compare(args)


if __name__ == '__main__':
    main()
