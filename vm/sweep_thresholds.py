#!/usr/bin/env python3
"""
Offline analysis of dumped logits -- CPU only, no GPU, no cluster.

Consumes the _logits_<model>_<split>.npz files written by
    python -m src.patchECG_finetune ... --no-finetuning --testing --dump_logits

and answers, without any retraining:

  1. THRESHOLD   How much of each model's F1 is lost to the hardcoded 0.5 cut in
                 src/core/support/tracking.py? Thresholds are tuned on VALID and
                 reported on TEST -- never tuned on test.
  2. MERGE       Does collapsing SNOMED synonyms (PVC/VPB/VEB, PAC/SVPB/BPAC) recover
                 performance on the Unified label space? Combines the existing output
                 units with noisy-OR; if this does not help, retraining on merged
                 labels will not help either.
  3. ENSEMBLE    Does averaging two models' probabilities beat either alone?

Usage
  python sweep_thresholds.py MODEL.npz [MODEL2.npz ...]        # auto-pairs valid/test
  python sweep_thresholds.py --classes PVC,PAC MODEL.npz
  python sweep_thresholds.py --merge MODEL.npz
  python sweep_thresholds.py --ensemble A_test.npz B_test.npz
"""
from __future__ import annotations
import argparse, os, sys, glob
import numpy as np

try:
    from sklearn.metrics import roc_auc_score, average_precision_score
except ImportError:
    sys.exit("needs scikit-learn:  pip install scikit-learn")

# SNOMED synonym groups -- see Datasets/README_PVC_PAC_inventory.md
MERGE_GROUPS = {
    "VENT_ECTOPY":  ["PVC", "VPB", "VEB", "VPVC", "VBig"],
    "ATRIAL_ECTOPY": ["PAC", "SVPB", "BPAC", "AB", "SVB"],
}
DEFAULT_CLASSES = ["PVC", "PAC", "VPB", "VEB", "SVPB", "BPAC"]


def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))


def load(path):
    d = np.load(path, allow_pickle=True)
    p = np.asarray(d["logits"], dtype=float)
    if p.ndim == 1:
        p = p[:, None]
    # predict() returns raw model output; convert only if it is not already in [0,1]
    if p.min() < 0.0 or p.max() > 1.0:
        p = sigmoid(p)
    return p, np.asarray(d["targets"], dtype=float), [str(c) for c in d["classes"]]


def prf(y, s, t):
    pred = (s >= t).astype(int)
    tp = float(((pred == 1) & (y == 1)).sum())
    fp = float(((pred == 1) & (y == 0)).sum())
    fn = float(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def best_threshold(y, s, grid=None):
    """Threshold maximising F1. Grid defaults to the unique score values (exact optimum)."""
    if y.sum() == 0:
        return 0.5, 0.0
    if grid is None:
        grid = np.unique(np.round(s, 4))
        if len(grid) > 2000:
            grid = np.quantile(s, np.linspace(0, 1, 2000))
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        _, _, f1 = prf(y, s, t)
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t, best_f1


def safe_auc(fn, y, s):
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return fn(y, s)


def report(name, classes_wanted, vp, vt, tp_, tt, classes):
    idx = {c: i for i, c in enumerate(classes)}
    want = [c for c in classes_wanted if c in idx]
    if not want:
        print(f"  (none of {classes_wanted} present; available: {classes[:12]}...)")
        return
    print(f"\n### {name}")
    print(f"{'class':<8}{'n_test':>7}{'F1@0.5':>9}{'F1@t*':>9}{'delta':>9}"
          f"{'t*':>8}{'prec':>8}{'rec':>8}{'AUROC':>8}{'AUPRC':>8}")
    for c in want:
        i = idx[c]
        yv, sv = vt[:, i], vp[:, i]
        yt, st = tt[:, i], tp_[:, i]
        t_star, _ = best_threshold(yv, sv)          # tuned on VALID only
        _, _, f1_half = prf(yt, st, 0.5)
        pr, rc, f1_star = prf(yt, st, t_star)
        print(f"{c:<8}{int(yt.sum()):>7}{f1_half:>9.3f}{f1_star:>9.3f}"
              f"{f1_star - f1_half:>+9.3f}{t_star:>8.3f}{pr:>8.3f}{rc:>8.3f}"
              f"{safe_auc(roc_auc_score, yt, st):>8.3f}"
              f"{safe_auc(average_precision_score, yt, st):>8.3f}")


def merged(p, y, classes, groups):
    """Noisy-OR combine synonym output units into one score + one label."""
    out_p, out_y, names = [], [], []
    idx = {c: i for i, c in enumerate(classes)}
    for gname, members in groups.items():
        cols = [idx[m] for m in members if m in idx]
        if len(cols) < 2:
            continue
        out_p.append(1.0 - np.prod(1.0 - p[:, cols], axis=1))   # noisy-OR
        out_y.append((y[:, cols].sum(axis=1) > 0).astype(float))
        names.append(f"{gname}({'+'.join(m for m in members if m in idx)})")
    if not names:
        return None
    return np.stack(out_p, 1), np.stack(out_y, 1), names


def pair_for(path):
    """Given either the valid or test npz, find its sibling."""
    for a, b in (("_test.npz", "_valid.npz"), ("_valid.npz", "_test.npz")):
        if path.endswith(a):
            sib = path[: -len(a)] + b
            return (path, sib) if a == "_valid.npz" else (sib, path)
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="+", help="_logits_*_test.npz (valid sibling auto-found)")
    ap.add_argument("--classes", default=",".join(DEFAULT_CLASSES))
    ap.add_argument("--merge", action="store_true", help="also run the synonym-merge analysis")
    ap.add_argument("--ensemble", action="store_true", help="average all given models (test only)")
    a = ap.parse_args()
    wanted = [c.strip() for c in a.classes.split(",") if c.strip()]

    loaded = []
    for path in a.npz:
        vpath, tpath = pair_for(path)
        if not vpath or not os.path.exists(vpath) or not os.path.exists(tpath):
            print(f"!! need both valid+test siblings for {path}; skipping")
            continue
        vp, vt, cls = load(vpath)
        tp_, tt, _ = load(tpath)
        name = os.path.basename(tpath).replace("_logits_", "").replace("_test.npz", "")
        loaded.append((name, vp, vt, tp_, tt, cls))
        report(name, wanted, vp, vt, tp_, tt, cls)

        if a.merge:
            m1 = merged(vp, vt, cls, MERGE_GROUPS)
            m2 = merged(tp_, tt, cls, MERGE_GROUPS)
            if m1 and m2:
                report(f"{name}  [SYNONYMS MERGED]", m1[2], m1[0], m1[1], m2[0], m2[1], m1[2])

    if a.ensemble and len(loaded) >= 2:
        cls = loaded[0][5]
        if all(l[5] == cls for l in loaded):
            vp = np.mean([l[1] for l in loaded], axis=0)
            tp_ = np.mean([l[3] for l in loaded], axis=0)
            report("ENSEMBLE(" + " + ".join(l[0] for l in loaded) + ")",
                   wanted, vp, loaded[0][2], tp_, loaded[0][4], cls)
        else:
            print("!! ensemble skipped: models have different class vocabularies")


if __name__ == "__main__":
    main()
