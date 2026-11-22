#!/usr/bin/env python3
"""Dry run for the whole final batch (docs/instructions_final_results.md: "One dry run for the whole batch"). No training, nothing submitted.

Splits (docs/DECISIONS.md, 2026-11-20): one identical split for every learned model: cv5_optB_leadV2_opt2p in folds 1, 4, 5, published cv5_optB
in folds 2, 3.
1. Loader: the real get_dls on every fold for lead_II, lead_V2 and two_lead. Every fold: the loader's records equal the manifest's, and the
   counts are the same for the three models. Fold 1 in full: windows and label mean and SD per lead per split; 200 random items per split
   equal the H5 channel(s) and label(s).
2. Jobs: every generated fine-tune job's python command through the real parser of src/patchECG_finetune.py, compared value by value with the
   published comparison run's params.txt (pretrained: R2 same mode and fold; supervised: R1b supervised same mode and fold). Allowed
   differences: model_name, save_path, root_path, seed (S2, S3 only), pretrained_model_path (the backbone named in jobs.csv). Also each
   job's data folder: one folder per fold for every model.
3. Step 1 recipe: not checkable on the VM (note only).
4. Job list, approvals and time estimate from R1b/R2 job_log.txt durations.
5. Gold-standard ST-deviation episodes per lead in each fold's validation set (docs/DECISIONS.md: before training).
Usage: <repo>/.venv/bin/python dry_run_final.py <generated sbatch folder> <output .md>. Also writes <output stem>_jobs.csv, a byte copy of jobs.csv."""
import hashlib
import json
import re
import shlex
import shutil
import subprocess
import sys
from argparse import Namespace
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

REPO = Path("/home/student/GIT/Robust-Automated-Cardiovascular-Arrhythmia-Detection")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent / "phase1_lib"))  # vendored phase-1 code (phase1_lib/MANIFEST.tsv)
from src.core.datasets.ecg_interface import ClassificationType  # noqa: E402
from src.core.utils.data_utils import get_dls  # noqa: E402
from ltst_pipeline.old_fsm_gs import MANUAL_OMIT  # noqa: E402  (the published scorer's mask list)

GEN, OUT_MD = Path(sys.argv[1]), Path(sys.argv[2])
FOLDS = REPO / "phase1_folds"
SAVE = REPO / "results/fine-tune/saved_models/patient-segment-regression"
OPT2P_MAN = REPO / "ptbxl_models_v2/cv5_optB_v2models_leadV2_fold_manifests"
BASE_MAN = REPO / "ptbxl_models_v2/cv5_optB_fold_manifests"
GS = Path("/home/student/GIT/transfer_bundle/gs_per_beat.csv")       # evaluate_test1.GS_PATH, the published scorer's gold standard
LEADS = {"lead_II": "II", "lead_V2": "V2", "two_lead": "II,V2"}
CH = {"lead_II": [0], "lead_V2": [1], "two_lead": [0, 1]}
CSV, H5 = "modified_dataset/modified_dataset.csv", "PatientSegmentRegressionDataset.h5"
out = []


def say(s=""):
    print(s, flush=True)
    out.append(s)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def folder(k):
    return FOLDS / (f"cv5_optB_leadV2_opt2p_fold{k:02d}" if k in (1, 4, 5) else f"cv5_optB_fold{k:02d}")


def manifest(k):
    return pd.read_csv((OPT2P_MAN if k in (1, 4, 5) else BASE_MAN) / f"fold{k:02d}.csv").set_index("patient_id").split


def loader(root, mode):
    p = Namespace(dset="patient-segment-regression", root_path=str(root), min_class_size=1, top_n_classes=None,
                  classification_type=ClassificationType.REGRESSION, custom_lead_selection=LEADS[mode], focal_loss=False, focal_alpha=None,
                  regression_loss="mse", alt_lead_ordering=False, trafos=["None"], batch_size=128, num_workers=0,
                  reset_strat_folds=False, distributed=False)
    return get_dls(p)


# ---------------------------------------------------------------- 1. loader
say(f"# Dry run, final thesis batch ({datetime.now():%Y-%m-%d %H:%M}, VM)")
say()
say("Made by `src/scripts/final/dry_run_final.py` on the VM. Splits as in `docs/DECISIONS.md` (2026-11-20): one identical split for every learned model, "
    "`cv5_optB_leadV2_opt2p_fold*` in folds 1, 4, 5 and the published `cv5_optB_fold02`/`03` in folds 2, 3. Nothing was trained or submitted.")
say()
say("## 1. Loader (real `get_dls`, `reset_strat_folds` off)")
say()
say("Labels: lead II = `target_lead0_mV`, V2 = `target_lead1_mV` (mV). Masked pairs in training keep their real labels, as published. "
    "In folds 1, 4, 5 the masked record (s20081, s20081, s20241) is unused for every model.")
say()
say("Fold 1 in full:")
say()
say("| model | split | records | windows | II label mean ± SD | V2 label mean ± SD | records |")
say("|---|---|--:|--:|---|---|---|")
rng = np.random.default_rng(0)
counts = {}
for k in range(1, 6):
    root = folder(k)
    full = pd.read_csv(root / CSV, usecols=["patient_id", "target_lead0_mV", "target_lead1_mV", "strat_fold_annotated"])
    m = manifest(k)
    h5 = h5py.File(root / H5, "r")["tracings"] if k == 1 else None
    for mode in LEADS:
        dl = loader(root, mode)
        for sp, d in (("train", dl.train), ("val", dl.valid), ("test", dl.test)):
            ds = d.dataset
            y = ds.data_y
            want = sorted(m[m == sp].index)
            got = sorted(y.patient_id.unique())
            assert got == want, f"fold {k} {mode} {sp}: loader {got} != manifest {want}"
            idx = y.original_index.to_numpy()
            assert len(ds) == int(full.patient_id.isin(want).sum())
            counts[(k, mode, sp)] = (len(ds), want)
            if k != 1:
                continue
            for i in rng.choice(len(ds), 200, replace=False):
                x, t = ds[int(i)]
                r = int(idx[i])
                assert np.array_equal(x.numpy().reshape(len(CH[mode]), -1), h5[r, CH[mode], :]), f"{mode} {sp}: item {i} != H5 row {r}"
                lab = full[["target_lead0_mV", "target_lead1_mV"]].to_numpy()[r, CH[mode]]
                assert np.allclose(t.numpy().reshape(-1), lab, atol=1e-6), f"{mode} {sp}: label mismatch at row {r}"
            xb, tb = next(iter(d))
            assert tuple(xb.shape[1:]) == (len(CH[mode]), 1000) and tuple(tb.shape[1:]) == (len(CH[mode]),), (mode, sp, xb.shape, tb.shape)
            lab = full.iloc[idx]
            say(f"| {mode} | {sp} | {len(want)} | {len(ds):,} | {lab.target_lead0_mV.mean():+.4f} ± {lab.target_lead0_mV.std():.4f} | "
                f"{lab.target_lead1_mV.mean():+.4f} ± {lab.target_lead1_mV.std():.4f} | {' '.join(want)} |")
        unused = sorted(m[m == "unused"].index)
        n_unused = int((full.strat_fold_annotated == -1).sum())
        assert n_unused == int(full.patient_id.isin(unused).sum())
        assert sum(counts[(k, mode, s)][0] for s in ("train", "val", "test")) + n_unused == len(full)
        counts[(k, mode, "unused")] = (n_unused, unused)
    for sp in ("train", "val", "test", "unused"):     # one identical split: the three models see the same records and windows
        assert counts[(k, "lead_II", sp)] == counts[(k, "lead_V2", sp)] == counts[(k, "two_lead", sp)], (k, sp)
say()
say("Checked for fold 1: 200 random items per split equal the H5 channel(s) and label(s) of the row the CSV names; one batch has shape "
    "(n, channels, 1000) / (n, channels). Every fold: loader records = manifest records in every split, for every model; split counts + unused rows = CSV rows.")
say()
say("### Every fold (loader counts, identical for lead_II, lead_V2 and two_lead: checked)")
say()
say("| fold | train / val / test windows | unused | validation records | change vs R2 |")
say("|--:|---|---|---|---|")
for k in range(1, 6):
    c = {sp: counts[(k, "lead_II", sp)] for sp in ("train", "val", "test", "unused")}
    u = f"{c['unused'][0]:,} ({' '.join(c['unused'][1])})" if c["unused"][1] else "–"
    note = ("none (published folder)" if k in (2, 3) else
            "R2b's split. New for lead II and two-lead: masked record out of validation (unused), moved record from training to validation")
    say(f"| {k} | {c['train'][0]:,} / {c['val'][0]:,} / {c['test'][0]:,} | {u} | {' '.join(c['val'][1])} | {note} |")

# ---------------------------------------------------------------- 2. flags through the real parser
say()
say("## 2. Every fine-tune job vs the published comparison run")
say()


def parse_text(t):
    return dict(re.findall(r"(\w+)=('[^']*'|\[[^\]]*\]|<[^>]*>|[^,)]+)", t))


def norm(v):
    """params.txt is written after the trainer turns some strings into enums (e.g. 'cls_token' -> <ClassToken.CLS_TOKEN: 1>):
    compare an enum by its member name, case-insensitively, and a string without quotes. Anything else is compared as written."""
    if v is None:
        return None
    m = re.fullmatch(r"<\w+\.(\w+): [^>]*>", v)
    return (m.group(1) if m else v.strip("'")).lower()


def job_words(p):
    body = p.read_text().split("-m src.patchECG_finetune")[1].split("\n\n")[0].replace("\\\n", " ")
    body = body.replace('"${STAGED_ROOT}"', "/tmp/none")
    return shlex.split(body)


def job_data(p):
    return re.search(r'\[\[ -f "([^"]+)/PatientSegmentRegressionDataset\.h5" \]\]', p.read_text()).group(1)


jobs = pd.read_csv(GEN / "jobs.csv")
ft = jobs[jobs.pipeline != "pretrain"]
code = ("import os, sys, json; batch = json.loads(sys.stdin.read()); out = []\n"
        "for words in batch:\n"
        "    sys.argv = ['x'] + words\n"
        "    if 'src.patchECG_finetune' in sys.modules: del sys.modules['src.patchECG_finetune']\n"
        "    import src.patchECG_finetune as F\n"
        "    out.append(repr(F.args))\n"
        "print('ARGS ' + json.dumps(out))\n")
batch = [job_words(GEN / r.group / f"{r.run_id}.sbatch") for r in ft.itertuples()]
res = subprocess.run([str(REPO / ".venv/bin/python"), "-c", code], input=json.dumps(batch), capture_output=True, text=True, cwd=REPO, timeout=1800)
line = next((x for x in res.stdout.splitlines() if x.startswith("ARGS ")), None)
assert line, res.stderr[-2000:]
parsed = [parse_text(a) for a in json.loads(line[5:])]
ALLOWED = {"model_name", "save_path", "root_path", "seed", "pretrained_model_path"}
rows, bad, data_by_fold = [], [], {}
for r, got in zip(ft.itertuples(), parsed):
    k = int(r.fold)
    arm = "supervised" if r.backbone == "none" else "pretrained"
    ref_run = f"R2_{r.mode}_pretrained_fold{k:02d}" if arm == "pretrained" else f"R1b_{r.mode}_supervised_fold{k:02d}"
    ref = parse_text((SAVE / ref_run / "params.txt").read_text())
    keys = sorted(set(ref) | set(got))
    got = dict(got, dset=got.get("dset", got["dset_finetune"]))   # the trainer sets args.dset = dset_finetune after parsing
    diff = {key: (ref.get(key), got.get(key)) for key in keys if norm(ref.get(key)) != norm(got.get(key))}
    unexpected = {key: v for key, v in diff.items() if key not in ALLOWED}
    seed_ok = int(got["seed"]) == int(r.seed) and (r.training_seed != "S1" or got["seed"] == ref["seed"])
    bb = got["pretrained_model_path"].strip("'")
    bb_ok = (bb == "None") if arm == "supervised" else bb == r.backbone
    if r.pipeline == "L2":
        bb_ok = bb_ok and Path(bb).name == Path(ref["pretrained_model_path"].strip("'")).name   # same backbone file as R2
    data = job_data(GEN / r.group / f"{r.run_id}.sbatch")
    data_ok = data == r.dataset and Path(data).name == folder(k).name
    data_by_fold.setdefault(k, set()).add(data)
    ok = not unexpected and seed_ok and bb_ok and data_ok
    rows.append((r.run_id, ref_run, ok))
    if not ok:
        bad.append((r.run_id, ref_run, unexpected, seed_ok, bb_ok, data_ok))
one_split = all(len(v) == 1 for v in data_by_fold.values())
say(f"{len(rows)} fine-tune jobs parsed by the real parser and compared with every value of the comparison run's params.txt (defaults included). "
    f"Allowed differences: {', '.join(sorted(ALLOWED))}. The seed must equal S1 = the published seed for S1 jobs. The backbone must be the one in "
    "jobs.csv (L2: the same file name as R2's). The data folder must be the fold's folder above.")
say()
say(f"**Result: {sum(ok for *_, ok in rows)} of {len(rows)} jobs match.** One data folder per fold for every model: **{'yes' if one_split else 'NO'}** "
    f"({'; '.join(f'fold {k}: {Path(next(iter(v))).name}' for k, v in sorted(data_by_fold.items())) if one_split else data_by_fold})." + ("" if not bad else " Failures:"))
for b in bad:
    say(f"- {b[0]} vs {b[1]}: unexpected {b[2]}, seed ok {b[3]}, backbone ok {b[4]}, data folder ok {b[5]}")

# ---------------------------------------------------------------- 3. step 1 recipe
say()
say("## 3. Step 1 recipe vs the `_rebuilt` backbones")
say()
say("Not checkable on the VM: the real `_rebuilt` params.txt are only on the cluster (the copies in an old VM session scratch are checker test "
    "fixtures, with paths such as '/old/ckpt'). Check on the cluster after step 1, needed only before L1-rebuilt can be called a one-change comparison: "
    "`diff <(tr ',' '\\n' < results/pre-train/saved_models/unified/patch_ecg_unified_ii_rebuilt/params.txt) <(tr ',' '\\n' < results/pre-train/saved_models/unified/patch_ecg_unified_ii_lossfix/params.txt)` "
    "and the same for v2: only model_name, save_path and root_path may differ; also compare each data_source.txt.")

# ---------------------------------------------------------------- 4. jobs and time
say()
say("## 4. Jobs, approvals and time estimate")
say()
dur = {}
for f in SAVE.glob("R[12]*/job_log.txt"):
    t = dict(x.split(": ", 1) for x in f.read_text().splitlines() if ": " in x)
    if "started" in t and "finished" in t:
        m = "two_lead" if "two_lead" in f.parent.name else "single"
        dur.setdefault(m, []).append((datetime.strptime(t["finished"], "%Y-%m-%d_%H:%M:%S") - datetime.strptime(t["started"], "%Y-%m-%d_%H:%M:%S")).seconds / 60)
med = {m: float(np.median(v)) for m, v in dur.items()}
say(f"Fine-tune duration from {sum(map(len, dur.values()))} R1b/R2/R2b job logs (Narval, a100_3g.20gb): single-lead median {med['single']:.1f} min "
    f"(range {min(dur['single']):.0f}–{max(dur['single']):.0f}), two-lead {med['two_lead']:.1f} min (range {min(dur['two_lead']):.0f}–{max(dur['two_lead']):.0f}). "
    "Pre-training: 50 min single-lead per pretrain_unified_v2.sh (the _fixed backbones on the same profile); walltime 2:30.")
say()
say("| group | jobs | approved (docs/DECISIONS.md) | GPU-hours | submit when |")
say("|---|--:|---|--:|---|")
need = {"step1": ("yes", "step 0 passes"), "step3": ("yes (S1)", "step 0 passes"), "step2_L2": ("yes", "step 0b passes"),
        "step2_L1": ("yes", "step 1 checks pass (loss near 1 and falling; backbone_masked_vs_visible.py, 3 seeds)"),
        "step2b": ("not yet", "only if approved"), "step3o": ("not yet", "only if approved")}
tot = 0.0
for g, d in jobs.groupby("group", sort=False):
    h = sum(50 / 60 if r.pipeline == "pretrain" else (med["two_lead"] if r.mode == "two_lead" else med["single"]) / 60 for r in d.itertuples())
    tot += h if not d.optional.any() else 0
    say(f"| {g} | {len(d)} | {need[g][0]} | {h:.1f} | {need[g][1]} |")
say()
say(f"Approved groups: {int((~jobs.optional).sum())} jobs, about {tot:.1f} GPU-hours. On 2026-11-08 the 15 R2 jobs ran side by side and all finished "
    "within 35 min of the first start, so wall-clock is set by queueing: roughly step 1 ≈ 1 h, then step 2 L1 ≈ 1 h; step 3 and step 2 L2 alongside.")
say()
jobs_copy = OUT_MD.with_name(OUT_MD.stem + "_jobs.csv")
shutil.copy2(GEN / "jobs.csv", jobs_copy)
assert sha(jobs_copy) == sha(GEN / "jobs.csv")
say(f"Full list: `{jobs_copy.name}` next to this report, a byte copy of the generated `jobs.csv` (sha256 `{sha(jobs_copy)}`): run ID, group, pipeline, "
    "mode, training seed, seed, fold, dataset, backbone, optional.")

# ---------------------------------------------------------------- 5. gold-standard episodes in validation
say()
say("## 5. Gold-standard ST-deviation episodes per lead in each fold's validation set")
say()
g = pd.read_csv(GS)
g["lead"] = g.lead_paper.astype(str)
assert len(g) == 101 and set(g.lead) == {"II", "V2"}
on_masked = sorted({(r, l) for r, l in zip(g.record, g.lead) if (r, l) in MANUAL_OMIT})
assert not on_masked, on_masked
ep = g.groupby(["record", "lead"]).size()
say(f"Gold standard: `{GS}` (the published scorer's file, md5 `{hashlib.md5(GS.read_bytes()).hexdigest()}`): {len(g)} episodes, "
    f"lead II {int((g.lead == 'II').sum())}, V2 {int((g.lead == 'V2').sum())}, on {g.record.nunique()} records. Masked pairs (`MANUAL_OMIT`: "
    f"{', '.join(f'{r} {l}' for r, l in sorted(MANUAL_OMIT))}) carry no episodes: checked. The split is the same for every learned model, so one table covers them all.")
say()
say("| fold | validation records | lead II episodes | V2 episodes | per record (II / V2) | masked leads in validation |")
say("|--:|---|--:|--:|---|---|")
tot_ii = tot_v2 = 0
for k in range(1, 6):
    m = manifest(k)
    val = sorted(m[m == "val"].index)
    assert val == counts[(k, "lead_II", "val")][1]
    masked = [f"{r} {l}" for r in val for l in ("II", "V2") if (r, l) in MANUAL_OMIT]
    assert not masked, (k, masked)      # docs/DECISIONS.md: no validation record has a masked lead
    n_ii = sum(int(ep.get((r, "II"), 0)) for r in val)
    n_v2 = sum(int(ep.get((r, "V2"), 0)) for r in val)
    tot_ii, tot_v2 = tot_ii + n_ii, tot_v2 + n_v2
    per = "; ".join(f"{r} {int(ep.get((r, 'II'), 0))} / {int(ep.get((r, 'V2'), 0))}" for r in val)
    say(f"| {k} | {' '.join(val)} | {n_ii} | {n_v2} | {per} | none |")
say()
say(f"Summed over the five validation sets: lead II {tot_ii}, V2 {tot_v2} (records that validate in two folds, s20011, s20141 and s20231, count once per fold).")

OUT_MD.parent.mkdir(parents=True, exist_ok=True)
OUT_MD.write_text("\n".join(out) + "\n")
print(f"\nwritten: {OUT_MD} and {jobs_copy}")
