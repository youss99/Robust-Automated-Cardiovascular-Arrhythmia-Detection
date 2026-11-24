#!/usr/bin/env python3
"""Dry run for step3e (docs/DECISIONS.md, 2026-11-24): the 45 end-to-end supervised jobs written by build_e2e_sup_sbatches.py.
No training, nothing submitted. Every check is applied to every job; section 5 shows that each check fails on an injected fault.

1. The existing final generator is unchanged: build_final_sbatches.py, re-run into a temporary folder, still writes the jobs.csv the
   2026-11-20 dry run approved (sha256 2133e920..., docs/reports/2026-11-20_dry_run_jobs.csv).
2. Job text: each step3e .sbatch vs its twin's .sbatch (step3/step3o, regenerated in 1). After replacing the run-ID prefix and the
   pipeline label, the only changed lines may be the two epoch lines (20 -> 0 head-only, 70 -> 90 full).
3. Flags: each job's python command through the real parser of src/patchECG_finetune.py, compared value by value (defaults included)
   with the twin's params.txt from the completed run (written on Narval, copied back). Allowed differences: model_name, save_path,
   root_path (Narval's per-job scratch), n_epochs_finetune_head (must be 20 -> 0), n_epochs_finetune (must be 70 -> 90).
4. Data: each job's data folder is its twin's (job_log.txt "dataset:"); the fold CSV here has the sha256 the twin logged on Narval
   ("dataset_csv_sha256:") and the one in phase1_folds/SHA256SUMS.txt; the H5 matches SHA256SUMS.txt; the CSV's split column assigns
   exactly the manifest's records to train / validation / test / unused.
5. Fault injection: one corrupted copy per check type must be flagged.
6. Time estimate from the twins' job_log.txt.
Usage: <repo>/.venv/bin/python src/scripts/final/dry_run_e2e.py <generated folder (FIN_OUT_E2E)> <output .md>
Also writes <output stem>_jobs.csv, a byte copy of the generated jobs.csv."""
import difflib
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
GEN, OUT_MD = Path(sys.argv[1]), Path(sys.argv[2])
SAVE = REPO / "results/fine-tune/saved_models/patient-segment-regression"
FOLDS = REPO / "phase1_folds"
MAN = {k: REPO / ("ptbxl_models_v2/cv5_optB_v2models_leadV2_fold_manifests" if k in (1, 4, 5) else "ptbxl_models_v2/cv5_optB_fold_manifests")
       / f"fold{k:02d}.csv" for k in range(1, 6)}
OLD_JOBS_SHA = "2133e92037d6c52239f4357d3b81b1440c8a88815d45bfb14f688895835480f8"
CSV, H5 = "modified_dataset/modified_dataset.csv", "PatientSegmentRegressionDataset.h5"
EPOCH_LINES = {"-  --n_epochs_finetune_head=20 \\", "-  --n_epochs_finetune=70 \\",
               "+  --n_epochs_finetune_head=0 \\", "+  --n_epochs_finetune=90 \\"}
ALLOWED = {"model_name", "save_path", "root_path"}
EPOCH_FLAGS = {"n_epochs_finetune_head": ("20", "0"), "n_epochs_finetune": ("70", "90")}
SPLIT = {1: "train", 3: "val", 2: "test", -1: "unused"}       # strat_fold_annotated codes (prepare_fold_dataset.py, build_phase1_folds.py)
out = []


def say(s=""):
    print(s, flush=True)
    out.append(s)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def norm_name(text, pipe):
    return text.replace(f"FIN_{pipe}_", "FIN_X_").replace(f"pipeline: {pipe};", "pipeline: X;")


def text_diff(new, new_pipe, twin, twin_pipe):
    """Changed lines between twin and new job after normalising the run ID; must be exactly EPOCH_LINES."""
    d = difflib.unified_diff(norm_name(twin, twin_pipe).splitlines(), norm_name(new, new_pipe).splitlines(), lineterm="", n=0)
    return {x for x in d if x[:1] in "+-" and not x.startswith(("+++", "---"))}


def parse_text(t):
    return dict(re.findall(r"(\w+)=('[^']*'|\[[^\]]*\]|<[^>]*>|[^,)]+)", t))


def norm(v):
    if v is None:
        return None
    m = re.fullmatch(r"<\w+\.(\w+): [^>]*>", v)
    return (m.group(1) if m else v.strip("'")).lower()


def flag_check(got, ref):
    """Returns (unexpected differences, epoch flags correct)."""
    got = dict(got, dset=got.get("dset", got["dset_finetune"]))   # the trainer sets args.dset = dset_finetune after parsing
    diff = {k: (ref.get(k), got.get(k)) for k in sorted(set(ref) | set(got)) if norm(ref.get(k)) != norm(got.get(k))}
    unexpected = {k: v for k, v in diff.items() if k not in ALLOWED | set(EPOCH_FLAGS)}
    epochs_ok = all((norm(ref.get(k)), norm(got.get(k))) == v for k, v in EPOCH_FLAGS.items())
    return unexpected, epochs_ok


def job_words(text):
    body = text.split("-m src.patchECG_finetune")[1].split("\n\n")[0].replace("\\\n", " ")
    return shlex.split(body.replace('"${STAGED_ROOT}"', "/tmp/none"))


def job_data(text):
    return re.search(r'\[\[ -f "([^"]+)/PatientSegmentRegressionDataset\.h5" \]\]', text).group(1)


def joblog(run):
    return dict(re.findall(r"^(\w+): (.*)$", (SAVE / run / "job_log.txt").read_text(), re.M))


def data_check(new_text, twin_log, sums):
    """Data folder = twin's; CSV sha here = twin's logged sha = SHA256SUMS.txt."""
    name = Path(job_data(new_text)).name
    same_folder = name == Path(twin_log["dataset"]).name
    here = sha(FOLDS / name / CSV)
    return same_folder and here == twin_log["dataset_csv_sha256"] == sums.get(f"{name}/{CSV}"), name, here


def parse_all(texts):
    code = ("import sys, json; batch = json.loads(sys.stdin.read()); out = []\n"
            "for words in batch:\n"
            "    sys.argv = ['x'] + words\n"
            "    if 'src.patchECG_finetune' in sys.modules: del sys.modules['src.patchECG_finetune']\n"
            "    import src.patchECG_finetune as F\n"
            "    out.append(repr(F.args))\n"
            "print('ARGS ' + json.dumps(out))\n")
    res = subprocess.run([str(REPO / ".venv/bin/python"), "-c", code], input=json.dumps([job_words(t) for t in texts]),
                         capture_output=True, text=True, cwd=REPO, timeout=1800)
    line = next((x for x in res.stdout.splitlines() if x.startswith("ARGS ")), None)
    assert line, res.stderr[-2000:]
    return [parse_text(a) for a in json.loads(line[5:])]


jobs = pd.read_csv(GEN / "jobs.csv")
say(f"# step3e dry run ({datetime.now():%Y-%m-%d %H:%M}, VM)")
say()
say(f"Generated folder: `{GEN}`. Commit: `{subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=REPO, capture_output=True, text=True).stdout.strip()}` "
    f"plus the uncommitted step3e files. {len(jobs)} jobs: "
    + ", ".join(f"{p} {m} {n}" for (p, m), n in jobs.groupby(["pipeline", "mode"]).size().items()) + ".")
assert len(jobs) == 45 and jobs.run_id.is_unique and jobs.twin.is_unique

# ---------------------------------------------------------------- 1. old generator unchanged
say()
say("## 1. The existing final generator is unchanged")
tmp = Path(tempfile.mkdtemp(prefix="e2e_old_"))
env = dict(os.environ, FIN_OUT=str(tmp))
r = subprocess.run([sys.executable, str(REPO / "src/scripts/custom_dataset/build_final_sbatches.py")], env=env, capture_output=True, text=True)
assert r.returncode == 0, r.stderr
old_sha = sha(tmp / "jobs.csv")
approved = sha(REPO / "docs/reports/2026-11-20_dry_run_jobs.csv")
ok1 = old_sha == OLD_JOBS_SHA == approved
say()
say(f"Re-run into a temporary folder: jobs.csv sha256 `{old_sha[:16]}...`; approved 2026-11-20 copy `{approved[:16]}...`; expected `{OLD_JOBS_SHA[:16]}...`. "
    f"**{'PASS' if ok1 else 'FAIL'}**")
old_jobs = pd.read_csv(tmp / "jobs.csv")
old_group = dict(zip(old_jobs.run_id, old_jobs.group))

# ---------------------------------------------------------------- 2. job text
say()
say("## 2. Job text vs the twin's job text")
texts, twins_text, bad2 = {}, {}, []
for j in jobs.itertuples():
    texts[j.run_id] = (GEN / j.group / f"{j.run_id}.sbatch").read_text()
    twins_text[j.run_id] = (tmp / old_group[j.twin] / f"{j.twin}.sbatch").read_text()
    tp = j.twin.split("_")[1]
    ch = text_diff(texts[j.run_id], j.pipeline, twins_text[j.run_id], tp)
    if ch != EPOCH_LINES:
        bad2.append((j.run_id, sorted(ch ^ EPOCH_LINES)))
say()
say(f"Twins: {', '.join(f'{g} {n}' for g, n in pd.Series([old_group[t] for t in jobs.twin]).value_counts().items())}. "
    f"Changed lines must be exactly the two epoch lines. **{45 - len(bad2)} of 45 PASS**" + (f"; failures: {bad2}" if bad2 else "."))

# ---------------------------------------------------------------- 3. flags vs twin params.txt
say()
say("## 3. Every flag through the real parser vs the twin's params.txt")
parsed = parse_all([texts[j.run_id] for j in jobs.itertuples()])
bad3, seen_diff = [], set()
for j, got in zip(jobs.itertuples(), parsed):
    ref = parse_text((SAVE / j.twin / "params.txt").read_text())
    unexpected, epochs_ok = flag_check(got, ref)
    seen_diff |= {k for k in set(ref) | set(got) if norm(ref.get(k)) != norm(dict(got, dset=got["dset_finetune"]).get(k))}
    seed_ok = got["seed"] == ref["seed"] == str(j.seed)
    bb_ok = got["pretrained_model_path"] == "None" == ref["pretrained_model_path"]
    if unexpected or not (epochs_ok and seed_ok and bb_ok):
        bad3.append((j.run_id, unexpected, epochs_ok, seed_ok, bb_ok))
say()
say(f"{len(parsed[0])} values per job compared. Allowed: {', '.join(sorted(ALLOWED))}; epoch flags must read 20 -> 0 and 70 -> 90; seed must equal "
    f"the twin's and jobs.csv; backbone None in both. Keys that differed in at least one job: {', '.join(sorted(seen_diff))}. "
    f"**{45 - len(bad3)} of 45 PASS**" + (f"; failures: {bad3}" if bad3 else "."))
key = parsed[0]
say()
say("Effective settings (job 1, from the parser): " + ", ".join(f"{k}={norm(key[k])}" for k in (
    "model", "d_model", "n_layers", "n_heads", "d_ff", "patch_len", "stride", "class_token", "revin_mode", "custom_lead_selection",
    "opt", "lr", "weight_decay", "layer_decay", "scheduler", "clip_grad", "sam", "batch_size", "regression_loss", "data_augmentation",
    "dropout", "head_dropout", "n_epochs_finetune_head", "n_epochs_finetune", "model_selection_metric", "pretrained_model_path")) + ". "
    "layer_decay is read only by prepare_optimizer, which runs only with a backbone (src/patchECG_finetune.py:304-305).")

# ---------------------------------------------------------------- 4. data
say()
say("## 4. Datasets and fold assignment")
sums = {ln.split("  ")[1]: ln.split("  ")[0] for ln in (FOLDS / "SHA256SUMS.txt").read_text().splitlines() if ln.strip()}
bad4, used = [], {}
for j in jobs.itertuples():
    ok, name, here = data_check(texts[j.run_id], joblog(j.twin), sums)
    used.setdefault(int(j.fold), set()).add((name, here))
    if not ok:
        bad4.append(j.run_id)
h5 = {n: sha(FOLDS / n / H5) for n in sorted({n for s in used.values() for n, _ in s})}
h5_ok = set(h5.values()) == {sums[H5]}
say()
say(f"Data folder = twin's, CSV sha here = twin's logged sha (Narval) = SHA256SUMS.txt: **{45 - len(bad4)} of 45 PASS**"
    + (f"; failures: {bad4}" if bad4 else ".") + f" H5 in all {len(h5)} folders = SHA256SUMS.txt (`{sums[H5][:16]}...`): **{'PASS' if h5_ok else 'FAIL'}**.")
say()
say("| fold | data folder | CSV sha256 | train / val / test / unused records (CSV = manifest) |")
say("|---|---|---|---|")
split_ok = True
for k in range(1, 6):
    assert len(used[k]) == 1, k
    (name, here), = used[k]
    md = pd.read_csv(FOLDS / name / CSV, usecols=["patient_id", "strat_fold_annotated"]).drop_duplicates()
    assert md.patient_id.is_unique, k
    csv_split = {p: SPLIT[int(s)] for p, s in zip(md.patient_id, md.strat_fold_annotated)}
    man = pd.read_csv(MAN[k])
    man_split = dict(zip(man.patient_id, man.split.str.lower()))
    same = csv_split == man_split
    split_ok &= same
    cnt = " / ".join(str(sum(v == s for v in csv_split.values())) for s in ("train", "val", "test", "unused"))
    say(f"| {k} | `{name}` | `{here[:16]}...` | {cnt} ({'same' if same else 'DIFFERENT'}) |")
say()
say(f"Split column = manifest in every fold: **{'PASS' if split_ok else 'FAIL'}**.")

# ---------------------------------------------------------------- 5. fault injection
say()
say("## 5. The checks fail on injected faults")
j0 = jobs.iloc[0]
t0, tw0 = texts[j0.run_id], twins_text[j0.run_id]
ref0 = parse_text((SAVE / j0.twin / "params.txt").read_text())
faults = [
    ("text: lr changed in the job", text_diff(t0.replace("--lr=0.001", "--lr=0.002"), j0.pipeline, tw0, "L1sup") != EPOCH_LINES),
    ("text: head epochs left at 20", text_diff(t0.replace("--n_epochs_finetune_head=0", "--n_epochs_finetune_head=20"), j0.pipeline, tw0, "L1sup") != EPOCH_LINES),
    ("flags: weight_decay 0.05 -> 0.0", bool(flag_check(dict(parsed[0], weight_decay="0.0"), ref0)[0])),
    ("flags: n_epochs_finetune 90 -> 70", not flag_check(dict(parsed[0], n_epochs_finetune="70"), ref0)[1]),
    ("flags: backbone added", bool(flag_check(dict(parsed[0], pretrained_model_path="'/x/bb.pt'"), ref0)[0])),
    ("data: fold-2 folder in a fold-1 job", not data_check(t0.replace("cv5_optB_leadV2_opt2p_fold01", "cv5_optB_fold02"), joblog(j0.twin), sums)[0]),
    ("data: logged CSV sha altered", not data_check(t0, dict(joblog(j0.twin), dataset_csv_sha256="0" * 64), sums)[0]),
]
say()
for n, caught in faults:
    say(f"- {n}: {'flagged' if caught else 'NOT FLAGGED'}")
ok5 = all(c for _, c in faults)

# ---------------------------------------------------------------- 6. time
say()
say("## 6. Time estimate")
dur = {}
for j in jobs.itertuples():
    lg = joblog(j.twin)
    f = lambda s: datetime.strptime(s.strip(), "%Y-%m-%d_%H:%M:%S")
    dur.setdefault(j.pipeline + " " + j.mode, []).append((f(lg["finished"]) - f(lg["started"])).total_seconds() / 60)
say()
say("Twins (20 head-only + 70 full) took, in minutes: " + "; ".join(f"{k}: median {pd.Series(v).median():.1f}, max {max(v):.1f}" for k, v in dur.items())
    + f". Total {sum(map(sum, dur.values())) / 60:.1f} GPU-hours. A head-only epoch has no backward pass through the encoder, so 90 full epochs cost "
    f"more; upper bound 90/70 of the twin: about {sum(map(sum, dur.values())) / 60 * 90 / 70:.1f} GPU-hours, longest job about "
    f"{max(map(max, dur.values())) * 90 / 70:.0f} min (walltime 45 min).")

# ---------------------------------------------------------------- summary
shutil.copyfile(GEN / "jobs.csv", OUT_MD.with_name(OUT_MD.stem + "_jobs.csv"))
allok = ok1 and not bad2 and not bad3 and not bad4 and h5_ok and split_ok and ok5
say()
say(f"## Result: {'ALL PASS' if allok else 'FAILURES ABOVE'}")
say()
say(f"`{OUT_MD.stem}_jobs.csv` next to this report is a byte copy of the generated jobs.csv, sha256 `{sha(GEN / 'jobs.csv')}`. "
    "Narval must reproduce this sha before submitting.")
OUT_MD.write_text("\n".join(out) + "\n")
shutil.rmtree(tmp)
sys.exit(0 if allok else 1)
