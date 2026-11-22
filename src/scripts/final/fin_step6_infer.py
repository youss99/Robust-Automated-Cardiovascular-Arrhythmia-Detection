#!/usr/bin/env python3
"""Final runs, VM step 6 (docs/instructions_final_results.md): out-of-fold predictions for the 90 final checkpoints, clean and noisy.

Pipelines (training report, section 4): L1 = FIN_L1_lead_II_* + FIN_L1_lead_V2_*, L2 = FIN_L2_two_lead_*, L1-sup / L2-sup = FIN_L1sup_*, FIN_L2sup_*;
seeds S1-S3, folds 1-5. Conditions: clean and NSTDB em seed 42 for every run; seeds 43 and 44 for S1 of L1 and L2 (DECISIONS.md, seeds).
Inference is phase 1's path, unchanged: ltst_pipeline.ml_inference.load_patchecg_model + predict_st_per_segment on the published 10 s grid
(250 Hz, window end = anchor), batch 1024, GPU; the trainer's own windows (training grid, clean) for regression, as p1_eval.infer_grid.
Noise: ltst_pipeline.noise.contaminate_ecg_with_motion_artifact(p_signal, seed=s, fs=250), as evaluate_test1.load_signal does for seed 42.

Controls before anything else (a failure stops the run):
  C1 this loader at seed 42 equals evaluate_test1.load_signal(rec, "noisy") exactly (s20051, both leads).
  C2 a published R1 checkpoint (two_lead pretrained, fold 1) through this code equals the published clean and noisy prediction files.
Checks after (written to CHECKS.txt; a failure writes FAILED_step6):
  G1 each run's predictions at its trainer's test rows reproduce its saved test MSE within 1 % (dataset folder from job_log.txt, local CSV sha256 checked).
  G2 window counts per record equal the published runs: grid = published R1 prediction files, training grid = published training-grid predictions.
  G3 each checkpoint's sha256 is in results/SHA256SUMS_final_copyback.txt.
Output: Tasks/final/runs/step6/ (parts/ per checkpoint, predictions_<cond>/, predictions_training_grid/, provenance.csv, STATUS.tsv, CHECKS.txt,
DONE_step6 or FAILED_step6). Finished parts are skipped on a re-launch.
Usage: setsid nohup <repo .venv>/bin/python -u fin_step6_infer.py > logs/step6.log 2>&1 < /dev/null &"""
import hashlib
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "phase1_lib"))  # vendored phase-1 code (phase1_lib/MANIFEST.tsv)
import p1_eval as P  # noqa: E402  (constants, published file paths, evaluate_test1 as P.T1)

T1, FS, WIN = P.T1, P.FS, P.WIN
REPO = Path("/home/student/GIT/Robust-Automated-Cardiovascular-Arrhythmia-Detection")
SAVE = REPO / "results/fine-tune/saved_models/patient-segment-regression"
SUMS = REPO / "results/SHA256SUMS_final_copyback.txt"
OUT = Path("/home/student/GIT/Tasks/final/runs/step6")
PIPES = [("L1", "lead_II"), ("L1", "lead_V2"), ("L2", "two_lead"), ("L1sup", "lead_II"), ("L1sup", "lead_V2"), ("L2sup", "two_lead")]
SEEDS = ("S1", "S2", "S3")
FOLDS = (1, 2, 3, 4, 5)
EXTRA = {"noisy43": 43, "noisy44": 44}  # S1 of L1 and L2 only
COND_SEED = {"clean": None, "noisy42": 42, **EXTRA}


def run_id(pipe, mode, s, k):
    return f"FIN_{pipe}_{mode}_{s}_fold{k:02d}"


def conds_for(pipe, s):
    return ("clean", "noisy42") + (tuple(EXTRA) if pipe in ("L1", "L2") and s == "S1" else ())


def log(*a):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "STATUS.tsv", "a") as f:
        f.write("\t".join(str(x) for x in (time.strftime("%F_%T"),) + a) + "\n")
    print(*a, flush=True)


def signal(rec, seed):
    import wfdb
    from ltst_pipeline.noise import contaminate_ecg_with_motion_artifact
    p = wfdb.rdrecord(str(T1.RECORDS_DIR / rec / rec)).p_signal
    if seed is not None:
        p, _ = contaminate_ecg_with_motion_artifact(p, seed=seed, fs=FS)
    return np.ascontiguousarray(p[:, [0, 1]]).astype(np.float32)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def frame(rec, k, samp, st):
    n = len(samp)
    nan = np.full(n, np.nan, np.float32)
    return pd.DataFrame({"record": rec, "fold": k, "anchor_samp": samp, "window_start_samp": samp - (WIN - 1), "pred_II_mV": st.get("II", nan), "pred_V2_mV": st.get("V2", nan)})


def predict_all(model, revin, device, mode, recs, k, sig, conds, offs):
    from ltst_pipeline.ml_inference import predict_st_per_segment
    out = {}
    for c in conds:
        parts = []
        for r in recs:
            samp, st = predict_st_per_segment(sig[r, c], FS, model, revin, device, lead_order=("II", "V2"), batch_size=1024, lead_mode=mode)
            parts.append(frame(r, k, samp, st))
        out[c] = pd.concat(parts, ignore_index=True)
    parts = []
    for r in recs:
        o = int(offs[r])
        n_seg = (len(sig[r, "clean"]) - o) // WIN
        samp, st = predict_st_per_segment(np.ascontiguousarray(sig[r, "clean"][o:o + n_seg * WIN]), FS, model, revin, device, lead_order=("II", "V2"), batch_size=1024, lead_mode=mode)
        f = frame(r, k, samp + o, st).rename(columns={"anchor_samp": "samp_end_raw"})
        f.insert(1, "seg_idx", np.arange(len(f)))
        parts.append(f)
    out["tg"] = pd.concat(parts, ignore_index=True)
    return out


def part(rid, c):
    return OUT / "parts" / c / f"{rid}.csv.gz"


def controls():
    from ltst_pipeline.ml_inference import load_patchecg_model
    lines, ok = [], True
    a, b = signal("s20051", 42), T1.load_signal("s20051", "noisy")[:, [0, 1]].astype(np.float32)
    good = a.shape == b.shape and np.array_equal(a, b)
    ok &= good
    lines.append(f"{'PASS' if good else 'FAIL'}  C1 seeded loader at seed 42 equals evaluate_test1.load_signal(noisy), s20051: shape {a.shape}")
    k, mode, arm = 1, "two_lead", "pretrained"
    recs = T1.FOLDS[k]
    sig = {(r, c): (signal(r, None) if c == "clean" else signal(r, 42)) for r in recs for c in ("clean", "noisy42")}
    model, revin, device = load_patchecg_model(str(P.ckpt("R1", mode, arm, k)), device="cuda", lead_mode=mode)
    offs = pd.read_csv(P.OFFSETS).set_index("record").offset_samples
    pr = predict_all(model, revin, device, mode, recs, k, sig, ("clean", "noisy42"), offs)
    for c, pc in (("clean", "clean"), ("noisy42", "noisy")):
        ref = pd.read_csv(P.grid_file("R1_published", arm, mode, pc))
        ref = ref[ref.record.isin(recs)]
        m = pr[c].merge(ref, on=["record", "anchor_samp"], suffixes=("", "_ref"))
        d = max(np.nanmax(np.abs(m[x] - m[x + "_ref"])) for x in ("pred_II_mV", "pred_V2_mV"))
        good = len(m) == len(ref) == len(pr[c]) and d < 1e-4
        ok &= good
        lines.append(f"{'PASS' if good else 'FAIL'}  C2 published R1 two_lead pretrained fold 1 through this code equals the published {pc} predictions: {len(m)}/{len(ref)} windows, worst |diff| {d:.2e} mV")
    del model
    return ok, lines


def infer():
    from ltst_pipeline.ml_inference import load_patchecg_model
    offs = pd.read_csv(P.OFFSETS).set_index("record").offset_samples
    prov, failed, t0 = [], [], time.time()
    for k in FOLDS:
        todo = [(pipe, mode, s) for pipe, mode in PIPES for s in SEEDS
                if not all(part(run_id(pipe, mode, s, k), c).exists() for c in conds_for(pipe, s) + ("tg",))]
        if not todo:
            continue
        recs = T1.FOLDS[k]
        need = sorted({c for pipe, _, s in todo for c in conds_for(pipe, s)}, key=list(COND_SEED).index)
        sig = {(r, c): signal(r, COND_SEED[c]) for r in recs for c in need}
        for pipe, mode, s in todo:
            rid = run_id(pipe, mode, s, k)
            try:
                p = SAVE / rid / f"{rid}_best.pt"
                model, revin, device = load_patchecg_model(str(p), device="cuda", lead_mode=mode)
                pr = predict_all(model, revin, device, mode, recs, k, sig, conds_for(pipe, s), offs)
                for c, df in pr.items():
                    P.write(df, part(rid, c))
                del model
                log("infer", rid, f"{time.time() - t0:.0f}s")
            except Exception:  # noqa: BLE001
                failed.append(rid)
                log("FAILED", rid, traceback.format_exc().replace("\n", " | "))
        del sig
    return failed


def assemble():
    for pipe, mode in PIPES:
        for s in SEEDS:
            stem = f"FIN_{pipe}_{mode}_{s}"
            for c in conds_for(pipe, s) + ("tg",):
                df = pd.concat([pd.read_csv(part(run_id(pipe, mode, s, k), c)) for k in FOLDS], ignore_index=True)
                if c == "tg":
                    P.write(df.sort_values(["record", "seg_idx"]), OUT / "predictions_training_grid" / f"{stem}.csv.gz")
                else:
                    P.write(df.sort_values(["record", "anchor_samp"]), OUT / f"predictions_{c}" / f"{stem}__grid.csv.gz")


def job_field(rid, key):
    for line in (SAVE / rid / "job_log.txt").read_text().splitlines():
        if line.startswith(key + ":"):
            return line.split(":", 1)[1].strip()
    raise KeyError(f"{rid}: {key}")


def checks():
    lines, ok, rows, prov = [], True, [], []
    sums = dict(reversed(line.split(None, 1)) for line in SUMS.read_text().splitlines())
    sums = {Path(k.strip()).name: v for k, v in sums.items()}
    for pipe, mode in PIPES:
        leads = P.OUT_LEADS[mode]
        for s in SEEDS:
            tg = pd.read_csv(OUT / "predictions_training_grid" / f"FIN_{pipe}_{mode}_{s}.csv.gz")
            for k in FOLDS:
                rid = run_id(pipe, mode, s, k)
                ds = REPO / "phase1_folds" / Path(job_field(rid, "dataset")).name
                csv = ds / "modified_dataset/modified_dataset.csv"
                csv_ok = sha(csv) == job_field(rid, "dataset_csv_sha256")
                md = pd.read_csv(csv, usecols=["patient_id", "segment_id", "strat_fold_annotated", *P.TARGET.values()])
                te = md[md.strat_fold_annotated == 2].copy()
                te["seg_idx"] = te.segment_id.str.slice(3).astype(int)
                m = te.merge(tg, left_on=["patient_id", "seg_idx"], right_on=["record", "seg_idx"], how="inner")
                pp = m[[P.COL[x] for x in leads]].to_numpy(float)
                y = m[[P.TARGET[x] for x in leads]].to_numpy(float)
                saved = float(pd.read_csv(SAVE / rid / f"_test_metrics_{rid}.csv").test_MSE.iloc[0])
                here = float(((pp - y) ** 2).mean())
                ck = SAVE / rid / f"{rid}_best.pt"
                ck_sha = sha(ck)
                rows.append(dict(run_id=rid, pipeline=pipe, mode=mode, seed=s, fold=k, dataset=ds.name, dataset_csv_sha_ok=csv_ok, n_test=len(te), n_matched=len(m),
                                 n_missing=int(np.isnan(pp).sum()), mse_here=here, mse_saved=saved, rel_diff=abs(here - saved) / saved, ckpt_sha_ok=sums.get(ck.name) == ck_sha))
                prov.append(dict(run_id=rid, pipeline=pipe, mode=mode, training_seed=s, seed=job_field(rid, "seed"), fold=k, checkpoint=str(ck), sha256=ck_sha,
                                 git_commit=job_field(rid, "git_commit"), dataset=ds.name, dataset_csv_sha256=job_field(rid, "dataset_csv_sha256"),
                                 backbone_sha256=job_field(rid, "backbone_sha256") if pipe in ("L1", "L2") else "none", slurm_job=job_field(rid, "slurm_job")))
    G = pd.DataFrame(rows)
    G["pass"] = (G.rel_diff <= 0.01) & (G.n_missing == 0) & (G.n_matched == G.n_test) & G.dataset_csv_sha_ok
    P.write(G, OUT / "gate_trainer_mse.csv")
    P.write(pd.DataFrame(prov), OUT / "provenance.csv")
    good = bool(G["pass"].all())
    ok &= good
    lines.append(f"{'PASS' if good else 'FAIL'}  G1 predictions at the trainer's test rows reproduce the saved test MSE within 1 %: {int(G['pass'].sum())}/{len(G)}, worst {G.rel_diff.max():.4%}; dataset CSV sha256 ok {int(G.dataset_csv_sha_ok.sum())}/{len(G)}")
    good = bool(G.ckpt_sha_ok.all())
    ok &= good
    lines.append(f"{'PASS' if good else 'FAIL'}  G3 checkpoint sha256 in the Narval copy-back list: {int(G.ckpt_sha_ok.sum())}/{len(G)}")
    # G2 window counts
    ref_grid = {c: pd.read_csv(P.grid_file("R1_published", "pretrained", "two_lead", c), usecols=["record"]).record.value_counts().sort_index() for c in ("clean", "noisy")}
    ref_tg = pd.read_csv(P.tg_file("R1_published", "pretrained", "two_lead"), usecols=["record"]).record.value_counts().sort_index()
    bad = []
    for pipe, mode in PIPES:
        for s in SEEDS:
            stem = f"FIN_{pipe}_{mode}_{s}"
            for c in conds_for(pipe, s):
                n = pd.read_csv(OUT / f"predictions_{c}" / f"{stem}__grid.csv.gz", usecols=["record"]).record.value_counts().sort_index()
                ref = ref_grid["clean" if c == "clean" else "noisy"]
                if not n.equals(ref):
                    bad.append(f"{stem} {c}")
            n = pd.read_csv(OUT / "predictions_training_grid" / f"{stem}.csv.gz", usecols=["record"]).record.value_counts().sort_index()
            if not n.equals(ref_tg):
                bad.append(f"{stem} training grid")
    good = not bad
    ok &= good
    lines.append(f"{'PASS' if good else 'FAIL'}  G2 window counts per record equal the published runs (grid {int(ref_grid['clean'].sum())} windows on {len(ref_grid['clean'])} records; "
                 f"training grid {int(ref_tg.sum())} on {len(ref_tg)}): mismatches {bad}")
    return ok, lines


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for m in ("DONE_step6", "FAILED_step6"):
        (OUT / m).unlink(missing_ok=True)
    if not (OUT / "CONTROLS_PASSED").exists():
        ok, lines = controls()
        (OUT / "CONTROLS.txt").write_text("\n".join(lines) + "\n")
        log("controls", "PASSED" if ok else "FAILED", *lines)
        if not ok:
            (OUT / "FAILED_step6").write_text("controls failed\n")
            return False
        (OUT / "CONTROLS_PASSED").write_text(time.strftime("%F %T") + "\n")
    failed = infer()
    if failed:
        (OUT / "FAILED_step6").write_text(f"inference failed: {failed}\n")
        return False
    assemble()
    log("assembled")
    ok, lines = checks()
    lines.append("ALL PASSED" if ok else "FAILED")
    (OUT / "CHECKS.txt").write_text("\n".join(lines) + "\n")
    log("checks", *lines)
    (OUT / ("DONE_step6" if ok else "FAILED_step6")).write_text(time.strftime("%F %T") + "\n")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
