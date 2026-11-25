#!/usr/bin/env python3
"""Podium-only (docs/DECISIONS.md 2026-11-24): Learned-1L S1 on the PUBLISHED folds, inference and scoring in the final podium configuration.

Runs: POD_L1_lead_{II,V2}_S1_fold0{1,4,5} (Narval, report docs/reports/2026-11-25_podium_training.md, copy-back list
results/SHA256SUMS_podium_copyback.txt); folds 2 and 3 reuse FIN_L1_lead_{II,V2}_S1_fold02/03 (already trained on the published folders).
Stage infer (GPU): fin_step6_infer's code path unchanged (signal, load_patchecg_model, predict_all; published 10 s grid and training grid,
clean and NSTDB em seed 42), its module paths redirected to Tasks/final/runs/podium/step6. Folds 2 and 3: the FIN S1 step-6 parts, copied.
  Controls: step 6's C1/C2 (seeded loader; published R1 through this code); G1 each POD run's predictions at its trainer's test rows reproduce
  its saved test MSE within 1 %; G2 window counts per record equal the published runs; G3 checkpoint sha256 in the podium copy-back list;
  F23 the assembled folds 2-3 rows equal the FIN S1 predictions.
Stage score: NeuroKit R-peaks for the Fiducial delineation and for the SQI gate (Phase A configuration: fin_nk_detector step-5 table, columns
  NKPR / sqi_grid / sqi_nk), 5-sample ST level, rounded values, 6 dB, noise seed 42; fin_step7_tier1's alarm path and scorer unchanged.
  Controls: Fiducial rows = RESULTS_NK Phase A (67/34/34, 67/43/34, 65/30/36); Monitor from the paper.
  Writes the section "Published folds, NeuroKit gate (final podium configuration)" at the end of docs/PODIUM_RESULTS.md (between markers);
  the existing .atr section is not changed (checked byte for byte).
Usage: <repo .venv>/bin/python -u -W ignore podium_published_folds.py infer|score"""
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "phase1_lib"))
import fin_step6_infer as S6  # noqa: E402

REPO, P = S6.REPO, S6.P
SAVE = S6.SAVE
FIN6 = Path("/home/student/GIT/Tasks/final/runs/step6")
OUT = Path("/home/student/GIT/Tasks/final/runs/podium/published_folds")
SUMS = REPO / "results/SHA256SUMS_podium_copyback.txt"
MD = REPO / "docs/PODIUM_RESULTS.md"
MODES = ("lead_II", "lead_V2")
POD_FOLDS, FIN_FOLDS = (1, 4, 5), (2, 3)
CONDS = ("clean", "noisy42")
TAGM = "published_folds_nk"


def rid(mode, k):
    return f"{'POD' if k in POD_FOLDS else 'FIN'}_L1_{mode}_S1_fold{k:02d}"


def redirect():
    S6.OUT = OUT / "step6"
    S6.SUMS = SUMS
    S6.PIPES, S6.SEEDS, S6.FOLDS = [("L1", m) for m in MODES], ("S1",), POD_FOLDS
    S6.run_id = lambda pipe, mode, s, k: f"POD_{pipe}_{mode}_{s}_fold{k:02d}"
    S6.conds_for = lambda pipe, s: CONDS


def pred_file(mode, c):
    return OUT / "step6" / ("predictions_training_grid" if c == "tg" else f"predictions_{c}") / (f"POD_L1_{mode}_S1" + ("" if c == "tg" else "__grid") + ".csv.gz")


def infer():
    redirect()
    (OUT / "step6").mkdir(parents=True, exist_ok=True)
    ok, lines = S6.controls()
    log = lines[:]
    if not ok:
        return finish(False, log)
    failed = S6.infer()
    if failed:
        return finish(False, log + [f"FAIL  inference: {failed}"])
    # assemble: POD parts (folds 1, 4, 5) + FIN S1 parts (folds 2, 3)
    for mode in MODES:
        for c in CONDS + ("tg",):
            parts = []
            for k in range(1, 6):
                src = S6.part(rid(mode, k), c) if k in POD_FOLDS else FIN6 / "parts" / c / f"{rid(mode, k)}.csv.gz"
                parts.append(pd.read_csv(src))
            df = pd.concat(parts, ignore_index=True)
            df = df.sort_values(["record", "seg_idx"] if c == "tg" else ["record", "anchor_samp"])
            P.write(df, pred_file(mode, c))
    # F23: the folds 2-3 rows of the assembled files equal the FIN S1 predictions
    bad = []
    for mode in MODES:
        for c in CONDS + ("tg",):
            a = pd.read_csv(pred_file(mode, c))
            b = pd.read_csv(FIN6 / ("predictions_training_grid" if c == "tg" else f"predictions_{c}") / (f"FIN_L1_{mode}_S1" + ("" if c == "tg" else "__grid") + ".csv.gz"))
            a, b = a[a.fold.isin(FIN_FOLDS)].reset_index(drop=True), b[b.fold.isin(FIN_FOLDS)].reset_index(drop=True)
            if not a.equals(b):
                bad.append(f"{mode} {c}")
    log.append(f"{'PASS' if not bad else 'FAIL'}  F23 folds 2-3 rows of the assembled predictions equal the FIN S1 predictions (clean, noisy42, training grid; both leads): mismatches {bad}")
    ok = not bad
    # G1 / G3 for the 6 POD runs (step 6's checks() logic, own file names)
    sums = {Path(line.split(None, 1)[1].strip()).name: line.split()[0] for line in SUMS.read_text().splitlines() if line.strip()}
    rows = []
    for mode in MODES:
        tg = pd.read_csv(pred_file(mode, "tg"))
        L = P.OUT_LEADS[mode]
        for k in POD_FOLDS:
            r = rid(mode, k)
            ds = REPO / "phase1_folds" / Path(S6.job_field(r, "dataset")).name
            csv = ds / "modified_dataset/modified_dataset.csv"
            md = pd.read_csv(csv, usecols=["patient_id", "segment_id", "strat_fold_annotated", *P.TARGET.values()])
            te = md[md.strat_fold_annotated == 2].copy()
            te["seg_idx"] = te.segment_id.str.slice(3).astype(int)
            m = te.merge(tg, left_on=["patient_id", "seg_idx"], right_on=["record", "seg_idx"], how="inner")
            pp, y = m[[P.COL[x] for x in L]].to_numpy(float), m[[P.TARGET[x] for x in L]].to_numpy(float)
            saved = float(pd.read_csv(SAVE / r / f"_test_metrics_{r}.csv").test_MSE.iloc[0])
            here = float(((pp - y) ** 2).mean())
            ck = SAVE / r / f"{r}_best.pt"
            rows.append(dict(run_id=r, dataset=ds.name, csv_sha_ok=S6.sha(csv) == S6.job_field(r, "dataset_csv_sha256"), n_test=len(te), n_matched=len(m),
                             n_missing=int(np.isnan(pp).sum()), mse_here=here, mse_saved=saved, rel_diff=abs(here - saved) / saved, ckpt_sha_ok=sums.get(ck.name) == S6.sha(ck)))
    G = pd.DataFrame(rows)
    G["pass"] = (G.rel_diff <= 0.01) & (G.n_missing == 0) & (G.n_matched == G.n_test) & G.csv_sha_ok & G.dataset.str.fullmatch(r"cv5_optB_fold0[145]")
    P.write(G, OUT / "step6" / "gate_trainer_mse_pod.csv")
    log.append(f"{'PASS' if G['pass'].all() else 'FAIL'}  G1 each POD run reproduces its saved test MSE within 1 % on the published fold's test rows: {int(G['pass'].sum())}/6, worst {G.rel_diff.max():.4%}")
    log.append(f"{'PASS' if G.ckpt_sha_ok.all() else 'FAIL'}  G3 POD checkpoint sha256 in the podium copy-back list: {int(G.ckpt_sha_ok.sum())}/6")
    ok &= bool(G["pass"].all() and G.ckpt_sha_ok.all())
    ref = {c: pd.read_csv(P.grid_file("R1_published", "pretrained", "two_lead", "clean" if c == "clean" else "noisy"), usecols=["record"]).record.value_counts().sort_index() for c in CONDS}
    ref_tg = pd.read_csv(P.tg_file("R1_published", "pretrained", "two_lead"), usecols=["record"]).record.value_counts().sort_index()
    bad = [f"{mode} {c}" for mode in MODES for c in CONDS + ("tg",)
           if not pd.read_csv(pred_file(mode, c), usecols=["record"]).record.value_counts().sort_index().equals(ref_tg if c == "tg" else ref[c])]
    log.append(f"{'PASS' if not bad else 'FAIL'}  G2 window counts per record equal the published runs: mismatches {bad}")
    ok &= not bad
    return finish(ok, log, "infer")


def finish(ok, lines, st="infer"):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"CHECKS_{st}.txt").write_text("\n".join(lines) + ("\nALL PASSED\n" if ok else "\nFAILED\n"))
    print("\n".join(lines), flush=True)
    (OUT / (f"DONE_{st}" if ok else f"FAILED_{st}")).write_text(time.strftime("%F %T") + "\n")
    return ok


# ------------------------------------------------------------------ score
def score():
    import fin_step7_tier1 as T
    import podium_results as PR
    V5 = Path("/home/student/GIT/Tasks/final/runs/nk_detector/step5")
    V = {c: pd.read_csv(V5 / f"nk_values_{c}.csv.gz") for c in ("clean", "noisy42")}
    Mn, Mg = T.memos(V["noisy42"], "anchor_nk", "sqi_nk"), T.memos(V["noisy42"], "anchor_grid", "sqi_grid")

    def frames(cond):
        c = "clean" if cond == "clean" else "noisy42"
        return {L: pd.read_csv(pred_file(m, c), usecols=["record", "anchor_samp", f"pred_{L}_mV"]).rename(columns={f"pred_{L}_mV": "pred"}) for m, L in (("lead_II", "II"), ("lead_V2", "V2"))}
    res = []
    for cond in ("clean", "noisy", "gated"):
        d = V["clean" if cond == "clean" else "noisy42"]
        fid = []
        for (rec, L), g in d[d.nk_window].groupby(["record", "lead"]):
            g = g.sort_values("k")
            fid += T.fsm(rec, L, g.anchor_nk, g.NKPR, Mn[(rec, L)] if cond == "gated" else None)
        for name, rows in (("Fiducial", fid), ("Learned-1L", T.learned_rows(frames(cond), cond, Mg if cond == "gated" else None))):
            al, c, m = T.score(rows)
            al.to_csv(OUT / f"alarms_{name}_{cond}.csv", index=False)
            r = dict(system=name, condition=cond, **m, coverage_eq4=c.pred_dur.sum() / c.gs_dur.sum(), coverage_merged_alarms=PR.merged_cov(al, False))
            if cond == "gated":
                tp = PR.per_lead_tp(al)
                r.update(TP_II=tp["II"][0], GS_II=tp["II"][1], TP_V2=tp["V2"][0], GS_V2=tp["V2"][1])
            res.append(r)
    R = pd.DataFrame(res)
    R.to_csv(OUT / "podium_published_folds_rows.csv", index=False)

    def row(s, c):
        return R[(R.system == s) & (R.condition == c)].iloc[0]
    lines, ok = [], True
    for cond, want in (("clean", (67, 34, 34)), ("noisy", (67, 43, 34)), ("gated", (65, 30, 36))):
        q = row("Fiducial", cond)
        got = (int(q.TP), int(q.FP), int(q.FN))
        ok &= got == want
        lines.append(f"{'PASS' if got == want else 'FAIL'}  Fiducial {PR.CONDN[cond]} = RESULTS_NK Phase A: {got} vs {want}")
    ep = []
    for mode, L in (("lead_II", "II"), ("lead_V2", "V2")):
        for k in range(1, 6):
            lg = (SAVE / rid(mode, k) / "job_log.txt").read_text()
            e = int(re.search(r"best_row_epoch: (\d+)", lg).group(1))
            v = float(re.search(r"best_valid_loss: ([0-9.eE+-]+)", lg).group(1))
            ep.append(dict(lead=L, fold=k, run_id=rid(mode, k), best_epoch=e, best_valid_loss=v))
    EP = pd.DataFrame(ep)
    EP.to_csv(OUT / "best_epochs.csv", index=False)
    before = MD.read_text()
    head = before.split(f"<!-- {TAGM} -->")[0].rstrip("\n")
    md = section(R, EP, lines)
    MD.write_text(head + "\n\n" + f"<!-- {TAGM} -->\n" + md.rstrip() + f"\n<!-- /{TAGM} -->\n")
    same = MD.read_text().startswith(head) and head.rstrip() == before.split(f"<!-- {TAGM} -->")[0].rstrip()
    lines.append(f"{'PASS' if same else 'FAIL'}  the existing .atr section of PODIUM_RESULTS.md is unchanged")
    ok &= same
    return finish(ok, lines, "score")


def section(R, EP, ctl):
    import podium_results as PR

    def row(s, c):
        return R[(R.system == s) & (R.condition == c)].iloc[0]
    o = ["## Published folds, NeuroKit gate (final podium configuration)", "",
         f"Generated {time.strftime('%F %T')} by `src/scripts/final/podium_published_folds.py` (VM). **Final podium configuration** (`docs/DECISIONS.md` 2026-11-24; "
         "decided before any result: this is what the podium talk and the corrected CMBEC paper use; thesis outputs are not affected).", "",
         "**Learned-1L (S1):** pretrained (`_lossfix` backbones), training seed S1, one model per lead, on the **published folds** `cv5_optB_fold01–05` with their published "
         "validation sets: POD_L1_lead_{II,V2}_S1_fold01/04/05 (Narval, `docs/reports/2026-11-25_podium_training.md`) and FIN_L1_lead_{II,V2}_S1_fold02/03 (already on the "
         "published folders). Everything else as the thesis runs. **Fiducial:** NeuroKit R-peaks and dwt delineation, 5-sample ST level at J+60 minus the PR-segment "
         "median, values rounded to 1e-6 mV. **Gate:** the paper's SQI from **NeuroKit R-peaks**, 6 dB, every gated row. Noise: NSTDB `em`, seed 42. Same PROSE FSM and "
         "scorer; union of II and V2, 101 episodes (Monitor: 99, copied from the paper); excluded record–lead pairs left out. The published V2 validation sets of folds 1, "
         "4 and 5 hold no V2 gold-standard episode (a property of the published folds).", "",
         "Controls: " + "; ".join(x[6:] for x in ctl) + ". Inference checks (`Tasks/final/runs/podium/published_folds/CHECKS_infer.txt`): each POD run reproduces its saved "
         "test MSE; folds 2–3 predictions equal the FIN S1 predictions; window counts equal the published runs; checkpoints match the copy-back list.", "",
         "### (a) Detection of ischemic episodes (paper Table 1 layout)", "",
         "| System | Condition | TP | FP | FN | Redundant | P | R | F1 | paper TP/FP/FN (P / R / F1) |", "|---|---|--:|--:|--:|--:|--:|--:|--:|---|"]
    mon = {"clean": (78, 66, 21), "noisy": (77, 171, 22), "gated": (76, 54, 23)}
    pf = {("Fiducial", "clean"): (.673, .653, .663), ("Fiducial", "noisy"): (.623, .653, .638), ("Fiducial", "gated"): (.727, .634, .677),
          ("Learned-1L", "clean"): (.652, .743, .694), ("Learned-1L", "noisy"): (.562, .723, .632), ("Learned-1L", "gated"): (.726, .733, .729)}
    for cond in ("clean", "noisy", "gated"):
        tp, fp, fn = mon[cond]
        p, r = tp / (tp + fp), tp / (tp + fn)
        o.append(f"| {'Monitor [1]' if cond == 'clean' else ''} | {PR.CONDN[cond]} | {tp} | {fp} | {fn} | n/a | {p:.3f} | {r:.3f} | {2 * p * r / (p + r):.3f} | same |")
    for s in ("Fiducial", "Learned-1L"):
        for cond in ("clean", "noisy", "gated"):
            q = row(s, cond)
            pp, x = PR.PAPER[s][cond], pf[(s, cond)]
            o.append(f"| {(s + (' (S1)' if s == 'Learned-1L' else '')) if cond == 'clean' else ''} | {PR.CONDN[cond]} | {int(q.TP)} | {int(q.FP)} | {int(q.FN)} | {int(q.Ignored)} | "
                     f"{q.precision:.3f} | {q.recall:.3f} | {q.F1:.3f} | {pp[0]}/{pp[1]}/{pp[2]} ({x[0]:.3f} / {x[1]:.3f} / {x[2]:.3f}) |")
    o += ["", "### (b) Temporal extent of the alarms (paper Table 2 layout)", "",
          "Sens. = TPt/(TPt+FNt); PPV = TPt/(TPt+FPt); IoU = TPt/(TPt+FPt+FNt); Cov. = (TPt+FPt)/(TPt+FNt) (paper eq. 4), per record–lead; Cov. merged = each record's "
          "II and V2 alarms merged into one timeline (the paper's 1.06 definition, checked in the section above). Paper values in brackets.", "",
          "| System | Condition | Sens. | PPV | IoU | Cov. | Cov. merged |", "|---|---|--:|--:|--:|--:|--:|"]
    pm_ = {("Learned-1L", "gated"): "1.06"}
    for s in ("Fiducial", "Learned-1L"):
        for cond in ("noisy", "gated"):
            q = row(s, cond)
            pp = PR.PAPER_T2[(s, cond)]
            o.append(f"| {(s + (' (S1)' if s == 'Learned-1L' else '')) if cond == 'noisy' else ''} | {PR.CONDN[cond]} | {q.sensitivity:.3f} ({pp[0]:.3f}) | {q.PPV_time:.3f} ({pp[1]:.3f}) | "
                     f"{q.IoU_whole_record:.3f} ({pp[2]:.3f}) | {q.coverage_eq4:.3f} ({pp[3]:.3f}) | {q.coverage_merged_alarms:.3f}" + (f" ({pm_[(s, cond)]})" if (s, cond) in pm_ else "") + " |")
    o += ["", "### (c) Gated TP per lead (NeuroKit gate)", "", "Each lead's alarms against that lead's episodes (69 on II, 32 on V2).", "",
          "| System | II TP | V2 TP | paper V2 |", "|---|--:|--:|---|"]
    for s, paper in (("Fiducial", "5 of 32"), ("Learned-1L", "22 of 32")):
        q = row(s, "gated")
        o.append(f"| {s}{' (S1)' if s == 'Learned-1L' else ''} | {int(q.TP_II)} of {int(q.GS_II)} | {int(q.TP_V2)} of {int(q.GS_V2)} | {paper} |")
    o += ["", "### (d) Learned-1L (S1) training: best epoch and best validation loss per fold", "",
          "Epochs 0–89 (0–19 head only, 20–89 full fine-tuning); validation loss = MSE (mV²) on the published validation set. Folds 2 and 3: the reused FIN runs.", "",
          "| Fold | II run | II best epoch | II best valid loss | V2 run | V2 best epoch | V2 best valid loss |", "|--:|---|--:|--:|---|--:|--:|"]
    for k in range(1, 6):
        a, b = EP[(EP.fold == k) & (EP.lead == "II")].iloc[0], EP[(EP.fold == k) & (EP.lead == "V2")].iloc[0]
        o.append(f"| {k} | {a.run_id} | {a.best_epoch} | {a.best_valid_loss:.6f} | {b.run_id} | {b.best_epoch} | {b.best_valid_loss:.6f} |")
    o += ["", "V2 fold 1 selects epoch 19, the last head-only epoch (as every FIN V2 fold-1 run did); the selection rule is unchanged."]
    return "\n".join(o) + "\n"


# ------------------------------------------------------------------ score_atr (user, 2026-11-25): sensitivity check, .atr gate
TAGA = "published_folds_atr"


def score_atr():
    """As score(), with the SQI gate from .atr beats at 6 dB (columns sqi_nk_atr / sqi_grid_atr, copied from step 5). Gated rows reported.
    Controls: Fiducial gated = 64/27/37 (the existing .atr section); Learned-1L baseline and artifact (no gate) = the final section (81/46/20, 79/56/22)."""
    import fin_step7_tier1 as T
    import podium_results as PR
    V5 = Path("/home/student/GIT/Tasks/final/runs/nk_detector/step5")
    V = {c: pd.read_csv(V5 / f"nk_values_{c}.csv.gz") for c in ("clean", "noisy42")}
    Mn, Mg = T.memos(V["noisy42"], "anchor_nk", "sqi_nk_atr"), T.memos(V["noisy42"], "anchor_grid", "sqi_grid_atr")

    def frames(cond):
        c = "clean" if cond == "clean" else "noisy42"
        return {L: pd.read_csv(pred_file(m, c), usecols=["record", "anchor_samp", f"pred_{L}_mV"]).rename(columns={f"pred_{L}_mV": "pred"}) for m, L in (("lead_II", "II"), ("lead_V2", "V2"))}
    res = {}
    for cond in ("clean", "noisy", "gated"):
        d = V["clean" if cond == "clean" else "noisy42"]
        fid = []
        for (rec, L), g in d[d.nk_window].groupby(["record", "lead"]):
            g = g.sort_values("k")
            fid += T.fsm(rec, L, g.anchor_nk, g.NKPR, Mn[(rec, L)] if cond == "gated" else None)
        for name, rows in (("Fiducial", fid), ("Learned-1L", T.learned_rows(frames(cond), cond, Mg if cond == "gated" else None))):
            al, c, m = T.score(rows)
            r = dict(system=name, condition=cond, **m, coverage_eq4=c.pred_dur.sum() / c.gs_dur.sum(), coverage_merged_alarms=PR.merged_cov(al, False))
            if cond == "gated":
                al.to_csv(OUT / f"alarms_{name}_gated_atr.csv", index=False)
                tp = PR.per_lead_tp(al)
                r.update(TP_II=tp["II"][0], GS_II=tp["II"][1], TP_V2=tp["V2"][0], GS_V2=tp["V2"][1])
            res[(name, cond)] = r
    R = pd.DataFrame(res.values())
    R.to_csv(OUT / "podium_published_folds_atr_rows.csv", index=False)
    lines, ok = [], True
    for (name, cond), want in ((("Fiducial", "gated"), (64, 27, 37)), (("Learned-1L", "clean"), (81, 46, 20)), (("Learned-1L", "noisy"), (79, 56, 22))):
        q = res[(name, cond)]
        got = (int(q["TP"]), int(q["FP"]), int(q["FN"]))
        ok &= got == want
        lines.append(f"{'PASS' if got == want else 'FAIL'}  {name} {PR.CONDN[cond]} = {'the existing .atr section' if name == 'Fiducial' else 'the final section'}: {got} vs {want}")
    o = ["## Published folds, .atr gate (sensitivity check)", "",
         f"Generated {time.strftime('%F %T')} by `src/scripts/final/podium_published_folds.py score_atr` (VM). Same as the final section above (Learned-1L S1 on the "
         "published folds; Fiducial on NeuroKit delineation, 5 samples, rounded; noise seed 42), except the SQI gate: `.atr` beats at 6 dB instead of NeuroKit R-peaks. "
         "Gated rows only.", "", "Controls: " + "; ".join(x[6:] for x in lines) + ".", "",
         "| System | TP | FP | FN | Redundant | P | R | F1 | Sens. | PPV | IoU | Cov. | Cov. merged | II TP | V2 TP |",
         "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for name in ("Fiducial", "Learned-1L"):
        q = res[(name, "gated")]
        o.append(f"| {name}{' (S1)' if name == 'Learned-1L' else ''} | {q['TP']} | {q['FP']} | {q['FN']} | {q['Ignored']} | {q['precision']:.3f} | {q['recall']:.3f} | {q['F1']:.3f} | "
                 f"{q['sensitivity']:.3f} | {q['PPV_time']:.3f} | {q['IoU_whole_record']:.3f} | {q['coverage_eq4']:.3f} | {q['coverage_merged_alarms']:.3f} | "
                 f"{q['TP_II']} of {q['GS_II']} | {q['TP_V2']} of {q['GS_V2']} |")
    before = MD.read_text()
    assert f"<!-- /{TAGM} -->" in before
    head = before.split(f"<!-- {TAGA} -->")[0].rstrip("\n")
    MD.write_text(head + "\n\n" + f"<!-- {TAGA} -->\n" + "\n".join(o) + f"\n<!-- /{TAGA} -->\n")
    same = MD.read_text().startswith(head + "\n") and before.startswith(head)
    lines.append(f"{'PASS' if same else 'FAIL'}  everything above this section in PODIUM_RESULTS.md is unchanged")
    ok &= same
    return finish(ok, lines, "score_atr")


if __name__ == "__main__":
    st = sys.argv[1]
    ok = infer() if st == "infer" else score() if st == "score" else score_atr() if st == "score_atr" else sys.exit(f"unknown stage {st}")
    sys.exit(0 if ok else 1)
