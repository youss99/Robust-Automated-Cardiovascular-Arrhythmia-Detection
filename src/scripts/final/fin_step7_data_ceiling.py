#!/usr/bin/env python3
"""Final runs, step 7 extras (user, 2026-11-22): data table, gold-standard ceiling, condition relabel of the .tex tables. One stage per call.

data     Pre-training set size per backbone (Unified v2 rows per lead, train / validation from the copied-back pretrain_split.csv; the two-lead
         backbone's count from the cluster report docs/reports/2026-11-20_step0_0b.md, its split file is not on the VM). Per fold: train / validation /
         test / unused records, windows (rows of the fold's dataset CSV, the rows the trainer used; every row has both targets), windows per lead
         without masked pairs, and gold-standard episodes per lead (101 = 69 II + 32 V2; none lies on a masked pair).
ceiling  The true window labels (set A: median .16a ee per 10 s grid window, anchors 2500k + 2499) through the same FSM, baseline condition (no
         noise, no gate), masked pairs out, union of II and V2, scored as T5. Control: equals phase 1's oracle union (Tasks/phase1/runs/_scores/oracle,
         TP/FP/FN/Ignored 100/9/1/3).
relabel  Condition names in docs/thesis_tables/*.tex: clean -> baseline, noisy -> artifact (cells, captions, notes). Exact replacements, each
         expected at least once where listed; afterwards no 'clean' / 'noisy' remains outside file names and labels. Idempotent: rerun after any
         table is regenerated (the generator scripts still write the old words).
Output: Tasks/final/runs/step7_extra/, docs/RESULTS_FINAL.md (data, ceiling blocks), docs/thesis_tables/T0_*.tex, T5b_gold_standard_ceiling.tex.
Usage: <repo .venv>/bin/python -u -W ignore fin_step7_data_ceiling.py data|ceiling|relabel"""
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))  # the other final drivers
import fin_step7_tier1 as T  # noqa: E402
import fin_step7_rest as Rr  # noqa: E402

Rr.SCRIPT = "src/scripts/final/fin_step7_data_ceiling.py"

P, K = T.P, T.K
REPO = P.REPO
OUT = Path("/home/student/GIT/Tasks/final/runs/step7_extra")
LEADS, MASK = T.LEADS, T.MASK
BB = REPO / "results/pre-train/saved_models/unified"
FOLDS = {1: "cv5_optB_leadV2_opt2p_fold01", 2: "cv5_optB_fold02", 3: "cv5_optB_fold03", 4: "cv5_optB_leadV2_opt2p_fold04", 5: "cv5_optB_leadV2_opt2p_fold05"}
SPLIT = {1: "train", 3: "validation", 2: "test", -1: "unused"}
ORDER = ["train", "validation", "test", "unused"]


def tex_write(name, caption, label, colspec, header, body, note):
    Rr.X.tex_write(name, caption, label, colspec, header, body, note)
    f = T.TEX / f"{name}.tex"
    f.write_text(f.read_text().replace("fin_step7_t10_fig1.py", "fin_step7_data_ceiling.py", 1))


# ------------------------------------------------------------------ data
def data():
    pre = []
    for name, leads, used in (("patch_ecg_unified_ii_lossfix", "II", "L1 lead II model"), ("patch_ecg_unified_v2_lossfix", "V2", "L1 lead V2 model")):
        s = pd.read_csv(BB / name / "pretrain_split.csv")
        src = s.filename.str.extract(r"100Hz/([^/]+)/")[0]
        ds = (BB / name / "data_source.txt").read_text()
        n_ds = int(re.search(r"rows trained on \(train \+ validation\): (\d+)", ds).group(1))
        assert n_ds == len(s), (name, n_ds, len(s))
        pre.append(dict(backbone=name, leads=leads, used_by=used, rows=len(s), train=int((s.strat_fold_unannotated == 1).sum()), validation=int((s.strat_fold_unannotated == 3).sum()),
                        chapman=int((src == "chapman").sum()), cinc2020=int((src == "cinc-2020").sum()), source="pretrain_split.csv + data_source.txt (VM copy)"))
    rep = (REPO / "docs/reports/2026-11-20_step0_0b.md").read_text()
    m = re.search(r"\| II,V2 \| ([\d,]+) of ([\d,]+) \| ([\d,]+) \(match\) \|", rep)
    pre.append(dict(backbone="patch_ecg_unified_ii_v2_rebuilt", leads="II + V2", used_by="L2", rows=int(m.group(3).replace(",", "")), train=46274, validation=11569,
                    chapman=39415, cinc2020=18428, source="Narval read-only check 2026-11-22 (pretrain_split.csv, data_source.txt; best.pt sha256 f3554f14...)"))
    assert pre[-1]["train"] + pre[-1]["validation"] == pre[-1]["rows"] == pre[-1]["chapman"] + pre[-1]["cinc2020"]
    PRE = pd.DataFrame(pre)
    total_v2 = int(re.search(r"\| II \| [\d,]+ of ([\d,]+) \|", rep).group(1).replace(",", ""))
    gs = K.load_gs(T.T1.GS_PATH)
    ep = gs.groupby(["record", "lead"]).size()
    assert not any((r, l) in MASK for r, l in zip(gs.record, gs.lead))
    rows = []
    for k, folder in FOLDS.items():
        md = pd.read_csv(REPO / "phase1_folds" / folder / "modified_dataset/modified_dataset.csv", usecols=["patient_id", "strat_fold_annotated", "target_lead0_mV", "target_lead1_mV"])
        assert md.target_lead0_mV.notna().all() and md.target_lead1_mV.notna().all()
        for code, g in md.groupby("strat_fold_annotated"):
            recs = sorted(g.patient_id.unique())
            r = dict(fold=k, dataset=folder, split=SPLIT[int(code)], n_records=len(recs), records=" ".join(recs), windows=len(g))
            for L in LEADS:
                keep = ~g.patient_id.map(lambda x: (x, L) in MASK)
                r[f"windows_{L}_no_masked"] = int(keep.sum())
                r[f"masked_{L}"] = " ".join(x for x in recs if (x, L) in MASK)
                r[f"episodes_{L}"] = int(sum(ep.get((x, L), 0) for x in recs))
            rows.append(r)
    F = pd.DataFrame(rows)
    F["split"] = pd.Categorical(F.split, ORDER, ordered=True)
    F = F.sort_values(["fold", "split"]).reset_index(drop=True)
    # checks: every fold's test sets partition the 15 records; episodes per fold sum to 69 / 32; windows per fold sum to the dataset rows
    chk = []
    tests = sorted(" ".join(F[F.split == "test"].records).split())
    chk.append(("test records over the 5 folds = the 15 scored records, each once", tests == sorted(K.RECORDS_15)))
    for k, g in F.groupby("fold"):
        chk.append((f"fold {k}: episodes over splits = 69 II / 32 V2", int(g.episodes_II.sum()) == 69 and int(g.episodes_V2.sum()) == 32))
        chk.append((f"fold {k}: 16 records over splits", int(g.n_records.sum()) == 16))
    chk.append(("pre-training total rows in the cluster report = 58,942 (Unified v2)", total_v2 == 58942))
    OUT.mkdir(parents=True, exist_ok=True)
    PRE.to_csv(OUT / "data_pretraining.csv", index=False)
    F.to_csv(OUT / "data_folds.csv", index=False)
    lines = [f"{'PASS' if ok else 'FAIL'}  {t}" for t, ok in chk]
    (OUT / "DATA_CHECKS.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    ok = all(o for _, o in chk)
    # markdown
    out = Rr.head("## Data", "Pre-training: Unified v2 (`Unified_Dataset_No_PTB_Segment_v2`, 58,942 10 s windows at 100 Hz from Chapman and CinC 2020, md5 `ac56fd0e…`); "
                  "rows where a selected lead is blank are removed. Fine-tuning: the fold dataset CSVs (`phase1_folds/`), one split for every learned model; "
                  "a window is a 10 s segment of the median LTST signal with both targets (every row has both). Masked record–lead pairs are in the training "
                  "rows (their targets are present) and are excluded from scoring; gold-standard episodes: 69 lead II + 32 lead V2 = 101, none on a masked pair.")
    out += ["### Pre-training set", "", "| backbone | leads | used by | rows | train | validation | Chapman | CinC 2020 |", "|---|---|---|--:|--:|--:|--:|--:|"]
    fmt = (lambda v: "–" if pd.isna(v) else f"{int(v):,}")
    for r in PRE.itertuples():
        out.append(f"| `{r.backbone}` | {r.leads} | {r.used_by} | {fmt(r.rows)} | {fmt(r.train)} | {fmt(r.validation)} | {fmt(r.chapman)} | {fmt(r.cinc2020)} |")
    out += ["", "The two-lead backbone's counts are from a read-only check on Narval (2026-11-22: its `pretrain_split.csv` and `data_source.txt`, `_best.pt` sha256 `f3554f14…`; the same check reproduced the two single-lead rows above). It holds the same 57,843 windows as the V2 backbone (lead II is never blank, so the II + V2 filter equals the V2 filter), with its own random 80/20 train / validation draw: the totals per split and per source are equal, the assignment differs (39,179 of 57,843 windows, 68 %, share a split; Narval's count).", "",
            "### Fine-tuning folds", "",
            "| fold | dataset | split | records | windows | II windows (no masked) | V2 windows (no masked) | II episodes | V2 episodes | record IDs (masked: II / V2) |",
            "|--:|---|---|--:|--:|--:|--:|--:|--:|---|"]
    for r in F.itertuples():
        mk = f" (masked: {r.masked_II or '–'} / {r.masked_V2 or '–'})" if (r.masked_II or r.masked_V2) else ""
        out.append(f"| {r.fold} | `{r.dataset}` | {r.split} | {r.n_records} | {r.windows:,} | {r.windows_II_no_masked:,} | {r.windows_V2_no_masked:,} | {r.episodes_II} | {r.episodes_V2} | {r.records}{mk} |")
    out += ["", "Checks: " + "; ".join(f"{t}: {'yes' if o else 'NO'}" for t, o in chk) + ".",
            "Note: the CSVs hold the windows the trainer used (step 6 reproduced every saved test MSE on them); the prediction grid of the trainer's H5 has 263 more "
            "10 s segments over the 15 records (119,126 vs 118,863), which have no CSV row. The scored set A grid (T1) is a separate 10 s grid from sample 0."]
    Rr.X.upsert("data", "\n".join(out))
    # tex
    body = [f"{r.leads} & {r.used_by} & {fmt(r.rows).replace(',', '{,}')} & {fmt(r.train).replace(',', '{,}')} & {fmt(r.validation).replace(',', '{,}')}" for r in PRE.itertuples()]
    body = [b.replace("–", "--") for b in body]
    tex_write("T0_pretraining_data", "Self-supervised pre-training data (Unified v2).", "tab:t0_pretraining",
              "llrrr", "Leads & Used by & Windows & Train & Validation", body,
              "Unified v2: 58{,}942 10\\,s windows at 100\\,Hz from Chapman and CinC 2020; windows with a blank selected lead removed. "
              "The II + V2 backbone uses the same windows as the V2 backbone with its own random 80/20 train/validation draw.")
    body = []
    for k, g in F.groupby("fold"):
        if k > 1:
            body.append("MIDRULE")
        for j, r in enumerate(g.itertuples()):
            body.append(f"{k if j == 0 else ''} & {r.split.capitalize()} & {r.n_records} & {r.windows:,} & {r.windows_II_no_masked:,} & {r.windows_V2_no_masked:,} & {r.episodes_II} & {r.episodes_V2}".replace(",", "{,}").replace("{,} ", ", "))
    notes = "; ".join(f"fold {k}: validation {g[g.split == 'validation'].records.iloc[0].replace('s', '')}, test {g[g.split == 'test'].records.iloc[0].replace('s', '')}"
                      + (f", unused {g[g.split == 'unused'].records.iloc[0].replace('s', '')}" if (g.split == "unused").any() else "") for k, g in F.groupby("fold"))
    tex_write("T0_fold_data", "Fine-tuning and evaluation data per fold (LTST, 16 records).", "tab:t0_folds",
              "llrrrrrr", "Fold & Split & Records & Windows & II w/o masked & V2 w/o masked & II episodes & V2 episodes", body,
              "Windows: 10\\,s segments with both targets (the rows the trainer used); masked record--lead pairs stay in training and are excluded from scoring "
              "(columns w/o masked). Episodes: gold-standard ST episodes per lead (69 II + 32 V2), none on a masked pair. Record IDs (LTST s-numbers): " + notes.replace("_", "\\_") + ".")
    return ok


# ------------------------------------------------------------------ ceiling
def ceiling():
    O = pd.read_csv(T.ORACLE).query("condition == 'clean'")
    rows = []
    for L in LEADS:
        for rec, g in O.groupby("record"):
            if (rec, L) in MASK:
                continue
            g = g.sort_values("anchor_samp")
            rows += T.fsm(rec, L, g.anchor_samp, g[f"oracle_{L}_mV"], None)
    al, c, met = T.score(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    al.to_csv(OUT / "ceiling_alarms_baseline.csv", index=False)
    c.to_csv(OUT / "ceiling_components_per_record.csv", index=False)
    got = (int(met["TP"]), int(met["FP"]), int(met["FN"]), int(met["Ignored"]))
    ref = pd.read_csv(P.SC / "oracle" / "alarms_clean.csv").query("run == 'oracle' and scheme == 'grid' and scope == 'FULL' and mode == 'union'").iloc[0]
    want = (int(ref.TP), int(ref.FP), int(ref.FN), int(ref.Ignored))
    ok = got == want and np.isclose(met["F1"], ref.F1)
    line = f"{'PASS' if ok else 'FAIL'}  ceiling (baseline, union) equals phase 1's oracle union: {got} vs {want}, F1 {met['F1']:.6f} vs {ref.F1:.6f}"
    (OUT / "CEILING_CHECK.txt").write_text(line + "\n")
    print(line)
    pd.DataFrame([dict(condition="baseline", **met)]).to_csv(OUT / "ceiling_metrics.csv", index=False)
    R5 = pd.read_csv(T.OUT / "T5_alarms_per_seed.csv").query("condition == 'clean'")
    out = Rr.head("## Gold-standard ceiling (baseline)", "The true window labels (set A: median `.16a` `ee` of each 10 s grid window, anchors 2500k + 2499) through the same "
                  "FSM and scorer as T5, baseline condition (no noise, no gate), masked pairs excluded, union of II and V2. It is the score of a perfect window-level "
                  f"estimator under this FSM. Control: equals phase 1's oracle union ({'yes' if ok else 'NO'}).")
    out += ["| pipeline | TP | FP | FN | precision | recall | F1 | IoU | false-alarm h | duration Se | duration +P | redundant |", "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
            f"| gold-standard labels | {got[0]} | {got[1]} | {got[2]} | {met['precision']:.3f} | {met['recall']:.3f} | {met['F1']:.3f} | {met['IoU_whole_record']:.3f} | "
            f"{met['false_hours']:.1f} | {met['sensitivity']:.3f} | {met['PPV_time']:.3f} | {got[3]} |"]
    for pipe in ("NK-PR", "NK-FP", "L1", "L2"):
        q = R5[R5.pipeline == pipe]
        out.append(f"| {pipe} (T5) | {T.pm(q.TP, 1)} | {T.pm(q.FP, 1)} | {T.pm(q.FN, 1)} | {T.pm(q.precision)} | {T.pm(q.recall)} | {T.pm(q.F1)} | {T.pm(q.IoU_whole_record)} | "
                   f"{T.pm(q.false_hours, 1)} | {T.pm(q.sensitivity)} | {T.pm(q.PPV_time)} | {T.pm(q.Ignored, 1)} |")
    out.append("\nWith the true labels as input, the 1 missed episode and 9 false alarms reflect differences between the FSM on 10 s window medians and "
               "the way the gold-standard episodes were built; their cause was not examined.")
    Rr.X.upsert("ceiling", "\n".join(out))
    body = [f"Gold-standard labels & {got[0]} & {got[1]} & {got[2]} & {met['precision']:.3f} & {met['recall']:.3f} & {met['F1']:.3f} & {met['IoU_whole_record']:.3f} & "
            f"{met['false_hours']:.1f} & {met['sensitivity']:.3f} & {met['PPV_time']:.3f} & {got[3]}", "MIDRULE"]
    for pipe in ("NK-PR", "NK-FP", "L1", "L2"):
        q = R5[R5.pipeline == pipe]
        body.append(f"{pipe} & {T.tpm(q.TP, 1)} & {T.tpm(q.FP, 1)} & {T.tpm(q.FN, 1)} & {T.tpm(q.precision)} & {T.tpm(q.recall)} & {T.tpm(q.F1)} & {T.tpm(q.IoU_whole_record)} & "
                    f"{T.tpm(q.false_hours, 1)} & {T.tpm(q.sensitivity)} & {T.tpm(q.PPV_time)} & {T.tpm(q.Ignored, 1)}")
    tex_write("T5b_gold_standard_ceiling", "Alarm ceiling: gold-standard window labels through the alarm FSM, baseline condition.", "tab:t5b_ceiling",
              "lrrrrrrrrrrr", "Estimate & TP & FP & FN & Precision & Recall & F1 & IoU & FA (h) & Dur.\\ Se & Dur.\\ +P & Redundant", body,
              "Gold-standard labels: median expert ST deviation of each 10\\,s window, through the same PROSE FSM and scorer; baseline condition, no gate; "
              "union of lead II and V2 alarms, 101 episodes, 15 records, masked pairs excluded. Pipeline rows repeated from Table T5 (baseline). "
              "L1, L2: mean $\\pm$ SD over three training seeds.")
    return ok


# ------------------------------------------------------------------ relabel
REPL = [  # (file glob, old, new): exact strings
    ("T5_alarms_main.tex", "alarms: clean, noisy and gated (T5)", "alarms: baseline, artifact and gated (T5)"),
    ("T5_alarms_main.tex", "\nClean & MON", "\nBaseline & MON"),
    ("T5_alarms_main.tex", "\nNoisy & MON", "\nArtifact & MON"),
    ("T5_alarms_main.tex", "Noisy: NSTDB em, seed 42; gated:", "Artifact: NSTDB em, seed 42; gated:"),
    ("T10_ablations_alarms.tex", " & Clean & ", " & Baseline & "),
    ("T10_ablations_alarms.tex", " & Noisy & ", " & Artifact & "),
    ("T10_ablations_alarms.tex", "; noisy: NSTDB em, seed 42;", "; artifact: NSTDB em, seed 42;"),
    ("*.tex", "clean signal", "baseline condition"),
    ("*.tex", "noisy predictions against clean labels", "predictions in the artifact condition against baseline labels"),
    ("*.tex", "noisy estimates against clean labels", "estimates in the artifact condition against baseline labels"),
    ("T4_noise_robustness.tex", "error on the noisy signal, ungated and gated", "error in the artifact condition, ungated and gated"),
]
MUST = {r[:2] for r in REPL if r[0] != "*.tex"} | {("*.tex", "clean signal"), ("*.tex", "noisy predictions against clean labels"), ("*.tex", "noisy estimates against clean labels")}


def relabel():
    files = sorted(T.TEX.glob("*.tex"))
    hits = {}
    for f in files:
        s = f.read_text()
        for pat, old, new in REPL:
            if pat != "*.tex" and f.name != pat:
                continue
            n = s.count(old)
            hits[(pat, old)] = hits.get((pat, old), 0) + n
            s = s.replace(old, new)
        f.write_text(s)
    left = []
    for f in files:
        for i, line in enumerate(f.read_text().splitlines(), 1):
            body = line if not line.startswith("%") else ""
            body = re.sub(r"\\label\{[^}]*\}", "", body)
            if re.search(r"(?i)\bclean\b|\bnoisy\b", body):
                left.append(f"{f.name}:{i}")
    lines = [f"{'ok  ' if n else 'NONE'}  {n:3d}  {pat}: {old!r}" for (pat, old), n in hits.items()]
    already = all(("Baseline" in (T.TEX / "T5_alarms_main.tex").read_text(), "Artifact" in (T.TEX / "T5_alarms_main.tex").read_text()))
    missing = [k for k in MUST if hits.get(k, 0) == 0]
    ok = not left and (not missing or already)
    lines.append(f"{'PASS' if not left else 'FAIL'}  no 'clean' / 'noisy' left outside comments and labels: {left}")
    lines.append(f"{'PASS' if (not missing or already) else 'FAIL'}  every listed replacement applied (or already applied): missing {missing}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "RELABEL.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return ok


if __name__ == "__main__":
    st = sys.argv[1]
    ok = {"data": data, "ceiling": ceiling, "relabel": relabel}[st]()
    (OUT / (f"DONE_{st}" if ok else f"FAILED_{st}")).write_text(time.strftime("%F %T") + "\n")
    sys.exit(0 if ok else 1)
