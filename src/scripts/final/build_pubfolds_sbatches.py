#!/usr/bin/env python3
"""Sbatch files for the published-folds extension of the podium retraining (docs/DECISIONS.md 2026-11-25). Nothing is submitted.

  pubfolds  21 jobs  the remaining pretrained runs on the PUBLISHED folds phase1_folds/cv5_optB_fold01/04/05:
                     POD_L1_lead_{II,V2}_{S2,S3}_fold0{1,4,5}   (12; _lossfix backbones)
                     POD_L2_two_lead_{S1,S2,S3}_fold0{1,4,5}    (9;  patch_ecg_unified_ii_v2_rebuilt, unaffected by the loss fix)
                     Not regenerated: POD_L1_lead_{II,V2}_S1_fold0{1,4,5} (build_podium_sbatches.py, trained 2026-11-25).
                     Not retrained: folds 2 and 3 (FIN_* runs already on the published folders cv5_optB_fold02/03).

Each job is its FIN twin FIN_<pipeline>_<mode>_<seed>_fold0X rendered by build_final_sbatches.render_finetune (imported, not changed), with exactly
the changes of the podium jobs: data folder cv5_optB_leadV2_opt2p_fold0X -> cv5_optB_fold0X, run ID FIN_ -> POD_, one backbone sha256 guard line.
Check: for L1 S1 this render is byte-identical to build_podium_sbatches.render (the 6 approved, trained podium jobs).
Overrides: as build_final_sbatches.py (FIN_REPO_ROOT, FIN_DS_ROOT, FIN_ACCOUNT, FIN_GPU, FIN_WALLTIME), plus FIN_OUT_PUB (output folder).
Usage: python3 src/scripts/final/build_pubfolds_sbatches.py   then   bash <FIN_OUT_PUB>/submit_pubfolds.sh pubfolds"""
import csv
import difflib
import os

import build_podium_sbatches as P  # noqa: E402  (same folder; it puts custom_dataset on sys.path and imports build_final_sbatches)

B = P.B
OUT = os.environ.get("FIN_OUT_PUB", f"{B.REPO_ROOT}/logs/sbatch_scripts/pubfolds")
GROUP = "pubfolds"
FOLDS = (1, 4, 5)
SEEDS = ("S1", "S2", "S3")
MODES = (("L1", "lead_II"), ("L1", "lead_V2"), ("L2", "two_lead"))
BB_SHA = dict(P.BB_SHA, two_lead="f3554f1436929b1e588df5bc33de1813e156f2755c738d4c877b4616b958a559")  # FIN_L2_two_lead_S*_fold0X/job_log.txt


def guard(pipe, mode):
    bb = B.backbone(pipe, mode)
    return f'[[ "$(sha256sum "{bb}" | cut -d" " -f1)" == "{BB_SHA[mode]}" ]] || {{ echo "ERROR: backbone sha256 differs from the FIN twin: {bb}" >&2; exit 2; }}\n'


def render(pipe, mode, s, k):
    twin_name, twin = B.render_finetune(pipe, mode, s, k)
    name = twin_name.replace("FIN_", "POD_", 1)
    old_data, new_data = B.dataset(mode, k), P.published(k)
    assert old_data.endswith(f"cv5_optB_leadV2_opt2p_fold{k:02d}") and twin.count(old_data) == 5, (twin_name, twin.count(old_data))
    body = twin.replace(old_data, new_data).replace(twin_name, name)
    anchor = f'[[ -f "{B.backbone(pipe, mode)}" ]]'
    i = body.index(anchor)
    i = body.index("\n", i) + 1
    body = body[:i] + guard(pipe, mode) + body[i:]
    return name, body, twin_name, twin


def changed_lines(body, twin):
    d = difflib.unified_diff(twin.splitlines(), body.splitlines(), lineterm="", n=0)
    return [x for x in d if x[:1] in "+-" and not x.startswith(("+++", "---"))]


def main():
    # the approved podium jobs come out of this render unchanged
    for mode in ("lead_II", "lead_V2"):
        for k in FOLDS:
            assert render("L1", mode, "S1", k)[:2] == P.render(mode, k)[:2], (mode, k)
    d = os.path.join(OUT, GROUP)
    os.makedirs(d, exist_ok=True)
    rows = []
    for pipe, mode in MODES:
        for s in SEEDS:
            if pipe == "L1" and s == "S1":
                continue  # POD_L1_*_S1 already trained (podium)
            for k in FOLDS:
                name, body, twin_name, twin = render(pipe, mode, s, k)
                back = body.replace(name, twin_name).replace(P.published(k), B.dataset(mode, k))
                assert changed_lines(back, twin) == ["+" + guard(pipe, mode).rstrip("\n")], name
                assert f"--seed={B.seed(s, k)} " in body and "--n_epochs_finetune_head=20" in body and "--n_epochs_finetune=70" in body
                assert "--model_selection_metric=valid_loss" in body and f'--model_name="{name}"' in body and "opt2p" not in body
                with open(os.path.join(d, f"{name}.sbatch"), "w") as f:
                    f.write(body)
                rows.append(dict(group=GROUP, pipeline=pipe, mode=mode, training_seed=s, seed=B.seed(s, k), fold=k, run_id=name, twin=twin_name,
                                 dataset=P.published(k), backbone=B.backbone(pipe, mode), backbone_sha256=BB_SHA[mode], head_epochs=20, full_epochs=70))
    assert len(rows) == 21 and len({r["run_id"] for r in rows}) == 21
    with open(os.path.join(OUT, "jobs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(OUT, "submit_pubfolds.sh"), "w") as f:
        f.write(B.SUBMIT % B.REPO_ROOT)
    print(f"{len(rows)} jobs written to {d}; jobs.csv and submit_pubfolds.sh in {OUT}. Reused: POD_L1_*_S1_fold0{{1,4,5}}, FIN_*_fold02/03. Nothing submitted.")


if __name__ == "__main__":
    main()
