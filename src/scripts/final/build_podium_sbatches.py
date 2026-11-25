#!/usr/bin/env python3
"""Sbatch files for the podium-only retraining (docs/DECISIONS.md 2026-11-24, podium-only exception). Nothing is submitted.

  podium  6 jobs  POD_L1_lead_{II,V2}_S1_fold0{1,4,5}: Learned-1L (pretrained, _lossfix backbones), training seed S1, on the PUBLISHED folds
                  phase1_folds/cv5_optB_fold01/04/05 (published validation sets). Folds 2 and 3 are not retrained: FIN_L1_lead_{II,V2}_S1_fold02/03
                  already ran on the published folders cv5_optB_fold02/03 (same CSV sha256 as phase1_folds/SHA256SUMS.txt) and are reused.

Each job is its FIN twin FIN_L1_<mode>_S1_fold0X, rendered by build_final_sbatches.render_finetune (imported, not changed), with exactly these changes:
  - data folder: cv5_optB_leadV2_opt2p_fold0X -> cv5_optB_fold0X (4 lines: the H5 check, the dataset and CSV sha256 lines of job_log.txt, the staging copy);
  - run ID: FIN_L1_ -> POD_L1_ (model_name, run folder, Slurm job name and log);
  - one added guard line: the job stops unless the backbone's sha256 is the one the FIN twin logged (ii b15f12ef..., v2 a62cb3ec...).
Everything else (seed 200 + fold - 1, 20 head-only + 70 full epochs, optimizer, flags, best-validation-loss selection, walltime, GPU, account,
staging, snapshot guard, job_log.txt with best epoch and best validation loss) is the twin's; asserted for every job below.
Overrides: as build_final_sbatches.py (FIN_REPO_ROOT, FIN_DS_ROOT, FIN_ACCOUNT, FIN_GPU, FIN_WALLTIME), plus FIN_OUT_POD (output folder).
Usage: python3 src/scripts/final/build_podium_sbatches.py   then   bash <FIN_OUT_POD>/submit_podium.sh podium"""
import csv
import difflib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "custom_dataset"))
import build_final_sbatches as B  # noqa: E402  (module level only defines constants and functions)

OUT = Path(os.environ.get("FIN_OUT_POD", f"{B.REPO_ROOT}/logs/sbatch_scripts/podium"))
GROUP = "podium"
FOLDS = (1, 4, 5)
MODES = ("lead_II", "lead_V2")
BB_SHA = {"lead_II": "b15f12efc615a860808b187c9ab9bd6ec9ed7fc0dffb44eab752d678d578bce3",   # FIN_L1_lead_II_S1_fold0X/job_log.txt
          "lead_V2": "a62cb3ec47ed9d5322a12fd172f6f6c33c7476b954fa1cf1ce2d281ff86d75e8"}   # FIN_L1_lead_V2_S1_fold0X/job_log.txt


def published(k):
    return f"{B.DS_ROOT}/cv5_optB_fold{k:02d}"


def guard(mode):
    bb = B.backbone("L1", mode)
    return f'[[ "$(sha256sum "{bb}" | cut -d" " -f1)" == "{BB_SHA[mode]}" ]] || {{ echo "ERROR: backbone sha256 differs from the FIN twin: {bb}" >&2; exit 2; }}\n'


def render(mode, k):
    twin_name, twin = B.render_finetune("L1", mode, "S1", k)
    name = twin_name.replace("FIN_L1_", "POD_L1_")
    old_data, new_data = B.dataset(mode, k), published(k)
    assert old_data.endswith(f"cv5_optB_leadV2_opt2p_fold{k:02d}") and twin.count(old_data) == 5, (twin_name, twin.count(old_data))  # 5 mentions on 4 lines: H5 check (+ its error text), dataset, CSV sha256, staging copy
    body = twin.replace(old_data, new_data).replace(twin_name, name)
    anchor = f'[[ -f "{B.backbone("L1", mode)}" ]]'
    i = body.index(anchor)
    i = body.index("\n", i) + 1
    body = body[:i] + guard(mode) + body[i:]
    return name, body, twin_name, twin


def changed_lines(body, twin):
    d = difflib.unified_diff(twin.splitlines(), body.splitlines(), lineterm="", n=0)
    return [x for x in d if x[:1] in "+-" and not x.startswith(("+++", "---"))]


def main():
    d = OUT / GROUP
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    for mode in MODES:
        for k in FOLDS:
            name, body, twin_name, twin = render(mode, k)
            # the job with the run ID and data folder put back is the twin plus the one guard line
            back = body.replace(name, twin_name).replace(published(k), B.dataset(mode, k))
            ch = changed_lines(back, twin)
            assert ch == ["+" + guard(mode).rstrip("\n")], (name, ch)
            assert f"--seed={B.seed('S1', k)} " in body and "--n_epochs_finetune_head=20" in body and "--n_epochs_finetune=70" in body
            assert "--model_selection_metric=valid_loss" in body and f'--model_name="{name}"' in body and "opt2p" not in body
            (d / f"{name}.sbatch").write_text(body)
            rows.append(dict(group=GROUP, pipeline="L1", mode=mode, training_seed="S1", seed=B.seed("S1", k), fold=k, run_id=name, twin=twin_name,
                             dataset=published(k), backbone=B.backbone("L1", mode), backbone_sha256=BB_SHA[mode], head_epochs=20, full_epochs=70))
    assert len(rows) == 6 and len({r["run_id"] for r in rows}) == 6
    with open(OUT / "jobs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")   # LF (.gitattributes text=auto)
        w.writeheader()
        w.writerows(rows)
    (OUT / "submit_podium.sh").write_text(B.SUBMIT % B.REPO_ROOT)
    print(f"{len(rows)} jobs written to {d}; jobs.csv and submit_podium.sh in {OUT}. Folds 2 and 3 reuse FIN_L1_lead_{{II,V2}}_S1_fold02/03. Nothing submitted.")


if __name__ == "__main__":
    main()
