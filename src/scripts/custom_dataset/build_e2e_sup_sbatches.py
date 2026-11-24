#!/usr/bin/env python3
"""Sbatch files for step3e: end-to-end supervised baselines without the head-only phase (docs/DECISIONS.md, 2026-11-24). Nothing is submitted.

  step3e 45 jobs  L1supE2E (lead_II, lead_V2) and L2supE2E (two_lead), supervised (no backbone), 5 folds x S1, S2, S3.

Each job is its step3/step3o twin (FIN_L1sup_* / FIN_L2sup_*), rendered by build_final_sbatches.render_finetune (imported, not changed),
with exactly two differences:
  - epochs: --n_epochs_finetune_head=0 --n_epochs_finetune=90 (twin: 20 head-only + 70 full). With 0 head epochs, Learner.fine_tune skips
    freeze() and the checkpoint reload, and trains the whole network for epochs 0-89 (src/core/learner.py, fine_tune);
  - run ID: FIN_L1supE2E_<mode>_<S>_fold<k> / FIN_L2supE2E_two_lead_<S>_fold<k> (own run folder, own log).
Everything else (seed, data folder, flags, walltime, GPU, staging, job_log.txt, snapshot guard) is the twin's. The script checks this for
every job by comparing the rendered text with the twin's, after replacing only those two things.
The existing generator, its 137 jobs and its jobs.csv (sha256 2133e920...) are untouched: this writes to its own folder.
Overrides: as build_final_sbatches.py (FIN_REPO_ROOT, FIN_DS_ROOT, FIN_ACCOUNT, FIN_GPU, FIN_WALLTIME), plus FIN_OUT_E2E (output folder).
Usage: python3 src/scripts/custom_dataset/build_e2e_sup_sbatches.py   then   bash <FIN_OUT_E2E>/submit_e2e.sh step3e"""
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_final_sbatches as B  # noqa: E402  (module level only defines constants and functions; main() is not run)

OUT = Path(os.environ.get("FIN_OUT_E2E", f"{B.REPO_ROOT}/logs/sbatch_scripts/final_e2e"))
GROUP = "step3e"
OLD = "  --n_epochs_finetune_head=20 \\\n  --n_epochs_finetune=70 \\\n"
NEW = "  --n_epochs_finetune_head=0 \\\n  --n_epochs_finetune=90 \\\n"
TWIN = {"L1supE2E": "L1sup", "L2supE2E": "L2sup"}
MODES = {"L1supE2E": ("lead_II", "lead_V2"), "L2supE2E": ("two_lead",)}
B.BACKBONE.update({"L1supE2E": {"lead_II": None, "lead_V2": None}, "L2supE2E": {"two_lead": None}})   # in memory only: no backbone


def norm(body, pipe):
    """The rendered job with the run-ID prefix and the pipeline label replaced, so a job and its twin can be compared."""
    return body.replace(f"FIN_{pipe}_", "FIN_X_").replace(f"pipeline: {pipe};", "pipeline: X;")


def main():
    d = OUT / GROUP
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    for pipe, twin_pipe in TWIN.items():
        for mode in MODES[pipe]:
            for s in B.SEED_BASE:
                for k in range(1, 6):
                    name, body = B.render_finetune(pipe, mode, s, k)
                    twin_name, twin = B.render_finetune(twin_pipe, mode, s, k)
                    assert body.count(OLD) == 1 and twin.count(OLD) == 1, name
                    body = body.replace(OLD, NEW)
                    assert norm(body, pipe) == norm(twin.replace(OLD, NEW), twin_pipe), f"{name}: differs from {twin_name} beyond epochs and run ID"
                    assert "--pretrained_model_path" not in body and 'echo "backbone: none"' in body, name
                    (d / f"{name}.sbatch").write_text(body)
                    rows.append(dict(group=GROUP, pipeline=pipe, mode=mode, training_seed=s, seed=B.seed(s, k), fold=k, run_id=name,
                                     twin=twin_name, dataset=B.dataset(mode, k), backbone="none", head_epochs=0, full_epochs=90))
    assert len(rows) == 45 and len({r["run_id"] for r in rows}) == 45 and len({r["twin"] for r in rows}) == 45
    for k in range(1, 6):   # one identical split for every model, as the final runs
        assert len({r["dataset"] for r in rows if r["fold"] == k}) == 1, k
    with open(OUT / "jobs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")   # LF (.gitattributes text=auto)
        w.writeheader()
        w.writerows(rows)
    (OUT / "submit_e2e.sh").write_text(B.SUBMIT % B.REPO_ROOT)
    print(f"{len(rows)} jobs written to {d}; jobs.csv and submit_e2e.sh in {OUT}. Nothing submitted.")


if __name__ == "__main__":
    main()
