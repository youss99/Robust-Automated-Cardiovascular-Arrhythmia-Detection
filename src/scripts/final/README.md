# Final thesis runs: drivers

The scripts that produced `docs/RESULTS_FINAL.md`, `docs/thesis_tables/` and `docs/thesis_figures/`. Run log, checks and caveats:
`docs/reports/2026-11-21_vm_steps4_7.md`. Plan and decisions: `docs/instructions_final_results.md`, `docs/DECISIONS.md`.

| script | what | interpreter |
|---|---|---|
| `dry_run_final.py` | 2026-11-20 dry run of the whole training batch (no training) | repo `.venv` |
| `fin_7a_r1_rescore.py` | step 7a: published R1 re-scored, must equal the paper | repo `.venv` |
| `fin_step4_seed44.py` | step 4: rodrigues2021 delineation, noise seed 44 | system `python3` (neurokit2 0.2.12) |
| `fin_step4_checks.py` | step 4: R-peak cache, same-setup check, `.atr` spot-check, per-fold detector rule | system `python3` |
| `fin_step5_nk.py` | step 5: Fiducial-FP fit, Fiducial-PR / Fiducial-FP values, SQI gate table (`fit`, `values`, `summarise`) | repo `.venv` |
| `fin_step6_infer.py` | step 6: out-of-fold predictions of the 90 checkpoints (GPU) | repo `.venv` |
| `fin_step7_tier1.py` | T1, T5, T7 | repo `.venv` |
| `fin_step7_t10_fig1.py` | T10, Fig. 1, threshold-sweep figure (`figures` redraws both figures from saved outputs) | repo `.venv` |
| `fin_step7_rest.py` | T2, T3, T4, T6 (`T2` … `T6`) | repo `.venv` |
| `fin_step7_data_reference.py` | data table, gold-standard ceiling, and `relabel` (condition and system names in the `.tex` files; rerun after regenerating any table) | repo `.venv` |
| `fin_step7_appendix.py` | appendix tables (`hyperparams`, `per_record`) | repo `.venv` |

Each script's docstring gives its inputs, controls and usage. Long runs were started detached (`setsid nohup … &`).

## `phase1_lib/`: vendored phase-1 code

Copies of every local module the drivers import (found by importing each driver and listing the loaded modules, plus the imports made
inside functions): `p1_eval`, the `C_neurokit` scripts `c3`–`c21b`, `evaluate_test1` / `evaluate_e1` / `evaluate_t1nk`, `score_alarms`,
`sweeps_all`, `nk_core`, and the `ltst_pipeline` modules used (`old_fsm_gs`, `ml_inference`, `noise`, `filters`, `delineation`,
`parse_ltst_16a_iso_jp_ee`). Copied 2026-11-22; `MANIFEST.tsv` gives each file's source path and sha256 (source and copy).

Only three kinds of line differ from the source (`CHANGES.tsv` lists all 32):
- `sys.path.insert(...)` lines that added the old code folders: replaced by `pass`, so every import resolves inside `phase1_lib`;
- `HERE = Path(__file__)...` in `evaluate_e1` / `evaluate_t1nk`: set to the original folder, because their data files live there;
- `ltst_pipeline/ml_inference.py`: the default PatchECG root is the repo this copy lives in (`PATCHECG_REPO_ROOT` still overrides it).

The drivers add `phase1_lib` to `sys.path` themselves. Check (2026-11-22): importing each of the ten drivers loads 130 local modules, all
from inside this repo.

## External data (not in git; paths are absolute, as on the VM)

| path | what |
|---|---|
| `/home/student/Desktop/Research Data/DATASET NK/` | LTST records (`ltst_records/`), NeuroKit delineations (`Delineation Data/`), phase-1 NeuroKit outputs (`C_neurokit/`, DMY tables), final R-peak cache and step-4 files (`final/`) |
| `<repo>/results/` (git-ignored) | pre-training backbones and the 90 `FIN_*` fine-tuning runs (`*_best.pt`, `params.txt`, metrics) |
| `<repo>/custom_datasets/` (git-ignored) and the H5 linked into `phase1_folds/` | fine-tuning datasets (`phase1_folds/README.md` has the link step and checksums) |
| `/home/student/GIT/transfer_bundle/gs_per_beat.csv` | gold-standard episodes (scorer) |
| `/home/student/GIT/results/neurokit/_sweep/record_durations.csv` | record durations (scorer) |
| `/home/student/GIT/Tasks/phase1/runs/` | published R1 predictions, oracle labels (`_scores/oracle/`), phase-1 results used as controls |
| `/home/student/GIT/4_ltst_beatcenter_test1/` | data folders of the `evaluate_*` modules |
| `/home/student/GIT/Tasks/final/runs/` | outputs of every driver (per-seed and per-record CSVs, alarm lists, predictions, `provenance.csv`, check files) |

Docs outputs (`RESULTS_FINAL.md`, `thesis_tables/`, `thesis_figures/`) go to the `docs/` of the repo the script is run from; everything
else is read from and written to the paths above.
