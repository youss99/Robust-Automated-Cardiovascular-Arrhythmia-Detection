#!/usr/bin/env python3
"""Phase 1 (README_phase1_runs.md): rebuild the published cv5_optB fold folders from the manifests, plus the option 2+ V2 folds, into phase1_folds/ so both machines use the same copy.

  phase1_folds/cv5_optB_fold01..05                 ptbxl_models_v2/cv5_optB_fold_manifests            R1b, R2
  phase1_folds/cv5_optB_leadV2_opt2p_fold01,04,05  ptbxl_models_v2/cv5_optB_v2models_leadV2_fold_manifests  R2b

Each folder is what prepare_fold_dataset.py writes, using that script unchanged: modified_dataset/modified_dataset.csv with the fold's split in strat_fold_annotated, and the H5.
The only addition is for the option 2+ folds: `unused` in a manifest becomes strat_fold_annotated = -1 (not train, val or test).
The CSVs are tracked in git. The H5 (1 GB, git-ignored) is hard-linked on the VM; on another machine run link_phase1_h5.sh, which checks it against SHA256SUMS.txt.
Usage, from the repo root: python3 src/scripts/custom_dataset/build_phase1_folds.py <source dataset root>"""
import hashlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).parent))
import prepare_fold_dataset as P  # noqa: E402

SRC = Path(sys.argv[1])
OUT = REPO / "phase1_folds"
BASE, OPT = REPO / "ptbxl_models_v2/cv5_optB_fold_manifests", REPO / "ptbxl_models_v2/cv5_optB_v2models_leadV2_fold_manifests"
P.SPLIT_LABEL_MAP["unused"] = -1


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


jobs = [(f"cv5_optB_fold{k:02d}", BASE / f"fold{k:02d}.csv") for k in range(1, 6)] + [(f"cv5_optB_leadV2_opt2p_fold{k:02d}", OPT / f"fold{k:02d}.csv") for k in (1, 4, 5)]
lines = []
for name, manifest in jobs:
    sys.argv = ["prepare_fold_dataset.py", "--manifest", str(manifest), "--src_root", str(SRC), "--dst_root", str(OUT / name)]
    rc = P.main()
    assert not rc, (name, rc)
    lines.append(f"{sha(OUT / name / 'modified_dataset/modified_dataset.csv')}  {name}/modified_dataset/modified_dataset.csv")
lines.append(f"{sha(SRC / 'PatientSegmentRegressionDataset.h5')}  PatientSegmentRegressionDataset.h5")
(OUT / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
