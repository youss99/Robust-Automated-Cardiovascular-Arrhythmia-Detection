#!/usr/bin/env python3
"""Final thesis runs (instructions of 2026-11-20, section 1 "Validation sets"): the lead II fold folders for option 2+.

SUPERSEDED on 2026-11-20 (docs/DECISIONS.md, "Splits"): every learned model now uses one identical split, the V2 option 2+ folders
cv5_optB_leadV2_opt2p_fold01, 04, 05 (masked record unused for every model). The lead II folders and manifests this script built are
kept in the repo, but no job uses them; their checksum lines were removed from phase1_folds/SHA256SUMS.txt and link_phase1_h5.sh
neither checks nor links them. Do not re-run this script (it stops anyway, because its output folders exist).

One validation manifest for every learned model: the published sets plus option 2+ in folds 1, 4, 5.
  lead_V2   phase1_folds/cv5_optB_leadV2_opt2p_fold01,04,05 (existing, R2b): masked V2 record unused, moved record in validation.
  two_lead  the same folders as lead_V2. User decision 2026-11-20: the masked V2 record is unused for the two-lead model too
            (a fold CSV can only drop whole records), so its split equals V2's. This script asserts that equality.
  lead_II   NEW here: phase1_folds/cv5_optB_leadII_opt2p_fold01,04,05: the published split with the moved record in validation;
            the masked V2 record stays in validation (lead II is not masked there).
  folds 2, 3: cv5_optB_fold02, 03 for every model (option 2+ changes nothing there).

The moved record is derived, not typed: the record that is `train` in cv5_optB and `val` in the V2 option 2+ manifest. It must be
exactly one record and match README_phase1_runs.md step 3 (s20231, s20071, s20141), or the script stops.
Writes ptbxl_models_v2/cv5_optB_leadII_opt2p_fold_manifests/fold01,04,05.csv and the three folders (prepare_fold_dataset.py, unchanged),
then appends their CSV checksums to phase1_folds/SHA256SUMS.txt after verifying every existing line. Existing files are never rewritten;
an existing target folder or manifest stops the script.
Usage, from the repo root: python3 src/scripts/custom_dataset/build_final_folds.py <source dataset root>"""
import hashlib
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).parent))
import prepare_fold_dataset as P  # noqa: E402

SRC = Path(sys.argv[1])
OUT = REPO / "phase1_folds"
BASE = REPO / "ptbxl_models_v2/cv5_optB_fold_manifests"
V2OPT = REPO / "ptbxl_models_v2/cv5_optB_v2models_leadV2_fold_manifests"
NEWMAN = REPO / "ptbxl_models_v2/cv5_optB_leadII_opt2p_fold_manifests"
EXPECT_MOVED = {1: "s20231", 4: "s20071", 5: "s20141"}
EXPECT_UNUSED = {1: "s20081", 4: "s20081", 5: "s20241"}
P.SPLIT_LABEL_MAP["unused"] = -1


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def man(p):
    return pd.read_csv(p).set_index("patient_id").split


# existing checksums must still hold before anything is added
sums = (OUT / "SHA256SUMS.txt").read_text().splitlines()
for line in sums:
    want, rel = line.split("  ", 1)
    if rel != "PatientSegmentRegressionDataset.h5":
        assert sha(OUT / rel) == want, f"existing checksum fails: {rel}"
assert sha(SRC / "PatientSegmentRegressionDataset.h5") == next(l.split()[0] for l in sums if l.endswith("PatientSegmentRegressionDataset.h5")), "source H5 differs from the published one"

NEWMAN.mkdir(parents=True, exist_ok=False)
new_lines = []
for k in (1, 4, 5):
    base, v2 = man(BASE / f"fold{k:02d}.csv"), man(V2OPT / f"fold{k:02d}.csv")
    assert sorted(base.index) == sorted(v2.index) and len(base) == 16
    moved = sorted(r for r in base.index if base[r] == "train" and v2[r] == "val")
    unused = sorted(v2[v2 == "unused"].index)
    assert moved == [EXPECT_MOVED[k]], (k, moved)
    assert unused == [EXPECT_UNUSED[k]] and base[unused[0]] == "val", (k, unused)
    diff = sorted(r for r in base.index if base[r] != v2[r])
    assert diff == sorted(moved + unused), (k, diff)          # option 2+ changes exactly these two records
    lead_ii = base.copy()
    lead_ii[moved[0]] = "val"
    # two-lead (user decision): masked V2 record unused, moved record in validation == the V2 manifest
    two_lead = lead_ii.copy()
    two_lead[unused[0]] = "unused"
    assert two_lead.sort_index().equals(v2.sort_index()), f"fold {k}: two-lead split differs from the V2 option 2+ manifest"
    mpath = NEWMAN / f"fold{k:02d}.csv"
    # same row order as the published manifest
    pd.DataFrame({"patient_id": base.index, "split": [lead_ii[r] for r in base.index]}).to_csv(mpath, index=False)
    name = f"cv5_optB_leadII_opt2p_fold{k:02d}"
    assert not (OUT / name).exists(), f"{name} exists already"
    sys.argv = ["prepare_fold_dataset.py", "--manifest", str(mpath), "--src_root", str(SRC), "--dst_root", str(OUT / name)]
    rc = P.main()
    assert not rc, (name, rc)
    print(f"fold {k}: lead II validation {sorted(lead_ii[lead_ii == 'val'].index)}; training loses {moved[0]}")
    new_lines.append(f"{sha(OUT / name / 'modified_dataset/modified_dataset.csv')}  {name}/modified_dataset/modified_dataset.csv")

h5_line = [l for l in sums if l.endswith("PatientSegmentRegressionDataset.h5")]
keep = [l for l in sums if not l.endswith("PatientSegmentRegressionDataset.h5")]
(OUT / "SHA256SUMS.txt").write_text("\n".join(keep + new_lines + h5_line) + "\n")
print("\n".join(new_lines))
