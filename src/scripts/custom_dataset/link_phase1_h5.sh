#!/usr/bin/env bash
# Phase 1 and final runs: put the published median H5 into the fold folders of phase1_folds/ that jobs use, after checking it and
# those folders' CSVs against phase1_folds/SHA256SUMS.txt. The cv5_optB_leadII_opt2p_fold* folders (an earlier per-lead split,
# superseded on 2026-11-20 by one split for every learned model, docs/DECISIONS.md) stay in the repo but are not checked or linked.
# Usage, from the repo root: bash src/scripts/custom_dataset/link_phase1_h5.sh /path/to/PatientSegmentRegressionDataset.h5
set -euo pipefail
H5=$(readlink -f "$1")
cd phase1_folds
grep -v "PatientSegmentRegressionDataset.h5" SHA256SUMS.txt | sha256sum -c --quiet
want=$(grep "PatientSegmentRegressionDataset.h5" SHA256SUMS.txt | cut -d' ' -f1)
got=$(sha256sum "$H5" | cut -d' ' -f1)
[ "$want" = "$got" ] || { echo "ERROR: H5 checksum differs: $got, expected $want" >&2; exit 2; }
dirs=(cv5_optB_fold0? cv5_optB_leadV2_opt2p_fold0?)
for d in "${dirs[@]}"; do ln -sfn "$H5" "$d/PatientSegmentRegressionDataset.h5"; done
echo "H5 verified and linked into ${#dirs[@]} folders: ${dirs[*]}"
