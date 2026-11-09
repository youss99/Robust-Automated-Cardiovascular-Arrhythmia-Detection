#!/usr/bin/env bash
# Phase 1: put the published median H5 into every folder of phase1_folds/, after checking it and the fold CSVs against phase1_folds/SHA256SUMS.txt.
# Usage, from the repo root: bash src/scripts/custom_dataset/link_phase1_h5.sh /path/to/PatientSegmentRegressionDataset.h5
set -euo pipefail
H5=$(readlink -f "$1")
cd phase1_folds
grep -v "PatientSegmentRegressionDataset.h5" SHA256SUMS.txt | sha256sum -c --quiet
want=$(grep "PatientSegmentRegressionDataset.h5" SHA256SUMS.txt | cut -d' ' -f1)
got=$(sha256sum "$H5" | cut -d' ' -f1)
[ "$want" = "$got" ] || { echo "ERROR: H5 checksum differs: $got, expected $want" >&2; exit 2; }
for d in cv5_optB_*; do ln -sfn "$H5" "$d/PatientSegmentRegressionDataset.h5"; done
echo "H5 verified and linked into: $(ls -d cv5_optB_* | wc -l) folders"
