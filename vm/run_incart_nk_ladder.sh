#!/bin/bash
# INCART beat model B, centring ladder (2026-10-15), trained on the VM A16, detached. Same recipe/seed as the cluster B.
#   ref       : reference-beat centres (cluster cache, byte-identical local rebuild)      -> reproduces cluster B
#   nk        : NeuroKit-peak centres (prepare_incart_nk), no jitter                        -> effect of centring
#   nk_jit50  : NeuroKit-peak centres + +-50 ms training jitter                             -> + jitter
# Progress: results/fine-tune/saved_models/incart_vm_logs/STATUS ; per-run logs next to it.
cd /home/student/GIT/Robust-Automated-Cardiovascular-Arrhythmia-Detection || exit 1
D=/home/student/GIT/Project_1/Datasets; PY=.venv/bin/python; L=results/fine-tune/saved_models/incart_vm_logs; mkdir -p $L
echo "start $(date)" > $L/STATUS
run() {  # name lead cache out_root extra
  $PY -m src.incart_beat.train --cache_dir $3 --lead $2 --runs B --out_root $4 $5 > $L/$1.log 2>&1 \
    && echo "done $1 $(date)" >> $L/STATUS || echo "FAILED $1 $(date)" >> $L/STATUS
}
run ii_ref      II $D/incart_beat_leadII_100hz    results/fine-tune/saved_models/incart_vm             "" &
run ii_nk       II $D/incart_beat_leadII_100hz_nk results/fine-tune/saved_models/incart_vm_nk          "" &
run ii_nk_jit50 II $D/incart_beat_leadII_100hz_nk results/fine-tune/saved_models/incart_vm_nk_jit50    "--jitter_ms 50" &
run v5_ref      V5 $D/incart_beat_leadV5_100hz    results/fine-tune/saved_models/incart_vm             "" &
run v5_nk       V5 $D/incart_beat_leadV5_100hz_nk results/fine-tune/saved_models/incart_vm_nk          "" &
run v5_nk_jit50 V5 $D/incart_beat_leadV5_100hz_nk results/fine-tune/saved_models/incart_vm_nk_jit50    "--jitter_ms 50" &
wait
echo "DONE $(date)" >> $L/STATUS
