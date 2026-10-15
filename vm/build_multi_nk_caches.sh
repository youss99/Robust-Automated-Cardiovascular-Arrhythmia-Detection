#!/bin/bash
# Detached build + verification of the multi-database NeuroKit-centred caches (V5 family first: smaller, then II).
# Output (Globus-shared): Desktop/Research Data/DATASET/beat_caches/multi_{V5,II}_nk/ (VERIFY.txt inside each);
# progress: beat_caches/STATUS, logs beat_caches/{build,verify}_{V5,II}.log
cd /home/student/GIT/Robust-Automated-Cardiovascular-Arrhythmia-Detection || exit 1
B="/home/student/Desktop/Research Data/DATASET/beat_caches"; DATA="/home/student/Desktop/Research Data/DATASET"
echo "start $(date)" > "$B/STATUS"
for F in V5 II; do
  rm -rf "$B/multi_${F}_nk"
  .venv/bin/python -m src.incart_beat.prepare_multi_nk --family $F --data_root "$DATA" --out_dir "$B/multi_${F}_nk" --workers 6 > "$B/build_${F}.log" 2>&1 \
    && echo "built $F $(date)" >> "$B/STATUS" || { echo "FAILED build $F $(date)" >> "$B/STATUS"; exit 1; }
  .venv/bin/python vm/verify_multi_nk_cache.py "$B/multi_${F}_nk" --data_root "$DATA" > "$B/verify_${F}.log" 2>&1 \
    && echo "verified $F $(date)" >> "$B/STATUS" || echo "VERIFY FAILED $F $(date)" >> "$B/STATUS"
done
echo "DONE $(date)" >> "$B/STATUS"
