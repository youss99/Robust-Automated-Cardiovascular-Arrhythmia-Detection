#!/bin/bash
# Detached evaluation of the multi-database beat models: validation thresholds, test per source, patient probabilities
# (vm/eval_multi_nk.py, GPU, repo venv), then v3 scoring on P21/P27/P36 (v3_multi.py, system python3).
# Progress: results/fine-tune/saved_models/incart_multi_eval/STATUS
cd /home/student/GIT/Robust-Automated-Cardiovascular-Arrhythmia-Detection || exit 1
E=results/fine-tune/saved_models/incart_multi_eval; H=/home/student/GIT/3_patients_pvc_apb/v3_eval_2026-10-14; mkdir -p $E
echo "start $(date)" > $E/STATUS
.venv/bin/python vm/eval_multi_nk.py > $E/eval_multi.log 2>&1 && echo "eval (val, test, patient probabilities) done $(date)" >> $E/STATUS || { echo "FAILED eval_multi_nk $(date)" >> $E/STATUS; exit 1; }
(cd $H && /usr/bin/python3 v3_multi.py) > $E/v3_multi.log 2>&1 && echo "v3 patient scoring done $(date)" >> $E/STATUS || { echo "FAILED v3_multi $(date)" >> $E/STATUS; exit 1; }
echo "DONE $(date)" >> $E/STATUS
