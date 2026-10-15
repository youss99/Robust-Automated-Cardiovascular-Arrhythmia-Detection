#!/bin/bash
# Waits for vm/run_incart_nk_ladder.sh to finish, then evaluates every model (INCART test with NeuroKit centres, and the
# patients with v3 scoring). Detached; progress in results/fine-tune/saved_models/incart_vm_logs/STATUS.
cd /home/student/GIT/Robust-Automated-Cardiovascular-Arrhythmia-Detection || exit 1
L=results/fine-tune/saved_models/incart_vm_logs
until grep -q "^DONE" $L/STATUS 2>/dev/null; do sleep 60; done
echo "eval start $(date)" >> $L/STATUS
.venv/bin/python vm/eval_incart_ladder.py > $L/eval_ladder.log 2>&1 && echo "eval INCART+patient probabilities done $(date)" >> $L/STATUS || { echo "FAILED eval_incart_ladder $(date)" >> $L/STATUS; exit 1; }
/usr/bin/python3 /home/student/GIT/3_patients_pvc_apb/v3_eval_2026-10-14/v3_ladder.py > $L/v3_ladder.log 2>&1 && echo "v3 ladder scoring done $(date)" >> $L/STATUS || { echo "FAILED v3_ladder $(date)" >> $L/STATUS; exit 1; }
echo "EVAL_DONE $(date)" >> $L/STATUS
