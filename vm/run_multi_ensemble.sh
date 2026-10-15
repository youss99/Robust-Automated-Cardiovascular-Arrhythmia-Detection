#!/bin/bash
# Detached: ensemble of the two multi runs (vm/eval_multi_ensemble.py, GPU) -> v3 patient scoring incl. multi_ens.
cd /home/student/GIT/Robust-Automated-Cardiovascular-Arrhythmia-Detection || exit 1
E=results/fine-tune/saved_models/incart_multi_eval; H=/home/student/GIT/3_patients_pvc_apb/v3_eval_2026-10-14
echo "ensemble start $(date)" >> $E/STATUS
.venv/bin/python vm/eval_multi_ensemble.py > $E/eval_ensemble.log 2>&1 && echo "ensemble probabilities done $(date)" >> $E/STATUS || { echo "FAILED ensemble $(date)" >> $E/STATUS; exit 1; }
(cd $H && /usr/bin/python3 v3_multi.py) > $E/v3_multi_ens.log 2>&1 && echo "v3 scoring with ensemble done $(date)" >> $E/STATUS || { echo "FAILED v3_multi (ensemble) $(date)" >> $E/STATUS; exit 1; }
echo "ENSEMBLE_DONE $(date)" >> $E/STATUS
