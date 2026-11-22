from __future__ import annotations

import numpy as np
import wfdb
from scipy.signal import resample_poly


'''def contaminate_ecg_with_motion_artifact(ecg_signal, seed: int = 42, fs: int = 250):
    """
    Contaminate ECG with NSTDB 'em' noise per your spec:
      - 1, 2, 4 min noise bursts
      - intervals ~ N(15 min, 5 min) between bursts
      - NOTE: your current code sets insert_time=0 (no clean first 10 min)
    """
    rng = np.random.default_rng(seed)

    x = np.asarray(ecg_signal, dtype=float)
    if x.ndim == 1:
        x = x[:, None]

    n_samples, n_leads = x.shape

    # Load and resample 'em' noise (360 Hz -> 250 Hz via 25/36)
    rec = wfdb.rdrecord("em", pn_dir="nstdb")
    noise_raw = rec.p_signal.T  # (2, N)
    noise = resample_poly(noise_raw, up=25, down=36, axis=1)
    n_noise = noise.shape[1]

    contaminated = x.copy()

    insert_time = 0  # you set this (instead of 10*60*fs)

    segment_durations_sec = [60, 120, 240]
    mean_interval_s = 15 * 60
    std_interval_s = 5 * 60
    min_gap_s = 0

    while insert_time < n_samples:
        seg_duration_s = int(rng.choice(segment_durations_sec))
        seg_samples = seg_duration_s * fs
        if insert_time + seg_samples > n_samples:
            break

        if seg_samples <= n_noise:
            start_idx = rng.integers(0, n_noise - seg_samples + 1)
            seg_noise = noise[:, start_idx:start_idx + seg_samples]
        else:
            reps = int(np.ceil(seg_samples / n_noise))
            seg_noise = np.tile(noise, reps)[:, :seg_samples]

        if n_leads >= 1:
            contaminated[insert_time:insert_time + seg_samples, 0] += seg_noise[0]
        if n_leads >= 2:
            contaminated[insert_time:insert_time + seg_samples, 1] += seg_noise[1]

        draw_s = rng.normal(loc=mean_interval_s, scale=std_interval_s)
        next_interval_s = max(draw_s, min_gap_s)
        next_interval_samples = int(round(next_interval_s * fs))

        insert_time += seg_samples + next_interval_samples

    return contaminated'''


def contaminate_ecg_with_motion_artifact(ecg_signal, seed: int = 42, fs: int = 250):
    import numpy as np
    import wfdb
    from scipy.signal import resample_poly

    rng = np.random.default_rng(seed)

    x = np.asarray(ecg_signal, dtype=float)
    if x.ndim == 1:
        x = x[:, None]

    n_samples, n_leads = x.shape

    # Load and resample 'em' noise (360 Hz -> 250 Hz via 25/36)
    rec = wfdb.rdrecord("em", pn_dir="nstdb")
    noise_raw = rec.p_signal.T  # (2, N)
    noise = resample_poly(noise_raw, up=25, down=36, axis=1)
    n_noise = noise.shape[1]

    contaminated = x.copy()

    insert_time = 0

    segment_durations_sec = [60, 120, 240]
    mean_interval_s = 15 * 60
    std_interval_s = 5 * 60
    min_gap_s = 0

    contamination_log = []

    while insert_time < n_samples:
        seg_duration_s = int(rng.choice(segment_durations_sec))
        seg_samples = seg_duration_s * fs
        if insert_time + seg_samples > n_samples:
            break

        start_sample = insert_time
        end_sample = insert_time + seg_samples  # exclusive end

        #print(f"      [noise] Contaminating samples {start_sample} to {end_sample} ({seg_duration_s}s)")

        if seg_samples <= n_noise:
            start_idx = rng.integers(0, n_noise - seg_samples + 1)
            seg_noise = noise[:, start_idx:start_idx + seg_samples]
        else:
            reps = int(np.ceil(seg_samples / n_noise))
            seg_noise = np.tile(noise, reps)[:, :seg_samples]

        contaminated[start_sample:end_sample, 0] += seg_noise[0]
        if n_leads >= 2:
            contaminated[start_sample:end_sample, 1] += seg_noise[1]

        contamination_log.append({
            "start_sample": int(start_sample),
            "end_sample": int(end_sample),   # exclusive
            "start_time_sec": start_sample / fs,
            "end_time_sec": end_sample / fs,
            "duration_sec": seg_duration_s,
            "leads_contaminated": list(range(min(n_leads, 2)))
        })

        draw_s = rng.normal(loc=mean_interval_s, scale=std_interval_s)
        next_interval_s = max(draw_s, min_gap_s)
        next_interval_samples = int(round(next_interval_s * fs))

        insert_time += seg_samples + next_interval_samples

    return contaminated, contamination_log