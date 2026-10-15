"""
Shared signal preprocessing for the central-beat classifier.

This is the ONLY place that defines how a raw ECG lead becomes model input. Training, validation, test and
patient inference must all go through `prepare_lead` + `map_samples` + `window_bounds`.

Units: the backbone (`patch_ecg_unified_ii_fixed`) was pretrained on UnifiedDataset.h5, which stores the raw
`val` ADC matrix from the Chapman / CinC-2020 .mat files (see chapman_preprocessing.extract_and_resample and
cinc_2020_preprocessing.extract_and_resample: `loadmat(filename)['val']`, gain/baseline never applied). Every
source header (Chapman, CPSC 2018, CPSC-extra, Georgia) declares gain 1000 adu/mV with baseline 0, so the stored
values are microvolts. We therefore feed the model microvolts.

Pipeline (exactly once each):
  physical signal in its native units (WFDB p_signal applies the header gain/baseline; never apply it again)
  -> microvolts (x1000 for mV)
  -> resample to 100 Hz with scipy.signal.resample_poly (zero-phase polyphase FIR)
  -> 10 s window [centre - 500, centre + 500), centre beat at window index 500.
Normalization (RevIN) happens inside the model wrapper, after this module.
"""
from fractions import Fraction

import numpy as np
from scipy import signal as sgn

TARGET_FS = 100
WINDOW_S = 10
WINDOW_LEN = TARGET_FS * WINDOW_S  # 1000 samples = 20 patches of 50
HALF_WINDOW = WINDOW_LEN // 2      # samples before the centre beat; window = [c - 500, c + 500)
INPUT_UNITS = 'uV'

_TO_MICROVOLTS = {'uV': 1.0, 'µV': 1.0, 'mV': 1e3, 'V': 1e6}

LEAD = 'II'
CLASS_NAMES = ['background', 'PAC', 'PVC']
LABEL_MAP = {'N': 0, 'R': 0, 'B': 0, 'j': 0, 'n': 0,  # background
             'A': 1,                                  # PAC / APB
             'V': 2}                                  # PVC
EXCLUDED_BEAT_SYMBOLS = ('F', 'Q', 'S')               # beat labels deliberately left out; anything else unmapped
                                                      # (e.g. '+' rhythm-change markers) is logged by symbol


def resample_ratio(fs):
    """Rational up/down factors taking `fs` to TARGET_FS (257 Hz -> up=100, down=257)."""
    frac = Fraction(TARGET_FS / float(fs)).limit_denominator(10000)
    return frac.numerator, frac.denominator


def to_microvolts(x, units):
    """Convert a PHYSICAL signal (already gain/baseline corrected) to microvolts."""
    if units not in _TO_MICROVOLTS:
        raise ValueError(f'Unknown signal units {units!r}; expected one of {sorted(_TO_MICROVOLTS)}')
    return np.asarray(x, dtype=np.float64) * _TO_MICROVOLTS[units]


def prepare_lead(x, fs, units):
    """1-D physical lead in `units` at `fs` Hz -> float32 microvolts at TARGET_FS."""
    x = to_microvolts(x, units)
    if not np.all(np.isfinite(x)):
        raise ValueError('Signal contains NaN/inf; impute or drop before preprocessing')
    up, down = resample_ratio(fs)
    if up != down:
        x = sgn.resample_poly(x, up, down)
    return x.astype(np.float32)


def map_samples(samples, fs):
    """Map sample indices at `fs` to the TARGET_FS grid produced by `prepare_lead`."""
    return np.rint(np.asarray(samples, dtype=np.float64) * TARGET_FS / float(fs)).astype(np.int64)


def window_bounds(centres, n_samples):
    """Start index of each centred window and whether the full window fits inside the signal."""
    starts = np.asarray(centres, dtype=np.int64) - HALF_WINDOW
    valid = (starts >= 0) & (starts + WINDOW_LEN <= n_samples)
    return starts, valid


def extract_windows(x, centres):
    """(n_valid, WINDOW_LEN) windows from a prepared 1-D signal; centres without a full window are dropped."""
    starts, valid = window_bounds(centres, len(x))
    idx = starts[valid, None] + np.arange(WINDOW_LEN)[None, :]
    return x[idx], valid


def load_wfdb_lead(record_path, lead=LEAD):
    """Read one lead of a WFDB record as physical units (gain/baseline from the header, applied once by WFDB)."""
    import wfdb
    rec = wfdb.rdrecord(record_path, channel_names=[lead], physical=True)
    if rec.sig_name != [lead]:
        raise ValueError(f'{record_path}: lead {lead} not found (got {rec.sig_name})')
    return rec.p_signal[:, 0], rec.fs, rec.units[0]


def preprocessing_config(lead=LEAD):
    """Saved with every checkpoint so inference can verify it uses the same input definition."""
    return {
        'input_units': INPUT_UNITS,
        'unit_conversion': 'physical signal (WFDB p_signal: (adc - baseline) / gain from the header) '
                           'x {mV: 1e3, uV: 1, V: 1e6}; applied exactly once',
        'pretrain_units_evidence': 'UnifiedDataset.h5 stores raw .mat val; Chapman/CPSC2018/CPSC-extra/Georgia '
                                   'headers are 1000 adu/mV, baseline 0 -> microvolts',
        'target_fs': TARGET_FS,
        'resampler': 'scipy.signal.resample_poly(x, up, down) on the whole recording',
        'annotation_mapping': 'rint(sample * target_fs / fs)',
        'window_len': WINDOW_LEN,
        'window': '[centre - 500, centre + 500); centre beat at index 500; incomplete windows dropped',
        'lead': lead,
        'filtering': 'none (matches pretraining)',
        'class_names': CLASS_NAMES,
        'label_map': LABEL_MAP,
    }
