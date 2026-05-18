import argparse
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

import h5py
import numpy as np
import pandas as pd
from scipy.signal import resample_poly
from sklearn.model_selection import train_test_split

from src.core.datasets.ecg_interface import Split


REGRESSION_SEGMENT_RE = re.compile(r"^(?P<patient_id>.+)_regression_(?P<segment_id>seg\d+)\.csv$")


@dataclass
class RegressionRecord:
    patient_id: str
    segment_id: str
    filename: str
    source_path: str
    source_leads: list[int]
    targets: dict[int, float]
    tracing: np.ndarray


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert patient_mode2_extracted_regression CSV folders into repo-compatible "
            "H5 + modified_dataset.csv files for ECG regression."
        )
    )
    parser.add_argument(
        "--input-root",
        required=True,
        help="Root containing record folders produced by mode 2 regression extraction.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Output dataset root. The script writes PatientSegmentRegressionDataset.h5 and modified_dataset/modified_dataset.csv.",
    )
    parser.add_argument(
        "--records",
        nargs="*",
        default=None,
        help="Optional record/patient IDs to include. If omitted, every record folder under --input-root is used.",
    )
    parser.add_argument(
        "--source-freq",
        type=int,
        default=250,
        help="Sampling rate of the extracted regression CSV signals.",
    )
    parser.add_argument(
        "--target-freq",
        type=int,
        default=100,
        help="Sampling rate of the generated H5 dataset.",
    )
    parser.add_argument(
        "--segment-seconds",
        type=int,
        default=10,
        help="Expected segment length in seconds.",
    )
    parser.add_argument(
        "--source-lead-indices",
        nargs="*",
        type=int,
        default=None,
        help=(
            "Lead IDs represented by the signal columns, in column order. "
            "If omitted, the script uses 0..n_leads-1 for each file."
        ),
    )
    parser.add_argument(
        "--select-lead-indices",
        nargs="*",
        type=int,
        default=None,
        help=(
            "Optional subset of lead IDs to keep in the H5 file and targets. "
            "If omitted, all source leads are kept."
        ),
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=200)
    parser.add_argument(
        "--split-manifest",
        type=str,
        default=None,
        help="Optional CSV with columns patient_id,split. Split may be train, validation, or test.",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="gzip",
        help="Compression algorithm for the output H5 dataset. Use 'none' to disable compression.",
    )
    parser.add_argument(
        "--h5-name",
        type=str,
        default="PatientSegmentRegressionDataset.h5",
        help="Name of the H5 file to write under --output-root.",
    )

    args = parser.parse_args()

    ratio_sum = args.train_ratio + args.valid_ratio + args.test_ratio
    if not math.isclose(ratio_sum, 1.0, rel_tol=1e-6):
        raise ValueError(f"Split ratios must sum to 1.0, received {ratio_sum}")
    if args.source_freq <= 0 or args.target_freq <= 0:
        raise ValueError("--source-freq and --target-freq must be positive.")
    if args.segment_seconds <= 0:
        raise ValueError("--segment-seconds must be positive.")
    if args.source_lead_indices is not None and len(set(args.source_lead_indices)) != len(args.source_lead_indices):
        raise ValueError("--source-lead-indices cannot contain duplicates.")
    if args.source_lead_indices is not None and len(args.source_lead_indices) == 0:
        raise ValueError("--source-lead-indices was provided but no lead IDs were listed.")
    if args.select_lead_indices is not None and len(set(args.select_lead_indices)) != len(args.select_lead_indices):
        raise ValueError("--select-lead-indices cannot contain duplicates.")
    if args.select_lead_indices is not None and len(args.select_lead_indices) == 0:
        raise ValueError("--select-lead-indices was provided but no lead IDs were listed.")

    return args


def resample_signal(signal: np.ndarray, source_freq: int, target_freq: int, target_length: int) -> np.ndarray:
    if signal.ndim != 1:
        raise ValueError(f"Expected a 1D signal, received shape {signal.shape}")

    if source_freq == target_freq:
        resampled = signal.astype(np.float32, copy=False)
    else:
        gcd = math.gcd(source_freq, target_freq)
        up = target_freq // gcd
        down = source_freq // gcd
        resampled = resample_poly(signal, up=up, down=down).astype(np.float32, copy=False)

    if resampled.shape[0] > target_length:
        resampled = resampled[:target_length]
    elif resampled.shape[0] < target_length:
        resampled = np.pad(resampled, (0, target_length - resampled.shape[0]))

    return resampled


def parse_regression_filename(path: Path) -> tuple[str, str]:
    match = REGRESSION_SEGMENT_RE.match(path.name)
    if match:
        return match.group("patient_id"), match.group("segment_id")
    return path.parent.name, path.stem


def load_regression_csv(csv_path: Path, args) -> RegressionRecord:
    df = pd.read_csv(csv_path, header=None)
    if df.empty:
        raise ValueError(f"CSV is empty: {csv_path}")
    if df.shape[1] < 2 or df.shape[1] % 2 != 0:
        raise ValueError(
            f"Expected an even number of columns: signal columns followed by target columns. "
            f"Received shape {df.shape} for {csv_path}"
        )

    n_file_leads = df.shape[1] // 2
    if args.source_lead_indices is None:
        source_leads = list(range(n_file_leads))
    else:
        if len(args.source_lead_indices) != n_file_leads:
            raise ValueError(
                f"{csv_path} has {n_file_leads} signal columns, but --source-lead-indices has "
                f"{len(args.source_lead_indices)} entries."
            )
        source_leads = list(args.source_lead_indices)

    selected_leads = list(args.select_lead_indices) if args.select_lead_indices is not None else source_leads
    missing = sorted(set(selected_leads) - set(source_leads))
    if missing:
        raise ValueError(f"Selected leads {missing} are not present in {csv_path}; source leads are {source_leads}")

    source_position = {lead_id: index for index, lead_id in enumerate(source_leads)}
    expected_length = args.source_freq * args.segment_seconds
    target_length = args.target_freq * args.segment_seconds

    signals = []
    targets = {}
    for lead_id in selected_leads:
        pos = source_position[lead_id]
        signal = pd.to_numeric(df.iloc[:, pos], errors="coerce").dropna().to_numpy(dtype=np.float32)
        if signal.shape[0] != expected_length:
            raise ValueError(
                f"Expected {expected_length} source samples for lead {lead_id}, "
                f"found {signal.shape[0]} in {csv_path}."
            )
        signals.append(resample_signal(signal, args.source_freq, args.target_freq, target_length))

        target_value = pd.to_numeric(pd.Series([df.iloc[0, n_file_leads + pos]]), errors="coerce").iloc[0]
        if pd.isna(target_value):
            raise ValueError(f"Missing target value for lead {lead_id} in first row of {csv_path}")
        targets[lead_id] = float(target_value)

    patient_id, segment_id = parse_regression_filename(csv_path)
    if patient_id != csv_path.parent.name:
        raise ValueError(
            f"Filename patient ID '{patient_id}' does not match parent folder '{csv_path.parent.name}' for {csv_path}"
        )

    return RegressionRecord(
        patient_id=patient_id,
        segment_id=segment_id,
        filename=csv_path.name,
        source_path=str(csv_path.resolve()),
        source_leads=selected_leads,
        targets=targets,
        tracing=np.stack(signals, axis=0),
    )


def build_records(input_root: Path, args) -> list[RegressionRecord]:
    records = []
    keep_records = set(args.records) if args.records is not None else None
    for patient_dir in sorted(path for path in input_root.iterdir() if path.is_dir()):
        if keep_records is not None and patient_dir.name not in keep_records:
            continue
        for csv_path in sorted(patient_dir.glob("*.csv")):
            records.append(load_regression_csv(csv_path=csv_path, args=args))
    return records


def load_split_manifest(manifest_path: Path) -> dict[str, int]:
    split_map = {
        "train": Split.TRAIN.value,
        "training": Split.TRAIN.value,
        "valid": Split.VALIDATION.value,
        "validation": Split.VALIDATION.value,
        "val": Split.VALIDATION.value,
        "test": Split.TEST.value,
        str(Split.TRAIN.value): Split.TRAIN.value,
        str(Split.VALIDATION.value): Split.VALIDATION.value,
        str(Split.TEST.value): Split.TEST.value,
    }

    df = pd.read_csv(manifest_path)
    required_columns = {"patient_id", "split"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"Split manifest must contain columns {required_columns}, received {list(df.columns)}")

    assignments = {}
    for _, row in df.iterrows():
        patient_id = str(row["patient_id"]).strip()
        split = str(row["split"]).strip().lower()
        if split not in split_map:
            raise ValueError(f"Unrecognized split value '{row['split']}' in manifest {manifest_path}")
        assignments[patient_id] = split_map[split]
    return assignments


def split_patient_ids(patient_ids: np.ndarray, train_ratio: float, valid_ratio: float, test_ratio: float,
                      seed: int) -> dict[str, int]:
    temp_ratio = valid_ratio + test_ratio
    if temp_ratio == 0 or len(patient_ids) < 3:
        return {patient_id: Split.TRAIN.value for patient_id in patient_ids}

    train_ids, temp_ids = train_test_split(patient_ids, test_size=temp_ratio, random_state=seed)
    assignments = {patient_id: Split.TRAIN.value for patient_id in train_ids}

    if valid_ratio == 0:
        for patient_id in temp_ids:
            assignments[patient_id] = Split.TEST.value
        return assignments

    if test_ratio == 0:
        for patient_id in temp_ids:
            assignments[patient_id] = Split.VALIDATION.value
        return assignments

    test_fraction_within_temp = test_ratio / temp_ratio
    valid_ids, test_ids = train_test_split(temp_ids, test_size=test_fraction_within_temp, random_state=seed)

    for patient_id in valid_ids:
        assignments[patient_id] = Split.VALIDATION.value
    for patient_id in test_ids:
        assignments[patient_id] = Split.TEST.value
    return assignments


def assign_patient_splits(records: list[RegressionRecord], args) -> dict[str, int]:
    patient_ids = np.asarray(sorted({record.patient_id for record in records}))
    if args.split_manifest:
        assignments = load_split_manifest(Path(args.split_manifest))
        missing = sorted(set(patient_ids) - set(assignments))
        if missing:
            raise ValueError(f"Split manifest is missing patient IDs: {missing[:10]}")
        return {patient_id: assignments[patient_id] for patient_id in patient_ids}

    return split_patient_ids(
        patient_ids=patient_ids,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )


def write_dataset(records: list[RegressionRecord], output_root: Path, args):
    output_root.mkdir(parents=True, exist_ok=True)
    metadata_root = output_root / "modified_dataset"
    metadata_root.mkdir(parents=True, exist_ok=True)

    records = sorted(records, key=lambda record: (record.patient_id, record.segment_id, record.filename))
    assignments = assign_patient_splits(records=records, args=args)

    metadata_rows = []
    for record in records:
        target_values = [record.targets[lead_id] for lead_id in record.source_leads]
        row = {
            "label": str(target_values),
            "filename": record.filename,
            "patient_id": record.patient_id,
            "segment_id": record.segment_id,
            "source_path": record.source_path,
            "source_leads": ",".join(str(lead_id) for lead_id in record.source_leads),
            "target_columns": ",".join(f"target_lead{lead_id}_mV" for lead_id in record.source_leads),
            "strat_fold_annotated": assignments[record.patient_id],
        }
        for lead_id in record.source_leads:
            row[f"target_lead{lead_id}_mV"] = record.targets[lead_id]
        metadata_rows.append(row)

    metadata = pd.DataFrame(metadata_rows)
    metadata.to_csv(metadata_root / "modified_dataset.csv", index=False)

    compression = None if args.compression.lower() == "none" else args.compression
    tracing_shape = (len(records),) + records[0].tracing.shape
    with h5py.File(output_root / args.h5_name, "w") as h5_file:
        dataset = h5_file.create_dataset(
            "tracings",
            shape=tracing_shape,
            dtype=np.float32,
            compression=compression,
        )
        for index, record in enumerate(records):
            dataset[index, :, :] = record.tracing

    split_counts = metadata.groupby("strat_fold_annotated").size().to_dict()
    patient_split_counts = metadata[["patient_id", "strat_fold_annotated"]].drop_duplicates() \
        .groupby("strat_fold_annotated").size().to_dict()

    print(f"\nWrote {len(records)} samples to {output_root}")
    print(f"H5 file: {output_root / args.h5_name}")
    print(f"Metadata: {metadata_root / 'modified_dataset.csv'}")
    print(f"Tracing shape: {tracing_shape}")
    print(f"Split counts by sample: {split_counts}")
    print(f"Split counts by patient: {patient_split_counts}")
    print("Patient-level split mapping:")
    print(f"  train={Split.TRAIN.value}, validation={Split.VALIDATION.value}, test={Split.TEST.value}")


def main():
    args = parse_args()
    input_root = Path(args.input_root)
    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    records = build_records(input_root=input_root, args=args)
    if not records:
        raise ValueError(f"No regression segment CSVs were found under {input_root}")

    write_dataset(records=records, output_root=Path(args.output_root), args=args)


if __name__ == "__main__":
    main()
