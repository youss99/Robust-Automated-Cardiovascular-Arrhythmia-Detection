import argparse
import math
import os
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


@dataclass
class SampleRecord:
    patient_id: str
    label: str
    segment_id: str
    filename: str
    source_path: str
    source_lead: str | None
    tracing: np.ndarray


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert custom binary ECG CSV folders into repo-compatible H5 + metadata datasets."
    )

    parser.add_argument("--single-lead-root", type=str, default=None,
                        help="Root for the single-lead task. Expected layout: root/patient_id/lead_0/*.csv and root/patient_id/lead_1/*.csv")
    parser.add_argument("--two-lead-root", type=str, default=None,
                        help="Root for the two-lead task. Expected layout: root/patient_id/*.csv where each file contains both leads.")
    parser.add_argument("--single-output-root", type=str, default=None,
                        help="Output root for the single-lead dataset.")
    parser.add_argument("--two-output-root", type=str, default=None,
                        help="Output root for the two-lead dataset.")
    parser.add_argument("--single-lead-separate-by-source-lead", action=argparse.BooleanOptionalAction, default=False,
                        help="If set, write one patient-split single-lead dataset per source lead folder, for example lead0 and lead1.")

    parser.add_argument("--source-freq", type=int, default=250,
                        help="Sampling rate of the source CSV signals.")
    parser.add_argument("--target-freq", type=int, default=100,
                        help="Sampling rate of the generated dataset.")
    parser.add_argument("--segment-seconds", type=int, default=10,
                        help="Expected segment length in seconds.")

    parser.add_argument("--signal-start-row", type=int, default=0,
                        help="Index of the first row that contains signal samples. Set this to 1 if row 0 is reserved for metadata/labels.")
    parser.add_argument("--label-source", type=str, choices=["filename", "csv", "auto"], default="filename",
                        help="Where to read labels from. 'filename' uses the final underscore-delimited token in the filename.")
    parser.add_argument("--single-label-column", type=int, default=1,
                        help="Label column for single-lead CSVs when --label-source is csv or auto.")
    parser.add_argument("--two-label-column", type=int, default=2,
                        help="Label column for two-lead CSVs when --label-source is csv or auto.")
    parser.add_argument("--single-signal-column", type=int, default=0,
                        help="Signal column index for the single-lead task.")
    parser.add_argument("--two-signal-columns", nargs=2, type=int, default=[0, 1],
                        help="Signal column indices for the two-lead task.")

    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=200)
    parser.add_argument("--single-split-manifest", type=str, default=None,
                        help="Optional CSV with columns patient_id,split for the single-lead task.")
    parser.add_argument("--two-split-manifest", type=str, default=None,
                        help="Optional CSV with columns patient_id,split for the two-lead task.")

    parser.add_argument("--compression", type=str, default="gzip",
                        help="Compression algorithm for the output H5 dataset. Use 'none' to disable compression.")

    args = parser.parse_args()

    ratio_sum = args.train_ratio + args.valid_ratio + args.test_ratio
    if not math.isclose(ratio_sum, 1.0, rel_tol=1e-6):
        raise ValueError(f"Split ratios must sum to 1.0, received {ratio_sum}")

    if not args.single_lead_root and not args.two_lead_root:
        raise ValueError("Provide at least one of --single-lead-root or --two-lead-root.")

    return args


def normalize_label(label) -> str:
    if pd.isna(label):
        return ""

    label_str = str(label).strip()
    try:
        numeric_label = float(label_str)
    except ValueError:
        return label_str

    if numeric_label.is_integer():
        return str(int(numeric_label))
    return str(numeric_label)


def parse_single_lead_filename(path: Path) -> tuple[str, str, str, str]:
    stem = path.stem
    try:
        patient_id, lead_token, label, segment_id = stem.rsplit("_", 3)
    except ValueError as exc:
        raise ValueError(
            f"Single-lead filenames must follow [patientid]_[lead#]_[label]_[segment#].csv. Received: {path.name}"
        ) from exc
    return patient_id, lead_token, normalize_label(label), segment_id


def parse_two_lead_filename(path: Path) -> tuple[str, str, str]:
    stem = path.stem
    try:
        patient_id, label, segment_id = stem.rsplit("_", 2)
    except ValueError as exc:
        raise ValueError(
            f"Two-lead filenames must follow [patientid]_[label]_[segment#].csv. Received: {path.name}"
        ) from exc
    return patient_id, normalize_label(label), segment_id


def extract_csv_label(df: pd.DataFrame, label_column: int, path: Path) -> str:
    if df.empty:
        raise ValueError(f"CSV is empty: {path}")
    if label_column >= df.shape[1]:
        raise ValueError(f"Label column {label_column} is out of bounds for {path}")
    label = df.iloc[0, label_column]
    if pd.isna(label):
        raise ValueError(f"Label column {label_column} is empty in {path}")
    return normalize_label(label)


def resolve_label(df: pd.DataFrame, path: Path, label_source: str, label_column: int, filename_label: str) -> str:
    if label_source == "filename":
        return filename_label

    if label_source == "csv":
        return extract_csv_label(df=df, label_column=label_column, path=path)

    csv_label = extract_csv_label(df=df, label_column=label_column, path=path)
    if filename_label != csv_label:
        raise ValueError(
            f"Filename label '{filename_label}' does not match CSV label '{csv_label}' for {path}"
        )
    return csv_label


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


def load_signal_column(df: pd.DataFrame, column: int, signal_start_row: int, path: Path) -> np.ndarray:
    if column >= df.shape[1]:
        raise ValueError(f"Signal column {column} is out of bounds for {path}")
    signal = pd.to_numeric(df.iloc[signal_start_row:, column], errors="coerce").dropna().to_numpy(dtype=np.float32)
    if signal.size == 0:
        raise ValueError(f"No signal values were found in column {column} for {path}. Check signal_start_row.")
    return signal


def build_single_lead_records(root: Path, args) -> list[SampleRecord]:
    records: list[SampleRecord] = []
    expected_length = args.source_freq * args.segment_seconds
    target_length = args.target_freq * args.segment_seconds

    for patient_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for lead_dir in sorted(path for path in patient_dir.iterdir() if path.is_dir()):
            for csv_path in sorted(lead_dir.glob("*.csv")):
                df = pd.read_csv(csv_path, header=None)
                filename_patient_id, filename_lead, filename_label, segment_id = parse_single_lead_filename(csv_path)
                if filename_patient_id != patient_dir.name:
                    raise ValueError(
                        f"Filename patient ID '{filename_patient_id}' does not match parent folder '{patient_dir.name}' "
                        f"for {csv_path}"
                    )
                if filename_lead != lead_dir.name:
                    raise ValueError(
                        f"Filename lead token '{filename_lead}' does not match parent folder '{lead_dir.name}' "
                        f"for {csv_path}"
                    )
                label = resolve_label(df=df, path=csv_path, label_source=args.label_source,
                                      label_column=args.single_label_column, filename_label=filename_label)
                signal = load_signal_column(df=df, column=args.single_signal_column,
                                            signal_start_row=args.signal_start_row, path=csv_path)
                if signal.shape[0] != expected_length:
                    raise ValueError(
                        f"Expected {expected_length} source samples but found {signal.shape[0]} in {csv_path}. "
                        f"Adjust --signal-start-row if row 0 is reserved for metadata."
                    )

                tracing = resample_signal(signal, args.source_freq, args.target_freq, target_length)[None, :]
                records.append(
                    SampleRecord(
                        patient_id=patient_dir.name,
                        label=label,
                        segment_id=segment_id,
                        filename=csv_path.name,
                        source_path=str(csv_path.resolve()),
                        source_lead=lead_dir.name,
                        tracing=tracing,
                    )
                )

    return records


def build_two_lead_records(root: Path, args) -> list[SampleRecord]:
    records: list[SampleRecord] = []
    expected_length = args.source_freq * args.segment_seconds
    target_length = args.target_freq * args.segment_seconds

    for patient_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for csv_path in sorted(patient_dir.glob("*.csv")):
            df = pd.read_csv(csv_path, header=None)
            filename_patient_id, filename_label, segment_id = parse_two_lead_filename(csv_path)
            if filename_patient_id != patient_dir.name:
                raise ValueError(
                    f"Filename patient ID '{filename_patient_id}' does not match parent folder '{patient_dir.name}' "
                    f"for {csv_path}"
                )
            label = resolve_label(df=df, path=csv_path, label_source=args.label_source,
                                  label_column=args.two_label_column, filename_label=filename_label)

            resampled_leads = []
            for column in args.two_signal_columns:
                signal = load_signal_column(df=df, column=column, signal_start_row=args.signal_start_row, path=csv_path)
                if signal.shape[0] != expected_length:
                    raise ValueError(
                        f"Expected {expected_length} source samples but found {signal.shape[0]} in {csv_path}. "
                        f"Adjust --signal-start-row if row 0 is reserved for metadata."
                    )
                resampled_leads.append(resample_signal(signal, args.source_freq, args.target_freq, target_length))

            tracing = np.stack(resampled_leads, axis=0)
            records.append(
                SampleRecord(
                    patient_id=patient_dir.name,
                    label=label,
                    segment_id=segment_id,
                    filename=csv_path.name,
                    source_path=str(csv_path.resolve()),
                    source_lead=None,
                    tracing=tracing,
                )
            )

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
        key = str(row["patient_id"]).strip()
        value = str(row["split"]).strip().lower()
        if value not in split_map:
            raise ValueError(f"Unrecognized split value '{row['split']}' in manifest {manifest_path}")
        assignments[key] = split_map[value]
    return assignments


def split_patient_ids(patient_ids: np.ndarray, train_ratio: float, valid_ratio: float, test_ratio: float, seed: int,
                      stratify_labels: np.ndarray | None = None) -> dict[str, int]:
    temp_ratio = valid_ratio + test_ratio

    if temp_ratio == 0:
        return {patient_id: Split.TRAIN.value for patient_id in patient_ids}

    if stratify_labels is not None:
        train_ids, temp_ids, _, temp_labels = train_test_split(
            patient_ids,
            stratify_labels,
            test_size=temp_ratio,
            stratify=stratify_labels,
            random_state=seed,
        )
    else:
        train_ids, temp_ids = train_test_split(
            patient_ids,
            test_size=temp_ratio,
            random_state=seed,
        )
        temp_labels = None

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
    if temp_labels is not None:
        valid_ids, test_ids = train_test_split(
            temp_ids,
            test_size=test_fraction_within_temp,
            stratify=temp_labels,
            random_state=seed,
        )
    else:
        valid_ids, test_ids = train_test_split(
            temp_ids,
            test_size=test_fraction_within_temp,
            random_state=seed,
        )

    for patient_id in valid_ids:
        assignments[patient_id] = Split.VALIDATION.value
    for patient_id in test_ids:
        assignments[patient_id] = Split.TEST.value

    return assignments


def assign_patient_splits(patient_summary: pd.DataFrame, manifest_path: str | None, train_ratio: float,
                          valid_ratio: float, test_ratio: float, seed: int) -> dict[str, int]:
    if manifest_path:
        assignments = load_split_manifest(Path(manifest_path))
        missing = sorted(set(patient_summary["patient_id"]) - set(assignments))
        if missing:
            raise ValueError(f"Split manifest is missing patient IDs: {missing[:10]}")
        return {patient_id: assignments[patient_id] for patient_id in patient_summary["patient_id"]}

    patient_ids = patient_summary["patient_id"].to_numpy()

    candidate_strata = [
        ("fine patient burden", patient_summary["stratum_fine"].to_numpy()),
        ("coarse patient burden", patient_summary["stratum_coarse"].to_numpy()),
        ("positive-presence", patient_summary["has_positive"].astype(int).to_numpy()),
        ("majority-label", patient_summary["majority_label"].to_numpy()),
        ("unstratified", None),
    ]

    last_error = None
    for strategy_name, stratify_labels in candidate_strata:
        try:
            assignments = split_patient_ids(
                patient_ids=patient_ids,
                train_ratio=train_ratio,
                valid_ratio=valid_ratio,
                test_ratio=test_ratio,
                seed=seed,
                stratify_labels=stratify_labels,
            )
            print(f"Assigned patient-level splits using strategy: {strategy_name}")
            return assignments
        except ValueError as exc:
            last_error = exc

    raise ValueError(
        "Could not create patient-level splits, even after falling back to a non-stratified split."
    ) from last_error


def build_patient_summary(records: list[SampleRecord]) -> pd.DataFrame:
    records_df = pd.DataFrame(
        {
            "patient_id": [record.patient_id for record in records],
            "label": [record.label for record in records],
        }
    )

    summary = records_df.groupby(["patient_id", "label"]).size().unstack(fill_value=0)
    for label in ["0", "1"]:
        if label not in summary.columns:
            summary[label] = 0
    summary = summary[["0", "1"]].rename(columns={"0": "negative_segments", "1": "positive_segments"}).reset_index()

    summary["total_segments"] = summary["negative_segments"] + summary["positive_segments"]
    summary["positive_fraction"] = summary["positive_segments"] / summary["total_segments"]
    summary["has_positive"] = summary["positive_segments"] > 0
    summary["majority_label"] = np.where(
        summary["positive_segments"] >= summary["negative_segments"], "1", "0"
    )

    def stratify_fine(row):
        if row["positive_segments"] == 0:
            return "all_negative"
        if row["negative_segments"] == 0:
            return "all_positive"
        if row["positive_fraction"] < (1 / 3):
            return "mixed_low_positive"
        if row["positive_fraction"] < (2 / 3):
            return "mixed_mid_positive"
        return "mixed_high_positive"

    def stratify_coarse(row):
        if row["positive_segments"] == 0:
            return "all_negative"
        if row["negative_segments"] == 0:
            return "all_positive"
        return "mixed"

    summary["stratum_fine"] = summary.apply(stratify_fine, axis=1)
    summary["stratum_coarse"] = summary.apply(stratify_coarse, axis=1)
    return summary


def write_dataset(records: list[SampleRecord], output_root: Path, h5_name: str, split_manifest: str | None, args,
                  patient_assignments: dict[str, int] | None = None):
    output_root.mkdir(parents=True, exist_ok=True)
    metadata_root = output_root / "modified_dataset"
    metadata_root.mkdir(parents=True, exist_ok=True)

    if patient_assignments is None:
        patient_summary = build_patient_summary(records)
        assignments = assign_patient_splits(
            patient_summary=patient_summary,
            manifest_path=split_manifest,
            train_ratio=args.train_ratio,
            valid_ratio=args.valid_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
    else:
        assignments = patient_assignments

    records = sorted(records, key=lambda record: (record.patient_id, record.filename, record.source_path))

    metadata_rows = []
    for record in records:
        metadata_rows.append(
            {
                "label": str([record.label]),
                "filename": record.filename,
                "patient_id": record.patient_id,
                "segment_id": record.segment_id,
                "source_path": record.source_path,
                "source_lead": record.source_lead,
                "strat_fold_annotated": assignments[record.patient_id],
            }
        )

    metadata = pd.DataFrame(metadata_rows)
    metadata.to_csv(metadata_root / "modified_dataset.csv", index=False)

    compression = None if args.compression.lower() == "none" else args.compression
    tracing_shape = (len(records),) + records[0].tracing.shape
    with h5py.File(output_root / h5_name, "w") as h5_file:
        dataset = h5_file.create_dataset(
            "tracings",
            shape=tracing_shape,
            dtype=np.float32,
            compression=compression,
        )
        for index, record in enumerate(records):
            dataset[index, :, :] = record.tracing

    split_counts = metadata.groupby("strat_fold_annotated").size().to_dict()
    print(f"\nWrote {len(records)} samples to {output_root}")
    print(f"H5 file: {output_root / h5_name}")
    print(f"Metadata: {metadata_root / 'modified_dataset.csv'}")
    print(f"Split counts by sample: {split_counts}")
    print("Patient-level split mapping:")
    print(f"  train={Split.TRAIN.value}, validation={Split.VALIDATION.value}, test={Split.TEST.value}")


def write_single_lead_outputs(records: list[SampleRecord], output_root: Path, split_manifest: str | None, args):
    if not args.single_lead_separate_by_source_lead:
        write_dataset(
            records=records,
            output_root=output_root,
            h5_name="SingleLeadBinaryDataset.h5",
            split_manifest=split_manifest,
            args=args,
        )
        return

    patient_summary = build_patient_summary(records)
    patient_assignments = assign_patient_splits(
        patient_summary=patient_summary,
        manifest_path=split_manifest,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    grouped_records: dict[str, list[SampleRecord]] = {}
    for record in records:
        lead_name = record.source_lead or "unknown_lead"
        grouped_records.setdefault(lead_name, []).append(record)

    print("\nWriting separate patient-split single-lead datasets by source lead:")
    for lead_name, lead_records in sorted(grouped_records.items()):
        print(f"  {lead_name}: {len(lead_records)} samples")
        write_dataset(
            records=lead_records,
            output_root=output_root / lead_name,
            h5_name="SingleLeadBinaryDataset.h5",
            split_manifest=split_manifest,
            args=args,
            patient_assignments=patient_assignments,
        )


def main():
    args = parse_args()

    if args.single_lead_root:
        if not args.single_output_root:
            raise ValueError("--single-output-root is required when --single-lead-root is provided.")
        single_records = build_single_lead_records(root=Path(args.single_lead_root), args=args)
        if not single_records:
            raise ValueError(f"No CSV files were found under {args.single_lead_root}")
        write_single_lead_outputs(
            records=single_records,
            output_root=Path(args.single_output_root),
            split_manifest=args.single_split_manifest,
            args=args,
        )

    if args.two_lead_root:
        if not args.two_output_root:
            raise ValueError("--two-output-root is required when --two-lead-root is provided.")
        two_lead_records = build_two_lead_records(root=Path(args.two_lead_root), args=args)
        if not two_lead_records:
            raise ValueError(f"No CSV files were found under {args.two_lead_root}")
        write_dataset(
            records=two_lead_records,
            output_root=Path(args.two_output_root),
            h5_name="TwoLeadBinaryDataset.h5",
            split_manifest=args.two_split_manifest,
            args=args,
        )


if __name__ == "__main__":
    main()
