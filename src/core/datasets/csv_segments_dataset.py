import os
import re
import numpy as np
import pandas as pd
import h5py
from pathlib import Path

from src.core.datasets.ecg_dataset import EcgDataset
from src.core.datasets.ecg_interface import Split


def _parse_label_value(raw) -> int:
    """
    Parse a label that may be stored as: 1, '1', "['1']", "[1]", etc.
    Returns integer 0 or 1.
    """
    if isinstance(raw, (int, np.integer)):
        v = int(raw)
        if v not in (0, 1):
            raise ValueError(f"Invalid label integer: {raw}")
        return v
    if isinstance(raw, float):
        v = int(raw)
        if v not in (0, 1):
            raise ValueError(f"Invalid label float: {raw}")
        return v
    if isinstance(raw, str):
        m = re.search(r"[01]", raw)
        if not m:
            raise ValueError(f"Cannot parse label from string: {raw!r}")
        return int(m.group(0))
    s = str(raw)
    m = re.search(r"[01]", s)
    if not m:
        raise ValueError(f"Cannot parse label from value: {raw!r}")
    return int(m.group(0))


def _resolve_csv_path(base_dir: str, filename_value: str) -> str:
    """
    Resolve a filename entry from manifest/labels to an actual CSV file path.
    Handles:
      - Absolute paths
      - Relative paths (joined with base_dir)
      - Missing .csv extension
    """
    candidate = str(filename_value).strip().strip('"').strip("'")
    if not candidate:
        raise ValueError("Empty filename/path in manifest")
    if os.path.isabs(candidate):
        path = candidate
    else:
        path = os.path.join(base_dir, candidate)
    if not os.path.splitext(path)[1]:
        path = path + ".csv"
    return path


def _read_embedded_label(csv_path: str):
    """
    Try to read an embedded label from either:
      - (row=0, col=2)  => third column, first row
      - (row=2, col=0)  => third row, first column
    Returns None if not found or parsing fails.
    """
    try:
        small = pd.read_csv(csv_path, header=None, nrows=3, usecols=[0, 1, 2])
    except Exception:
        return None
    # (0,2)
    try:
        if small.shape[1] >= 3 and small.shape[0] >= 1:
            return _parse_label_value(small.iloc[0, 2])
    except Exception:
        pass
    # (2,0)
    try:
        if small.shape[0] >= 3 and small.shape[1] >= 1:
            return _parse_label_value(small.iloc[2, 0])
    except Exception:
        pass
    return None


class CsvSegmentsDataset(EcgDataset):
    """
    Dataset for CSV segments with 2 leads, 10s duration (supports 100 Hz or 250 Hz).
    CSV format:
      - First two columns are the two ECG leads (I, II conceptually).
      - Labels are provided in a manifest CSV (preferred) with columns: 'label' and one of
        {'filename','file','path','filepath','csv'}, OR optionally embedded in each segment CSV
        at either (row=0, col=2) or (row=2, col=0) as '0' or '1'.
    """

    def __init__(self, root_path, **dataset_kwargs):
        # Create H5 file from CSV files if it doesn't exist
        self._h5_path = os.path.join(root_path, f"{self.__class__.__name__}.h5")
        self._csv_folder = root_path

        if not os.path.exists(self._h5_path):
            print(f"Creating H5 file from CSVs at {self._h5_path}")
            self._create_h5_from_csvs()

        self._h5_file = h5py.File(self._h5_path, 'r')['tracings']

        super().__init__(root_path=root_path, **dataset_kwargs)

    def _find_manifest_csv(self) -> str | None:
        """
        Prefer a mapping file that lists segment CSV paths and labels.
        Tries, in order:
          - <root>/manifest.csv
          - <root>/labels.csv
        Returns path or None if not found.
        """
        candidates = [
            os.path.join(self._csv_folder, "manifest.csv"),
            os.path.join(self._csv_folder, "labels.csv"),
        ]
        for c in candidates:
            if os.path.exists(c):
                try:
                    df = pd.read_csv(c, nrows=1)
                    # must have label and a filename-like column
                    has_label = "label" in df.columns
                    has_path = any(col in df.columns for col in ("filename", "file", "path", "filepath", "csv"))
                    if has_label and has_path:
                        return c
                except Exception:
                    continue
        return None

    def _create_h5_from_csvs(self):
        """Convert CSV files specified in manifest/labels to H5 format expected by the framework."""
        manifest_path = self._find_manifest_csv()

        if manifest_path is None:
            raise ValueError(
                f"No suitable manifest found in {self._csv_folder}. "
                f"Expected a CSV with 'label' and one of "
                f"{{'filename','file','path','filepath','csv'}}."
            )

        df_manifest = pd.read_csv(manifest_path)
        # Determine path column
        fname_col = None
        for cand in ("filename", "file", "path", "filepath", "csv"):
            if cand in df_manifest.columns:
                fname_col = cand
                break
        if fname_col is None or "label" not in df_manifest.columns:
            raise ValueError(
                f"Manifest must contain 'label' and one of "
                f"{{'filename','file','path','filepath','csv'}} columns"
            )

        # Build entries (absolute paths + labels)
        entries: list[tuple[str, int]] = []
        for _, row in df_manifest.iterrows():
            path = _resolve_csv_path(self._csv_folder, row[fname_col])
            if not os.path.exists(path):
                # Try not adding .csv if already had it added incorrectly
                alt = str(row[fname_col]).strip()
                alt = alt if os.path.isabs(alt) else os.path.join(self._csv_folder, alt)
                if os.path.exists(alt):
                    path = alt
                else:
                    print(f"[WARN] CSV file not found, skipping: {path}")
                    continue
            # Prefer manifest label; fallback to embedded if manifest is NaN/empty
            try:
                lbl = _parse_label_value(row["label"])
            except Exception:
                emb = _read_embedded_label(path)
                if emb is None:
                    print(f"[WARN] Could not parse label for {path}, skipping.")
                    continue
                lbl = int(emb)
            entries.append((path, int(lbl)))

        if len(entries) == 0:
            raise ValueError("No valid entries found in manifest after resolving paths and labels.")

        # Helper: read first two leads robustly (handles header row lead_0/lead_1)
        def _read_two_leads(csv_path: str) -> np.ndarray:
            """
            Returns array of shape (2, n_rows) with numeric values only.
            Tries header-aware read with named columns; falls back to header=None.
            """
            try:
                dfh = pd.read_csv(csv_path)
                if {"lead_0", "lead_1"}.issubset(dfh.columns):
                    dff = dfh[["lead_0", "lead_1"]].copy()
                else:
                    # Use first two columns if names aren't present
                    dff = dfh.iloc[:, :2].copy()
            except Exception:
                dff = pd.read_csv(csv_path, header=None, usecols=[0, 1])

            # Coerce to numeric and drop non-numeric rows (e.g., headers)
            for i in range(2):
                dff.iloc[:, i] = pd.to_numeric(dff.iloc[:, i], errors="coerce")
            dff = dff.dropna(subset=[dff.columns[0], dff.columns[1]], how="any")

            return dff.to_numpy(dtype=np.float32).T  # (2, n_rows)

        # Infer number of samples from first entry AFTER removing header/non-numeric rows
        first_path = entries[0][0]
        first_arr = _read_two_leads(first_path)
        n_samples = int(first_arr.shape[1])
        n_leads = 12  # Framework expects 12 leads, we'll pad with zeros

        print(f"Found {len(entries)} segment CSVs (using manifest: {os.path.basename(manifest_path)})")
        print(f"Inferred signal length: {n_samples} samples per lead")

        # Prepare arrays
        all_data = []
        all_labels = []
        all_filenames = []

        for csv_path, lbl in entries:
            ecg_data = _read_two_leads(csv_path)  # Shape: (2, n_rows)

            # Ensure consistent length (pad or trim to n_samples)
            if ecg_data.shape[1] < n_samples:
                # Pad with zeros if shorter
                padding = np.zeros((2, n_samples - ecg_data.shape[1]), dtype=np.float32)
                ecg_data = np.concatenate([ecg_data, padding], axis=1)
            elif ecg_data.shape[1] > n_samples:
                # Trim if longer
                ecg_data = ecg_data[:, :n_samples]

            # Pad to 12 leads with zeros
            padded_data = np.zeros((n_leads, n_samples), dtype=np.float32)
            padded_data[:2, :] = ecg_data

            all_data.append(padded_data)
            # Format label as string list for compatibility with framework (e.g., "['0']" or "['1']")
            all_labels.append(f"['{int(lbl)}']")
            all_filenames.append(Path(csv_path).stem)

        # Stack all data
        all_data = np.stack(all_data, axis=0)  # Shape: (n_files, 12, n_samples)

        # Save to H5
        with h5py.File(self._h5_path, 'w') as f:
            f.create_dataset('tracings', data=all_data, dtype=np.float32)

        # Create CSV metadata file (parent class will create original_index from DataFrame index)
        csv_path = os.path.join(self._csv_folder, f'{self.__class__.__name__}')
        os.makedirs(csv_path, exist_ok=True)

        metadata_df = pd.DataFrame({
            'label': all_labels,
            'filename': all_filenames
        })
        metadata_df.to_csv(os.path.join(csv_path, 'labels.csv'), index=False)

        print(f"Created H5 file with {len(all_data)} samples")
        print(f"Label distribution: {pd.Series(all_labels).value_counts().to_dict()}")

    def preprocess_labeled_data(self, label_encoder, n_hot_vector):
        """
        Override parent method to handle 1D arrays from LabelEncoder for multi-class classification.
        The parent class expects 2D arrays (multi-label), but LabelEncoder returns 1D arrays.
        """
        # For multi-class with LabelEncoder, n_hot_vector is 1D (class indices)
        # Convert it to 2D for compatibility with parent class logic
        if n_hot_vector.ndim == 1:
            # Get unique classes and their counts
            unique_classes, counts = np.unique(n_hot_vector, return_counts=True)

            # Filter classes by minimum size
            valid_classes = unique_classes[counts >= self._min_class_size]

            if len(valid_classes) == 0:
                raise ValueError(f"No classes have at least {self._min_class_size} samples")

            # Get indices of samples belonging to valid classes
            df_indices = np.where(np.isin(n_hot_vector, valid_classes))[0]

            # Update label encoder to only include valid classes
            label_encoder.classes_ = np.array([label_encoder.classes_[c] for c in valid_classes])

            # Re-encode the labels with the filtered classes
            filtered_labels = n_hot_vector[df_indices]
            # Remap to new indices
            class_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(valid_classes)}
            n_hot_vector = np.array([class_mapping[label] for label in filtered_labels])

            return df_indices, n_hot_vector, label_encoder
        else:
            # Fall back to parent implementation for multi-label case
            return super().preprocess_labeled_data(label_encoder, n_hot_vector)

    @property
    def h5_file(self):
        return self._h5_file

    @property
    def dataset_folder(self):
        return 'CsvSegmentsDataset'

    @property
    def csv_path(self):
        return 'labels.csv'

    def __len__(self):
        return len(self._data_y)
