import argparse
import random
from pathlib import Path

import pandas as pd


RECORDS = [
    "s20011", "s20051", "s20061", "s20071",
    "s20081", "s20091", "s20101", "s20111",
    "s20121", "s20131", "s20141", "s20201",
    "s20211", "s20221", "s20231", "s20241",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a reproducible LTSTDB patient-level split manifest."
    )
    parser.add_argument("--output", type=str, required=True, help="Output CSV path.")
    parser.add_argument("--seed", type=int, default=200, help="Random seed.")
    parser.add_argument("--train-count", type=int, default=10)
    parser.add_argument("--validation-count", type=int, default=3)
    parser.add_argument("--test-count", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    expected_total = args.train_count + args.validation_count + args.test_count
    if expected_total != len(RECORDS):
        raise ValueError(
            f"Split counts must sum to {len(RECORDS)}, received {expected_total}."
        )

    records = RECORDS.copy()
    rng = random.Random(args.seed)
    rng.shuffle(records)

    train_records = records[:args.train_count]
    validation_records = records[args.train_count:args.train_count + args.validation_count]
    test_records = records[args.train_count + args.validation_count:]

    rows = (
        [{"patient_id": patient_id, "split": "train"} for patient_id in sorted(train_records)]
        + [{"patient_id": patient_id, "split": "validation"} for patient_id in sorted(validation_records)]
        + [{"patient_id": patient_id, "split": "test"} for patient_id in sorted(test_records)]
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)

    print(f"Wrote split manifest to {output_path}")
    print("Train ({}): {}".format(len(train_records), ", ".join(sorted(train_records))))
    print("Validation ({}): {}".format(len(validation_records), ", ".join(sorted(validation_records))))
    print("Test ({}): {}".format(len(test_records), ", ".join(sorted(test_records))))


if __name__ == "__main__":
    main()
