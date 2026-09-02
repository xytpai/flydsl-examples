import argparse
import csv
from pathlib import Path


def list_csv_files(directory: str | Path) -> list[Path]:
    directory = Path(directory)
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower() == ".csv"
        and not path.stem.endswith("_dedup")
        and path.name != "_missing.csv"
    )


KEY_COLS = [
    "gfx",
    "cu_num",
    "M",
    "N",
    "K",
    "bias",
    "dtype",
    "outdtype",
    "scaleAB",
    "bpreshuffle",
]
TUNING_INPUT_COLS = KEY_COLS[2:]


US_COL = "us"


def aggregate_csv_data(input_files):
    best = {}
    fname_best = {}
    header = None
    for input_file in input_files:
        with open(input_file, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or all(x.strip() == "" for x in row):
                    continue
                row = [x.strip() for x in row]
                if row[0] == "gfx":
                    header = row
                    continue
                if header is None:
                    raise RuntimeError("CSV header not found")
                if len(row) != len(header):
                    continue
                record = dict(zip(header, row))
                key = tuple(record[col] for col in KEY_COLS)
                us = float(record[US_COL])
                if key not in best or us < float(best[key][US_COL]):
                    best[key] = record
                    fname_best[key] = input_file
    return best, fname_best


def replace_csv_rows(
    input_directory: str | Path,
    output_directory: str | Path,
    aggregate_data: dict[tuple[str, ...], dict[str, str]],
    gfx: str | None = None,
) -> None:
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    missing_rows = set()
    for input_file in list_csv_files(input_directory):
        with open(input_file, "r", encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)
            with open(
                output_directory / input_file.name,
                "w",
                encoding="utf-8",
                newline="",
            ) as target:
                writer = csv.DictWriter(target, fieldnames=reader.fieldnames)
                writer.writeheader()
                for row in reader:
                    key = tuple(row[column].strip() for column in KEY_COLS)
                    if row["gfx"].strip() == gfx and key not in aggregate_data:
                        missing_row = tuple(
                            row[column].strip() for column in TUNING_INPUT_COLS
                        )
                        missing_rows.add(missing_row)
                    writer.writerow(aggregate_data.get(key, row))

    with open(
        output_directory / "_missing.csv",
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.writer(f)
        writer.writerow(TUNING_INPUT_COLS)
        writer.writerows(sorted(missing_rows))


def main():
    parser = argparse.ArgumentParser(description="Analyze tuned GEMM CSV files.")
    parser.add_argument(
        "-d",
        "--directory",
        type=Path,
        required=True,
        help="Directory containing CSV files used to build the aggregate.",
    )
    parser.add_argument(
        "-r",
        "--replace-directory",
        type=Path,
        required=True,
        help="Directory containing CSV rows to replace.",
    )
    parser.add_argument(
        "-o",
        "--output-directory",
        type=Path,
        required=True,
        help="New directory for the replaced CSV files.",
    )
    parser.add_argument(
        "-g",
        "--gfx",
        help="Require matching aggregate rows for this GPU architecture.",
    )
    args = parser.parse_args()
    csv_files = list_csv_files(args.directory)

    print(f"\nCSV files in {args.directory.resolve()} ({len(csv_files)}):")
    if not csv_files:
        print("  (none)")
    for index, csv_file in enumerate(csv_files, start=1):
        print(f"  {index:>2}. {csv_file.name}")

    aggregate_data, _ = aggregate_csv_data(csv_files)
    flydsl_count = sum(
        "flydsl_hgemm_" in row["kernelName"] for row in aggregate_data.values()
    )
    total_count = len(aggregate_data)
    ratio = flydsl_count / total_count if total_count else 0
    print(f"\nFlyDSL HGEMM: {flydsl_count}/{total_count} ({ratio:.2%})")

    replace_csv_rows(
        args.replace_directory,
        args.output_directory,
        aggregate_data,
        args.gfx,
    )


if __name__ == "__main__":
    main()


# python update_aiter_configs.py -d .\new_tuned -r .\old_tuned -o .\replaced -g gfx950
