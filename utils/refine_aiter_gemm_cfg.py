import csv
from pathlib import Path


INPUT_FILES = [
    'dsv4.csv',
    'kimi.csv',
    'kimik2.csv',
]

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

US_COL = "us"


def dedup_csv(input_files):
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
    for input_file in input_files:
        p = Path(input_file)
        output_file = str(p.with_name(p.stem + "_dedup.csv"))
        with open(output_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            for key, record in best.items():
                if fname_best[key] == input_file:
                    writer.writerow(record)


def main():
    dedup_csv(INPUT_FILES)


if __name__ == "__main__":
    main()
