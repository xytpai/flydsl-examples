import glob
import json


def merge_tuned_data(prefix, out_prefix):
    files = glob.glob(f"{prefix}_m_*.jsonl")
    merged_data = []
    for file in files:
        print(file, flush=True)
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                merged_data.append(data)
    with open(f"{out_prefix}.jsonl", "w", encoding="utf-8") as f:
        for data in merged_data:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
            f.flush()


if __name__ == '__main__':
    merge_tuned_data('temp/hgemm_tuned', 'hgemm_tuned')
