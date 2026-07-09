from pathlib import Path
import csv

BENCHMARKS_DIR = Path(__file__).resolve().parent
RAW_DIR = BENCHMARKS_DIR / "raw" / "clams" / "ru_evalset"
OUT_DIR = BENCHMARKS_DIR / "processed" / "clams_ru"

BOOLS = {"True", "False"}


def parse_clams_file(path: Path):
    tokens = path.read_text(encoding="utf-8").strip().split()

    examples = []
    i = 0

    while i < len(tokens):
        label = tokens[i]

        if label not in BOOLS:
            raise ValueError(f"Expected True/False in {path}, got {label!r}")

        i += 1
        sent = []

        while i < len(tokens) and tokens[i] not in BOOLS:
            sent.append(tokens[i])
            i += 1

        examples.append((label, sent))

    if len(examples) % 2 != 0:
        raise ValueError(f"Odd number of examples in {path}")

    pairs = []

    for j in range(0, len(examples), 2):
        label_good, good_words = examples[j]
        label_bad, bad_words = examples[j + 1]

        if label_good != "True" or label_bad != "False":
            raise ValueError(
                f"Expected True/False pair in {path}, got "
                f"{label_good}/{label_bad} at pair {j // 2}"
            )

        pairs.append((good_words, bad_words))

    return pairs


def convert_file(path: Path):
    pairs = parse_clams_file(path)

    max_words = max(
        max(len(good_words), len(bad_words))
        for good_words, bad_words in pairs
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{path.stem}.csv"

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)

        header = [f"stim{i + 1}" for i in range(max_words + 2)]
        writer.writerow(header)

        for i, (good_words, bad_words) in enumerate(pairs):
            writer.writerow([2 * i + 1] + good_words + ["+"])
            writer.writerow([2 * i + 2] + bad_words + ["-"])

    print(f"Wrote {out_path} with {len(pairs)} pairs")


def main():
    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"Could not find {RAW_DIR}. Did you initialise the CLAMS submodule?"
        )

    for path in sorted(RAW_DIR.glob("*.txt")):
        convert_file(path)


if __name__ == "__main__":
    main()
