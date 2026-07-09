from pathlib import Path
import csv

import pandas as pd
from datasets import get_dataset_config_names, load_dataset


DATASET_ID = "jmichaelov/blimp_nl"

BENCHMARKS_DIR = Path(__file__).resolve().parent

RAW_DIR = BENCHMARKS_DIR / "raw" / "blimp-nl"
PROCESSED_DIR = BENCHMARKS_DIR / "processed" / "blimp-nl"

CLEAR_OLD_RAW = True
CLEAR_OLD_PROCESSED = True


def sentence_to_words(sentence):
    return str(sentence).strip().split()


def download_raw():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if CLEAR_OLD_RAW:
        for old_file in RAW_DIR.glob("*.csv"):
            old_file.unlink()

    configs = get_dataset_config_names(DATASET_ID)

    print(f"Found {len(configs)} BLiMP-NL subsets:")
    for config in configs:
        print(" ", config)

    for config in configs:
        out_path = RAW_DIR / f"{config}.csv"

        print(f"\nDownloading subset: {config}")

        ds = load_dataset(DATASET_ID, config, split="test")
        df = ds.to_pandas()

        required = {"sentence_good", "sentence_bad"}
        missing = required - set(df.columns)

        if missing:
            raise ValueError(
                f"{config} is missing required columns: {missing}. "
                f"Available columns: {list(df.columns)}"
            )

        df.to_csv(out_path, index=False)
        print(f"Saved {len(df)} rows to {out_path}")


def convert_suite(suite_name):
    raw_path = RAW_DIR / f"{suite_name}.csv"
    processed_path = PROCESSED_DIR / f"{suite_name}.csv"

    df = pd.read_csv(raw_path)

    required = {"sentence_good", "sentence_bad"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{raw_path} is missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    good_sentences = []
    bad_sentences = []
    max_words = 0

    for _, row in df.iterrows():
        good_words = sentence_to_words(row["sentence_good"])
        bad_words = sentence_to_words(row["sentence_bad"])

        good_sentences.append(good_words)
        bad_sentences.append(bad_words)

        max_words = max(max_words, len(good_words), len(bad_words))

    # stim1 = item id
    # stim2...stimN = sentence words
    # final stim column = '+' or '-'
    n_cols = max_words + 2
    header = [f"stim{i + 1}" for i in range(n_cols)]

    with open(processed_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for i, (good_words, bad_words) in enumerate(
            zip(good_sentences, bad_sentences)
        ):
            good_row = [2 * i + 1] + good_words + ["+"]
            bad_row = [2 * i + 2] + bad_words + ["-"]

            good_row += [""] * (n_cols - len(good_row))
            bad_row += [""] * (n_cols - len(bad_row))

            writer.writerow(good_row)
            writer.writerow(bad_row)

    print(
        f"Converted {suite_name}: "
        f"{len(df)} pairs / {2 * len(df)} rows -> {processed_path}"
    )


def convert_all():
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Raw directory does not exist: {RAW_DIR}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if CLEAR_OLD_PROCESSED:
        for old_file in PROCESSED_DIR.glob("*.csv"):
            old_file.unlink()

    raw_files = sorted(RAW_DIR.glob("*.csv"))

    if not raw_files:
        raise FileNotFoundError(
            f"No raw BLiMP-NL CSV files found in {RAW_DIR}"
        )

    for raw_file in raw_files:
        suite_name = raw_file.stem
        convert_suite(suite_name)


def main():
    print("=" * 80)
    print("BLiMP-NL raw download + stimuli conversion")
    print("Dataset:", DATASET_ID)
    print("Raw dir:", RAW_DIR)
    print("Processed dir:", PROCESSED_DIR)
    print("=" * 80)

    download_raw()
    convert_all()

    print("\nDone.")
    print(f"Raw files: {RAW_DIR}")
    print(f"Processed files: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
