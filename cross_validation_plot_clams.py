import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LABEL_MAP = {
    "simple_agrmt": "Simple agreement",
    "obj_rel_within_anim": "Object relative\nwithin animate",
    "long_vp_coord": "Long VP coordination",
    "vp_coord": "VP coordination",
    "subj_rel": "Subject relative",
    "obj_rel_across_anim": "Object relative\nacross animate",
    "prep_anim": "Prepositional animate",
}


ORDER = [
    "simple_agrmt",
    "obj_rel_within_anim",
    "long_vp_coord",
    "vp_coord",
    "subj_rel",
    "obj_rel_across_anim",
    "prep_anim",
]


def pretty_label(name):
    return LABEL_MAP.get(name, name.replace("_", " ").title())


def parse_filename(path, dataset):
    """
    Handles filenames like:
    cross-validation_clams_ru_shuffled_gemma-3-4b-pt_1.0%_2-fold.txt
    """
    fname = path.name
    prefix = f"cross-validation_{dataset}_"
    suffix = ".txt"

    if not fname.startswith(prefix) or not fname.endswith(suffix):
        raise ValueError(f"Unexpected filename format: {fname}")

    rest = fname[len(prefix):-len(suffix)]
    model_name, perc_str, num_folds = rest.rsplit("_", 2)

    percentage = float(perc_str.replace("%", ""))
    folds = int(num_folds.replace("-fold", ""))

    return model_name, percentage, folds


def read_cross_validation_file(path):
    """
    Reads lines like:
    559 (64.25%) simple_agrmt
    """
    rows = []

    line_re = re.compile(
        r"^\s*(?P<count>\d+)\s+\((?P<percent>[\d.]+)%\)\s+(?P<phenomenon>.+?)\s*$"
    )

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            match = line_re.match(line)

            if match is None:
                print(f"WARNING: could not parse line: {line}")
                continue

            rows.append(
                {
                    "count": int(match.group("count")),
                    "percent": float(match.group("percent")),
                    "phenomenon": match.group("phenomenon"),
                }
            )

    if not rows:
        raise ValueError(f"No valid rows parsed from {path}")

    df = pd.DataFrame(rows)

    ordered = [x for x in ORDER if x in set(df["phenomenon"])]
    extras = sorted(set(df["phenomenon"]) - set(ordered))
    final_order = ordered + extras

    order_map = {phenomenon: i for i, phenomenon in enumerate(final_order)}
    df["order"] = df["phenomenon"].map(order_map)

    df = df.sort_values("order").drop(columns=["order"])
    return df


def find_input_files(directory, dataset):
    pattern = f"cross-validation_{dataset}_*.txt"
    files = sorted(directory.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No cross-validation files found in {directory} for dataset={dataset}. "
            f"Expected pattern: {pattern}"
        )

    return files


def plot_clams(df, model_name, percentage, folds, title, out_prefix):
    plot_df = df.copy()
    plot_df["label"] = plot_df["phenomenon"].map(pretty_label)

    random_row = pd.DataFrame(
        [
            {
                "count": np.nan,
                "percent": percentage,
                "phenomenon": "random",
                "label": "Random\nexpected",
            }
        ]
    )

    plot_df = pd.concat([plot_df, random_row], ignore_index=True)

    y = np.arange(len(plot_df))

    fig_height = max(4.5, 0.55 * len(plot_df) + 1.8)
    fig, ax = plt.subplots(figsize=(8.5, fig_height))

    bars = ax.barh(
        y,
        plot_df["percent"],
        height=0.72,
        edgecolor="black",
        linewidth=0.8,
        zorder=2,
    )

    bars[-1].set_hatch("//")

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["label"], fontsize=10)
    ax.invert_yaxis()

    max_val = float(plot_df["percent"].max())
    x_max = max(10, np.ceil((max_val + 8) / 10) * 10)

    ax.set_xlim(0, x_max)
    ax.set_xlabel("Percentage of units", fontsize=11)

    if title:
        ax.set_title(title, fontsize=13)
    else:
        ax.set_title(
            f"{folds}-fold overlap on CLAMS-RU ({model_name}, top {percentage:g}% units)",
            fontsize=13,
        )

    ax.grid(axis="x", linestyle=":", alpha=0.4, zorder=0)

    for i, row in plot_df.iterrows():
        val = float(row["percent"])
        ax.text(
            val + 0.6,
            i,
            f"{val:.2f}%",
            va="center",
            ha="left",
            fontsize=9,
        )

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("black")

    fig.tight_layout()

    png_path = out_prefix.with_suffix(".png")
    pdf_path = out_prefix.with_suffix(".pdf")

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    plt.close(fig)

    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        default="clams_ru_shuffled",
        help="Dataset name as it appears in the filename.",
    )

    parser.add_argument(
        "--directory",
        default="multilingual/clams/ru",
        help="Directory containing the cross-validation .txt file.",
    )

    parser.add_argument(
        "--title",
        default="2-fold overlap (gemma-3-4b-pt)",
        help="Plot title.",
    )

    parser.add_argument(
        "--out-name",
        default=None,
        help="Optional output filename stem without extension.",
    )

    args = parser.parse_args()

    directory = Path(args.directory)
    files = find_input_files(directory, args.dataset)

    for path in files:
        model_name, percentage, folds = parse_filename(path, args.dataset)

        print(f"Reading {path}")
        print(f"  model={model_name}")
        print(f"  percentage={percentage:g}%")
        print(f"  folds={folds}")

        df = read_cross_validation_file(path)

        if args.out_name is not None:
            out_prefix = directory / args.out_name
        else:
            out_prefix = directory / (
                f"cross_validation_{args.dataset}_{model_name}_{percentage:g}pct_{folds}fold"
            )

        plot_clams(
            df=df,
            model_name=model_name,
            percentage=percentage,
            folds=folds,
            title=args.title,
            out_prefix=out_prefix,
        )


if __name__ == "__main__":
    main()
