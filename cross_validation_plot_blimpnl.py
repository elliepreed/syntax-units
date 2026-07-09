import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from categories import CATEGORIES


def pretty_name(name: str) -> str:
    special = {
        "blimp-nl": "BLiMP-NL",
        "wh_movement": "Wh-Movement",
        "wh_movement_restrictions": "Wh-Movement Restrictions",
        "r_words": "R-Words",
        "adpositional_phrases": "Adpositional Phrases",
        "adverbial_modification": "Adverbial Modification",
        "anaphor_agreement": "Anaphor Agreement",
        "argument_structure": "Argument Structure",
        "auxiliaries": "Auxiliaries",
        "binding_principle_a": "Binding Principle A",
        "complementive": "Complementive",
        "crossing_dependencies": "Crossing Dependencies",
        "determiners": "Determiners",
        "extraposition": "Extraposition",
        "finite_argument_clause": "Finite Argument Clause",
        "infinitival_argument_clause": "Infinitival Argument Clause",
        "nominalization": "Nominalization",
        "parasitic_gaps": "Parasitic Gaps",
        "quantifiers": "Quantifiers",
        "relativization": "Relativization",
        "topicalization": "Topicalization",
        "verb_second": "Verb Second",
    }

    if name in special:
        return special[name]

    return (
        name.replace("__", ": ")
        .replace("_", " ")
        .replace("-", " ")
        .title()
    )


def find_result_file(
    directory: str,
    dataset: str,
    percentage: float,
    num_folds: int,
) -> Path:
    directory = Path(directory)
    dataset = dataset.lower()

    patterns = [
        f"cross-validation_{dataset}_*_{percentage}%_{num_folds}-fold.txt",
        f"cross-validation_{dataset}_*.txt",
    ]

    for pattern in patterns:
        matches = sorted(directory.glob(pattern))
        if matches:
            return matches[-1]

    raise FileNotFoundError(
        f"No cross-validation file found in {directory} for dataset={dataset}"
    )


def build_suite_to_category(dataset: str) -> Dict[str, str]:
    category_map = CATEGORIES[dataset.lower()]

    suite_to_category = {}

    for category, suites in category_map.items():
        for suite in suites:
            suite_to_category[suite] = category

    return suite_to_category


def extract_number_excluding_suite(line: str, suite: Optional[str] = None) -> Optional[float]:
    """
    Extract the actual overlap value from a result line.

    Important: BLiMP-NL paradigm names contain numbers, e.g.
    wh_movement__stranding_2
    infinitival_argument_clause__bare_verb_type_3

    So we first remove the suite name before looking for numbers.
    """
    cleaned = line

    if suite is not None:
        cleaned = cleaned.replace(suite, "")

    cleaned = cleaned.replace("%", "")

    nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", cleaned)

    if not nums:
        return None

    values = [float(x) for x in nums]

    # Prefer plausible percentage values.
    percentage_like = [v for v in values if 0 <= v <= 100]

    if percentage_like:
        return percentage_like[-1]

    return values[-1]


def parse_cross_validation_file(
    path: Path,
    valid_suites: List[str],
) -> Tuple[pd.DataFrame, Optional[float]]:
    rows = []
    random_value = None

    # Longest first avoids matching a shorter suite inside a longer suite name.
    valid_suites = sorted(valid_suites, key=len, reverse=True)

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for raw_line in lines:
        line = raw_line.strip()

        if not line:
            continue

        lower = line.lower()

        if lower.startswith("random") or " random " in f" {lower} ":
            val = extract_number_excluding_suite(line)

            if val is not None:
                random_value = val

            continue

        for suite in valid_suites:
            if suite in line:
                val = extract_number_excluding_suite(line, suite=suite)

                if val is not None:
                    rows.append(
                        {
                            "suite": suite,
                            "overlap": val,
                            "raw_line": line,
                        }
                    )

                break

    if not rows:
        raise ValueError(
            f"Could not parse any suite overlaps from {path}. "
            "Open the txt file and check its format."
        )

    df = pd.DataFrame(rows).drop_duplicates(subset=["suite"], keep="last")

    # If values are fractions, convert to percentages.
    if df["overlap"].max() <= 1.0:
        df["overlap"] *= 100.0

    if random_value is not None and random_value <= 1.0:
        random_value *= 100.0

    return df, random_value


def make_category_colors(categories: List[str]) -> Dict[str, tuple]:
    cmaps = [
        plt.get_cmap("tab20"),
        plt.get_cmap("tab20b"),
        plt.get_cmap("tab20c"),
    ]

    colors = []

    for cmap in cmaps:
        colors.extend([cmap(i) for i in range(cmap.N)])

    return {
        category: colors[i % len(colors)]
        for i, category in enumerate(categories)
    }


def plot_cross_validation(
    df: pd.DataFrame,
    random_value: Optional[float],
    suite_to_category: Dict[str, str],
    dataset: str,
    directory: str,
    title: str,
    aggregate_by_category: bool,
):
    df = df.copy()
    df["category"] = df["suite"].map(suite_to_category)

    missing = df[df["category"].isna()]["suite"].tolist()

    if missing:
        raise ValueError(
            "These suites are in the cross-validation result but not in categories.py:\n"
            + "\n".join(missing)
        )

    if aggregate_by_category:
        plot_df = (
            df.groupby("category", as_index=False)["overlap"]
            .mean()
            .rename(columns={"category": "label"})
        )
        plot_df["category"] = plot_df["label"]
    else:
        plot_df = df.rename(columns={"suite": "label"})

    plot_df = plot_df.sort_values("overlap", ascending=False).reset_index(drop=True)

    categories_in_plot = list(dict.fromkeys(plot_df["category"].tolist()))
    category_to_color = make_category_colors(categories_in_plot)

    if random_value is not None:
        random_row = pd.DataFrame(
            {
                "label": ["Random"],
                "overlap": [random_value],
                "category": ["Random"],
            }
        )

        plot_df = pd.concat([random_row, plot_df], ignore_index=True)
        category_to_color["Random"] = "#BDBDBD"

    n = len(plot_df)

    if aggregate_by_category:
        fig_height = max(7, 0.35 * n + 1.5)
        label_fontsize = 10
        value_fontsize = 11
    else:
        fig_height = max(10, 0.18 * n + 1.5)
        label_fontsize = 5.5
        value_fontsize = 8

    fig, ax = plt.subplots(figsize=(15, fig_height))

    y = np.arange(n)

    bar_labels = [pretty_name(x) for x in plot_df["label"]]
    bar_colors = [category_to_color[c] for c in plot_df["category"]]

    bars = ax.barh(
        y,
        plot_df["overlap"],
        color=bar_colors,
        edgecolor="black",
        linewidth=0.8,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(bar_labels, fontsize=label_fontsize)
    ax.invert_yaxis()

    max_val = float(plot_df["overlap"].max())
    x_max = min(100, max_val + 18)
    ax.set_xlim(0, x_max)

    ax.set_title(title, fontsize=24)
    ax.set_xlabel("")
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", length=0)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
        spine.set_color("black")

    for bar, val in zip(bars, plot_df["overlap"]):
        ax.text(
            val + 1.0,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}%",
            va="center",
            ha="left",
            fontsize=value_fontsize,
        )

    legend_categories = categories_in_plot.copy()

    if random_value is not None:
        legend_categories.append("Random")

    legend_handles = [
        Patch(
            facecolor=category_to_color[cat],
            edgecolor="black",
            label=pretty_name(cat),
        )
        for cat in legend_categories
    ]

    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=True,
        fontsize=11 if len(legend_handles) > 16 else 14,
    )

    fig.tight_layout()

    dataset_key = dataset.lower()
    suffix = "category" if aggregate_by_category else "paradigm"

    out_png = Path(directory) / f"cross_validation_{dataset_key}_{suffix}.png"
    out_pdf = Path(directory) / f"cross_validation_{dataset_key}_{suffix}.pdf"

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")

    print(f"Saved {out_png}")
    print(f"Saved {out_pdf}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="BLiMP-NL")
    p.add_argument("--directory", default="multilingual/blimpnl")
    p.add_argument("--title", default="2-fold overlap (gemma-3-4b-pt)")
    p.add_argument("--percentage", type=float, default=1.0)
    p.add_argument("--num-folds", type=int, default=2)
    p.add_argument("--aggregate-by-category", action="store_true")
    args = p.parse_args()

    dataset_key = args.dataset.lower()

    suite_to_category = build_suite_to_category(dataset_key)
    valid_suites = list(suite_to_category.keys())

    path = find_result_file(
        directory=args.directory,
        dataset=dataset_key,
        percentage=args.percentage,
        num_folds=args.num_folds,
    )

    print(f"Reading {path}")

    df, random_value = parse_cross_validation_file(path, valid_suites)

    print("\nParsed values:")
    print(df[["suite", "overlap"]].sort_values("overlap", ascending=False).to_string(index=False))
    print(f"\nParsed {len(df)} rows")
    print(f"Random value: {random_value}")

    plot_cross_validation(
        df=df,
        random_value=random_value,
        suite_to_category=suite_to_category,
        dataset=dataset_key,
        directory=args.directory,
        title=args.title,
        aggregate_by_category=args.aggregate_by_category,
    )


if __name__ == "__main__":
    main()
