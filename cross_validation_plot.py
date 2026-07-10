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
        "blimp": "BLiMP",
        "blimp-nl": "BLiMP-NL",
        "turblimp": "TurBLiMP",
        "clams_ru": "CLAMS-RU",

        "wh_movement": "Wh-Movement",
        "wh_movement_restrictions": "Wh-Movement Restrictions",
        "r_words": "R-Words",
        "adpositional_phrases": "Adpositional Phrases",
        "adverbial_modification": "Adverbial Modification",
        "anaphor_agreement": "Anaphor Agreement",
        "argument_structure": "Argument Structure",
        "argument_structure_ditransitive": "Argument Structure Ditransitive",
        "argument_structure_transitive": "Argument Structure Transitive",
        "auxiliaries": "Auxiliaries",
        "binding": "Binding",
        "binding_principle_a": "Binding Principle A",
        "complementive": "Complementive",
        "crossing_dependencies": "Crossing Dependencies",
        "determiners": "Determiners",
        "ellipsis": "Ellipsis",
        "extraposition": "Extraposition",
        "finite_argument_clause": "Finite Argument Clause",
        "infinitival_argument_clause": "Infinitival Argument Clause",
        "irregular_forms": "Irregular Forms",
        "island_effects": "Island Effects",
        "nominalization": "Nominalization",
        "npi_licensing": "NPI Licensing",
        "parasitic_gaps": "Parasitic Gaps",
        "passive": "Passive",
        "quantifiers": "Quantifiers",
        "question_formation": "Question Formation",
        "relativization": "Relativization",
        "suspended_affixation": "Suspended Affixation",
        "topicalization": "Topicalization",
        "verb_agreement": "Verb Agreement",
        "verb_second": "Verb Second",
        "word_order": "Word Order",
        "word_structure": "Word Structure",
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
    directory_path = Path(directory)
    dataset_key = dataset.lower()

    patterns = [
        f"cross-validation_{dataset_key}_*_{percentage}%_{num_folds}-fold.txt",
        f"cross-validation_{dataset_key}_*.txt",
    ]

    for pattern in patterns:
        matches = sorted(directory_path.glob(pattern))
        if matches:
            return matches[-1]

    raise FileNotFoundError(
        f"No cross-validation file found in {directory_path} for dataset={dataset_key}"
    )


def build_suite_to_category(dataset: str) -> Dict[str, str]:
    dataset_key = dataset.lower()

    if dataset_key not in CATEGORIES:
        raise KeyError(
            f"{dataset_key!r} is not in CATEGORIES. "
            f"Add CATEGORIES[{dataset_key!r}] to categories.py first."
        )

    suite_to_category = {}

    for category, suites in CATEGORIES[dataset_key].items():
        for suite in suites:
            suite_to_category[suite] = category

    return suite_to_category


def extract_percentage(line: str, suite: Optional[str] = None) -> Optional[float]:
    """
    Prefer the value inside parentheses, e.g.
        559 (64.25%) simple_agrmt
    Falls back to extracting numbers after removing the suite name.
    """
    paren_match = re.search(r"\(([-+]?\d*\.\d+|[-+]?\d+)%?\)", line)
    if paren_match:
        return float(paren_match.group(1))

    cleaned = line

    if suite is not None:
        cleaned = cleaned.replace(suite, "")

    cleaned = cleaned.replace("%", "")

    nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", cleaned)

    if not nums:
        return None

    values = [float(x) for x in nums]
    plausible = [v for v in values if 0 <= v <= 100]

    if plausible:
        return plausible[-1]

    return values[-1]


def parse_cross_validation_file(
    path: Path,
    valid_suites: List[str],
) -> Tuple[pd.DataFrame, Optional[float]]:
    rows = []
    random_value = None

    valid_suites = sorted(valid_suites, key=len, reverse=True)

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for raw_line in lines:
        line = raw_line.strip()

        if not line:
            continue

        lower = line.lower()

        if lower.startswith("random") or " random " in f" {lower} ":
            val = extract_percentage(line)
            if val is not None:
                random_value = val
            continue

        matched_suite = None

        for suite in valid_suites:
            if suite in line:
                matched_suite = suite
                break

        if matched_suite is None:
            # Fallback: assume the final whitespace-separated token is the suite name.
            parts = line.split()
            if parts:
                matched_suite = parts[-1]

        if matched_suite is None:
            continue

        val = extract_percentage(line, suite=matched_suite)

        if val is not None:
            rows.append(
                {
                    "suite": matched_suite,
                    "overlap": val,
                    "raw_line": line,
                }
            )

    if not rows:
        raise ValueError(
            f"Could not parse any suite overlaps from {path}. "
            "Open the text file and check its format."
        )

    df = pd.DataFrame(rows).drop_duplicates(subset=["suite"], keep="last")

    # Convert fractions to percentages if needed.
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

    # If a suite is not in categories.py, treat it as its own category.
    df["category"] = df["category"].fillna(df["suite"])

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
        fig_height = max(7, 0.4 * n + 1.5)
        label_fontsize = 10
        value_fontsize = 10
    else:
        fig_height = max(7, 0.35 * n + 1.5)
        label_fontsize = 9
        value_fontsize = 9

    fig, ax = plt.subplots(figsize=(12, fig_height))

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
    x_max = min(100, max(10, np.ceil((max_val + 10) / 10.0) * 10.0))
    ax.set_xlim(0, x_max)

    ax.set_title(title, fontsize=18)
    ax.set_xlabel("Percentage of units", fontsize=12)
    ax.set_ylabel(f"{pretty_name(dataset.lower())} Category", fontsize=12)

    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", length=0)

    ax.grid(axis="x", linestyle=":", alpha=0.4, zorder=0)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.1)
        spine.set_color("black")

    for bar, val in zip(bars, plot_df["overlap"]):
        ax.text(
            val + 0.8,
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

    # Put legend outside if there are many categories.
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=True,
        fontsize=9 if len(legend_handles) > 12 else 11,
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
    p.add_argument("--dataset", default="TurBLiMP")
    p.add_argument("--directory", default="multilingual/turblimp")
    p.add_argument("--title", default="Average 2-fold overlap in TurBLiMP")
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
    print(
        df[["suite", "overlap"]]
        .sort_values("overlap", ascending=False)
        .to_string(index=False)
    )

    print(f"\nParsed {len(df)} rows")
    print(f"Average overlap: {df['overlap'].mean():.2f}%")
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
