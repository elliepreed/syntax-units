import argparse
from pathlib import Path

import matplotlib.pyplot as plt
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
        "passive": "Passive",
    }

    if name in special:
        return special[name]

    return name.replace("__", ": ").replace("_", " ").replace("-", " ").title()


def find_overlap_file(directory: str, dataset: str, percentage: float) -> Path:
    directory = Path(directory)
    dataset = dataset.lower()

    pattern = f"cross-overlap_{dataset}_{dataset}_*_{percentage}%.csv"
    matches = sorted(directory.glob(pattern))

    if not matches:
        pattern = f"cross-overlap_{dataset}_{dataset}_*.csv"
        matches = sorted(directory.glob(pattern))

    if not matches:
        raise FileNotFoundError(
            f"No within-dataset cross-overlap file found in {directory}"
        )

    return matches[-1]


def convert_counts_to_percent_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    values = df.to_numpy(dtype=float)
    max_val = np.nanmax(values)

    if max_val <= 100:
        print("Matrix appears to already be percentages.")
        return df

    diagonal = np.diag(values)
    num_units = float(np.nanmedian(diagonal))

    if num_units <= 0:
        raise ValueError("Could not infer number of units from diagonal.")

    print(f"Matrix appears to be raw counts. Converting by num_units={num_units:.0f}.")
    return df / num_units * 100.0


def within_values(df: pd.DataFrame, suites):
    suites = [s for s in suites if s in df.index and s in df.columns]

    if len(suites) < 2:
        return np.array([], dtype=float)

    sub = df.loc[suites, suites].to_numpy(dtype=float)
    mask = ~np.eye(len(suites), dtype=bool)

    return sub[mask]


def outbound_values(df: pd.DataFrame, suites, all_suites):
    suites = [s for s in suites if s in df.index]
    outside = [s for s in all_suites if s not in suites and s in df.columns]

    if not suites or not outside:
        return np.array([], dtype=float)

    return df.loc[suites, outside].to_numpy(dtype=float).ravel()


def finite_mean(vals):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]

    if len(vals) == 0:
        return np.nan

    return float(vals.mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="BLiMP-NL")
    p.add_argument("--directory", default="multilingual/blimpnl")
    p.add_argument("--title", default="Cross-phenomenon overlap in BLiMP-NL")
    p.add_argument("--percentage", type=float, default=1.0)
    args = p.parse_args()

    dataset_key = args.dataset.lower()
    category_map = CATEGORIES[dataset_key]

    path = find_overlap_file(args.directory, dataset_key, args.percentage)
    print(f"Reading {path}")

    df = pd.read_csv(path, index_col=0)
    df = convert_counts_to_percent_if_needed(df)

    all_suites = [suite for suites in category_map.values() for suite in suites]

    rows = []

    for category, suites in category_map.items():
        w = within_values(df, suites)
        o = outbound_values(df, suites, all_suites)

        rows.append(
            {
                "category": category,
                "within": finite_mean(w),
                "outbound": finite_mean(o),
                "n_paradigms": len(suites),
                "n_within_pairs": len(w),
                "n_outbound_pairs": len(o),
            }
        )

    plot_df = pd.DataFrame(rows)
    plot_df["diff"] = plot_df["within"] - plot_df["outbound"]

    # Categories with valid within-category values first; singleton categories last.
    plot_df = plot_df.sort_values(
        by=["within", "diff"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)

    print("\nValues used in plot:")
    print(
        plot_df[
            [
                "category",
                "n_paradigms",
                "n_within_pairs",
                "within",
                "outbound",
            ]
        ].to_string(index=False)
    )

    y = np.arange(len(plot_df))
    width = 0.38

    fig_height = max(8, 0.42 * len(plot_df) + 2)
    fig, ax = plt.subplots(figsize=(9, fig_height))

    ax.barh(
        y - width / 2,
        plot_df["within"],
        height=width,
        label="Overlap within category",
        color="#89CCF1",
        edgecolor="black",
        linewidth=0.9,
        zorder=2,
    )

    ax.barh(
        y + width / 2,
        plot_df["outbound"],
        height=width,
        label="Overlap with other categories",
        color="#FFB668",
        edgecolor="black",
        linewidth=0.9,
        zorder=2,
    )

    ax.set_yticks(y)
    ax.set_yticklabels([pretty_name(c) for c in plot_df["category"]], fontsize=11)
    ax.invert_yaxis()

    # Fixed x-axis to reduce empty space.
    ax.set_xlim(0, 30)

    ax.set_xlabel("Percentage of units", fontsize=13)
    ax.set_ylabel("BLiMP-NL Category", fontsize=13)
    ax.set_title(args.title, fontsize=16)
    ax.grid(axis="x", linestyle=":", alpha=0.5, zorder=0)

    for i, row in plot_df.iterrows():
        if np.isfinite(row["within"]):
            ax.text(
                row["within"] + 0.8,
                i - width / 2,
                f"{row['within']:.2f}%",
                va="center",
                ha="left",
                fontsize=9,
            )
        else:
            ax.text(
                0.5,
                i - width / 2,
                "N/A",
                va="center",
                ha="left",
                fontsize=9,
            )

        if np.isfinite(row["outbound"]):
            ax.text(
                row["outbound"] + 0.8,
                i + width / 2,
                f"{row['outbound']:.2f}%",
                va="center",
                ha="left",
                fontsize=9,
            )

    ax.legend(loc="lower right", frameon=True, fontsize=10)

    for spine in ax.spines.values():
        spine.set_linewidth(1.1)
        spine.set_color("black")

    fig.tight_layout()

    out_png = Path(args.directory) / f"cross_overlap_{dataset_key}_fixed_{args.percentage}%.png"
    out_pdf = Path(args.directory) / f"cross_overlap_{dataset_key}_fixed_{args.percentage}%.pdf"

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")

    print(f"\nSaved {out_png}")
    print(f"Saved {out_pdf}")


if __name__ == "__main__":
    main()
