import argparse
import os
from typing import Dict, List, Sequence, Tuple

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from categories import CATEGORIES
from plot_utils import (
    build_model_style_maps,
    add_model_scatter,
    pretty_category_name,
    get_canonical_order,
    load_model_matrices,
)


def get_blimpnl_category_map() -> Dict[str, List[str]]:
    """
    Broad BLiMP-NL categories for cross-phenomenon overlap plots.

    Blue bar:
        mean overlap between different phenomena within the same broad group

    Orange bar:
        mean overlap between phenomena in this group and phenomena outside this group

    Diagonal/self-overlap is excluded, so blue bars are not trivially 100%.
    """
    return {
        "agreement_and_binding": [
            "anaphor_agreement",
            "binding_principle_a",
        ],
        "argument_structure_and_voice": [
            "argument_structure",
            "passive",
        ],
        "clausal_arguments": [
            "finite_argument_clause",
            "infinitival_argument_clause",
            "complementive",
        ],
        "movement_and_word_order": [
            "crossing_dependencies",
            "extraposition",
            "parasitic_gaps",
            "relativization",
            "topicalization",
            "verb_second",
            "wh_movement",
            "wh_movement_restrictions",
        ],
        "nominal_domain": [
            "adpositional_phrases",
            "determiners",
            "nominalization",
            "quantifiers",
            "r_words",
        ],
        "verbal_functional_domain": [
            "adverbial_modification",
            "auxiliaries",
        ],
    }


def get_category_map(dataset: str) -> Dict[str, List[str]]:
    dataset_key = dataset.lower()

    if dataset_key == "blimp-nl":
        return get_blimpnl_category_map()

    return CATEGORIES[dataset_key]


def clean_suites(df: pd.DataFrame, suites: Sequence[str]) -> List[str]:
    return [s for s in suites if s in df.index and s in df.columns]


def category_within_values(df: pd.DataFrame, suites: Sequence[str]) -> np.ndarray:
    suites = clean_suites(df, suites)

    if len(suites) < 2:
        return np.array([], dtype=float)

    sub = df.loc[suites, suites]

    # Exclude diagonal/self-overlap.
    # This is the important part: no 100% self-comparison.
    mask = ~np.eye(len(suites), dtype=bool)

    return sub.values[mask]


def category_outbound_values(
    df: pd.DataFrame,
    suites: Sequence[str],
    canonical_order: List[str],
) -> np.ndarray:
    suites = [s for s in suites if s in df.index]
    outside = [s for s in canonical_order if s not in suites and s in df.columns]

    if not suites or not outside:
        return np.array([], dtype=float)

    sub = df.loc[suites, outside]
    return sub.values.ravel()


def finite_mean(vals: Sequence[float] | np.ndarray) -> float:
    arr = np.asarray(vals, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return np.nan

    return float(arr.mean())


def aggregate_across_models(
    matrices: Dict[str, pd.DataFrame],
    category_map: Dict[str, List[str]],
    canonical_order: List[str],
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    rows = []
    first_df = next(iter(matrices.values()))

    per_model_within = {}
    per_model_outbound = {}

    for cat, suites in category_map.items():
        per_model_within_counts = {}
        per_model_outbound_counts = {}

        for model_name, df in matrices.items():
            w_vals = category_within_values(df, suites)
            o_vals = category_outbound_values(df, suites, canonical_order)

            w_mean = finite_mean(w_vals)
            o_mean = finite_mean(o_vals)

            per_model_within_counts[model_name] = w_mean
            per_model_outbound_counts[model_name] = o_mean

        per_model_within[cat] = per_model_within_counts
        per_model_outbound[cat] = per_model_outbound_counts

        within_avg = finite_mean(list(per_model_within_counts.values()))
        outbound_avg = finite_mean(list(per_model_outbound_counts.values()))

        rows.append(
            {
                "category": cat,
                "within": within_avg,
                "outbound": outbound_avg,
                "diff": within_avg - outbound_avg,
                "n_suites": len(suites),
                "cells_within_per_model": len(category_within_values(first_df, suites)),
                "cells_outbound_per_model": len(
                    category_outbound_values(first_df, suites, canonical_order)
                ),
            }
        )

    summary_df = pd.DataFrame(rows).set_index("category")

    return summary_df, per_model_within, per_model_outbound


def display_category_name(cat: str) -> str:
    custom = {
        "agreement_and_binding": "Agreement and Binding",
        "argument_structure_and_voice": "Argument Structure and Voice",
        "clausal_arguments": "Clausal Arguments",
        "movement_and_word_order": "Movement and Word Order",
        "nominal_domain": "Nominal Domain",
        "verbal_functional_domain": "Verbal/Functional Domain",
    }

    if cat in custom:
        return custom[cat]

    try:
        return pretty_category_name(cat)
    except Exception:
        return cat.replace("_", " ").replace("-", " ").title()


def plot_horizontal_bars(
    summary_df: pd.DataFrame,
    per_model_within: Dict[str, Dict[str, float]],
    per_model_outbound: Dict[str, Dict[str, float]],
    dataset: str,
    directory: str,
    title: str,
    add_model_markers: bool,
    percentage: float,
    width: float = 0.35,
    seed: int = 42,
):
    data = summary_df.copy()
    data = data.sort_values(["within", "diff"], ascending=[False, False])

    model_names = sorted({m for d in per_model_within.values() for m in d.keys()})
    model_list, model_to_color, model_to_marker = build_model_style_maps(model_names)

    y = np.arange(len(data))
    offset = width / 2

    fig_height = max(4.8, 0.55 * len(data) + 1.8)
    fig, ax = plt.subplots(figsize=(9.0, fig_height))

    ax.barh(
        y - offset,
        data["within"],
        height=width,
        label="Overlap within broad category",
        color="#89CCF1",
        edgecolor="black",
        zorder=2,
    )

    ax.barh(
        y + offset,
        data["outbound"],
        height=width,
        label="Overlap with other categories",
        color="#FFB668",
        edgecolor="black",
        zorder=2,
    )

    ax.set_yticks(y)
    ax.set_yticklabels([display_category_name(c) for c in data.index], fontsize=10)
    ax.set_ylabel(f"{dataset} Category", fontsize=12)
    ax.invert_yaxis()

    max_val = np.nanmax(data[["within", "outbound"]].to_numpy(dtype=float))
    x_lim = max(10, int(np.ceil((max_val + 5) / 10.0) * 10.0))

    ax.set_xlim(0, x_lim)
    ax.set_xlabel("Percentage of units", fontsize=12)
    ax.grid(axis="x", linestyle=":", alpha=0.5, zorder=0)

    ax.set_title(title, fontsize=13)

    val_d = max(0.6, x_lim * 0.015)

    for series_name, dy in [
        ("within", -offset),
        ("outbound", offset),
    ]:
        for yi, cat in zip(y, data.index):
            val = data.loc[cat, series_name]

            if not np.isfinite(val):
                continue

            ax.text(
                val + val_d,
                yi + dy,
                f"{val:.2f}%",
                va="center",
                ha="left",
                fontsize=8,
                zorder=9,
            )

    if add_model_markers:
        rng = np.random.default_rng(seed)

        for yi, cat in zip(y, data.index):
            add_model_scatter(
                per_model_within[cat],
                center=yi - offset,
                ax=ax,
                model_to_color=model_to_color,
                model_to_marker=model_to_marker,
                rng=rng,
                jitter=0.07,
            )
            add_model_scatter(
                per_model_outbound[cat],
                center=yi + offset,
                ax=ax,
                model_to_color=model_to_color,
                model_to_marker=model_to_marker,
                rng=rng,
                jitter=0.07,
            )

    bar_legend = ax.legend(
        loc="lower right",
        frameon=True,
        title="Averages",
        fontsize=8,
        title_fontsize=9,
    )
    ax.add_artist(bar_legend)

    if add_model_markers:
        model_handles = [
            Line2D(
                [],
                [],
                marker=model_to_marker[m],
                color=model_to_color[m],
                linestyle="None",
                markersize=6,
                markeredgecolor="black",
                markeredgewidth=0.4,
                label=m,
            )
            for m in model_list
        ]

        ax.legend(
            handles=model_handles,
            loc="center right",
            frameon=True,
            title="Models",
            fontsize=8,
            title_fontsize=9,
            ncol=1,
        )

    fig.tight_layout()

    out_png = f"{directory}/cross_overlap_{dataset.lower()}_broad_{percentage}%.png"
    out_pdf = f"{directory}/cross_overlap_{dataset.lower()}_broad_{percentage}%.pdf"

    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)

    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="BLiMP")
    p.add_argument("--directory", default="english/cross-overlap")
    p.add_argument("--title", default="Cross-phenomenon overlap in BLiMP")
    p.add_argument("--add-model-markers", action="store_true")
    p.add_argument("--percentage", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()

    dataset_key = args.dataset.lower()

    category_map = get_category_map(args.dataset)
    canonical_order = get_canonical_order(category_map)

    matrices = load_model_matrices(
        args.directory,
        dataset_key,
        dataset_key,
        canonical_order,
        canonical_order,
        args.percentage,
    )

    summary, per_model_within, per_model_outbound = aggregate_across_models(
        matrices,
        category_map,
        canonical_order,
    )

    print(summary.to_string())

    plot_horizontal_bars(
        summary,
        per_model_within,
        per_model_outbound,
        dataset=args.dataset,
        directory=args.directory,
        title=args.title,
        add_model_markers=args.add_model_markers,
        seed=args.seed,
        percentage=args.percentage,
    )


if __name__ == "__main__":
    main()
