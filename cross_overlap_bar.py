import argparse
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


def category_within_values(df: pd.DataFrame, suites: Sequence[str]) -> np.ndarray:
    sub = df.loc[suites, suites]
    mask = ~np.eye(len(suites), dtype=bool)
    return sub.values[mask]


def category_outbound_values(
    df: pd.DataFrame, suites: Sequence[str], canonical_order: List[str]
) -> np.ndarray:
    outside = [s for s in canonical_order if s not in suites]
    sub = df.loc[suites, outside]
    return sub.values.ravel()


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

            w_mean = w_vals.mean()
            o_mean = o_vals.mean()

            per_model_within_counts[model_name] = w_mean
            per_model_outbound_counts[model_name] = o_mean

        per_model_within[cat] = per_model_within_counts
        per_model_outbound[cat] = per_model_outbound_counts

        within_avg = float(np.mean(list(per_model_within_counts.values())))
        outbound_avg = float(np.mean(list(per_model_outbound_counts.values())))

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


def plot_horizontal_bars(
    summary_df: pd.DataFrame,
    per_model_within: Dict[str, Dict[str, float]],
    per_model_outbound: Dict[str, Dict[str, float]],
    dataset: str,
    directory: str,
    title: str,
    add_model_markers: bool,
    percentage: float,
    width: float = 0.45,
    x_lim: int = 100,  # 60
    val_d: int = 8,  # 1
    seed: int = 42,
):
    data = summary_df.sort_values("diff", ascending=False)

    model_names = sorted({m for d in per_model_within.values() for m in d.keys()})
    model_list, model_to_color, model_to_marker = build_model_style_maps(model_names)

    y = np.linspace(0, len(data), num=len(data))
    offset = width / 2

    fig, ax = plt.subplots(figsize=(9.0, 10.0))
    ax.barh(
        y - offset,
        data["within"],
        height=width,
        label="Overlap within category",
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
    ax.set_yticklabels([pretty_category_name(c) for c in data.index], fontsize=12)
    ax.set_ylabel(f"{dataset} Category", fontsize=14)
    ax.invert_yaxis()

    ax.set_xlim(0, x_lim)
    ax.set_xlabel("Percentage of units", fontsize=14)
    ax.grid(axis="x", linestyle=":", alpha=0.5, zorder=0)

    ax.set_title(title, fontsize=15)

    for series_name, dy in [
        ("within", -offset),
        ("outbound", offset),
    ]:
        for yi, cat in zip(y, data.index):
            val = data.loc[cat, series_name]
            ax.text(
                val + val_d,
                yi + dy,
                f"{val:.2f}%",
                va="center",
                ha="left",
                fontsize=10,
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
                jitter=0.09,
            )
            add_model_scatter(
                per_model_outbound[cat],
                center=yi + offset,
                ax=ax,
                model_to_color=model_to_color,
                model_to_marker=model_to_marker,
                rng=rng,
                jitter=0.09,
            )

    bar_legend = ax.legend(
        loc="lower right", frameon=True, title="Averages", fontsize=10
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
                markersize=7,
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
            fontsize=10,
            ncol=1,
        )

    fig.tight_layout()
    fig.savefig(f"{directory}/cross_overlap_{dataset.lower()}_{percentage}%.png", dpi=300)
    fig.savefig(f"{directory}/cross_overlap_{dataset.lower()}_{percentage}%.pdf")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="BLiMP")
    p.add_argument("--directory", default=f"english/cross-overlap")
    p.add_argument("--title", default=f"Cross-phenomenon overlap in BLiMP")
    p.add_argument("--add-model-markers", action="store_true")
    p.add_argument("--percentage", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    category_map = CATEGORIES[args.dataset.lower()]
    canonical_order = get_canonical_order(category_map)

    matrices = load_model_matrices(
        args.directory,
        args.dataset.lower(),
        args.dataset.lower(),
        canonical_order,
        canonical_order,
        args.percentage,
    )
    
    summary, per_model_within, per_model_outbound = aggregate_across_models(
        matrices, category_map, canonical_order
    )

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
