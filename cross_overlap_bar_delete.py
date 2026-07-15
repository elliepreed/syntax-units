cd /local/scratch/ep757/syntax-units

cat > cross_overlap_bar.py <<'PY'
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

    if len(suites) <= 1:
        return np.array([np.nan])

    mask = ~np.eye(len(suites), dtype=bool)
    return sub.values[mask]


def category_outbound_values(
    df: pd.DataFrame,
    suites: Sequence[str],
    canonical_order: List[str],
) -> np.ndarray:
    outside = [s for s in canonical_order if s not in suites]

    if len(outside) == 0:
        return np.array([np.nan])

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

            w_mean = float(np.nanmean(w_vals))
            o_mean = float(np.nanmean(o_vals))

            per_model_within_counts[model_name] = w_mean
            per_model_outbound_counts[model_name] = o_mean

        per_model_within[cat] = per_model_within_counts
        per_model_outbound[cat] = per_model_outbound_counts

        within_avg = float(np.nanmean(list(per_model_within_counts.values())))
        outbound_avg = float(np.nanmean(list(per_model_outbound_counts.values())))

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


def fmt_val(x: float) -> str:
    if np.isnan(x):
        return "N/A"
    return f"{x:.2f}%"


def plot_horizontal_bars(
    summary_df: pd.DataFrame,
    per_model_within: Dict[str, Dict[str, float]],
    per_model_outbound: Dict[str, Dict[str, float]],
    dataset: str,
    directory: str,
    title: str,
    add_model_markers: bool,
    percentage: float,
    every_other: bool,
    seed: int = 42,
):
    data = summary_df.sort_values("diff", ascending=False)

    if every_other:
        data = data.iloc[::2].copy()

    model_names = sorted({m for d in per_model_within.values() for m in d.keys()})
    model_list, model_to_color, model_to_marker = build_model_style_maps(model_names)

    y = np.arange(len(data))
    width = 0.36
    offset = width / 2

    max_val = np.nanmax(data[["within", "outbound"]].values)
    x_lim = max(10, int(np.ceil((max_val + 3) / 5) * 5))
    val_d = x_lim * 0.015

    fig_h = max(4.2, 0.42 * len(data) + 1.8)
    fig, ax = plt.subplots(figsize=(8.0, fig_h))

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
    ax.set_yticklabels([pretty_category_name(c) for c in data.index], fontsize=9)
    ax.set_ylabel(f"{dataset} Category", fontsize=11)
    ax.invert_yaxis()

    ax.set_xlim(0, x_lim)
    ax.set_xlabel("Percentage of units", fontsize=11)
    ax.grid(axis="x", linestyle=":", alpha=0.5, zorder=0)

    ax.set_title(title, fontsize=12)

    for series_name, dy in [
        ("within", -offset),
        ("outbound", offset),
    ]:
        for yi, cat in zip(y, data.index):
            val = data.loc[cat, series_name]

            if np.isnan(val):
                text_x = 0.3
            else:
                text_x = val + val_d

            ax.text(
                text_x,
                yi + dy,
                fmt_val(val),
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

    suffix = "_every_other" if every_other else ""
    out_png = f"{directory}/cross_overlap_{dataset.lower()}_{percentage}%{suffix}.png"
    out_pdf = f"{directory}/cross_overlap_{dataset.lower()}_{percentage}%{suffix}.pdf"

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")

    print(f"Saved {out_png}")
    print(f"Saved {out_pdf}")


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--dataset", default="BLiMP")
    p.add_argument("--directory", default="english/cross-overlap")
    p.add_argument("--title", default="Cross-phenomenon overlap")
    p.add_argument("--add-model-markers", action="store_true")
    p.add_argument("--percentage", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)

    # Keeps every other category row. Both bars for retained categories remain.
    p.add_argument("--every-other", action="store_true")

    args = p.parse_args()

    dataset_key = args.dataset.lower()

    category_map = CATEGORIES[dataset_key]
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
        every_other=args.every_other,
    )


if __name__ == "__main__":
    main()
PY
