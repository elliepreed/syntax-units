import os
import argparse

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from categories import CATEGORIES
from model_utils import get_num_blocks, get_hidden_dim
from plot_utils import CATEGORY_COLORS
from utils import random_overlap_expected


# Thresholds from comparison datasets
SLING_MIN = 46.32
RUBLIMP_MIN = 50.46


def norm_name(x):
    return str(x).strip().lower().replace(" ", "_")


def display_name(name):
    return str(name).replace("_", " ").replace("-", " ").title()


def add_score(table, model_name, suite, score):
    if suite not in table:
        table[suite] = {}
    table[suite][model_name] = score


def build_category_map(dataset_key):
    """
    Build:
        suite/paradigm name -> BLiMP-NL category name
    """
    category_map = {}

    for cat, suites in CATEGORIES[dataset_key].items():
        for suite in suites:
            category_map[norm_name(suite)] = norm_name(cat)

    return category_map


def build_category_colors(dataset_key, categories_in_plot):
    if dataset_key in CATEGORY_COLORS:
        palette = CATEGORY_COLORS[dataset_key]
    else:
        palette = list(plt.cm.tab20.colors)

    color_map = {}
    for i, cat in enumerate(categories_in_plot):
        color_map[cat] = palette[i % len(palette)]

    return color_map


def threshold_boundary_y(entries, threshold):
    """
    Bars are sorted descending.
    Return the y-position between bars >= threshold and bars < threshold.
    """
    vals = list(entries.values())
    n_above = sum(v >= threshold for v in vals)
    return n_boundary_y(entries, threshold):
    """
    Bars are sorted descending.
    Return the y-position between bars >= threshold and bars < threshold.
    """
    vals = list(entries.values())
    n_above = sum(v >= threshold for v in vals)
    return n_above - 0.5


def parse_filename(fname, dataset_key):
    """
    Expected:
    cross-validation_blimp-nl_gemma-3-4b-pt_1.0%_2-fold.txt
    """
    prefix = f"cross-validation_{dataset_key}_"
    suffix = ".txt"

    if not fname.startswith(prefix) or not fname.endswith(suffix):
        return None

    rest = fname[len(prefix):-len(suffix)]
    model_name, perc_str, folds_str = rest.rsplit("_", 2)

    percent = float(perc_str.replace("%", "")) / 100
    n_folds = int(folds_str.replace("-fold", ""))

    return model_name, percent, n_folds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="BLiMP-NL")
    parser.add_argument("--directory", default="multilingual/blimpnl")
    parser.add_argument("--title", default="2-fold overlap (gemma-3-4b-pt)")
    args = parser.parse_args()

    dataset_key = args.dataset.lower()

    if dataset_key != "blimp-nl":
        raise ValueError("This script is intended for BLiMP-NL only.")

    model_names = set()
    suite_rows = {}
    percentage_units = {}

    for fname in os.listdir(args.directory):
        parsed = parse_filename(fname, dataset_key)

        if parsed is None:
            continue

        model_name, percent, n_folds = parsed
        model_names.add(model_name)

        print(fname)

        n_units_total = get_num_blocks(model_name) * get_hidden_dim(model_name)
        k_units = int(n_units_total * percent)
        percentage_units[model_name] = k_units

        with open(os.path.join(args.directory, fname)) as fh:
            for ln in fh:
                parts = ln.strip().split()

                if len(parts) != 3:
                    continue

                raw_overlap = float(parts[0])
                suite_name = parts[2]

                add_score(suite_rows, model_name, suite_name, raw_overlap)

        rand_score = random_overlap_expected(
            n_units_total,
            k_units,
            n_folds=n_folds,
        )

        add_score(suite_rows, model_name, "Random", float(rand_score))

    if not suite_rows:
        raise FileNotFoundError(
            f"No cross-validation files found in {args.directory} "
            f"for dataset={dataset_key}"
        )

    model_list = sorted(model_names)

    avg_entries = {}

    for suite, individual_scores in suite_rows.items():
        vals = []

        for model in model_list:
            if model not in individual_scores:
                continue

            vals.append(
                100 * individual_scores[model] / percentage_units[model]
            )

        if vals:
            avg_entries[suite] = float(np.mean(vals))

    print(
        "Average overlap:",
        np.mean(
            [
                avg_entries[suite]
                for suite in avg_entries
                if suite != "Random"
            ]
        ),
    )

    category_map = build_category_map(dataset_key)

    suite_to_plot_category = {}

    for suite in avg_entries:
        suite_key = norm_name(suite)

        if suite == "Random":
            suite_to_plot_category[suite] = "random"
        elif suite_key in category_map:
            suite_to_plot_category[suite] = category_map[suite_key]
        else:
            suite_to_plot_category[suite] = suite_key

    # Sort bars descending by overlap.
    avg_entries = dict(
        sorted(avg_entries.items(), key=lambda kv: kv[1], reverse=True)
    )

    categories_in_plot = []

    for suite in avg_entries:
        cat = suite_to_plot_category[suite]

        if cat != "random" and cat not in categories_in_plot:
            categories_in_plot.append(cat)

    cat2color = build_category_colors(dataset_key, categories_in_plot)

    GREY_RAND = "#bbbbbb"

    n_bars = len(avg_entries)
    bar_h = 1.0

    fig_height = max(5, n_bars * 0.18)
    fig, ax = plt.subplots(figsize=(8, fig_height))

    y_pos = np.arange(n_bars)
    ax.set_xlim(0, 100)

    # ----------------------------
    # Horizontal threshold regions
    # ----------------------------
    y_rublimp = threshold_boundary_y(avg_entries, RUBLIMP_MIN)
    y_sling = threshold_boundary_y(avg_entries, SLING_MIN)

    # Green: BLiMP-NL paradigms at/above the lowest RuBLiMP value.
    ax.axhspan(
        -0.5,
        y_rublimp,
        color="#d4edda",
        alpha=0.28,
        zorder=0,
    )

    # Yellow: between RuBLiMP and SLING minima.
    ax.axhspan(
        y_rublimp,
        y_sling,
        color="#fff3cd",
        alpha=0.22,
        zorder=0,
    )

    # Red: below the lowest SLING value.
    ax.axhspan(
        y_sling,
        n_bars - 0.5,
        color="#f8d7da",
        alpha=0.28,
        zorder=0,
    )

    # Dotted threshold lines.
    ax.axhline(
        y_rublimp,
        color="#1e8449",
        linestyle=":",
        linewidth=2.5,
        zorder=1,
    )

    ax.axhline(
        y_sling,
        color="#c0392b",
        linestyle=":",
        linewidth=2.5,
        zorder=1,
    )

    # Readable threshold textbox inside plot.
    threshold_text = (
        f"RuBLiMP minimum = {RUBLIMP_MIN:.2f}%\n"
        f"SLING minimum = {SLING_MIN:.2f}%"
    )

    ax.text(
        0.03,
        0.04,
        threshold_text,
        transform=ax.transAxes,
        fontsize=10,
        ha="left",
        va="bottom",
        color="black",
        bbox=dict(
            facecolor="white",
            edgecolor="black",
            alpha=0.92,
            boxstyle="round,pad=0.35",
        ),
        zorder=10,
    )

    # ----------------------------
    # Bars
    # ----------------------------
    for i, (suite, avg_score) in enumerate(avg_entries.items()):
        cat = suite_to_plot_category[suite]

        if cat == "random":
            color = GREY_RAND
        else:
            color = cat2color.get(cat, "#1f77b4")

        ax.barh(
            y_pos[i],
            avg_score,
            height=bar_h,
            color=color,
            edgecolor="black",
            zorder=2,
        )

        ax.text(
            1.03,
            y_pos[i],
            f"{avg_score:.2f}%",
            transform=ax.get_yaxis_transform(),
            va="center",
            ha="left",
            fontsize=10,
            clip_on=False,
            zorder=3,
        )

        if len(model_list) > 1:
            for j, model in enumerate(model_list):
                if model not in suite_rows[suite]:
                    continue

                score = suite_rows[suite][model] / percentage_units[model] * 100
                offset = (j - (len(model_list) - 1) / 2) * 0.04

                ax.scatter(
                    score,
                    y_pos[i] + offset,
                    color="black",
                    marker="o",
                    s=10,
                    zorder=4,
                )

    ax.set_xlabel("Percentage of Units", fontsize=12)
    ax.tick_params(axis="x", labelsize=10)

    ax.set_yticks([])
    ax.invert_yaxis()

    ax.set_title(args.title, fontsize=14)

    handles = []

    for cat in categories_in_plot:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="s",
                linestyle="",
                markersize=10,
                markerfacecolor=cat2color[cat],
                markeredgecolor="black",
                label=display_name(cat),
            )
        )

    if "random" in suite_to_plot_category.values():
        handles.append(
            Line2D(
                [0],
                [0],
                marker="s",
                linestyle="",
                markersize=10,
                markerfacecolor=GREY_RAND,
                markeredgecolor="black",
                label="Random",
            )
        )

    handles.extend(
        [
            Line2D(
                [0],
                [0],
                color="#1e8449",
                linestyle=":",
                linewidth=2.5,
                label="RuBLiMP minimum",
            ),
            Line2D(
                [0],
                [0],
                color="#c0392b",
                linestyle=":",
                linewidth=2.5,
                label="SLING minimum",
            ),
            Patch(
                facecolor="#d4edda",
                edgecolor="none",
                alpha=0.28,
                label="At/above RuBLiMP minimum",
            ),
            Patch(
                facecolor="#f8d7da",
                edgecolor="none",
                alpha=0.28,
                label="Below SLING minimum",
            ),
        ]
    )

    ax.legend(
        handles=handles,
        bbox_to_anchor=(1.22, 1),
        loc="upper left",
        frameon=True,
    )

    fig.subplots_adjust(
        left=0.06,
        right=0.62,
        top=0.90,
        bottom=0.12,
    )

    out_pdf = f"{args.directory}/cross_validation_blimp-nl_thresholds.pdf"
    out_png = f"{args.directory}/cross_validation_blimp-nl_thresholds.png"

    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")

    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
