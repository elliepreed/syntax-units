import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from categories import CATEGORIES
from model_utils import get_num_blocks, get_hidden_dim
from plot_utils import pretty_category_name, CATEGORY_COLORS


# -----------------------------
# Fixed thresholds to compare against BLiMP-NL
# -----------------------------
RUBLIMP_MIN = 50.46
SLING_MIN = 46.32

GREEN_LINE = "#3F8F5A"
RED_LINE = "#C84D3A"
GREEN_SHADE = "#DFF0E3"
RED_SHADE = "#F7E1E1"

GREY_RAND = "#D3D3D3"


def add_score(table, model_name, suite, score):
    if suite not in table:
        table[suite] = {}
    table[suite][model_name] = score


def random_overlap_expected(n_units_total, k_units, n_folds=2):
    """
    Expected random overlap count if each fold independently selects k_units
    out of n_units_total.
    """
    p = k_units / n_units_total
    return n_units_total * (p ** n_folds)


def parse_result_filename(fname, dataset_key, percentage, num_folds):
    """
    Expected pattern:
      cross-validation_<dataset>_<model>_<percentage>%_<num_folds>-fold.txt

    Example:
      cross-validation_blimp-nl_gemma-3-4b-pt_1.0%_2-fold.txt
    """
    prefix = f"cross-validation_{dataset_key}_"
    suffix = f"_{percentage}%_{num_folds}-fold.txt"

    if not fname.startswith(prefix):
        return None
    if not fname.endswith(suffix):
        return None

    model_name = fname[len(prefix):-len(suffix)]
    return model_name


def build_suite_to_category(dataset_key):
    """
    Build mapping:
        suite/paradigm -> broad category
    Works whether CATEGORIES[dataset_key] is:
      - dict[category] = list_of_suites
      - dict[category] = dict[subsuite] = ...
    """
    suite_to_cat = {}
    dataset_categories = CATEGORIES[dataset_key]

    for cat, value in dataset_categories.items():
        if isinstance(value, dict):
            for suite_name in value.keys():
                suite_to_cat[suite_name] = cat
        else:
            for suite_name in value:
                suite_to_cat[suite_name] = cat

    return suite_to_cat


def get_category_colors(dataset_key, ordered_cats):
    if dataset_key in CATEGORY_COLORS:
        palette = CATEGORY_COLORS[dataset_key]
    else:
        palette = list(plt.cm.tab20.colors)

    color_map = {}
    for i, cat in enumerate(ordered_cats):
        color_map[cat] = palette[i % len(palette)]
    return color_map


def find_threshold_boundary(sorted_entries, threshold):
    """
    sorted_entries: list[(suite, score)] in descending order.
    Returns y-value BETWEEN bars, so that the horizontal line sits between:
      last bar >= threshold
      first bar < threshold

    Bars are centered at y = 0,1,2,...
    The boundary between bar i and i+1 is at i + 0.5
    """
    values = [score for _, score in sorted_entries]

    last_ge_idx = None
    for i, v in enumerate(values):
        if v >= threshold:
            last_ge_idx = i

    if last_ge_idx is None:
        # everything is below threshold
        return -0.5

    if last_ge_idx == len(values) - 1:
        # everything is above threshold
        return len(values) - 0.5

    return last_ge_idx + 0.5


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="BLiMP-NL")
    p.add_argument("--directory", default="multilingual/blimpnl")
    p.add_argument("--title", default="2-fold overlap (gemma-3-4b-pt)")
    p.add_argument("--percentage", type=float, default=1.0)
    p.add_argument("--num-folds", type=int, default=2)
    args = p.parse_args()

    dataset_key = args.dataset.lower()

    if dataset_key not in CATEGORIES:
        raise KeyError(
            f"Dataset '{dataset_key}' not found in CATEGORIES.\n"
            f"Available keys: {list(CATEGORIES.keys())}"
        )

    model_names = set()
    suite_rows = {}
    percentage_units = {}

    for fname in os.listdir(args.directory):
        model_name = parse_result_filename(
            fname=fname,
            dataset_key=dataset_key,
            percentage=args.percentage,
            num_folds=args.num_folds,
        )
        if model_name is None:
            continue

        print(fname)
        model_names.add(model_name)

        n_units_total = get_num_blocks(model_name) * get_hidden_dim(model_name)
        k_units = int(n_units_total * args.percentage / 100)
        percentage_units[model_name] = k_units

        with open(os.path.join(args.directory, fname)) as fh:
            for ln in fh:
                parts = ln.strip().split()
                if len(parts) < 3:
                    continue

                raw_overlap = float(parts[0])
                suite_name = parts[2]
                add_score(suite_rows, model_name, suite_name, raw_overlap)

        rand_score = random_overlap_expected(
            n_units_total=n_units_total,
            k_units=k_units,
            n_folds=args.num_folds,
        )
        add_score(suite_rows, model_name, "Random", float(rand_score))

    if not suite_rows:
        raise FileNotFoundError(
            f"No cross-validation files found in {args.directory} "
            f"for dataset={dataset_key}, percentage={args.percentage}, "
            f"num_folds={args.num_folds}"
        )

    model_list = sorted(model_names)

    # Average percentage overlap across models for each suite
    avg_entries = {}
    for suite, individual_scores in suite_rows.items():
        vals = []
        for model in model_list:
            if model not in individual_scores:
                continue
            vals.append(100 * individual_scores[model] / percentage_units[model])
        if vals:
            avg_entries[suite] = float(np.mean(vals))

    # Print overall average excluding random
    non_random = [v for k, v in avg_entries.items() if k != "Random"]
    if non_random:
        print("Average overlap:", np.mean(non_random))

    suite_to_cat = build_suite_to_category(dataset_key)

    # Preserve category order from categories.py as much as possible
    ordered_cats = []
    for cat in CATEGORIES[dataset_key].keys():
        ordered_cats.append(cat)

    cat2color = get_category_colors(dataset_key, ordered_cats)

    # Sort descending
    avg_entries = dict(sorted(avg_entries.items(), key=lambda kv: kv[1], reverse=True))
    sorted_entries = list(avg_entries.items())

    n_bars = len(sorted_entries)
    fig_height = max(7, n_bars * 0.22)
    fig, ax = plt.subplots(figsize=(10.5, fig_height))

    y_pos = np.arange(n_bars)

    # Horizontal threshold boundaries
    y_rublimp = find_threshold_boundary(sorted_entries, RUBLIMP_MIN)
    y_sling = find_threshold_boundary(sorted_entries, SLING_MIN)

    # Background shading
    ax.axhspan(-0.5, y_rublimp, facecolor=GREEN_SHADE, alpha=0.45, zorder=0)
    ax.axhspan(y_sling, n_bars - 0.5, facecolor=RED_SHADE, alpha=0.40, zorder=0)

    # Bars
    for i, (suite, avg_score) in enumerate(sorted_entries):
        if suite == "Random":
            color = GREY_RAND
        else:
            category = suite_to_cat.get(suite, None)
            color = cat2color.get(category, "#1f77b4")

        ax.barh(
            y_pos[i],
            avg_score,
            height=0.9,
            color=color,
            edgecolor="black",
            zorder=2,
        )

        # Percentage labels OUTSIDE the plotting area
        ax.text(
            104,
            y_pos[i],
            f"{avg_score:.2f}%",
            va="center",
            ha="left",
            fontsize=10,
            zorder=5,
        )

    # Threshold lines (horizontal, between bars)
    ax.hlines(
        y=y_rublimp,
        xmin=0,
        xmax=100,
        colors=GREEN_LINE,
        linestyles=":",
        linewidth=2.5,
        zorder=4,
    )
    ax.hlines(
        y=y_sling,
        xmin=0,
        xmax=100,
        colors=RED_LINE,
        linestyles=":",
        linewidth=2.5,
        zorder=4,
    )

    # Text boxes for thresholds, placed INSIDE the plot so they are readable
    green_box = dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=GREEN_LINE, linewidth=1.2)
    red_box = dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=RED_LINE, linewidth=1.2)

    # Put them in the open space on the right side of the bar area
    ax.text(
        73,
        max(0.5, y_rublimp - 0.8),
        f"Lowest RuBLiMP\n= {RUBLIMP_MIN:.2f}%",
        color=GREEN_LINE,
        fontsize=10,
        ha="left",
        va="center",
        bbox=green_box,
        zorder=6,
    )
    ax.text(
        73,
        min(n_bars - 1.0, y_sling + 0.8),
        f"Lowest SLING\n= {SLING_MIN:.2f}%",
        color=RED_LINE,
        fontsize=10,
        ha="left",
        va="center",
        bbox=red_box,
        zorder=6,
    )

    ax.set_xlabel("Percentage of Units", fontsize=14)
    ax.set_xlim(0, 115)
    ax.tick_params(axis="x", labelsize=11)

    # No left-side paradigm labels
    ax.set_yticks([])
    ax.invert_yaxis()

    ax.set_title(args.title, fontsize=18)

    # Category legend
    legend_handles = []
    seen_cats = set()

    for suite, _ in sorted_entries:
        if suite == "Random":
            continue
        cat = suite_to_cat.get(suite, None)
        if cat is None or cat in seen_cats:
            continue
        seen_cats.add(cat)

        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="s",
                linestyle="",
                markersize=10,
                markerfacecolor=cat2color.get(cat, "#1f77b4"),
                markeredgecolor="black",
                label=pretty_category_name(cat),
            )
        )

    # Random handle
    legend_handles.append(
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

    ax.legend(
        handles=legend_handles,
        bbox_to_anchor=(1.21, 1.0),
        loc="upper left",
        frameon=True,
        fontsize=11,
    )

    plt.tight_layout()

    out_base = os.path.join(
        args.directory,
        f"cross_validation_{dataset_key}_thresholds"
    )
    fig.savefig(f"{out_base}.pdf", bbox_inches="tight")
    fig.savefig(f"{out_base}.png", dpi=300, bbox_inches="tight")

    print(f"Saved {out_base}.pdf")
    print(f"Saved {out_base}.png")


if __name__ == "__main__":
    main()
