import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from categories import CATEGORIES
from model_utils import get_num_blocks, get_hidden_dim
from plot_utils import pretty_category_name, CATEGORY_COLORS
from utils import random_overlap_expected


# ----------------------------
# Thresholds for background
# ----------------------------
SLING_MIN = 46.32
RUBLIMP_MIN = 50.46


def add_score(table, model_name, suite, score):
    if suite not in table:
        table[suite] = {}
    table[suite][model_name] = score


def get_dataset_key(dataset_name: str) -> str:
    return dataset_name.lower()


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


def get_category_colors(dataset_key, unique_cats):
    if dataset_key in CATEGORY_COLORS:
        palette = CATEGORY_COLORS[dataset_key]
    else:
        palette = list(plt.cm.tab20.colors)

    color_map = {}
    for i, cat in enumerate(unique_cats):
        color_map[cat] = palette[i % len(palette)]
    return color_map


def parse_result_filename(fname, dataset_key, percentage, num_folds):
    """
    Expected:
      cross-validation_<dataset>_<model>_<percentage>%_<num_folds>-fold.txt
    """
    prefix = f"cross-validation_{dataset_key}_"
    suffix = f"_{percentage}%_{num_folds}-fold.txt"

    if not fname.startswith(prefix):
        return None
    if not fname.endswith(suffix):
        return None

    model_name = fname[len(prefix):-len(suffix)]
    return model_name


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="BLiMP-NL")
    p.add_argument("--directory", default="multilingual/blimpnl")
    p.add_argument("--title", default="2-fold overlap (gemma-3-4b-pt)")
    p.add_argument("--percentage", type=float, default=1.0)
    p.add_argument("--num-folds", type=int, default=2)
    args = p.parse_args()

    dataset_key = get_dataset_key(args.dataset)

    if dataset_key not in CATEGORIES:
        raise KeyError(
            f"Dataset '{dataset_key}' not found in CATEGORIES. "
            f"Available keys: {list(CATEGORIES.keys())}"
        )

    model_names = set()
    suite_rows = {}
    percentage_units = {}

    for fname in os.listdir(args.directory):
        model_name = parse_result_filename(
            fname,
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
            n_units_total, k_units, n_folds=args.num_folds
        )
        add_score(suite_rows, model_name, "Random", float(rand_score))

    if not suite_rows:
        raise FileNotFoundError(
            f"No cross-validation files found in {args.directory} "
            f"for dataset={dataset_key}, percentage={args.percentage}, "
            f"num_folds={args.num_folds}"
        )

    model_list = sorted(model_names)

    avg_entries = {}
    for suite, individual_scores in suite_rows.items():
        vals = []
        for model in model_list:
            if model not in individual_scores:
                continue
            vals.append(100 * individual_scores[model] / percentage_units[model])
        if vals:
            avg_entries[suite] = float(np.mean(vals))

    non_random = [v for k, v in avg_entries.items() if k != "Random"]
    if non_random:
        print("Average overlap:", np.mean(non_random))

    suite_to_cat = build_suite_to_category(dataset_key)

    unique_cats = []
    for suite in avg_entries:
        if suite == "Random":
            continue
        if suite in suite_to_cat and suite_to_cat[suite] not in unique_cats:
            unique_cats.append(suite_to_cat[suite])

    cat2color = get_category_colors(dataset_key, unique_cats)

    GREY_RAND = "#D3D3D3"

    avg_entries = dict(sorted(avg_entries.items(), key=lambda kv: kv[1], reverse=True))

    n_bars = len(avg_entries)
    fig_height = max(7, n_bars * 0.22)
    fig, ax = plt.subplots(figsize=(9.5, fig_height))

    y_pos = np.arange(n_bars)

    x_max = 100

    # ----------------------------
    # Background shading
    # ----------------------------
    ax.axvspan(0, SLING_MIN, color="#f8d7da", alpha=0.30, zorder=0)           # light red
    ax.axvspan(SLING_MIN, RUBLIMP_MIN, color="#fff3cd", alpha=0.25, zorder=0) # light neutral band
    ax.axvspan(RUBLIMP_MIN, x_max, color="#d4edda", alpha=0.30, zorder=0)     # light green

    # Reference lines
    ax.axvline(SLING_MIN, color="#c0392b", linestyle=":", linewidth=2, zorder=1)
    ax.axvline(RUBLIMP_MIN, color="#1e8449", linestyle=":", linewidth=2, zorder=1)

    for i, (suite, avg_score) in enumerate(avg_entries.items()):
        if suite == "Random":
            color = GREY_RAND
        else:
            color = cat2color.get(suite_to_cat.get(suite, ""), "#1f77b4")

        ax.barh(
            y_pos[i],
            avg_score,
            height=0.9,
            color=color,
            edgecolor="black",
            zorder=2,
        )

        # percentage labels outside bars
        ax.text(
            avg_score + 1.5,
            y_pos[i],
            f"{avg_score:.2f}%",
            va="center",
            ha="left",
            fontsize=10,
            zorder=3,
        )

    ax.set_xlabel("Percentage of Units", fontsize=12)
    ax.set_xlim(0, x_max + 15)  # extra room for labels
    ax.tick_params(axis="x", labelsize=10)

    ax.set_yticks([])
    ax.invert_yaxis()

    ax.set_title(args.title, fontsize=14)

    # Legend for categories
    handles = []
    for cat in unique_cats:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="s",
                linestyle="",
                markersize=10,
                markerfacecolor=cat2color[cat],
                markeredgecolor="black",
                label=pretty_category_name(cat),
            )
        )

    if "Random" in avg_entries:
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

    # Threshold legend entries
    handles.extend([
        Line2D(
            [0], [0],
            color="#c0392b",
            linestyle=":",
            linewidth=2,
            label=f"Lowest SLING = {SLING_MIN:.2f}%",
        ),
        Line2D(
            [0], [0],
            color="#1e8449",
            linestyle=":",
            linewidth=2,
            label=f"Lowest RuBLiMP = {RUBLIMP_MIN:.2f}%",
        ),
        Patch(facecolor="#f8d7da", edgecolor="none", alpha=0.30, label="Below SLING minimum"),
        Patch(facecolor="#d4edda", edgecolor="none", alpha=0.30, label="Above RuBLiMP minimum"),
    ])

    ax.legend(
        handles=handles,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=True,
        fontsize=10,
    )

    plt.tight_layout()

    out_base = os.path.join(
        args.directory,
        f"cross_validation_{dataset_key}_paradigm_thresholds"
    )
    fig.savefig(f"{out_base}.pdf", bbox_inches="tight")
    fig.savefig(f"{out_base}.png", dpi=300, bbox_inches="tight")

    print(f"Saved {out_base}.pdf")
    print(f"Saved {out_base}.png")


if __name__ == "__main__":
    main()
