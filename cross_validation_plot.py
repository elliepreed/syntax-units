import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from categories import CATEGORIES
from model_utils import get_num_blocks, get_hidden_dim
from plot_utils import CATEGORY_COLORS
from utils import random_overlap_expected


TURBLIMP_LABELS = {
    "anaphor_agreement": "Anaphor Agreement",
    "argument_structure_transitive": "Arg. Struct. Trans.",
    "argument_structure_ditransitive": "Arg. Struct. Ditrans.",
    "binding": "Binding",
    "determiners": "Determiners",
    "ellipsis": "Ellipsis",
    "irregular_forms": "Irregular Forms",
    "island_effects": "Island Effects",
    "nominalization": "Nominalization",
    "npi_licensing": "NPI Licensing",
    "passives": "Passives",
    "quantifiers": "Quantifiers",
    "relative_clauses": "Relative Clauses",
    "scrambling": "Scrambling",
    "subject_agreement": "Subject Agreement",
    "suspended_affixation": "Suspended Affixation",
    "random": "Random",
    "blimp-control_(avg.)": "BLiMP-Control",
}


TURBLIMP_LEGEND_ORDER = [
    "anaphor_agreement",
    "argument_structure_transitive",
    "argument_structure_ditransitive",
    "binding",
    "determiners",
    "ellipsis",
    "irregular_forms",
    "island_effects",
    "nominalization",
    "npi_licensing",
    "passives",
    "quantifiers",
    "relative_clauses",
    "scrambling",
    "subject_agreement",
    "suspended_affixation",
]


def norm_name(x):
    return str(x).strip().lower().replace(" ", "_")


def display_name(name):
    key = norm_name(name)

    if key in TURBLIMP_LABELS:
        return TURBLIMP_LABELS[key]

    return (
        str(name)
        .replace("_", " ")
        .replace("-", " ")
        .title()
    )


def read_control_file(path: str, expect_many: bool):
    records = []

    with open(path) as f:
        for line in f:
            spl = line.strip().split()

            if len(spl) < 1:
                continue

            records.append(int(spl[0]))

    if not records:
        raise RuntimeError(f"{path} is empty?")

    avg = float(np.mean(records))
    df = pd.DataFrame({"LangOverlap": records}) if expect_many else None

    return avg, df


def add_score(table, model_name, suite, score):
    if suite not in table:
        table[suite] = {}

    table[suite][model_name] = score


def build_category_map(dataset_key):
    """
    Build:
        normalised suite name -> normalised category/phenomenon name

    If categories.py has one-suite-per-category, this still works.
    """
    category_map = {}

    if dataset_key not in CATEGORIES:
        return category_map

    for cat, subdict in CATEGORIES[dataset_key].items():
        cat_key = norm_name(cat)

        if isinstance(subdict, dict):
            suite_names = subdict.keys()
        else:
            suite_names = subdict

        for suite_name in suite_names:
            category_map[norm_name(suite_name)] = cat_key

    return category_map


def build_category_colors(dataset_key, categories_in_plot):
    """
    Assign one colour per phenomenon/category.
    For TurBLiMP, use a stable order so legend/bar colours remain consistent.
    """
    if dataset_key in CATEGORY_COLORS:
        palette = CATEGORY_COLORS[dataset_key]
    else:
        palette = list(plt.cm.tab20.colors)

    color_map = {}

    if dataset_key == "turblimp":
        ordered = [
            c for c in TURBLIMP_LEGEND_ORDER
            if c in categories_in_plot
        ]
        ordered += [
            c for c in categories_in_plot
            if c not in ordered
        ]
    else:
        ordered = categories_in_plot

    for i, cat in enumerate(ordered):
        color_map[cat] = palette[i % len(palette)]

    return color_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="BLiMP")
    parser.add_argument("--directory", default="english/cross-validation")
    parser.add_argument("--title", default="Average 2-fold overlap")
    args = parser.parse_args()

    dataset_key = args.dataset.lower()

    model_names = set()
    suite_rows = {}
    percentage_units = {}

    for fname in os.listdir(args.directory):
        if fname.startswith(f"cross-validation_{dataset_key}_") and fname.endswith(
            "txt"
        ):
            _, dataset, model_name, perc_str, num_folds = fname.split("_")

            percent = float(perc_str[:-1]) / 100
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

            for ctrl_file in os.listdir(args.directory):
                if f"cross-validation_blimp-control_{model_name}" in ctrl_file:
                    ctrl_path = os.path.join(args.directory, ctrl_file)
                    score, _ = read_control_file(ctrl_path, expect_many=False)

                    add_score(
                        suite_rows,
                        model_name,
                        "BLiMP-Control (Avg.)",
                        float(score),
                    )

            rand_score = random_overlap_expected(
                n_units_total,
                k_units,
                n_folds=int(num_folds[:-9]),
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
                if "Control" not in suite and "Random" not in suite
            ]
        ),
    )

    category_map = build_category_map(dataset_key)

    # Map every plotted suite to a phenomenon/category.
    # If a suite is missing from categories.py, treat the suite itself as its category.
    suite_to_plot_category = {}

    for suite in avg_entries:
        suite_key = norm_name(suite)

        if suite == "Random":
            suite_to_plot_category[suite] = "random"
        elif suite.startswith("BLiMP-Control"):
            suite_to_plot_category[suite] = "blimp-control_(avg.)"
        elif suite_key in category_map:
            suite_to_plot_category[suite] = category_map[suite_key]
        else:
            suite_to_plot_category[suite] = suite_key

    # Sort bars descending by overlap.
    avg_entries = dict(
        sorted(avg_entries.items(), key=lambda kv: kv[1], reverse=True)
    )

    # Build legend categories using desired TurBLiMP order where possible.
    categories_in_plot = []

    if dataset_key == "turblimp":
        for cat in TURBLIMP_LEGEND_ORDER:
            if cat in suite_to_plot_category.values():
                categories_in_plot.append(cat)

        for cat in suite_to_plot_category.values():
            if cat not in categories_in_plot and cat not in {
                "random",
                "blimp-control_(avg.)",
            }:
                categories_in_plot.append(cat)
    else:
        for suite in avg_entries:
            cat = suite_to_plot_category[suite]

            if cat not in {"random", "blimp-control_(avg.)"} and cat not in categories_in_plot:
                categories_in_plot.append(cat)

    cat2color = build_category_colors(dataset_key, categories_in_plot)

    GREY_CTRL = "#888888"
    GREY_RAND = "#bbbbbb"

    n_bars = len(avg_entries)
    bar_h = 1.0

    fig_height = max(5, n_bars * 0.18)
    fig, ax = plt.subplots(figsize=(8, fig_height))

    y_pos = np.arange(n_bars)

    # Keep the main grid fixed at 0--100.
    ax.set_xlim(0, 100)

    for i, (suite, avg_score) in enumerate(avg_entries.items()):
        cat = suite_to_plot_category[suite]

        if cat == "blimp-control_(avg.)":
            color = GREY_CTRL
        elif cat == "random":
            color = GREY_RAND
        else:
            color = cat2color.get(cat, "#1f77b4")

        ax.barh(
            y_pos[i],
            avg_score,
            height=bar_h,
            color=color,
            edgecolor="black",
        )

        # Percentage labels outside the main grid.
        ax.text(
            1.03,
            y_pos[i],
            f"{avg_score:.2f}%",
            transform=ax.get_yaxis_transform(),
            va="center",
            ha="left",
            fontsize=10,
            clip_on=False,
        )

        if len(model_list) > 1:
            for j, model in enumerate(model_list):
                if model not in suite_rows[suite]:
                    continue

                score = suite_rows[suite][model] / percentage_units[model] * 100
                offset = (j - 3.5) * 0.04

                ax.scatter(
                    score,
                    y_pos[i] + offset,
                    color="black",
                    marker="o",
                    s=10,
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

    if "blimp-control_(avg.)" in suite_to_plot_category.values():
        handles.append(
            Line2D(
                [0],
                [0],
                marker="s",
                linestyle="",
                markersize=10,
                markerfacecolor=GREY_CTRL,
                markeredgecolor="black",
                label="BLiMP-Control",
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

    out_pdf = f"{args.directory}/cross_validation_{dataset_key}.pdf"
    out_png = f"{args.directory}/cross_validation_{dataset_key}.png"

    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")

    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")
