import os
import argparse
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from categories import CATEGORIES
from model_utils import get_num_blocks, get_hidden_dim
from plot_utils import pretty_category_name, CATEGORY_COLORS
from utils import random_overlap_expected


BLIMPNL_CATEGORIES: Dict[str, List[str]] = {
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


BLIMPNL_CATEGORY_COLORS = [
    "#89CCF1",
    "#FFB668",
    "#8ECA7A",
    "#BC9E92",
    "#C0C0C0",
    "#D6A5E8",
]


def get_category_map(dataset: str) -> Dict[str, List[str]]:
    dataset_key = dataset.lower()

    if dataset_key == "blimp-nl":
        return BLIMPNL_CATEGORIES

    return CATEGORIES[dataset_key]


def get_category_colors(dataset: str, unique_cats: List[str]) -> Dict[str, str]:
    dataset_key = dataset.lower()

    if dataset_key == "blimp-nl":
        return {
            cat: BLIMPNL_CATEGORY_COLORS[i % len(BLIMPNL_CATEGORY_COLORS)]
            for i, cat in enumerate(unique_cats)
        }

    return {
        cat: CATEGORY_COLORS[dataset_key][i]
        for i, cat in enumerate(unique_cats)
    }


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


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="BLiMP")
    p.add_argument("--directory", default="english/cross-validation")
    p.add_argument("--title", default="Average 2-fold overlap")
    args = p.parse_args()

    dataset_key = args.dataset.lower()

    model_names = set()
    suite_rows = {}
    percentage_units = {}

    num = 0
    total = 0.0

    for fname in os.listdir(args.directory):
        if fname.startswith(f"cross-validation_{dataset_key}_") and fname.endswith(
            "txt"
        ):
            parts = fname.split("_")

            if len(parts) != 5:
                print(f"Skipping unexpected filename format: {fname}")
                continue

            _, dataset, model_name, perc_str, num_folds = parts

            percent = float(perc_str[:-1]) / 100
            model_names.add(model_name)

            print(fname)

            n_units_total = get_num_blocks(model_name) * get_hidden_dim(model_name)
            k_units = int(n_units_total * percent)

            percentage_units[model_name] = k_units

            with open(os.path.join(args.directory, fname)) as fh:
                for ln in fh:
                    row = ln.strip().split()

                    if len(row) != 3:
                        continue

                    overlap_count = float(row[0])
                    suite = row[2]

                    total += overlap_count / k_units
                    num += 1

                    add_score(suite_rows, model_name, suite, overlap_count)

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

    model_list = sorted(model_names)

    if not model_list:
        raise RuntimeError(
            f"No cross-validation files found in {args.directory} "
            f"for dataset {dataset_key}."
        )

    avg_entries = {
        suite: np.mean(
            [
                100 * individual_scores[m] / percentage_units[m]
                for m in model_list
                if m in individual_scores
            ]
        )
        for suite, individual_scores in suite_rows.items()
    }

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

    category_map = {}

    for cat, suites in get_category_map(args.dataset).items():
        for suite_name in suites:
            category_map[suite_name] = cat

    unique_cats = sorted(
        set(category_map[s] for s in avg_entries if s in category_map)
    )

    cat2color = get_category_colors(args.dataset, unique_cats)

    GREY_CTRL = "#888888"
    GREY_RAND = "#bbbbbb"

    avg_entries = dict(
        sorted(avg_entries.items(), key=lambda kv: kv[1], reverse=True)
    )

    n_bars = len(avg_entries)
    bar_h = 1.0

    fig_height = max(4.0, n_bars * 0.18)
    fig, ax = plt.subplots(figsize=(8, fig_height))

    y_pos = np.arange(n_bars)

    for i, (suite, avg_score) in enumerate(avg_entries.items()):
        if suite.startswith("BLiMP-Control"):
            color = GREY_CTRL
        elif suite == "Random":
            color = GREY_RAND
        else:
            color = cat2color.get(category_map.get(suite), "#1f77b4")

        ax.barh(
            y_pos[i],
            avg_score,
            height=bar_h,
            color=color,
            edgecolor="black",
        )

        ax.text(
            103,
            y_pos[i],
            f"{avg_score:.2f}%",
            va="center",
            fontsize=10,
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
                )

    ax.set_xlabel("Percentage of Units", fontsize=12)
    ax.set_xlim(0, 110)
    ax.tick_params(axis="x", labelsize=10)

    ax.set_yticks([])
    ax.invert_yaxis()

    ax.set_title(args.title, fontsize=14)

    handles = (
        [
            Line2D(
                [0],
                [0],
                marker="s",
                linestyle="",
                markersize=10,
                markerfacecolor=cat2color[cat],
                markeredgecolor="black",
                label=display_category_name(cat),
            )
            for cat in unique_cats
        ]
        + (
            [
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
            ]
            if "BLiMP-Control (Avg.)" in suite_rows
            else []
        )
        + [
            Line2D(
                [0],
                [0],
                marker="s",
                linestyle="",
                markersize=10,
                markerfacecolor=GREY_RAND,
                markeredgecolor="black",
                label="Random",
            ),
        ]
    )

    ax.legend(
        handles=handles,
        bbox_to_anchor=(1.2, 1),
        loc="upper left",
        frameon=True,
    )

    plt.tight_layout()

    fig.savefig(
        f"{args.directory}/cross_validation_{dataset_key}.pdf",
        bbox_inches="tight",
    )
    fig.savefig(
        f"{args.directory}/cross_validation_{dataset_key}.png",
        dpi=300,
        bbox_inches="tight",
    )

    print(f"Saved: {args.directory}/cross_validation_{dataset_key}.pdf")
    print(f"Saved: {args.directory}/cross_validation_{dataset_key}.png")
