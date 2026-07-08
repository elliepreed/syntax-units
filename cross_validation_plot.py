import os
import argparse
from typing import Dict, List

import numpy as np
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


BLIMPNL_CATEGORY_COLORS = {
    "agreement_and_binding": "#89CCF1",
    "argument_structure_and_voice": "#FFB668",
    "clausal_arguments": "#8ECA7A",
    "movement_and_word_order": "#BC9E92",
    "nominal_domain": "#C0C0C0",
    "verbal_functional_domain": "#D6A5E8",
}


def get_category_map(dataset: str) -> Dict[str, List[str]]:
    dataset_key = dataset.lower()

    if dataset_key == "blimp-nl":
        return BLIMPNL_CATEGORIES

    return CATEGORIES[dataset_key]


def get_category_colors(dataset: str, unique_cats: List[str]) -> Dict[str, str]:
    dataset_key = dataset.lower()

    if dataset_key == "blimp-nl":
        return {
            cat: BLIMPNL_CATEGORY_COLORS.get(cat, "#1f77b4")
            for cat in unique_cats
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


def display_suite_name(suite: str) -> str:
    return suite.replace("_", " ").replace("-", " ").title()


def add_score(table, model_name, suite, score):
    if suite not in table:
        table[suite] = {}
    table[suite][model_name] = score


def parse_cv_filename(fname: str, dataset_key: str):
    """
    Expected:
    cross-validation_blimp-nl_gemma-3-4b-pt_1.0%_2-fold.txt
    """
    prefix = f"cross-validation_{dataset_key}_"

    if not fname.startswith(prefix) or not fname.endswith(".txt"):
        return None

    rest = fname[len(prefix):]
    try:
        model_name, perc_str, folds_str = rest.rsplit("_", 2)
    except ValueError:
        return None

    percent = float(perc_str.replace("%", "")) / 100.0
    num_folds = int(folds_str.replace("-fold.txt", ""))

    return model_name, percent, num_folds


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

    for fname in os.listdir(args.directory):
        parsed = parse_cv_filename(fname, dataset_key)

        if parsed is None:
            continue

        model_name, percent, num_folds = parsed

        print(fname)
        model_names.add(model_name)

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

                add_score(suite_rows, model_name, suite, overlap_count)

        rand_score = random_overlap_expected(
            n_units_total,
            k_units,
            n_folds=num_folds,
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
                if suite != "Random" and "Control" not in suite
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

    GREY_RAND = "#bbbbbb"

    avg_entries = dict(
        sorted(avg_entries.items(), key=lambda kv: kv[1], reverse=True)
    )

    n_bars = len(avg_entries)
    fig_height = max(6.0, n_bars * 0.32)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    y_pos = np.arange(n_bars)

    labels = []

    for i, (suite, avg_score) in enumerate(avg_entries.items()):
        if suite == "Random":
            color = GREY_RAND
            label = "Random"
        else:
            cat = category_map.get(suite)
            color = cat2color.get(cat, "#1f77b4")
            label = display_suite_name(suite)

        labels.append(label)

        ax.barh(
            y_pos[i],
            avg_score,
            height=0.75,
            color=color,
            edgecolor="black",
        )

        ax.text(
            avg_score + 0.4,
            y_pos[i],
            f"{avg_score:.2f}%",
            va="center",
            fontsize=8,
        )

        if len(model_list) > 1:
            for j, model in enumerate(model_list):
                if model not in suite_rows[suite]:
                    continue

                score = suite_rows[suite][model] / percentage_units[model] * 100
                offset = (j - (len(model_list) - 1) / 2) * 0.05

                ax.scatter(
                    score,
                    y_pos[i] + offset,
                    color="black",
                    marker="o",
                    s=12,
                    zorder=4,
                )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()

    max_score = max(avg_entries.values())
    x_max = max(10, int(np.ceil((max_score + 5) / 5) * 5))

    ax.set_xlabel("Percentage of units", fontsize=11)
    ax.set_xlim(0, x_max)
    ax.tick_params(axis="x", labelsize=9)
    ax.grid(axis="x", linestyle=":", alpha=0.4, zorder=0)

    ax.set_title(args.title, fontsize=13)

    handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="",
            markersize=9,
            markerfacecolor=cat2color[cat],
            markeredgecolor="black",
            label=display_category_name(cat),
        )
        for cat in unique_cats
    ]

    if "Random" in avg_entries:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="s",
                linestyle="",
                markersize=9,
                markerfacecolor=GREY_RAND,
                markeredgecolor="black",
                label="Random",
            )
        )

    ax.legend(
        handles=handles,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=True,
        fontsize=8,
    )

    plt.tight_layout()

    out_pdf = f"{args.directory}/cross_validation_{dataset_key}.pdf"
    out_png = f"{args.directory}/cross_validation_{dataset_key}.png"

    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")

    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")
