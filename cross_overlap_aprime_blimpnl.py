from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from categories import CATEGORIES
from plot_utils import (
    build_model_style_maps,
    add_model_scatter,
)


APRIME_CATEGORIES: Dict[str, List[str]] = {
    "wh_movement": [
        "wh_movement__filler_effect_gap",
        "wh_movement__filler_effect_no_gap",
        "wh_movement__hierarchy",
        "wh_movement__question_formation",
        "wh_movement__stranding_1",
        "wh_movement__stranding_2",
    ],
    "wh_movement_restrictions": [
        "wh_movement_restrictions__bridge_verb_1",
        "wh_movement_restrictions__bridge_verb_2",
        "wh_movement_restrictions__island_1",
        "wh_movement_restrictions__island_2",
        "wh_movement_restrictions__resumptive_prolepsis",
        "wh_movement_restrictions__superiority",
    ],
    "relativization": [
        "relativization__island",
        "relativization__pied_piping",
        "relativization__resumptive_prolepsis",
    ],
    "topicalization": [
        "topicalization__island",
        "topicalization__question_similarity_1",
        "topicalization__question_similarity_2",
        "topicalization__resumptive_prolepsis",
    ],
    "parasitic_gaps": [
        "parasitic_gaps__scrambling",
        "parasitic_gaps__structure_type_1",
        "parasitic_gaps__structure_type_2",
        "parasitic_gaps__structure_type_3",
    ],
}


def pretty_name(name: str) -> str:
    special = {
        "blimp-nl": "BLiMP-NL",
        "wh_movement": "Wh-Movement",
        "wh_movement_restrictions": "Wh-Movement Restrictions",
        "relativization": "Relativization",
        "topicalization": "Topicalization",
        "parasitic_gaps": "Parasitic Gaps",
    }

    if name in special:
        return special[name]

    return name.replace("__", ": ").replace("_", " ").replace("-", " ").title()


def find_matrices(
    directory: str,
    dataset: str,
    percentage: float,
) -> Dict[str, pd.DataFrame]:
    directory_path = Path(directory)
    dataset_key = dataset.lower()

    pattern = f"cross-overlap_{dataset_key}_{dataset_key}_*_{percentage}%.csv"
    paths = sorted(directory_path.glob(pattern))

    if not paths:
        pattern = f"cross-overlap_{dataset_key}_{dataset_key}_*.csv"
        paths = sorted(directory_path.glob(pattern))

    if not paths:
        raise FileNotFoundError(
            f"No within-dataset cross-overlap matrix found in {directory_path} "
            f"for dataset={dataset_key}."
        )

    matrices = {}

    prefix = f"cross-overlap_{dataset_key}_{dataset_key}_"
    suffix = f"_{percentage}%.csv"

    for path in paths:
        name = path.name

        if name.startswith(prefix) and name.endswith(suffix):
            model_name = name[len(prefix) : -len(suffix)]
        else:
            model_name = path.stem.replace(prefix, "")

        df = pd.read_csv(path, index_col=0)
        df = convert_counts_to_percent_if_needed(df, model_name=model_name)
        matrices[model_name] = df

    return matrices


def convert_counts_to_percent_if_needed(
    df: pd.DataFrame,
    model_name: str,
) -> pd.DataFrame:
    values = df.to_numpy(dtype=float)
    max_val = np.nanmax(values)

    if max_val <= 100:
        print(f"{model_name}: matrix already appears to be percentages.")
        return df

    diagonal = np.diag(values)
    num_units = float(np.nanmedian(diagonal))

    if num_units <= 0:
        raise ValueError(
            f"{model_name}: could not infer number of units from diagonal."
        )

    print(
        f"{model_name}: matrix appears to be raw counts. "
        f"Converting by num_units={num_units:.0f}."
    )

    return df / num_units * 100.0


def finite_mean(vals: Sequence[float] | np.ndarray) -> float:
    arr = np.asarray(vals, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return np.nan

    return float(arr.mean())


def clean_suites(df: pd.DataFrame, suites: Sequence[str]) -> List[str]:
    return [s for s in suites if s in df.index and s in df.columns]


def within_values(df: pd.DataFrame, suites: Sequence[str]) -> np.ndarray:
    suites = clean_suites(df, suites)

    if len(suites) < 2:
        return np.array([], dtype=float)

    sub = df.loc[suites, suites].to_numpy(dtype=float)
    mask = ~np.eye(len(suites), dtype=bool)

    return sub[mask]


def cross_values(
    df: pd.DataFrame,
    suites_from: Sequence[str],
    suites_to: Sequence[str],
) -> np.ndarray:
    suites_from = [s for s in suites_from if s in df.index]
    suites_to = [s for s in suites_to if s in df.columns]

    if not suites_from or not suites_to:
        return np.array([], dtype=float)

    return df.loc[suites_from, suites_to].to_numpy(dtype=float).ravel()


def validate_aprime_suites(dataset: str):
    dataset_key = dataset.lower()
    all_dataset_suites = {
        suite
        for suites in CATEGORIES[dataset_key].values()
        for suite in suites
    }

    aprime_suites = {
        suite
        for suites in APRIME_CATEGORIES.values()
        for suite in suites
    }

    missing = sorted(aprime_suites - all_dataset_suites)

    if missing:
        raise ValueError(
            "These A′ suites are not listed in categories.py:\n"
            + "\n".join(missing)
        )


def aggregate(
    matrices: Dict[str, pd.DataFrame],
    dataset: str,
) -> Tuple[
    pd.DataFrame,
    Dict[str, Dict[str, float]],
    Dict[str, Dict[str, float]],
    Dict[str, Dict[str, float]],
]:
    dataset_key = dataset.lower()
    cat_map = CATEGORIES[dataset_key]

    all_dataset_suites = [
        suite
        for suites in cat_map.values()
        for suite in suites
    ]

    all_aprime_suites = [
        suite
        for suites in APRIME_CATEGORIES.values()
        for suite in suites
    ]

    non_aprime_suites = [
        suite
        for suite in all_dataset_suites
        if suite not in all_aprime_suites
    ]

    per_within = {}
    per_other_aprime = {}
    per_non_aprime = {}
    rows = []

    for category, suites in APRIME_CATEGORIES.items():
        other_aprime = [
            suite
            for other_cat, other_suites in APRIME_CATEGORIES.items()
            if other_cat != category
            for suite in other_suites
        ]

        w_by_model = {}
        oa_by_model = {}
        na_by_model = {}

        for model_name, df in matrices.items():
            w_by_model[model_name] = finite_mean(within_values(df, suites))
            oa_by_model[model_name] = finite_mean(cross_values(df, suites, other_aprime))
            na_by_model[model_name] = finite_mean(cross_values(df, suites, non_aprime_suites))

        per_within[category] = w_by_model
        per_other_aprime[category] = oa_by_model
        per_non_aprime[category] = na_by_model

        rows.append(
            {
                "category": category,
                "within": finite_mean(list(w_by_model.values())),
                "other_aprime": finite_mean(list(oa_by_model.values())),
                "non_aprime": finite_mean(list(na_by_model.values())),
                "n_paradigms": len(suites),
            }
        )

    summary = pd.DataFrame(rows).set_index("category")
    summary["aprime_advantage"] = summary["other_aprime"] - summary["non_aprime"]

    return summary, per_within, per_other_aprime, per_non_aprime


def print_domain_summary(
    matrices: Dict[str, pd.DataFrame],
    dataset: str,
):
    dataset_key = dataset.lower()
    cat_map = CATEGORIES[dataset_key]

    all_dataset_suites = [
        suite
        for suites in cat_map.values()
        for suite in suites
    ]

    all_aprime_suites = [
        suite
        for suites in APRIME_CATEGORIES.values()
        for suite in suites
    ]

    non_aprime_suites = [
        suite
        for suite in all_dataset_suites
        if suite not in all_aprime_suites
    ]

    print("\nOverall A′-domain summary:")

    for model_name, df in matrices.items():
        aprime_internal = within_values(df, all_aprime_suites)
        aprime_to_non = cross_values(df, all_aprime_suites, non_aprime_suites)

        print(
            f"  {model_name}: "
            f"A′↔A′ = {finite_mean(aprime_internal):.2f}%, "
            f"A′↔non-A′ = {finite_mean(aprime_to_non):.2f}%"
        )


def plot(
    summary: pd.DataFrame,
    per_within: Dict[str, Dict[str, float]],
    per_other_aprime: Dict[str, Dict[str, float]],
    per_non_aprime: Dict[str, Dict[str, float]],
    dataset: str,
    directory: str,
    title: str,
    add_model_markers: bool,
    seed: int = 42,
):
    summary = summary.copy()
    summary = summary.sort_values("aprime_advantage", ascending=False)

    show_cols = ["within", "other_aprime", "non_aprime"]

    colors = ["#89CCF1", "#FFB668", "#C0C0C0"]

    labels = [
        f"Within A′ category in {dataset}",
        f"With other A′ categories in {dataset}",
        f"With non-A′ categories in {dataset}",
    ]

    y = np.arange(len(summary))
    width = 0.18
    offsets = np.linspace(-width, width, len(show_cols))

    fig_h = max(4.5, 1.0 + len(summary) * 0.75)
    fig, ax = plt.subplots(figsize=(10.5, fig_h))

    for i, col in enumerate(show_cols):
        ax.barh(
            y + offsets[i],
            summary[col],
            height=width,
            color=colors[i],
            edgecolor="black",
            linewidth=0.9,
            label=labels[i],
            zorder=2,
        )

    ax.set_yticks(y)
    ax.set_yticklabels([pretty_name(c) for c in summary.index], fontsize=9)
    ax.invert_yaxis()

    max_val = np.nanmax(summary[show_cols].to_numpy(dtype=float))
    x_max = min(100, max(10, np.ceil((max_val + 8) / 10.0) * 10.0))

    ax.set_xlim(0, x_max)
    ax.set_xlabel("Percentage of units", fontsize=9)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="x", linestyle=":", alpha=0.4, zorder=0)
    ax.set_title(title, fontsize=11)

    rng = np.random.default_rng(seed)
    model_names = sorted({m for d in per_within.values() for m in d})
    model_list, model_to_color, model_to_marker = build_model_style_maps(model_names)

    metric_maps = [
        per_within,
        per_other_aprime,
        per_non_aprime,
    ]

    for yi, category in zip(y, summary.index):
        if add_model_markers:
            for j, metric_map in enumerate(metric_maps):
                add_model_scatter(
                    metric_map[category],
                    yi + offsets[j],
                    ax,
                    model_to_color,
                    model_to_marker,
                    rng,
                    s=20,
                    jitter=0.03,
                )

        for j, col in enumerate(show_cols):
            val = summary.loc[category, col]

            if not np.isfinite(val):
                label = "N/A"
                x = 0.5
            else:
                label = f"{val:.2f}%"
                x = val + 0.8

            ax.text(
                x,
                yi + offsets[j],
                label,
                va="center",
                ha="left",
                fontsize=7,
                zorder=9,
            )

    bar_handles = [
        Patch(facecolor=colors[i], edgecolor="black", label=labels[i])
        for i in range(len(show_cols))
    ]

    bar_legend = ax.legend(
        handles=bar_handles,
        loc="lower right",
        fontsize=7,
        frameon=True,
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
                markersize=5,
                markeredgecolor="black",
                markeredgewidth=0.4,
                label=m,
            )
            for m in model_list
        ]

        ax.legend(
            handles=model_handles,
            loc="upper right",
            title="Models",
            fontsize=7,
            title_fontsize=8,
            frameon=True,
        )

    for spine in ax.spines.values():
        spine.set_linewidth(1.1)
        spine.set_color("black")

    fig.tight_layout()

    dataset_key = dataset.lower()
    out_png = Path(directory) / f"cross_overlap_aprime_{dataset_key}.png"
    out_pdf = Path(directory) / f"cross_overlap_aprime_{dataset_key}.pdf"

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")

    print(f"\nSaved {out_png}")
    print(f"Saved {out_pdf}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="BLiMP-NL")
    p.add_argument("--directory", default="multilingual/blimpnl")
    p.add_argument("--title", default="Cross-phenomenon overlap in BLiMP-NL (A′-dependencies)")
    p.add_argument("--percentage", type=float, default=1.0)
    p.add_argument("--add-model-markers", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    dataset_key = args.dataset.lower()

    validate_aprime_suites(dataset_key)

    matrices = find_matrices(
        directory=args.directory,
        dataset=dataset_key,
        percentage=args.percentage,
    )

    summary, per_within, per_other_aprime, per_non_aprime = aggregate(
        matrices=matrices,
        dataset=dataset_key,
    )

    print("\nValues used in plot:")
    print(
        summary[
            [
                "n_paradigms",
                "within",
                "other_aprime",
                "non_aprime",
                "aprime_advantage",
            ]
        ].to_string()
    )

    print_domain_summary(matrices, dataset_key)

    plot(
        summary=summary,
        per_within=per_within,
        per_other_aprime=per_other_aprime,
        per_non_aprime=per_non_aprime,
        dataset=args.dataset,
        directory=args.directory,
        title=args.title,
        add_model_markers=args.add_model_markers,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
