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
from plot_utils import build_model_style_maps, add_model_scatter


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
        "blimp": "BLiMP",
        "wh_movement": "Wh-Movement",
        "wh_movement_restrictions": "Wh-Movement Restrictions",
        "relativization": "Relativization",
        "topicalization": "Topicalization",
        "parasitic_gaps": "Parasitic Gaps",
    }

    if name in special:
        return special[name]

    return name.replace("__", ": ").replace("_", " ").replace("-", " ").title()


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


def extract_model_name(path: Path, prefix: str, percentage: float) -> str:
    name = path.name
    suffix = f"_{percentage}%.csv"

    if name.startswith(prefix) and name.endswith(suffix):
        return name[len(prefix) : -len(suffix)]

    stem = path.stem
    if stem.startswith(prefix):
        return stem[len(prefix) :]

    return stem


def find_within_matrices(
    directory: str,
    dataset: str,
    percentage: float,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, float], Dict[str, bool]]:
    directory_path = Path(directory)
    dataset_key = dataset.lower()

    prefix = f"cross-overlap_{dataset_key}_{dataset_key}_"
    pattern = f"{prefix}*_{percentage}%.csv"

    paths = sorted(directory_path.glob(pattern))

    if not paths:
        paths = sorted(directory_path.glob(f"{prefix}*.csv"))

    if not paths:
        raise FileNotFoundError(
            f"No within-dataset cross-overlap matrix found in {directory_path} "
            f"for dataset={dataset_key}."
        )

    matrices = {}
    normalizers = {}
    raw_flags = {}

    for path in paths:
        model = extract_model_name(path, prefix, percentage)
        df = pd.read_csv(path, index_col=0)

        values = df.to_numpy(dtype=float)
        max_val = np.nanmax(values)
        diag = np.diag(values)

        normalizer = float(np.nanmedian(diag))
        is_raw = max_val > 100 or normalizer > 100

        if is_raw:
            print(
                f"{model}: within matrix appears to be raw counts. "
                f"Converting by num_units={normalizer:.0f}."
            )
            df = df / normalizer * 100.0
        else:
            print(f"{model}: within matrix appears to already be percentages.")

        matrices[model] = df
        normalizers[model] = normalizer
        raw_flags[model] = is_raw

    return matrices, normalizers, raw_flags


def find_cross_matrices_to_blimp(
    directory: str,
    dataset: str,
    percentage: float,
    normalizers: Dict[str, float],
    raw_flags: Dict[str, bool],
) -> Dict[str, pd.DataFrame]:
    """
    Returns matrices with:
      rows    = BLiMP-NL suites
      columns = BLiMP suites

    It handles either saved orientation:
      cross-overlap_blimp_blimp-nl_...
      cross-overlap_blimp-nl_blimp_...
    """
    directory_path = Path(directory)
    dataset_key = dataset.lower()

    direct_prefix = f"cross-overlap_{dataset_key}_blimp_"
    reverse_prefix = f"cross-overlap_blimp_{dataset_key}_"

    direct_paths = sorted(directory_path.glob(f"{direct_prefix}*_{percentage}%.csv"))
    reverse_paths = sorted(directory_path.glob(f"{reverse_prefix}*_{percentage}%.csv"))

    if not direct_paths:
        direct_paths = sorted(directory_path.glob(f"{direct_prefix}*.csv"))

    if not reverse_paths:
        reverse_paths = sorted(directory_path.glob(f"{reverse_prefix}*.csv"))

    paths_with_orientation = []

    for path in direct_paths:
        paths_with_orientation.append((path, "direct"))

    for path in reverse_paths:
        paths_with_orientation.append((path, "reverse"))

    if not paths_with_orientation:
        raise FileNotFoundError(
            f"No BLiMP cross-overlap matrix found in {directory_path}. "
            f"Expected cross-overlap_blimp_{dataset_key}_... or "
            f"cross-overlap_{dataset_key}_blimp_..."
        )

    matrices = {}

    for path, orientation in paths_with_orientation:
        prefix = direct_prefix if orientation == "direct" else reverse_prefix
        model = extract_model_name(path, prefix, percentage)

        df = pd.read_csv(path, index_col=0)

        if orientation == "reverse":
            df = df.T

        if model not in normalizers:
            fallback_model = next(iter(normalizers))
            print(
                f"{model}: no matching within normalizer found; "
                f"using {fallback_model} normalizer."
            )
            normalizer = normalizers[fallback_model]
            is_raw = raw_flags[fallback_model]
        else:
            normalizer = normalizers[model]
            is_raw = raw_flags[model]

        if is_raw:
            print(
                f"{model}: cross matrix assumed raw counts. "
                f"Converting by num_units={normalizer:.0f}."
            )
            df = df / normalizer * 100.0
        else:
            print(f"{model}: cross matrix assumed already percentages.")

        matrices[model] = df

    return matrices


def get_all_suites(cat_map: Dict[str, List[str]]) -> List[str]:
    return [suite for suites in cat_map.values() for suite in suites]


def get_blimp_aprime_categories(cat_blimp: Dict[str, List[str]]) -> List[str]:
    """
    BLiMP does not use the same category names as BLiMP-NL.
    For A′/extraction-style dependencies, the relevant BLiMP categories are
    normally filler-gap dependencies and island effects.
    """
    preferred = [
        "filler_gap_dependency",
        "island_effects",
    ]

    found = [cat for cat in preferred if cat in cat_blimp]

    if found:
        return found

    fallback = [
        cat
        for cat in cat_blimp
        if any(key in cat for key in ["filler", "gap", "island", "wh", "relative"])
    ]

    if not fallback:
        print("\nAvailable BLiMP categories:")
        for cat in cat_blimp:
            print(" ", cat)

        raise ValueError(
            "Could not identify BLiMP A′ categories. "
            "Expected filler_gap_dependency and/or island_effects."
        )

    return fallback


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
    mats_lang: Dict[str, pd.DataFrame],
    mats_cross: Dict[str, pd.DataFrame],
    dataset: str,
) -> Tuple[
    pd.DataFrame,
    Dict[str, Dict[str, float]],
    Dict[str, Dict[str, float]],
    Dict[str, Dict[str, float]],
    Dict[str, Dict[str, float]],
    Dict[str, Dict[str, float]],
]:
    dataset_key = dataset.lower()
    cat_src = CATEGORIES[dataset_key]
    cat_blimp = CATEGORIES["blimp"]

    all_src_suites = get_all_suites(cat_src)
    all_aprime_src = get_all_suites(APRIME_CATEGORIES)

    non_aprime_src = [
        suite for suite in all_src_suites if suite not in all_aprime_src
    ]

    blimp_aprime_cats = get_blimp_aprime_categories(cat_blimp)
    all_aprime_blimp = [
        suite for cat in blimp_aprime_cats for suite in cat_blimp[cat]
    ]

    all_blimp_suites = get_all_suites(cat_blimp)
    non_aprime_blimp = [
        suite for suite in all_blimp_suites if suite not in all_aprime_blimp
    ]

    print("\nUsing BLiMP A′ categories:")
    for cat in blimp_aprime_cats:
        print(f"  {cat}: {len(cat_blimp[cat])} suites")

    per_w = {}
    per_other_aprime = {}
    per_non_aprime = {}
    per_blimp_aprime = {}
    per_blimp_non_aprime = {}

    rows = []

    for category, suites in APRIME_CATEGORIES.items():
        other_aprime = [
            suite
            for other_cat, other_suites in APRIME_CATEGORIES.items()
            if other_cat != category
            for suite in other_suites
        ]

        w = {}
        oa = {}
        na = {}
        ba = {}
        bn = {}

        for model, df in mats_lang.items():
            w[model] = finite_mean(within_values(df, suites))
            oa[model] = finite_mean(cross_values(df, suites, other_aprime))
            na[model] = finite_mean(cross_values(df, suites, non_aprime_src))

            if model in mats_cross:
                df_cross = mats_cross[model]
            else:
                df_cross = mats_cross[next(iter(mats_cross))]

            ba[model] = finite_mean(cross_values(df_cross, suites, all_aprime_blimp))
            bn[model] = finite_mean(cross_values(df_cross, suites, non_aprime_blimp))

        per_w[category] = w
        per_other_aprime[category] = oa
        per_non_aprime[category] = na
        per_blimp_aprime[category] = ba
        per_blimp_non_aprime[category] = bn

        rows.append(
            {
                "category": category,
                "within": finite_mean(list(w.values())),
                "other_aprime": finite_mean(list(oa.values())),
                "non_aprime": finite_mean(list(na.values())),
                "blimp_aprime": finite_mean(list(ba.values())),
                "blimp_non_aprime": finite_mean(list(bn.values())),
                "n_paradigms": len(suites),
            }
        )

    summary = pd.DataFrame(rows).set_index("category")
    summary["aprime_advantage"] = summary["other_aprime"] - summary["non_aprime"]

    return (
        summary,
        per_w,
        per_other_aprime,
        per_non_aprime,
        per_blimp_aprime,
        per_blimp_non_aprime,
    )


def plot(
    summary: pd.DataFrame,
    pw: Dict[str, Dict[str, float]],
    poa: Dict[str, Dict[str, float]],
    pna: Dict[str, Dict[str, float]],
    pba: Dict[str, Dict[str, float]],
    pbn: Dict[str, Dict[str, float]],
    dataset: str,
    directory: str,
    title: str,
    add_model_markers: bool,
    seed: int = 42,
):
    summary = summary.copy()
    summary = summary.sort_values("aprime_advantage", ascending=False)

    show_cols = [
        "within",
        "other_aprime",
        "non_aprime",
        "blimp_aprime",
        "blimp_non_aprime",
    ]

    colors = ["#89CCF1", "#FFB668", "#C0C0C0", "#8ECA7A", "#BC9E92"]

    labels = [
        f"Within-category in {dataset}",
        f"With other A′ categories in {dataset}",
        f"With non-A′ categories in {dataset}",
        "With A′ categories in BLiMP",
        "With non-A′ categories in BLiMP",
    ]

    y = np.arange(len(summary))
    width = 0.14
    offsets = np.linspace(-2 * width, 2 * width, len(show_cols))

    fig_h = max(4.8, 1.0 + len(summary) * 0.75)
    fig, ax = plt.subplots(figsize=(11.5, fig_h))

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
    model_names = sorted({m for d in pw.values() for m in d})
    model_list, model_to_color, model_to_marker = build_model_style_maps(model_names)

    metric_maps = [pw, poa, pna, pba, pbn]

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
    out_png = Path(directory) / f"cross_overlap_aprime_{dataset_key}_5bar.png"
    out_pdf = Path(directory) / f"cross_overlap_aprime_{dataset_key}_5bar.pdf"

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")

    print(f"\nSaved {out_png}")
    print(f"Saved {out_pdf}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="BLiMP-NL")
    p.add_argument("--directory", default="multilingual/blimpnl")
    p.add_argument(
        "--title",
        default="Cross-phenomenon overlap in BLiMP-NL (A′-dependencies)",
    )
    p.add_argument("--percentage", type=float, default=1.0)
    p.add_argument("--add-model-markers", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    dataset_key = args.dataset.lower()

    validate_aprime_suites(dataset_key)

    mats_lang, normalizers, raw_flags = find_within_matrices(
        directory=args.directory,
        dataset=dataset_key,
        percentage=args.percentage,
    )

    mats_cross = find_cross_matrices_to_blimp(
        directory=args.directory,
        dataset=dataset_key,
        percentage=args.percentage,
        normalizers=normalizers,
        raw_flags=raw_flags,
    )

    summary, pw, poa, pna, pba, pbn = aggregate(
        mats_lang=mats_lang,
        mats_cross=mats_cross,
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
                "blimp_aprime",
                "blimp_non_aprime",
                "aprime_advantage",
            ]
        ].to_string()
    )

    plot(
        summary=summary,
        pw=pw,
        poa=poa,
        pna=pna,
        pba=pba,
        pbn=pbn,
        dataset=args.dataset,
        directory=args.directory,
        title=args.title,
        add_model_markers=args.add_model_markers,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
