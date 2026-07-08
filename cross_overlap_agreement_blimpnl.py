from __future__ import annotations

import argparse
import math
import os
import re
from glob import glob
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from categories import CATEGORIES


# ============================================================
# BLiMP-NL category setup
# ============================================================
# BLiMP-NL appears to have one clearly explicit agreement suite:
#   anaphor_agreement
#
# If you later decide that another BLiMP-NL suite should count as agreement,
# add it here, e.g.
#   "determiner_agreement": ["determiners"]
#
BLIMPNL_AGREEMENT_CATEGORIES: Dict[str, List[str]] = {
    "anaphor_agreement": ["anaphor_agreement"],
}


def is_agreement_category(cat: str) -> bool:
    cat_l = cat.lower()
    return (
        "agreement" in cat_l
        or cat_l in {"anaphor_number", "anaphor_gender"}
    )


def flatten(xs: Sequence[Sequence[str]]) -> List[str]:
    return [x for sub in xs for x in sub]


def pretty_name(name: str) -> str:
    replacements = {
        "anaphor_agreement": "Anaphor Agreement",
        "det_n_agreement": "DET-N Agreement",
        "s_v_agreement": "S-V Agreement",
        "subject_verb_agreement": "S-V Agreement",
        "np_agreement": "NP Agreement",
        "floating_quantifier_agreement": "Floating Quantifier Agreement",
        "subject_predicate_agreement": "Subject-Predicate Agreement",
    }
    if name in replacements:
        return replacements[name]
    return name.replace("_", " ").replace("-", " ").title()


def finite_mean(values: np.ndarray | Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    return float(arr.mean())


# ============================================================
# Matrix loading
# ============================================================

def model_name_from_path(path: str, dataset_a: str, dataset_b: str) -> str:
    base = os.path.basename(path)
    base = base[:-4] if base.endswith(".csv") else base

    for prefix in [
        f"cross-overlap_{dataset_a}_{dataset_b}_",
        f"cross-overlap_{dataset_b}_{dataset_a}_",
    ]:
        if base.startswith(prefix):
            rest = base[len(prefix):]
            rest = re.sub(r"_\d+(?:\.\d+)?%$", "", rest)
            return rest

    return base


def orient_matrix(
    df: pd.DataFrame,
    desired_rows: Sequence[str],
    desired_cols: Sequence[str],
    path: str,
) -> pd.DataFrame:
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)

    desired_rows = list(desired_rows)
    desired_cols = list(desired_cols)

    rows_in_index = all(x in df.index for x in desired_rows)
    cols_in_cols = all(x in df.columns for x in desired_cols)

    # Already row = desired_rows, col = desired_cols
    if rows_in_index and cols_in_cols:
        return df.loc[desired_rows, desired_cols]

    # Transposed case: row = desired_cols, col = desired_rows
    cols_in_index = all(x in df.index for x in desired_cols)
    rows_in_cols = all(x in df.columns for x in desired_rows)

    if cols_in_index and rows_in_cols:
        return df.loc[desired_cols, desired_rows].T

    missing_rows = [x for x in desired_rows if x not in df.index and x not in df.columns]
    missing_cols = [x for x in desired_cols if x not in df.index and x not in df.columns]

    raise ValueError(
        f"Could not orient matrix: {path}\n"
        f"Wanted rows: {desired_rows[:5]} ...\n"
        f"Wanted cols: {desired_cols[:5]} ...\n"
        f"Missing row labels: {missing_rows[:20]}\n"
        f"Missing col labels: {missing_cols[:20]}\n"
        f"Actual index sample: {list(df.index[:10])}\n"
        f"Actual columns sample: {list(df.columns[:10])}"
    )


def load_pair_matrices(
    directory: str,
    dataset_a: str,
    dataset_b: str,
    desired_rows: Sequence[str],
    desired_cols: Sequence[str],
) -> Dict[str, pd.DataFrame]:
    patterns = [
        os.path.join(directory, f"cross-overlap_{dataset_a}_{dataset_b}_*.csv"),
        os.path.join(directory, f"cross-overlap_{dataset_b}_{dataset_a}_*.csv"),
    ]

    paths = []
    for pat in patterns:
        paths.extend(glob(pat))

    paths = sorted(set(paths))

    if not paths:
        raise FileNotFoundError(
            f"No cross-overlap CSV found in {directory} for "
            f"{dataset_a} × {dataset_b}. Tried:\n"
            + "\n".join(patterns)
        )

    matrices = {}

    for path in paths:
        model = model_name_from_path(path, dataset_a, dataset_b)
        df = pd.read_csv(path, index_col=0)
        df = orient_matrix(df, desired_rows, desired_cols, path)
        matrices[model] = df

    return matrices


# ============================================================
# Aggregation
# ============================================================

def within_values(df: pd.DataFrame, suites: Sequence[str]) -> np.ndarray:
    suites = [s for s in suites if s in df.index and s in df.columns]

    # If a category has only one suite, there are no distinct within-category
    # pairs. We deliberately do NOT use the diagonal, because that would be
    # self-overlap and artificially high.
    if len(suites) < 2:
        return np.array([], dtype=float)

    sub = df.loc[suites, suites]
    mask = ~np.eye(len(suites), dtype=bool)
    return sub.values[mask]


def cross_values(
    df: pd.DataFrame,
    suites_from: Sequence[str],
    suites_to: Sequence[str],
) -> np.ndarray:
    suites_from = [s for s in suites_from if s in df.index]
    suites_to = [s for s in suites_to if s in df.columns]

    if not suites_from or not suites_to:
        return np.array([], dtype=float)

    return df.loc[suites_from, suites_to].values.ravel()


def aggregate(
    mats_lang: Dict[str, pd.DataFrame],
    mats_cross: Dict[str, pd.DataFrame],
    cat_src: Dict[str, List[str]],
    cat_blimp: Dict[str, List[str]],
) -> Tuple[pd.DataFrame, dict, dict, dict, dict, dict]:
    agree_src = [cat for cat in cat_src if is_agreement_category(cat)]
    all_agree_src = flatten([cat_src[c] for c in agree_src])

    # Use the rows in the actual BLiMP-NL matrix to define non-agreement suites.
    first_lang_df = next(iter(mats_lang.values()))
    all_src_suites = list(first_lang_df.index)
    non_agree_src = [s for s in all_src_suites if s not in all_agree_src]

    agree_blimp = [cat for cat in cat_blimp if is_agreement_category(cat)]
    all_agree_blimp = flatten([cat_blimp[c] for c in agree_blimp])

    first_cross_df = next(iter(mats_cross.values()))
    all_blimp_suites = list(first_cross_df.columns)
    non_agree_blimp = [s for s in all_blimp_suites if s not in all_agree_blimp]

    per_w, per_oa, per_na, per_ba, per_bn = {}, {}, {}, {}, {}
    rows = []

    for cat in agree_src:
        suites = cat_src[cat]
        other_agree = [
            s
            for other_cat in agree_src
            if other_cat != cat
            for s in cat_src[other_cat]
        ]

        w, oa, na, ba, bn = {}, {}, {}, {}, {}

        for model, df in mats_lang.items():
            w[model] = finite_mean(within_values(df, suites))
            oa[model] = finite_mean(cross_values(df, suites, other_agree))
            na[model] = finite_mean(cross_values(df, suites, non_agree_src))

            df_cross = mats_cross[model]
            ba[model] = finite_mean(cross_values(df_cross, suites, all_agree_blimp))
            bn[model] = finite_mean(cross_values(df_cross, suites, non_agree_blimp))

        per_w[cat] = w
        per_oa[cat] = oa
        per_na[cat] = na
        per_ba[cat] = ba
        per_bn[cat] = bn

        rows.append(
            {
                "category": cat,
                "within": finite_mean(list(w.values())),
                "other_agree": finite_mean(list(oa.values())),
                "non_agree": finite_mean(list(na.values())),
                "blimp_agree": finite_mean(list(ba.values())),
                "blimp_non_agree": finite_mean(list(bn.values())),
            }
        )

    summary = pd.DataFrame(rows).set_index("category")
    return summary, per_w, per_oa, per_na, per_ba, per_bn


# ============================================================
# Plotting
# ============================================================

def model_styles(model_names: Sequence[str]):
    markers = ["o", "s", "^", "D", "P", "X", "v", "*", "<", ">"]
    cmap = plt.get_cmap("tab10")

    m2c = {}
    m2m = {}

    for i, m in enumerate(model_names):
        m2c[m] = cmap(i % 10)
        m2m[m] = markers[i % len(markers)]

    return m2c, m2m


def add_model_points(
    values_by_model: Dict[str, float],
    y: float,
    ax,
    m2c: Dict[str, object],
    m2m: Dict[str, str],
    rng: np.random.Generator,
):
    for model, value in values_by_model.items():
        if not np.isfinite(value):
            continue
        jitter = rng.normal(0, 0.015)
        ax.scatter(
            value,
            y + jitter,
            s=22,
            color=m2c[model],
            marker=m2m[model],
            edgecolor="black",
            linewidth=0.4,
            zorder=5,
        )


def plot(
    summary: pd.DataFrame,
    pw: Dict[str, Dict[str, float]],
    poa: Dict[str, Dict[str, float]],
    pna: Dict[str, Dict[str, float]],
    pba: Dict[str, Dict[str, float]],
    pbn: Dict[str, Dict[str, float]],
    dataset_label: str,
    directory: str,
    title: str,
    add_model_markers: bool,
    seed: int,
    xlim: float | None,
):
    show_cols = [
        "within",
        "other_agree",
        "non_agree",
        "blimp_agree",
        "blimp_non_agree",
    ]

    colors = ["#89CCF1", "#FFB668", "#C0C0C0", "#8ECA7A", "#BC9E92"]
    labels = [
        f"Within-category in {dataset_label}",
        f"With other agreement categories in {dataset_label}",
        f"With non-agreement categories in {dataset_label}",
        "With agreement categories in BLiMP",
        "With non-agreement categories in BLiMP",
    ]

    # Sort like original: within minus other agreement.
    # If within/other_agree are NaN, send that category to the bottom.
    summary = summary.copy()
    summary["diff"] = summary["within"] - summary["other_agree"]
    summary["diff_for_sort"] = summary["diff"].fillna(-1e9)
    summary = summary.sort_values("diff_for_sort", ascending=False)

    y = np.arange(len(summary))
    width = 0.15
    offs = np.linspace(-2 * width, 2 * width, len(show_cols))

    valid_values = summary[show_cols].to_numpy(dtype=float).ravel()
    valid_values = valid_values[np.isfinite(valid_values)]

    if xlim is None:
        if valid_values.size:
            xlim = max(10, math.ceil((float(valid_values.max()) + 5) / 10) * 10)
        else:
            xlim = 10

    fig_h = max(2.2, 1.2 + len(summary) * 0.75)
    fig, ax = plt.subplots(figsize=(9, fig_h))

    for i, col in enumerate(show_cols):
        values = summary[col].to_numpy(dtype=float)

        for yi, val in zip(y, values):
            if not np.isfinite(val):
                continue

            ax.barh(
                yi + offs[i],
                val,
                height=width,
                color=colors[i],
                edgecolor="black",
                label=labels[i] if yi == y[0] else None,
                zorder=2,
            )

            ax.text(
                val + 0.8,
                yi + offs[i],
                f"{val:.2f}%",
                va="center",
                ha="left",
                fontsize=7,
                zorder=9,
            )

    rng = np.random.default_rng(seed)

    model_names = sorted(
        set(
            list(next(iter(pw.values())).keys())
            if pw
            else []
        )
    )
    m2c, m2m = model_styles(model_names)

    if add_model_markers:
        dicts = [pw, poa, pna, pba, pbn]

        for yi, cat in zip(y, summary.index):
            for i, dct in enumerate(dicts):
                add_model_points(dct[cat], yi + offs[i], ax, m2c, m2m, rng)

    ax.set_yticks(y)
    ax.set_yticklabels([pretty_name(c) for c in summary.index], fontsize=9)
    ax.invert_yaxis()

    ax.set_xlim(0, xlim)
    ax.set_xlabel("Percentage of units", fontsize=9)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="x", linestyle=":", alpha=0.4, zorder=0)
    ax.set_title(title, fontsize=11)

    bar_handles = [
        Patch(facecolor=colors[i], edgecolor="black", label=labels[i])
        for i in range(len(show_cols))
    ]
    bar_leg = ax.legend(
        handles=bar_handles,
        loc="lower right",
        fontsize=8,
        frameon=True,
    )
    ax.add_artist(bar_leg)

    if add_model_markers and model_names:
        model_handles = [
            Line2D(
                [],
                [],
                marker=m2m[m],
                color=m2c[m],
                linestyle="None",
                markersize=5,
                markeredgecolor="black",
                markeredgewidth=0.4,
                label=m,
            )
            for m in model_names
        ]
        ax.legend(
            handles=model_handles,
            loc="upper right",
            title="Models",
            fontsize=7,
            title_fontsize=8,
            frameon=True,
        )

    fig.tight_layout()

    out_png = os.path.join(directory, "cross_overlap_agreement_blimp-nl.png")
    out_pdf = os.path.join(directory, "cross_overlap_agreement_blimp-nl.pdf")

    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)

    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


# ============================================================
# Main
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--directory", default=".")
    p.add_argument(
        "--title",
        default="Cross-phenomenon overlap in BLiMP-NL (agreement, gemma-3-4b-pt)",
    )
    p.add_argument("--add-model-markers", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--xlim", type=float, default=None)
    args = p.parse_args()

    directory = args.directory

    lang = "blimp-nl"
    dataset_label = "BLiMP-NL"

    # BLiMP-NL agreement categories are defined locally above.
    cat_src = BLIMPNL_AGREEMENT_CATEGORIES

    # Use existing BLiMP categories from the repo.
    cat_blimp = CATEGORIES["blimp"]

    # Get exact BLiMP-NL order from the within-language matrix.
    # This avoids having to hard-code all 22 BLiMP-NL suite names.
    within_paths = glob(os.path.join(directory, "cross-overlap_blimp-nl_blimp-nl_*.csv"))
    if not within_paths:
        raise FileNotFoundError(
            f"No BLiMP-NL within-overlap CSV found in {directory}.\n"
            f"Expected something like:\n"
            f"  cross-overlap_blimp-nl_blimp-nl_gemma-3-4b-pt_1.0%.csv"
        )

    tmp = pd.read_csv(within_paths[0], index_col=0)
    blimpnl_order = list(tmp.index.astype(str))

    # BLiMP order comes from existing category map.
    blimp_order = flatten(list(cat_blimp.values()))

    mats_lang = load_pair_matrices(
        directory=directory,
        dataset_a="blimp-nl",
        dataset_b="blimp-nl",
        desired_rows=blimpnl_order,
        desired_cols=blimpnl_order,
    )

    mats_cross = load_pair_matrices(
        directory=directory,
        dataset_a="blimp",
        dataset_b="blimp-nl",
        desired_rows=blimpnl_order,
        desired_cols=blimp_order,
    )

    summary, pw, poa, pna, pba, pbn = aggregate(
        mats_lang=mats_lang,
        mats_cross=mats_cross,
        cat_src=cat_src,
        cat_blimp=cat_blimp,
    )

    print("\nSummary:")
    print(summary.to_string())

    print("\nNote:")
    print(
        "BLiMP-NL currently has only one explicit agreement suite in this script: "
        "anaphor_agreement. Therefore within-category and other-agreement bars may "
        "be NaN/blank unless you add more BLiMP-NL agreement suites."
    )

    plot(
        summary=summary,
        pw=pw,
        poa=poa,
        pna=pna,
        pba=pba,
        pbn=pbn,
        dataset_label=dataset_label,
        directory=directory,
        title=args.title,
        add_model_markers=args.add_model_markers,
        seed=args.seed,
        xlim=args.xlim,
    )


if __name__ == "__main__":
    main()
