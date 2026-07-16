from __future__ import annotations

import argparse
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
    pretty_category_name,
    get_canonical_order,
    load_model_matrices,
)


def agreement_cats(cat_map: Dict[str, List[str]]) -> List[str]:
    return [cat for cat in cat_map if "agreement" in cat]


def within_values(df: pd.DataFrame, suites: Sequence[str]) -> np.ndarray:
    sub = df.loc[suites, suites]

    if len(suites) <= 1:
        return np.array([np.nan])

    mask = ~np.eye(len(suites), dtype=bool)
    return sub.values[mask]


def cross_values(
    df: pd.DataFrame,
    suites_from: Sequence[str],
    suites_to: Sequence[str],
) -> np.ndarray:
    if len(suites_from) == 0 or len(suites_to) == 0:
        return np.array([np.nan])

    return df.loc[suites_from, suites_to].values.ravel()


def aggregate(
    mats_lang: Dict[str, pd.DataFrame],
    mats_cross: Dict[str, pd.DataFrame] | None,
    cat_src: Dict[str, List[str]],
    cat_blimp: Dict[str, List[str]] | None = None,
) -> Tuple[
    pd.DataFrame,
    Dict[str, Dict[str, float]],
    Dict[str, Dict[str, float]],
    Dict[str, Dict[str, float]],
    Dict[str, Dict[str, float]] | None,
    Dict[str, Dict[str, float]] | None,
]:
    def agree_cats(cat_map: Dict[str, List[str]]) -> List[str]:
        return [
            c
            for c in cat_map
            if "agreement" in c or c == "anaphor_number" or c == "anaphor_gender"
        ]

    agree_src = agree_cats(cat_src)
    all_agree_src = [s for c in agree_src for s in cat_src[c]]

    non_agree_src = [
        s
        for suites in cat_src.values()
        for s in suites
        if s not in all_agree_src
    ]

    if mats_cross is not None:
        if cat_blimp is None:
            raise ValueError("cat_blimp must be provided when mats_cross is not None.")

        agree_blimp = agree_cats(cat_blimp)
        all_agree_blimp = [s for c in agree_blimp for s in cat_blimp[c]]
        non_agree_blimp = [
            s
            for suites in cat_blimp.values()
            for s in suites
            if s not in all_agree_blimp
        ]
    else:
        all_agree_blimp = []
        non_agree_blimp = []

    per_w, per_oa, per_na = {}, {}, {}
    per_ba, per_bn = ({}, {}) if mats_cross is not None else (None, None)
    rows = []

    for cat in agree_src:
        suites = cat_src[cat]
        other_agree = [s for c in agree_src if c != cat for s in cat_src[c]]

        w, oa, na = {}, {}, {}
        ba, bn = ({}, {}) if mats_cross is not None else (None, None)

        for model, df in mats_lang.items():
            w[model] = float(np.nanmean(within_values(df, suites)))
            oa[model] = float(np.nanmean(cross_values(df, suites, other_agree)))
            na[model] = float(np.nanmean(cross_values(df, suites, non_agree_src)))

            if mats_cross is not None:
                df_cross = mats_cross[model]
                ba[model] = float(
                    np.nanmean(cross_values(df_cross, suites, all_agree_blimp))
                )
                bn[model] = float(
                    np.nanmean(cross_values(df_cross, suites, non_agree_blimp))
                )

        per_w[cat], per_oa[cat], per_na[cat] = w, oa, na

        if mats_cross is not None:
            per_ba[cat], per_bn[cat] = ba, bn

        row = {
            "category": cat,
            "within": float(np.nanmean(list(w.values()))),
            "other_agree": float(np.nanmean(list(oa.values()))),
            "non_agree": float(np.nanmean(list(na.values()))),
        }

        if mats_cross is not None:
            row["blimp_agree"] = float(np.nanmean(list(ba.values())))
            row["blimp_non_agree"] = float(np.nanmean(list(bn.values())))

        rows.append(row)

    summary_df = pd.DataFrame(rows).set_index("category")
    return summary_df, per_w, per_oa, per_na, per_ba, per_bn


def plot(
    summary: pd.DataFrame,
    pw: Dict[str, Dict[str, float]],
    poa: Dict[str, Dict[str, float]],
    pna: Dict[str, Dict[str, float]],
    pba: Dict[str, Dict[str, float]] | None,
    pbn: Dict[str, Dict[str, float]] | None,
    dataset: str,
    directory: str,
    title: str,
    add_model_markers: bool,
    seed: int = 42,
):
    cols_core = ["within", "other_agree", "non_agree"]
    extra_cols = []

    if pba is not None:
        extra_cols = ["blimp_agree", "blimp_non_agree"]

    show_cols = cols_core + extra_cols

    colors = ["#89CCF1", "#FFB668", "#C0C0C0", "#8ECA7A", "#BC9E92"]

    labels = [
        f"Within-category in {dataset}",
        f"With other agreement categories in {dataset}",
        f"With non-agreement categories in {dataset}",
        "With agreement categories in BLiMP",
        "With non-agreement categories in BLiMP",
    ][: len(show_cols)]

    summary = summary.copy()
    summary["diff"] = summary["within"] - summary["other_agree"]
    summary = summary.sort_values("diff", ascending=False)

    y = np.arange(len(summary))

    # More space because there are up to five bars per category.
    fig_h = max(5.8, 1.20 * len(summary) + 1.8)
    fig_w = 12.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    bar_h = 0.14
    offsets = np.linspace(-0.32, 0.32, len(show_cols))

    for i, col in enumerate(show_cols):
        ax.barh(
            y + offsets[i],
            summary[col],
            height=bar_h,
            color=colors[i],
            edgecolor="black",
            label=labels[i],
            zorder=2,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(
        [pretty_category_name(c) for c in summary.index],
        fontsize=11,
    )
    ax.invert_yaxis()

    max_val = float(np.nanmax(summary[show_cols].values))
    x_lim = max(60, int(np.ceil((max_val + 12) / 10) * 10))
    ax.set_xlim(0, x_lim)

    ax.set_xlabel("Percentage of units", fontsize=12)
    ax.tick_params(axis="x", labelsize=10)
    ax.grid(axis="x", linestyle=":", alpha=0.4, zorder=0)
    ax.set_title(title, fontsize=13)

    rng = np.random.default_rng(seed)
    model_names = sorted({m for d in pw.values() for m in d})
    model_list, model_to_color, model_to_marker = build_model_style_maps(model_names)

    for yi, cat in zip(y, summary.index):
        per_col_model_dicts = [pw[cat], poa[cat], pna[cat]]

        if pba is not None:
            per_col_model_dicts += [pba[cat], pbn[cat]]

        for j, col in enumerate(show_cols):
            val = summary.loc[cat, col]

            ax.text(
                val + 0.8,
                yi + offsets[j],
                f"{val:.2f}%",
                va="center",
                ha="left",
                fontsize=8,
                zorder=9,
            )

            if add_model_markers:
                add_model_scatter(
                    per_col_model_dicts[j],
                    center=yi + offsets[j],
                    ax=ax,
                    model_to_color=model_to_color,
                    model_to_marker=model_to_marker,
                    rng=rng,
                    s=22,
                    jitter=0.025,
                )

    bar_handles = [
        Patch(facecolor=colors[i], edgecolor="black", label=labels[i])
        for i in range(len(show_cols))
    ]

    # Legend outside the plot so it does not compress the bars.
    bar_leg = ax.legend(
        handles=bar_handles,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=9,
        frameon=True,
        title="Overlap type",
        title_fontsize=10,
    )
    ax.add_artist(bar_leg)

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
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            title="Models",
            fontsize=8,
            title_fontsize=9,
            frameon=True,
        )

    fig.tight_layout(rect=[0, 0, 0.78, 1])

    out_png = f"{directory}/cross_overlap_agreement_{dataset.lower()}_clean.png"
    out_pdf = f"{directory}/cross_overlap_agreement_{dataset.lower()}_clean.pdf"

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")

    print(f"Saved {out_png}")
    print(f"Saved {out_pdf}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="BLiMP")
    p.add_argument("--directory", default="english/cross-overlap")
    p.add_argument("--title", default="Cross-phenomenon overlap in BLiMP (agreement)")
    p.add_argument("--add-model-markers", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    lang = args.dataset.lower()

    cat_src = CATEGORIES[lang]
    cat_blimp = CATEGORIES["blimp"]

    row_order = get_canonical_order(cat_src)
    col_order = row_order
    col_blimp = get_canonical_order(cat_blimp)

    mats_lang = load_model_matrices(
        args.directory,
        lang,
        lang,
        row_order,
        col_order,
    )

    mats_cross = (
        load_model_matrices(
            args.directory,
            "blimp",
            lang,
            row_order,
            col_blimp,
        )
        if lang != "blimp"
        else None
    )

    summary, pw, poa, pna, pba, pbn = aggregate(
        mats_lang,
        mats_cross,
        cat_src,
        cat_blimp,
    )

    if mats_cross is not None:

        def agree_suites(cat_map: Dict[str, List[str]]) -> List[str]:
            return [
                s
                for c, suites in cat_map.items()
                if ("agreement" in c) or (c in {"anaphor_number", "anaphor_gender"})
                for s in suites
            ]

        src_agree_suites = agree_suites(cat_src)
        blimp_agree_suites = agree_suites(cat_blimp)

        per_model_means = {}

        for model, df in mats_cross.items():
            sub = df.loc[src_agree_suites, blimp_agree_suites].to_numpy().ravel()
            sub = sub[~np.isnan(sub)]

            if sub.size:
                per_model_means[model] = float(sub.mean())
            else:
                per_model_means[model] = np.nan

        overall_avg = float(np.nanmean(list(per_model_means.values())))

        print(f"Per-model mean overlap ({args.dataset} agreement / BLiMP agreement):")

        for m, v in sorted(per_model_means.items()):
            print(f"  {m}: {v:.2f}%")

        print(f"Overall average: {overall_avg:.2f}%")

    plot(
        summary,
        pw,
        poa,
        pna,
        pba,
        pbn,
        dataset=args.dataset,
        directory=args.directory,
        title=args.title,
        add_model_markers=args.add_model_markers,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
