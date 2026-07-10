from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


INPUT_PATH = Path(
    "multilingual/turblimp/"
    "cross-overlap_turblimp_turblimp_gemma-3-4b-pt_1.0%.csv"
)

OUT_PNG = Path("multilingual/turblimp/cross_overlap_subject_agreement_turblimp.png")
OUT_PDF = Path("multilingual/turblimp/cross_overlap_subject_agreement_turblimp.pdf")

TARGET = "subject_agreement"


LABELS = {
    "anaphor_agreement": "Anaphor Agreement",
    "argument_structure_ditransitive": "Arg. Struct. Ditrans.",
    "argument_structure_transitive": "Arg. Struct. Trans.",
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
}


def pretty_name(name: str) -> str:
    if name in LABELS:
        return LABELS[name]

    return name.replace("_", " ").replace("-", " ").title()


def convert_to_percent_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    values = df.to_numpy(dtype=float)
    max_val = np.nanmax(values)
    diagonal = np.diag(values)
    normalizer = float(np.nanmedian(diagonal))

    if max_val > 100 or normalizer > 100:
        print(f"Matrix appears to be raw counts. Converting by {normalizer:.0f} units.")
        return df / normalizer * 100.0

    print("Matrix appears to already be percentages.")
    return df


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Could not find {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH, index_col=0)
    df = convert_to_percent_if_needed(df)

    if TARGET not in df.index:
        print("Available rows:")
        for row in df.index:
            print(" ", row)
        raise KeyError(f"{TARGET!r} not found in matrix rows.")

    if TARGET not in df.columns:
        print("Available columns:")
        for col in df.columns:
            print(" ", col)
        raise KeyError(f"{TARGET!r} not found in matrix columns.")

    rows = []

    for category in df.columns:
        if category == TARGET:
            continue

        # Within-dataset cross-overlap should be symmetric, but average row/column
        # values just in case the saved matrix has small orientation differences.
        vals = []

        if category in df.columns:
            vals.append(float(df.loc[TARGET, category]))

        if category in df.index:
            vals.append(float(df.loc[category, TARGET]))

        vals = [v for v in vals if np.isfinite(v)]

        if not vals:
            continue

        rows.append(
            {
                "category": category,
                "label": pretty_name(category),
                "overlap": float(np.mean(vals)),
            }
        )

    plot_df = pd.DataFrame(rows).sort_values("overlap", ascending=False)

    print("\nSubject Agreement overlap with other TurBLiMP phenomena:")
    print(plot_df[["category", "overlap"]].to_string(index=False))

    fig_height = max(5, 0.35 * len(plot_df) + 1.5)
    fig, ax = plt.subplots(figsize=(8.5, fig_height))

    y = np.arange(len(plot_df))

    ax.barh(
        y,
        plot_df["overlap"],
        height=0.75,
        edgecolor="black",
        linewidth=0.9,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["label"], fontsize=10)
    ax.invert_yaxis()

    max_val = float(plot_df["overlap"].max())
    x_max = max(25, np.ceil((max_val + 5) / 5) * 5)

    ax.set_xlim(0, x_max)
    ax.set_xlabel("Percentage of units", fontsize=12)
    ax.set_title(
        "Overlap of Subject Agreement with other TurBLiMP phenomena\n"
        "(gemma-3-4b-pt, top 1% units)",
        fontsize=13,
    )

    ax.grid(axis="x", linestyle=":", alpha=0.4, zorder=0)

    # Random top-1% overlap baseline.
    ax.axvline(1.0, linestyle="--", linewidth=1.0)
    ax.text(
        1.2,
        -0.65,
        "Random ≈ 1%",
        va="center",
        ha="left",
        fontsize=9,
    )

    # Percent labels outside bars.
    for i, val in enumerate(plot_df["overlap"]):
        ax.text(
            val + 0.4,
            i,
            f"{val:.2f}%",
            va="center",
            ha="left",
            fontsize=9,
        )

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("black")

    fig.tight_layout()

    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")

    print(f"\nSaved {OUT_PNG}")
    print(f"Saved {OUT_PDF}")


if __name__ == "__main__":
    main()
