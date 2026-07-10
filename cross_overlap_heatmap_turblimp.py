from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


INPUT_PATH = Path(
    "multilingual/turblimp/"
    "cross-overlap_turblimp_turblimp_gemma-3-4b-pt_1.0%.csv"
)

OUT_PNG = Path("multilingual/turblimp/cross_overlap_heatmap_turblimp.png")
OUT_PDF = Path("multilingual/turblimp/cross_overlap_heatmap_turblimp.pdf")

MASK_DIAGONAL = True

LABELS = {
    "anaphor_agreement": "Anaphor\nAgreement",
    "argument_structure_ditransitive": "Arg. Struct.\nDitrans.",
    "argument_structure_transitive": "Arg. Struct.\nTrans.",
    "binding": "Binding",
    "determiners": "Determiners",
    "ellipsis": "Ellipsis",
    "irregular_forms": "Irregular\nForms",
    "island_effects": "Island\nEffects",
    "nominalization": "Nominalization",
    "npi_licensing": "NPI\nLicensing",
    "passives": "Passives",
    "quantifiers": "Quantifiers",
    "relative_clauses": "Relative\nClauses",
    "scrambling": "Scrambling",
    "subject_agreement": "Subject\nAgreement",
    "suspended_affixation": "Suspended\nAffixation",
}


ORDER = [
    "anaphor_agreement",
    "subject_agreement",
    "argument_structure_transitive",
    "argument_structure_ditransitive",
    "suspended_affixation",
    "scrambling",
    "relative_clauses",
    "island_effects",
    "binding",
    "determiners",
    "quantifiers",
    "npi_licensing",
    "ellipsis",
    "nominalization",
    "passives",
    "irregular_forms",
]


def pretty_name(name: str) -> str:
    return LABELS.get(name, name.replace("_", " ").title())


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


def text_colour_for_background(rgba):
    """
    Choose white text on dark cells and black text on light cells.
    Uses perceived luminance from the cell background colour.
    """
    r, g, b, _ = rgba
    luminance = 0.299 * r + 0.587 * g + 0.114 * b

    if luminance < 0.50:
        return "white"

    return "black"


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Could not find {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH, index_col=0)
    df = convert_to_percent_if_needed(df)

    # Keep only categories present in the matrix, in a stable order.
    ordered = [x for x in ORDER if x in df.index and x in df.columns]
    ordered += [x for x in df.index if x not in ordered and x in df.columns]

    df = df.loc[ordered, ordered]

    data = df.to_numpy(dtype=float)

    if MASK_DIAGONAL:
        plot_data = data.copy()
        np.fill_diagonal(plot_data, np.nan)
        off_diag_vals = plot_data[np.isfinite(plot_data)]
        vmax = max(5, float(np.nanmax(off_diag_vals)))
    else:
        plot_data = data
        vmax = float(np.nanmax(plot_data))

    vmin = 0.0

    # Use a dark-to-light map. Text colour is chosen automatically per cell.
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad(color="#E6E6E6")

    norm = Normalize(vmin=vmin, vmax=vmax)

    fig_size = max(9, 0.62 * len(df))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    im = ax.imshow(
        plot_data,
        cmap=cmap,
        norm=norm,
        aspect="equal",
    )

    labels = [pretty_name(x) for x in df.index]

    ax.set_xticks(np.arange(len(df)))
    ax.set_yticks(np.arange(len(df)))

    ax.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
    ax.set_yticklabels(labels, fontsize=8)

    ax.set_title(
        "Pairwise unit overlap between TurBLiMP phenomena\n"
        "(gemma-3-4b-pt, top 1% units)",
        fontsize=13,
        pad=14,
    )

    # Cell annotations.
    for i in range(len(df)):
        for j in range(len(df)):
            val = data[i, j]

            if MASK_DIAGONAL and i == j:
                ax.text(
                    j,
                    i,
                    "—",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                )
                continue

            rgba = cmap(norm(plot_data[i, j]))
            text_color = text_colour_for_background(rgba)

            ax.text(
                j,
                i,
                f"{val:.1f}",
                ha="center",
                va="center",
                fontsize=7,
                color=text_color,
            )

    # Grid lines between cells.
    ax.set_xticks(np.arange(-0.5, len(df), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(df), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Percentage of units", fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    if MASK_DIAGONAL:
        ax.text(
            0.0,
            -0.12,
            "Diagonal masked because it is self-overlap.",
            transform=ax.transAxes,
            fontsize=8,
            ha="left",
            va="top",
        )

    fig.tight_layout()

    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")

    print(f"Saved {OUT_PNG}")
    print(f"Saved {OUT_PDF}")


if __name__ == "__main__":
    main()
