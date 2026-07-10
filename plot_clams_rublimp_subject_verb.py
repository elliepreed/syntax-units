import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from categories import CATEGORIES

OUTDIR = "multilingual/clams/ru"
MODEL = "gemma-3-4b-pt"
PERCENTAGE = 1.0

# Existing files
RUBLIMP_SELF = "multilingual/rublimp/cross-overlap_rublimp_rublimp_gemma-3-4b-pt_1.0%.csv"

RUBLIMP_CLAMS = (
    "multilingual/clams/ru/"
    "cross-overlap_rublimp_subject_predicate_agreement_clams_ru_shuffled_gemma-3-4b-pt_1.0%.csv"
)

CLAMS_SELF = (
    "multilingual/clams/ru/"
    "cross-overlap_clams_ru_shuffled_clams_ru_shuffled_gemma-3-4b-pt_1.0%.csv"
)

# Only this RuBLiMP phenomenon
RUBLIMP_SUBJECT_PREDICATE = CATEGORIES["rublimp"]["subject_predicate_agreement"]

# All CLAMS Russian suites are subject-verb agreement
CLAMS_SUITES = [
    "long_vp_coord",
    "obj_rel_across_anim",
    "obj_rel_within_anim",
    "prep_anim",
    "simple_agrmt",
    "subj_rel",
    "vp_coord",
]


def read_matrix(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    print(f"Reading {path}")
    return pd.read_csv(path, index_col=0)


def check_suites_exist(df, suites, name):
    rows = set(df.index)
    cols = set(df.columns)

    missing_rows = [s for s in suites if s not in rows]
    missing_cols = [s for s in suites if s not in cols]

    if missing_rows or missing_cols:
        print(f"\nAvailable rows in {name}:")
        print(list(df.index))
        print(f"\nAvailable columns in {name}:")
        print(list(df.columns))
        raise ValueError(
            f"Missing suites in {name}.\n"
            f"Missing rows: {missing_rows}\n"
            f"Missing cols: {missing_cols}"
        )


def mean_offdiag_submatrix(df, suites, name):
    check_suites_exist(df, suites, name)
    sub = df.loc[suites, suites].to_numpy(dtype=float)

    if sub.shape[0] != sub.shape[1]:
        raise ValueError(f"{name} submatrix is not square: {sub.shape}")

    mask = ~np.eye(sub.shape[0], dtype=bool)
    return float(sub[mask].mean())


def mean_cross_submatrix(df, row_suites=None, col_suites=None):
    if row_suites is not None:
        missing = [s for s in row_suites if s not in df.index]
        if missing:
            print("Available rows:", list(df.index))
            raise ValueError(f"Missing cross rows: {missing}")
        df = df.loc[row_suites]

    if col_suites is not None:
        missing = [s for s in col_suites if s not in df.columns]
        if missing:
            print("Available columns:", list(df.columns))
            raise ValueError(f"Missing cross cols: {missing}")
        df = df[col_suites]

    return float(df.to_numpy(dtype=float).mean())


rublimp_self = read_matrix(RUBLIMP_SELF)
rublimp_clams = read_matrix(RUBLIMP_CLAMS)
clams_self = read_matrix(CLAMS_SELF)

# Target 1% unit set size. Diagonal self-overlap is the selected-unit count.
selected_units = float(np.median(np.diag(rublimp_self.to_numpy(dtype=float))))

# Phenomenon-level values
rublimp_value = mean_offdiag_submatrix(
    rublimp_self,
    RUBLIMP_SUBJECT_PREDICATE,
    "RuBLiMP subject_predicate_agreement",
)

# If your RuBLiMP × CLAMS file already has only subject-predicate rows,
# this mean over the whole matrix is correct.
cross_value = mean_cross_submatrix(rublimp_clams)

# CLAMS has only the agreement suites, so off-diagonal mean over all CLAMS suites.
clams_value = mean_offdiag_submatrix(
    clams_self,
    CLAMS_SUITES,
    "CLAMS Russian subject-verb agreement",
)

random_value = selected_units * (PERCENTAGE / 100)

labels = [
    "RuBLiMP ×\nCLAMS",
    "RuBLiMP",
    "CLAMS",
    "Random",
]

values = [
    cross_value,
    rublimp_value,
    clams_value,
    random_value,
]

percentages = [100 * v / selected_units for v in values]

print("\nPhenomenon-level overlap values")
print(f"Selected units: {selected_units:.0f}")
for label, value, pct in zip(labels, values, percentages):
    print(f"{label.replace(chr(10), ' '):20s} {value:8.2f} units ({pct:5.2f}%)")

fig, ax = plt.subplots(figsize=(8.5, 5.8))

x = np.arange(len(labels))

colors = [
    "#f4a261",  # cross-benchmark
    "#8ecae6",  # within-benchmark
    "#8ecae6",  # within-benchmark
    "#bdbdbd",  # random
]

ax.bar(
    x,
    values,
    color=colors,
    edgecolor="black",
    linewidth=1.0,
    zorder=2,
)

# One marker because only Gemma has been run.
ax.scatter(
    x,
    values,
    marker=">",
    s=95,
    color="#8c2d04",
    edgecolor="black",
    linewidth=0.7,
    label="Gemma-3-4B",
    zorder=3,
)

for xi, value, pct in zip(x, values, percentages):
    ax.text(
        xi,
        value + max(values) * 0.035,
        f"{pct:.1f}%",
        ha="center",
        va="bottom",
        fontsize=13,
    )

ax.set_title("Subject-verb agreement", fontsize=18)
ax.set_ylabel("Number of Units", fontsize=15)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=12)
ax.tick_params(axis="y", labelsize=12)
ax.legend(title="Model", fontsize=11, title_fontsize=12, loc="upper center")
ax.set_ylim(0, max(values) * 1.35)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()

os.makedirs(OUTDIR, exist_ok=True)

out_png = f"{OUTDIR}/rublimp_clams_subject_verb_phenomenon_overlap_{MODEL}_{PERCENTAGE}%.png"
out_pdf = f"{OUTDIR}/rublimp_clams_subject_verb_phenomenon_overlap_{MODEL}_{PERCENTAGE}%.pdf"

fig.savefig(out_png, dpi=300)
fig.savefig(out_pdf)

print("\nSaved:")
print(out_png)
print(out_pdf)
