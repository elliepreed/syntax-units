import os
import numpy as np
import pandas as pd
from scipy import stats

from categories import CATEGORIES

PERCENTAGE = 1.0
MODEL = "gemma-3-4b-pt"

RUBLIMP_SELF = "multilingual/rublimp/cross-overlap_rublimp_rublimp_gemma-3-4b-pt_1.0%.csv"

RUBLIMP_CLAMS = (
    "multilingual/clams/ru/"
    "cross-overlap_rublimp_subject_predicate_agreement_clams_ru_shuffled_gemma-3-4b-pt_1.0%.csv"
)

CLAMS_SELF = (
    "multilingual/clams/ru/"
    "cross-overlap_clams_ru_shuffled_clams_ru_shuffled_gemma-3-4b-pt_1.0%.csv"
)

RUBLIMP_SUITES = CATEGORIES["rublimp"]["subject_predicate_agreement"]

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
        raise FileNotFoundError(path)
    return pd.read_csv(path, index_col=0)


def upper_triangle_values(df, suites):
    sub = df.loc[suites, suites].to_numpy(dtype=float)
    idx = np.triu_indices_from(sub, k=1)
    return sub[idx]


def cross_values(df, row_suites=None, col_suites=None):
    if row_suites is not None:
        row_suites = [s for s in row_suites if s in df.index]
        df = df.loc[row_suites]

    if col_suites is not None:
        col_suites = [s for s in col_suites if s in df.columns]
        df = df[col_suites]

    return df.to_numpy(dtype=float).flatten()


def one_sample_greater(x, mu):
    t, p_two = stats.ttest_1samp(x, popmean=mu)
    # one-sided p-value for mean(x) > mu
    if t >= 0:
        p_one = p_two / 2
    else:
        p_one = 1 - p_two / 2
    return t, p_one


rublimp_self = read_matrix(RUBLIMP_SELF)
rublimp_clams = read_matrix(RUBLIMP_CLAMS)
clams_self = read_matrix(CLAMS_SELF)

# Size of target top-1% unit set.
# Diagonal self-overlap = selected unit count.
selected_units = float(np.median(np.diag(rublimp_self.to_numpy(dtype=float))))
random_baseline = selected_units * (PERCENTAGE / 100)

# Cross-benchmark: all RuBLiMP SVA × CLAMS SVA pairs.
cross = cross_values(
    rublimp_clams,
    row_suites=RUBLIMP_SUITES,
    col_suites=CLAMS_SUITES,
)

# Within-benchmark: unordered pairs of distinct suites.
rublimp_within = upper_triangle_values(rublimp_self, RUBLIMP_SUITES)
clams_within = upper_triangle_values(clams_self, CLAMS_SUITES)

# Test cross-benchmark overlap against random.
t_cross, p_cross = one_sample_greater(cross, random_baseline)

# Optional descriptive comparisons.
t_cross_vs_rublimp, p_cross_vs_rublimp = stats.ttest_ind(
    cross, rublimp_within, equal_var=False
)
t_cross_vs_clams, p_cross_vs_clams = stats.ttest_ind(
    cross, clams_within, equal_var=False
)

print("\n=== Russian subject-verb / subject-predicate agreement overlap ===")
print(f"Model: {MODEL}")
print(f"Target selected units: {selected_units:.0f}")
print(f"Random baseline: {random_baseline:.2f} units")
print()

print("Counts of pairwise overlaps:")
print(f"RuBLiMP × CLAMS pairs: {len(cross)}")
print(f"RuBLiMP within pairs:  {len(rublimp_within)}")
print(f"CLAMS within pairs:    {len(clams_within)}")
print()

def summary(name, x):
    pct = 100 * np.mean(x) / selected_units
    print(
        f"{name:18s} "
        f"mean={np.mean(x):8.2f} units "
        f"({pct:5.2f}%), "
        f"sd={np.std(x, ddof=1):8.2f}, "
        f"min={np.min(x):.0f}, max={np.max(x):.0f}"
    )

summary("RuBLiMP × CLAMS", cross)
summary("RuBLiMP within", rublimp_within)
summary("CLAMS within", clams_within)
print(f"{'Random':18s} mean={random_baseline:8.2f} units ({100 * random_baseline / selected_units:5.2f}%)")
print()

print("One-sample t-test: RuBLiMP × CLAMS > random")
print(f"t = {t_cross:.3f}, p = {p_cross:.5f}")
print()

print("Descriptive Welch tests, interpret cautiously because pair overlaps are not fully independent:")
print(f"Cross vs RuBLiMP within: t = {t_cross_vs_rublimp:.3f}, p = {p_cross_vs_rublimp:.5f}")
print(f"Cross vs CLAMS within:   t = {t_cross_vs_clams:.3f}, p = {p_cross_vs_clams:.5f}")
