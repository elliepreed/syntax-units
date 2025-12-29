import os
from typing import List, Dict

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from model_utils import get_num_blocks, get_hidden_dim


MARKER_STYLES = ["o", "s", "^", "D", "v", ">", "<"]


def build_model_style_maps(model_names: List[str]):
    model_list = sorted(model_names)
    cmap = plt.cm.get_cmap("Set1")
    base_colors = np.array(cmap.colors)
    subset_idx = [0, 2, 3, 7, 5, 6, 8]
    palette = base_colors[subset_idx]
    color_cycle = palette[np.arange(len(model_list)) % len(palette)]
    model_to_color = {m: color_cycle[i] for i, m in enumerate(model_list)}
    model_to_marker = {
        m: MARKER_STYLES[i % len(MARKER_STYLES)] for i, m in enumerate(model_list)
    }
    return model_list, model_to_color, model_to_marker


def add_model_scatter(
    model_to_value: Dict[str, float],
    center: float,
    ax: plt.Axes,
    model_to_color: Dict[str, tuple],
    model_to_marker: Dict[str, str],
    rng: np.random.Generator,
    s: int = 40,
    jitter: float = 0.1,
    vertical: bool = False,
):
    for model, val in model_to_value.items():
        jit = rng.normal(center, jitter)
        (arg1, arg2) = (jit, val) if vertical else (val, jit)
        ax.scatter(
            arg1,
            arg2,
            s=s,
            color=model_to_color[model],
            marker=model_to_marker[model],
            edgecolor="black",
            linewidth=0.4,
            alpha=0.9,
            zorder=8,
        )


def pretty_category_name(cat: str) -> str:
    mapping = {
        # BLiMP
        "anaphor_agreement": "Anaphor Agreement",
        "argument_structure": "Argument Structure",
        "binding": "Binding",
        "control_raising": "Control/Raising",
        "determiner_noun_agreement": "DET-N Agreement",
        "ellipsis": "Ellipsis",
        "irregular_forms": "Irregular Forms",
        "island_effects": "Island Effects",
        "npi_licensing": "NPI Licensing",
        "filler_gap": "Filler–Gap",
        "subject_verb_agreement": "S-V Agreement",
        "quantifiers": "Quantifiers",
        # RuBLiMP
        "subject_predicate_agreement": "Subject-Predicate Agreement",
        "floating_quantifier_agreement": "Floating Quantifier Agreement",
        "np_agreement": "NP Agreement",
        "reflexives": "Reflexives",
        "negation": "Negation",
        "government": "Government",
        "aspect": "Aspect",
        "tense": "Tense",
        "word_formation": "Word Formation",
        "word_inflection": "Word Inflection",
        # SLING
        "alternative_question": "Alternative Question",
        "anaphor_gender": "Anaphor Gender Agreement",
        "anaphor_number": "Anaphor Number Agreement",
        "classifier_noun_agreement": "CLS-N Agreement",
        "definiteness": "Definiteness",
        "polarity_item": "Polarity Item",
        "relative_clause": "Relative Clause",
        "wh_fronting": "Wh-Fronting",
    }
    return mapping.get(cat, cat)


def get_canonical_order(cat_map: Dict[str, List[str]]) -> List[str]:
    order: List[str] = []
    for suites in cat_map.values():
        order.extend(suites)
    return order


def load_model_matrices(
    directory: str,
    lang_src: str,
    lang_tgt: str,
    row_order: List[str],
    col_order: List[str],
    percentage: float = 1.0,
) -> Dict[str, pd.DataFrame]:
    prefix = f"cross-overlap_{lang_src}_{lang_tgt}_"
    suffix = "%.csv"

    matrices = {}

    for fname in sorted(os.listdir(directory)):
        if not (fname.startswith(prefix) and fname.endswith(suffix)):
            continue
        print(fname)

        parts = fname.split("_")
        model_name = "_".join(parts[3:-1])
        path = os.path.join(directory, fname)

        df = pd.read_csv(path, index_col=0)
        df.index = df.index.astype(str)
        df.columns = df.columns.astype(str)

        if set(row_order).issubset(df.columns) and set(col_order).issubset(df.index):
            df = df.T

        df = df.reindex(index=row_order, columns=col_order)

        total_units = get_num_blocks(model_name) * get_hidden_dim(model_name)
        percentage_units = int(total_units * percentage / 100)

        df = df / percentage_units * 100

        matrices[model_name] = df

    return matrices


CATEGORY_COLORS = {
    "blimp": [
        "#F27979",  # light-red     anaphor agreement
        "#18b826",  # dark-green    argument structure
        "#CC3D3D",  # dark-red      binding
        "#91D2F2",  # light-blue    control/raising
        "#FFB668",  # orange        det-n agreement
        "#BF9D8F",  # brown         ellipsis
        "#FFCCD5",  # pink          filler-gap
        "#A8D998",  # lght-green    irregular forms
        "#AB52CC",  # purple        island effects
        "#FFFFFF",  # white         npi licensing
        "#296DCC",  # dark-blue     quantifiers
        "#FFFFB3",  # yellow        s-v agreement
    ],
    "rublimp": [
        "#CC3D3D",  # dark-red      anaphor agreement
        "#AB52CC",  # purple        argument structure
        "#A8D998",  # light-green   aspect
        "#F27979",  # light-red     floating quantifier agreement
        "#FFCCD5",  # pink          government
        "#BF9D8F",  # brown         negation
        "#FFB668",  # orange        np agreement
        "#FFFFFF",  # white         reflexives
        "#FFFFB3",  # yellow        s-p agreement
        "#18b826",  # dark-green    tense
        "#91D2F2",  # light-blue    word formation
        "#296DCC",  # dark-blue     word inflection
    ],
    "sling": [
        "#FFFFFF",  # white         alternative question
        "#CC3D3D",  # dark-red      anaphor gender agreement
        "#F27979",  # light-red     anaphor number agreement
        "#A8D998",  # light-green   aspect
        "#FFB668",  # orange        cls-n agreement
        "#FFFFB3",  # yellow        definiteness
        "#91D2F2",  # light-blue    polarity item
        "#AB52CC",  # purple        relative clause
        "#BF9D8F",  # brown         wh-fronting
    ],
}

