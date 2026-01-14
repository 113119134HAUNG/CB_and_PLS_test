# pls_estimate.py
import numpy as np
import pandas as pd
from typing import Optional, Dict
from pls_project.pls_core import (
    run_plspm_python,
    corr_items_vs_scores,
    get_outer_results,
    get_path_results,
    apply_sign_to_outer,
    apply_sign_to_paths,
    get_sign_map_by_anchors,
    quality_paper_table,
)

from pls_project.sign_fix import (
    choose_anchors_by_max_abs_loading,
)

def _apply_sign_to_scores(scores_df: pd.DataFrame, sign_map: dict) -> pd.DataFrame:
    """Flip LV scores using sign_map (+1/-1)."""
    out = scores_df.copy()
    for lv, s in sign_map.items():
        if lv in out.columns and int(s) == -1:
            out[lv] = -out[lv]
    return out

def estimate_pls_basic_paper(
    cog,
    *,
    Xpls: pd.DataFrame,
    item_cols: list[str],
    path_df: pd.DataFrame,
    lv_blocks: dict,
    lv_modes: dict,
    order: list[str],
    anchor_overrides: Optional[Dict[str, str]] = None,   # ✅ NEW
):
    ...
    sign_fix_on = bool(getattr(cfg, "SIGN_FIX", True))
    if sign_fix_on:
        anchors = choose_anchors_by_max_abs_loading(Xpls, scores, lv_blocks)

        # ✅ NEW: allow override (e.g., Commitment -> "CCO_score")
        if anchor_overrides:
            anchors.update(anchor_overrides)

        sign_map = get_sign_map_by_anchors(Xpls, scores, anchors)
        scores = _apply_sign_to_scores(scores, sign_map)
    else:
        anchors = {}
        sign_map = {}

    # 4) cross-loadings (inspection only)
    PLS_cross = corr_items_vs_scores(
        Xpls[item_cols],
        scores[order],
        method=cross_method
    ).round(dec)
    PLS_cross.index.name = "Item"

    # 5) outer results: STRICT from model API
    outer = get_outer_results(
        model,
        Xpls[item_cols],
        scores[order],
        lv_blocks,
        lv_modes,
        strict=True,
    )
    if sign_fix_on:
        outer = apply_sign_to_outer(outer, sign_map)
    PLS_outer = outer.round(dec)

    # 6) CR/AVE from model-based outer loadings
    PLS_quality = quality_paper_table(outer, lv_modes, order=order).round(dec)

    # 7) paths: STRICT from model API
    pe = get_path_results(model, path_df, strict=True)
    if sign_fix_on:
        pe = apply_sign_to_paths(pe, sign_map)

    key = pe[["from", "to"]].copy()
    est = pe["estimate"].astype(float).values

    return {
        "model": model,
        "scores": scores.round(dec),
        "anchors": anchors,
        "sign_map": sign_map,
        "PLS_cross": PLS_cross,
        "PLS_outer": PLS_outer,
        "PLS_quality": PLS_quality,
        "paths_long": pe.round(dec),
        "key": key,
        "est": est,
    }
