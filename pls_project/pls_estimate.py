# pls_project/pls_estimate.py
from __future__ import annotations

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
    apply_sign_to_scores,          # ✅ use shared helper
    quality_paper_table,
)
from pls_project.sign_fix import choose_anchors_by_max_abs_loading


def estimate_pls_basic_paper(
    cog,
    *,
    Xpls: pd.DataFrame,
    item_cols: list[str],
    path_df: pd.DataFrame,
    lv_blocks: dict,
    lv_modes: dict,
    order: list[str],
    anchor_overrides: Optional[Dict[str, str]] = None,
):
    """
    One-shot clean output:
      - plspm model API provides estimation (outer/path) [strict=True]
      - cross-loadings by correlation (inspection only)
      - sign orientation uses ONE sign_map for scores/outer/paths (no double-corr)
      - optional anchor_overrides, e.g. {"Commitment": "CCO_score"}
    """
    cfg = cog.cfg.pls

    for k in ["PAPER_DECIMALS", "PLS_CROSS_CORR_METHOD", "PLS_STANDARDIZED", "SIGN_FIX"]:
        if not hasattr(cfg, k):
            raise AttributeError(f"Config.pls must define {k}.")

    dec = int(cfg.PAPER_DECIMALS)
    cross_method = str(cfg.PLS_CROSS_CORR_METHOD)

    model, scores = run_plspm_python(
        cog,
        Xpls,
        path_df,
        lv_blocks,
        lv_modes,
        scaled=bool(cfg.PLS_STANDARDIZED),
    )
    if model is None:
        raise ValueError("plspm model is None (clean mode should not allow scores-only schemes).")

    missing = [lv for lv in order if lv not in scores.columns]
    if missing:
        raise KeyError(
            f"LV scores missing columns from plspm output: {missing}. "
            f"Available={list(scores.columns)}"
        )
    scores = scores[order].copy()

    sign_fix_on = bool(cfg.SIGN_FIX)
    if sign_fix_on:
        anchors = choose_anchors_by_max_abs_loading(Xpls, scores, lv_blocks)
        if anchor_overrides:
            anchors.update(anchor_overrides)

        sign_map = get_sign_map_by_anchors(Xpls, scores, anchors)
        scores = apply_sign_to_scores(scores, sign_map)  # ✅ shared
    else:
        anchors = {}
        sign_map = {}

    PLS_cross = corr_items_vs_scores(Xpls[item_cols], scores[order], method=cross_method).round(dec)
    PLS_cross.index.name = "Item"

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

    PLS_quality = quality_paper_table(outer, lv_modes, order=order).round(dec)

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
