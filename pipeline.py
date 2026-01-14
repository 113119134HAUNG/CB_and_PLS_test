import numpy as np
import pandas as pd

from pls_project.pls_core import (
    run_plspm_python,
    corr_items_vs_scores,
    get_outer_results,
    get_path_results,
    apply_sign_to_outer,
    apply_sign_to_paths,
    get_sign_map_by_anchors,
    quality_paper_table,   # 你若還沒有，等一下我也給你版本
)

from pls_project.sign_fix import (
    choose_anchors_by_max_abs_loading,
    sign_fix_scores_by_anchors,
)

def estimate_pls_basic_paper(
    cog,
    *,
    Xpls: pd.DataFrame,
    item_cols: list[str],
    path_df: pd.DataFrame,
    lv_blocks: dict,
    lv_modes: dict,
    order: list[str],
):
    """
    Clean + SmartPLS4-aligned:
      - estimation from plspm model API only (outer/path)
      - no OLS(paths) fallback, no corr(outer) fallback
      - sign-fix optional (cfg.pls.SIGN_FIX)
      - paper tables returned
    """
    cfg = cog.cfg.pls
    dec = int(getattr(cfg, "PAPER_DECIMALS", 3))

    # 1) run model
    model, scores = run_plspm_python(
        cog,
        Xpls,
        path_df,
        lv_blocks,
        lv_modes,
        scaled=bool(getattr(cfg, "PLS_STANDARDIZED", True)),
        scheme=str(getattr(cfg, "PLS_SCHEME", "PATH")),
    )

    # 2) strict LV alignment (乾淨：不允許 fallback 用前幾欄)
    missing = [lv for lv in order if lv not in scores.columns]
    if missing:
        raise KeyError(f"LV scores missing columns from plspm output: {missing}")
    scores = scores[order].copy()

    # 3) sign-fix (optional)
    sign_fix_on = bool(getattr(cfg, "SIGN_FIX", True))
    if sign_fix_on:
        anchors = choose_anchors_by_max_abs_loading(Xpls, scores, lv_blocks)
        sign_map = get_sign_map_by_anchors(Xpls, scores, anchors)
        scores = sign_fix_scores_by_anchors(scores, Xpls, anchors)
    else:
        anchors = {}
        sign_map = {}

    # 4) cross-loadings (corr; inspection only)
    PLS_cross = corr_items_vs_scores(Xpls[item_cols], scores[order], method=str(getattr(cfg, "HTMT_CORR_METHOD", "pearson"))).round(dec)

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

    out = {
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
    return out
