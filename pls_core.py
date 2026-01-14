# -*- pls_core.py -*-
from __future__ import annotations

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
    compute_cr_ave,
)
from pls_project.sign_fix import (
    choose_anchors_by_max_abs_loading,
    sign_fix_scores_by_anchors,
)
from pls_project.io_utils import safe_t, ci_sig


# ---------------------------------------------------------
# Paper helper: CR/AVE table (如果你已經有就刪掉這段)
# ---------------------------------------------------------
def quality_paper_table(outer_tbl: pd.DataFrame, lv_modes: dict, order: list[str]) -> pd.DataFrame:
    """
    Paper-style quality table (reflective only):
      Construct, Mode, CR, AVE
    Formative: CR/AVE = NaN
    """
    rows = []
    for lv in order:
        mode = str(lv_modes.get(lv, "A")).upper()
        if mode != "A":
            rows.append({"Construct": lv, "Mode": "B(formative)", "CR": np.nan, "AVE": np.nan})
            continue
        lam = outer_tbl.loc[outer_tbl["Construct"] == lv, "OuterLoading"].dropna().values.astype(float)
        cr, ave = compute_cr_ave(lam)
        rows.append({"Construct": lv, "Mode": "A(reflective)", "CR": cr, "AVE": ave})
    return pd.DataFrame(rows)


# ---------------------------------------------------------
# (A) 你那段 1~7：乾淨版（Config 驅動、strict、paper style）
# ---------------------------------------------------------
def estimate_pls_basic_stage(
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
    Clean + SmartPLS4-aligned estimation:
      - run plspm -> model + scores
      - strict LV alignment (no silent fallback)
      - optional sign-fix (cfg.pls.SIGN_FIX)
      - cross-loadings via corr (inspection only)
      - outer + paths strictly from model API (strict=True)
      - paper tables: Cross, Outer, Quality, Paths
    """
    cfg = cog.cfg.pls
    dec = int(getattr(cfg, "PAPER_DECIMALS", 3))
    corr_method = str(getattr(cfg, "HTMT_CORR_METHOD", "pearson"))
    sign_fix_on = bool(getattr(cfg, "SIGN_FIX", True))

    # 1) 跑模型（只用 plspm；回傳 model + scores）
    model, scores = run_plspm_python(
        cog,
        Xpls,
        path_df,
        lv_blocks,
        lv_modes,
        scaled=bool(getattr(cfg, "PLS_STANDARDIZED", True)),
        scheme=str(getattr(cfg, "PLS_SCHEME", "PATH")),
    )

    # 2) 對齊 LV 順序（乾淨：strict，不允許用 iloc 偷切）
    missing = [lv for lv in order if lv not in scores.columns]
    if missing:
        raise KeyError(f"LV scores missing from plspm output: {missing}. "
                       f"Available: {list(scores.columns)}")
    scores = scores[order].copy()

    # 3) sign-fix（只乘 -1，不產生新數值；且 sign 會同步套到 outer/path）
    if sign_fix_on:
        anchors = choose_anchors_by_max_abs_loading(Xpls, scores, lv_blocks)
        sign_map = get_sign_map_by_anchors(Xpls, scores, anchors)
        scores = sign_fix_scores_by_anchors(scores, Xpls, anchors)
    else:
        anchors = {}
        sign_map = {}

    # 4) cross-loadings（可用 corr；檢視表，不是 outer loading 來源）
    PLS_cross = corr_items_vs_scores(Xpls[item_cols], scores[order], method=corr_method).round(dec)

    # 5) Outer：只從 model 取（strict=True）
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

    # 6) CR/AVE：用 outer loadings（model-based）
    PLS_quality = quality_paper_table(outer, lv_modes, order=order).round(dec)

    # 7) Paths：只從 model 取（strict=True）
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


# ---------------------------------------------------------
# (B) summarize_direct：CI 判顯著（無 p-value、無分配假設）
# ---------------------------------------------------------
def summarize_direct_ci(cog, keys_df: pd.DataFrame, point_est: np.ndarray, boot: np.ndarray) -> pd.DataFrame:
    """
    Paper-style bootstrap summary (distribution-free):
      estimate, boot_mean, boot_se, t, CI(lo), CI(hi), Sig
    Sig: CI does not include 0
    """
    cfg = cog.cfg.pls
    dec = int(getattr(cfg, "PAPER_DECIMALS", 3))

    qlo = float(getattr(cfg, "BOOT_CI_LO", 0.025))
    qhi = float(getattr(cfg, "BOOT_CI_HI", 0.975))

    se = np.nanstd(boot, axis=0, ddof=1)
    t = safe_t(point_est, se)

    ci_l = np.nanquantile(boot, qlo, axis=0)
    ci_u = np.nanquantile(boot, qhi, axis=0)
    sig = ci_sig(ci_l, ci_u)

    # 欄名用百分比（paper 直觀）
    lo_label = f"CI{qlo*100:.1f}"
    hi_label = f"CI{qhi*100:.1f}"

    out = keys_df.copy()
    out["estimate"] = np.asarray(point_est, dtype=float)
    out["boot_mean"] = np.nanmean(boot, axis=0)
    out["boot_se"] = se
    out["t"] = t
    out[lo_label] = ci_l
    out[hi_label] = ci_u
    out["Sig"] = sig  # True/False
    return out.round(dec)
