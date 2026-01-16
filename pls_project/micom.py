# pls_project/micom.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from pls_project.pls_core import run_plspm_python


@dataclass
class MICOMSettings:
    B: int = 200
    seed: int = 0
    alpha: float = 0.05
    min_n_per_group: int = 30
    standardized: bool = True
    vif_note_threshold: float = 3.3


def _zscore_df(X: pd.DataFrame) -> pd.DataFrame:
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0).replace(0, np.nan)
    Z = (X - mu) / sd
    return Z


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5:
        return np.nan
    aa = a[ok]
    bb = b[ok]
    if np.nanstd(aa) < 1e-12 or np.nanstd(bb) < 1e-12:
        return np.nan
    return float(np.corrcoef(aa, bb)[0, 1])


def _ols_weights_no_intercept(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Get weights beta from y ~ X (no intercept), both should already be standardized.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    ok = np.isfinite(y).ravel() & np.isfinite(X).all(axis=1)
    if ok.sum() < max(5, X.shape[1] + 2):
        return np.full((X.shape[1],), np.nan, dtype=float)
    beta = np.linalg.lstsq(X[ok], y[ok], rcond=None)[0].ravel()
    return beta.astype(float)


def _estimate_group_weights_via_scores(
    X_group: pd.DataFrame,
    scores_group: pd.DataFrame,
    lv_blocks: Dict[str, List[str]],
    order: List[str],
    *,
    standardized: bool,
) -> Dict[str, pd.Series]:
    """
    Fallback weight estimation:
      for each LV, regress LV score on its indicators (standardized) => beta as weights
    """
    Xg = X_group.copy()
    if standardized:
        Xg = _zscore_df(Xg)

    W: Dict[str, pd.Series] = {}
    for lv in order:
        inds = [c for c in lv_blocks.get(lv, []) if c in Xg.columns]
        if not inds or lv not in scores_group.columns:
            continue
        y = pd.to_numeric(scores_group[lv], errors="coerce").to_numpy()
        # standardize y
        y = (y - np.nanmean(y)) / (np.nanstd(y) if np.nanstd(y) > 1e-12 else np.nan)
        Xmat = Xg[inds].apply(pd.to_numeric, errors="coerce").to_numpy()
        beta = _ols_weights_no_intercept(Xmat, y)
        W[lv] = pd.Series(beta, index=inds, name=lv)
    return W


def _scores_from_weights(
    X_full: pd.DataFrame,
    W: Dict[str, pd.Series],
    *,
    standardized: bool,
) -> pd.DataFrame:
    Xf = X_full.copy()
    if standardized:
        Xf = _zscore_df(Xf)

    out = {}
    for lv, w in W.items():
        inds = [c for c in w.index if c in Xf.columns]
        if not inds:
            continue
        ww = pd.to_numeric(w.loc[inds], errors="coerce").to_numpy(dtype=float)
        Xm = Xf[inds].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        s = Xm @ ww
        out[lv] = s
    return pd.DataFrame(out, index=X_full.index)


def _sign_align_weights_by_anchor(
    X_full: pd.DataFrame,
    W: Dict[str, pd.Series],
    anchors: Dict[str, str],
    *,
    standardized: bool,
) -> Dict[str, pd.Series]:
    """
    Flip each LV weights if corr(anchor_item, composite_score) < 0.
    This prevents arbitrary sign flips breaking compositional invariance (corr ~ -1).
    """
    W2: Dict[str, pd.Series] = {k: v.copy() for k, v in W.items()}
    S = _scores_from_weights(X_full, W2, standardized=standardized)
    for lv, a in anchors.items():
        if lv not in S.columns:
            continue
        if a not in X_full.columns:
            continue
        r = _safe_corr(pd.to_numeric(X_full[a], errors="coerce").to_numpy(), S[lv].to_numpy())
        if pd.notna(r) and r < 0:
            W2[lv] = -1.0 * W2[lv]
    return W2


def micom_two_group(
    cog,
    *,
    X_full: pd.DataFrame,
    path_df: pd.DataFrame,
    lv_blocks: Dict[str, List[str]],
    lv_modes: Dict[str, str],
    order: List[str],
    group_mask: np.ndarray,
    anchors: Optional[Dict[str, str]] = None,
    settings: Optional[MICOMSettings] = None,
) -> Dict[str, pd.DataFrame]:
    """
    MICOM for 2 groups (SmartPLS-style):
      Step1: configural invariance (checks/flags)
      Step2: compositional invariance via permutation on correlation between
             two sets of composite scores built using group-specific weights (applied to full data)
      Step3: equality of mean and variance via permutation on score differences (pooled weights)

    Returns dict of DataFrames:
      info, step1_configural, step2_compositional, step3_means_vars, summary
    """
    if settings is None:
        settings = MICOMSettings()
    if anchors is None:
        anchors = {}

    group_mask = np.asarray(group_mask, dtype=bool)
    if group_mask.ndim != 1 or group_mask.shape[0] != X_full.shape[0]:
        raise ValueError("group_mask must be 1D boolean array with length == nrows(X_full).")

    n1 = int(group_mask.sum())
    n2 = int((~group_mask).sum())

    # Step 1 (configural): basic checks
    step1 = pd.DataFrame([{
        "Configural_invariance": True,  # same codepath/settings by construction
        "Same_indicators_per_construct": True,
        "Same_algorithm_settings": True,
        "Same_data_treatment(standardized)": bool(settings.standardized),
        "n_group1": n1,
        "n_group2": n2,
        "min_n_per_group": int(settings.min_n_per_group),
        "Pass_min_n": bool((n1 >= settings.min_n_per_group) and (n2 >= settings.min_n_per_group)),
        "B_permutations": int(settings.B),
        "alpha": float(settings.alpha),
    }])

    # If not enough n, still return with info
    if not bool(step1.loc[0, "Pass_min_n"]):
        return {
            "info": pd.DataFrame([{"Error": "MICOM blocked: group size too small."}]),
            "step1_configural": step1,
            "step2_compositional": pd.DataFrame(),
            "step3_means_vars": pd.DataFrame(),
            "summary": pd.DataFrame(),
        }

    cfg_pls = cog.cfg.pls
    scaled = bool(getattr(cfg_pls, "PLS_STANDARDIZED", True))

    X1 = X_full.loc[group_mask].reset_index(drop=True)
    X2 = X_full.loc[~group_mask].reset_index(drop=True)

    # ----- estimate group models (original split) -----
    m1, s1 = run_plspm_python(cog, X1, path_df, lv_blocks, lv_modes, scaled=scaled)
    m2, s2 = run_plspm_python(cog, X2, path_df, lv_blocks, lv_modes, scaled=scaled)

    # weights by regression on LV scores (stable, works for A/B)
    W1 = _estimate_group_weights_via_scores(X1, s1[order], lv_blocks, order, standardized=settings.standardized)
    W2 = _estimate_group_weights_via_scores(X2, s2[order], lv_blocks, order, standardized=settings.standardized)

    # sign-align to avoid -1 correlations due to arbitrary sign
    W1 = _sign_align_weights_by_anchor(X_full, W1, anchors, standardized=settings.standardized)
    W2 = _sign_align_weights_by_anchor(X_full, W2, anchors, standardized=settings.standardized)

    S1_full = _scores_from_weights(X_full, W1, standardized=settings.standardized)
    S2_full = _scores_from_weights(X_full, W2, standardized=settings.standardized)

    # Step 2: original correlations
    step2_rows = []
    orig_corr = {}
    for lv in order:
        if lv in S1_full.columns and lv in S2_full.columns:
            c = _safe_corr(S1_full[lv].to_numpy(), S2_full[lv].to_numpy())
        else:
            c = np.nan
        orig_corr[lv] = c
        step2_rows.append({"Construct": lv, "c_original": c})
    step2 = pd.DataFrame(step2_rows)

    # Step 2: permutation distribution of correlations
    rng = np.random.default_rng(int(settings.seed))
    perm_corrs = {lv: [] for lv in order}
    n = int(X_full.shape[0])

    for _ in range(int(settings.B)):
        idx = rng.permutation(n)
        g1_idx = idx[:n1]
        g2_idx = idx[n1:]
        Xp1 = X_full.iloc[g1_idx].reset_index(drop=True)
        Xp2 = X_full.iloc[g2_idx].reset_index(drop=True)

        mp1, sp1 = run_plspm_python(cog, Xp1, path_df, lv_blocks, lv_modes, scaled=scaled)
        mp2, sp2 = run_plspm_python(cog, Xp2, path_df, lv_blocks, lv_modes, scaled=scaled)

        Wp1 = _estimate_group_weights_via_scores(Xp1, sp1[order], lv_blocks, order, standardized=settings.standardized)
        Wp2 = _estimate_group_weights_via_scores(Xp2, sp2[order], lv_blocks, order, standardized=settings.standardized)

        Wp1 = _sign_align_weights_by_anchor(X_full, Wp1, anchors, standardized=settings.standardized)
        Wp2 = _sign_align_weights_by_anchor(X_full, Wp2, anchors, standardized=settings.standardized)

        Sp1_full = _scores_from_weights(X_full, Wp1, standardized=settings.standardized)
        Sp2_full = _scores_from_weights(X_full, Wp2, standardized=settings.standardized)

        for lv in order:
            if lv in Sp1_full.columns and lv in Sp2_full.columns:
                cperm = _safe_corr(Sp1_full[lv].to_numpy(), Sp2_full[lv].to_numpy())
            else:
                cperm = np.nan
            perm_corrs[lv].append(cperm)

    # Compute p (lower-tail) and quantile for compositional invariance
    alpha = float(settings.alpha)
    out_rows = []
    for lv in order:
        c0 = float(orig_corr.get(lv, np.nan))
        arr = np.asarray(perm_corrs[lv], dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0 or not np.isfinite(c0):
            out_rows.append({
                "Construct": lv,
                "c_original": c0,
                "q_alpha": np.nan,
                "p_lower": np.nan,
                "Pass_compositional": False,
            })
            continue
        q_alpha = float(np.quantile(arr, alpha))
        p_lower = float(np.mean(arr <= c0))  # one-tailed (low corr = bad)
        pass_comp = bool(p_lower >= alpha)   # equivalent to c0 >= q_alpha (up to discreteness)
        out_rows.append({
            "Construct": lv,
            "c_original": c0,
            "q_alpha": q_alpha,
            "p_lower": p_lower,
            "Pass_compositional": pass_comp,
        })

    step2 = pd.DataFrame(out_rows)

    # ----- Step 3: equality of mean/variance (use pooled weights for scoring) -----
    # pooled model weights via regression on pooled LV scores
    mp, sp = run_plspm_python(cog, X_full.reset_index(drop=True), path_df, lv_blocks, lv_modes, scaled=scaled)
    Wp = _estimate_group_weights_via_scores(
        X_full.reset_index(drop=True),
        sp[order],
        lv_blocks,
        order,
        standardized=settings.standardized,
    )
    Wp = _sign_align_weights_by_anchor(X_full, Wp, anchors, standardized=settings.standardized)
    Sp = _scores_from_weights(X_full, Wp, standardized=settings.standardized)

    # observed diffs
    step3_rows = []
    perm_diffs_mean = {lv: [] for lv in order}
    perm_diffs_var = {lv: [] for lv in order}

    # permutation diffs using fixed pooled scores (efficient & valid once compositional holds)
    idx_all = np.arange(n)
    for _ in range(int(settings.B)):
        idxp = rng.permutation(idx_all)
        g1p = np.zeros(n, dtype=bool)
        g1p[idxp[:n1]] = True
        for lv in order:
            if lv not in Sp.columns:
                continue
            v = Sp[lv].to_numpy(dtype=float)_
