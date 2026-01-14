# pls_project/pls_core.py
from __future__ import annotations

import sys
import subprocess
import warnings
from typing import Dict, List, Tuple, Optional, Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA

from .io_utils import safe_t_p


# =========================================================
# 0) Graph / path utilities
# =========================================================
def topo_sort(nodes: Iterable[str], edges: List[Tuple[str, str]]) -> List[str]:
    """Topological sort for a DAG; falls back to input order if cycle detected."""
    nodes = list(nodes)
    adj = {n: [] for n in nodes}
    indeg = {n: 0 for n in nodes}
    for a, b in edges:
        if a in adj and b in adj:
            adj[a].append(b)
            indeg[b] += 1

    q = [n for n in nodes if indeg[n] == 0]
    out = []
    while q:
        n = q.pop(0)
        out.append(n)
        for m in adj[n]:
            indeg[m] -= 1
            if indeg[m] == 0:
                q.append(m)

    return out if len(out) == len(nodes) else nodes


def make_plspm_path(nodes_ordered: List[str], edges_from_to: List[Tuple[str, str]]) -> pd.DataFrame:
    """
    Build lower-triangular path matrix used by plspm:
      path.loc[to, from] = 1 if from -> to exists
    """
    path = pd.DataFrame(0, index=nodes_ordered, columns=nodes_ordered, dtype=int)
    for fr, to in edges_from_to:
        if fr in path.columns and to in path.index:
            path.loc[to, fr] = 1

    A = path.values
    if np.any(np.triu(A, k=0) != 0):
        bad = np.where(np.triu(A, k=0) != 0)
        raise ValueError(
            f"path_matrix not lower triangular. Bad at row={bad[0][0]}, col={bad[1][0]}."
        )
    return path


# =========================================================
# 1) plspm dependency + SmartPLS4-aligned runner
# =========================================================
def ensure_plspm(auto_install: bool = True):
    """
    Import plspm. If missing and auto_install=True, pip install it.
    Returns: (config_module, Plspm, Mode, Scheme)
    """
    try:
        import plspm.config as c
        from plspm.plspm import Plspm
        from plspm.mode import Mode
        from plspm.scheme import Scheme
        return c, Plspm, Mode, Scheme
    except Exception as e:
        if not auto_install:
            raise ImportError("plspm not installed. Please pip install plspm.") from e
        subprocess.check_call([sys.executable, "-m", "pip", "-q", "install", "plspm"])
        import plspm.config as c
        from plspm.plspm import Plspm
        from plspm.mode import Mode
        from plspm.scheme import Scheme
        return c, Plspm, Mode, Scheme


def _zscore_df(X: pd.DataFrame) -> pd.DataFrame:
    X = X.astype(float)
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0).replace(0, np.nan)
    Z = (X - mu) / sd
    return Z


def _scheme_to_plspm_scheme(s: str) -> str:
    """
    Map cfg scheme string to plspm Scheme attribute name.
    SmartPLS4 offers PATH/FACTOR/PCA and removed CENTROID; we keep CENTROID for backward compatibility.
    """
    s = (s or "").strip().upper()
    if s in ("PATH", "PATH_WEIGHTING", "PATHWEIGHTING"):
        return "PATH"
    if s in ("FACTOR", "FACTORIAL", "FACTOR_WEIGHTING"):
        return "FACTORIAL"
    if s in ("CENTROID",):
        return "CENTROID"
    if s in ("PCA", "PRINCIPAL_COMPONENTS", "PRINCIPAL_COMPONENT_ANALYSIS"):
        return "PCA"
    # default: align SmartPLS4 default (path weighting)
    return "PATH"


def _pca_construct_scores(
    X: pd.DataFrame,
    lv_blocks: Dict[str, List[str]],
    lv_modes: Dict[str, str],
    scaled: bool = True,
    seed: int = 0,
) -> pd.DataFrame:
    """
    SmartPLS4 offers a PCA option. Here we approximate it as:
      - for each LV, compute first PCA component score of its indicators (after z-scoring if scaled=True).
      - formative/reflective are treated the same for score construction (score extraction only).
    Note: This is a pragmatic approximation for construct scores; it is not the full SmartPLS4 PCA pipeline.
    """
    Xuse = _zscore_df(X) if scaled else X.astype(float)
    scores = {}
    rng = np.random.default_rng(seed)

    for lv, inds in lv_blocks.items():
        cols = [c for c in inds if c in Xuse.columns]
        if len(cols) == 0:
            scores[lv] = np.full((len(Xuse),), np.nan)
            continue

        M = Xuse[cols].to_numpy()
        # If all-NaN column(s) exist, PCA will fail; fill with column mean (0 after z-score)
        if np.isnan(M).any():
            col_mean = np.nanmean(M, axis=0)
            col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
            inds_nan = np.where(np.isnan(M))
            M[inds_nan] = np.take(col_mean, inds_nan[1])

        pca = PCA(n_components=1, random_state=int(rng.integers(0, 2**31 - 1)))
        s = pca.fit_transform(M).ravel()

        # sign orientation: make sum of loadings positive
        load = pca.components_[0]
        if np.nansum(load) < 0:
            s = -s
        scores[lv] = s

    return pd.DataFrame(scores, index=X.index)


def run_plspm_python(
    cog,
    X: pd.DataFrame,
    path_df: pd.DataFrame,
    lv_blocks: Dict[str, List[str]],
    lv_modes: Dict[str, str],
    *,
    scaled: bool = True,
    auto_install: bool = True,
):
    """
    SmartPLS4-aligned PLS runner wrapper.

    - SmartPLS4 defaults: PATH weighting, 3000 iterations, stop criterion 1e-7. :contentReference[oaicite:3]{index=3}
    - plspm (python) supports schemes: PATH/FACTORIAL/CENTROID (version dependent).
    - We:
        (1) map cfg scheme to plspm
        (2) try SmartPLS4-like (3000, 1e-7) first, then cfg values, then fallbacks
        (3) optionally support scheme='PCA' by a PCA score approximation (returns model=None)

    Returns:
      (model, scores_df)
        - model may be None if scheme == 'PCA'
        - scores_df columns are latent variables (LVs)
    """
    cfg = cog.cfg.pls
    scheme = _scheme_to_plspm_scheme(getattr(cfg, "PLS_SCHEME", "PATH"))

    # SmartPLS4 removed centroid: warn if user selects it
    if scheme == "CENTROID":
        warnings.warn(
            "CENTROID scheme is not available in SmartPLS 4 (legacy option). "
            "Use PATH (default) or FACTORIAL to align with SmartPLS4."
        )

    # PCA option (SmartPLS4 provides PCA-based estimation option)
    if scheme == "PCA":
        scores = _pca_construct_scores(
            X, lv_blocks=lv_blocks, lv_modes=lv_modes, scaled=scaled, seed=getattr(cfg, "PLS_SEED", 0)
        )
        scores = scores.apply(pd.to_numeric, errors="coerce")
        return None, scores

    c, Plspm, Mode, Scheme = ensure_plspm(auto_install=auto_install)
    scheme_attr = scheme  # PATH / FACTORIAL / CENTROID
    scheme_obj = getattr(Scheme, scheme_attr, Scheme.PATH)

    # SmartPLS4 fixed settings (prefer first)
    SMARTPLS4_MAX_ITER = 3000
    SMARTPLS4_TOL = 1e-7

    max_iter_cfg = int(getattr(cfg, "PLSPM_MAX_ITER", SMARTPLS4_MAX_ITER) or SMARTPLS4_MAX_ITER)
    tol_cfg = float(getattr(cfg, "PLSPM_TOL", SMARTPLS4_TOL) or SMARTPLS4_TOL)

    try_settings = [
        (SMARTPLS4_MAX_ITER, SMARTPLS4_TOL),  # align SmartPLS4 first
        (max_iter_cfg, tol_cfg),              # then user's cfg
        (2000, 1e-6),
        (2000, 1e-5),
        (1000, 1e-5),
    ]

    last_err = None

    for iters, tol in try_settings:
        try:
            config = c.Config(path_df, scaled=scaled)
            for lv, inds in lv_blocks.items():
                mode = str(lv_modes.get(lv, "A")).upper()
                mode_obj = Mode.A if mode == "A" else Mode.B
                mvs = [c.MV(col) for col in inds]
                config.add_lv(lv, mode_obj, *mvs)
        except Exception as e:
            last_err = e
            continue

        model = None
        # plspm versions differ in keyword args; try multiple
        for kw in (
            {"iterations": iters, "tolerance": tol},
            {"max_iter": iters, "tol": tol},
            {},
        ):
            try:
                model = Plspm(X, config, scheme_obj, **kw)
                break
            except TypeError:
                continue
            except Exception as e:
                last_err = e
                model = None
                break

        if model is None:
            continue

        try:
            scores_raw = model.scores() if callable(getattr(model, "scores", None)) else model.scores
            scores = (
                scores_raw.copy()
                if isinstance(scores_raw, pd.DataFrame)
                else pd.DataFrame(scores_raw, index=X.index)
            )
            scores = scores.apply(pd.to_numeric, errors="coerce")
            return model, scores
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"plspm failed after retries. Last error: {last_err}")


# =========================================================
# 2) Measurement helpers
# =========================================================
def corr_items_vs_scores(X_items: pd.DataFrame, scores: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    tmp = pd.concat([X_items, scores], axis=1)
    C = tmp.corr(method=method)
    return C.loc[X_items.columns, scores.columns]


def compute_cr_ave(loadings: np.ndarray) -> Tuple[float, float]:
    """
    CR (composite reliability) and AVE for reflective constructs.
    Uses standardized outer loadings.
    """
    lam = np.asarray(loadings, dtype=float)
    lam = lam[~np.isnan(lam)]
    if lam.size == 0:
        return np.nan, np.nan
    ave = float(np.mean(lam**2))
    denom = (lam.sum() ** 2) + np.sum(1 - lam**2)
    cr = float((lam.sum() ** 2) / denom) if denom != 0 else np.nan
    return cr, ave


def htmt_matrix(
    X_items: pd.DataFrame,
    group_items_dict: Dict[str, List[str]],
    groups_list: List[str],
    method: str = "pearson",
) -> pd.DataFrame:
    """
    HTMT as defined by Henseler et al.:
      mean(|r_ij| i in A, j in B) / sqrt(mean(|r_ij| i<j in A) * mean(|r_ij| i<j in B))
    """
    R = X_items.corr(method=method).abs()
    out = pd.DataFrame(index=groups_list, columns=groups_list, dtype=float)

    for i, ga in enumerate(groups_list):
        A = group_items_dict[ga]
        RA = R.loc[A, A].values
        triA = RA[np.triu_indices_from(RA, k=1)]
        mA = np.nanmean(triA) if triA.size else np.nan

        for j, gb in enumerate(groups_list):
            if i == j:
                out.loc[ga, gb] = 1.0
                continue

            B = group_items_dict[gb]
            RB = R.loc[B, B].values
            triB = RB[np.triu_indices_from(RB, k=1)]
            mB = np.nanmean(triB) if triB.size else np.nan

            HAB = R.loc[A, B].values
            mAB = np.nanmean(HAB) if HAB.size else np.nan

            denom = np.sqrt(mA * mB) if (pd.notna(mA) and pd.notna(mB) and mA > 0 and mB > 0) else np.nan
            out.loc[ga, gb] = float(mAB / denom) if pd.notna(denom) else np.nan

    return out.round(3)


# =========================================================
# 3) Structural helpers
# =========================================================
def preds_for_endogenous(path_df: pd.DataFrame, to: str) -> List[str]:
    return [c for c in path_df.columns if path_df.loc[to, c] == 1]


def ols_fit(y: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    OLS with intercept:
      y = b0 + b1 x1 + ... + bp xp
    Returns (beta, R2)
    """
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    X = np.asarray(X, dtype=float)
    X1 = np.column_stack([np.ones((X.shape[0], 1)), X])

    beta = np.linalg.lstsq(X1, y, rcond=None)[0].flatten()
    yhat = (X1 @ beta.reshape(-1, 1)).flatten()
    y0 = y.flatten()

    sse = float(np.sum((y0 - yhat) ** 2))
    sst = float(np.sum((y0 - np.mean(y0)) ** 2))
    r2 = 1 - sse / sst if sst > 1e-12 else np.nan

    return beta, float(r2)


def path_estimates_from_scores(scores_df: pd.DataFrame, path_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for to in path_df.index:
        preds = [p for p in preds_for_endogenous(path_df, to) if p in scores_df.columns]
        if len(preds) == 0 or to not in scores_df.columns:
            continue
        beta, _ = ols_fit(scores_df[to].values, scores_df[preds].values)
        for j, fr in enumerate(preds):
            rows.append({"from": fr, "to": to, "estimate": float(beta[1 + j])})
    return pd.DataFrame(rows)


def r2_f2_from_scores(scores_df: pd.DataFrame, path_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    r2_rows, f2_rows = [], []
    for to in path_df.index:
        preds = [p for p in preds_for_endogenous(path_df, to) if p in scores_df.columns]
        if len(preds) == 0 or to not in scores_df.columns:
            continue

        _, r2_full = ols_fit(scores_df[to].values, scores_df[preds].values)
        r2_rows.append({"Construct": to, "R2": r2_full, "Predictors": ", ".join(preds)})

        for p in preds:
            preds_ex = [x for x in preds if x != p]
            if len(preds_ex) == 0:
                r2_ex = 0.0
            else:
                _, r2_ex = ols_fit(scores_df[to].values, scores_df[preds_ex].values)
            denom = (1 - r2_full)
            f2 = (r2_full - r2_ex) / denom if denom > 1e-12 else np.nan
            f2_rows.append({"from": p, "to": to, "f2": f2, "R2_full": r2_full, "R2_excluded": r2_ex})

    return pd.DataFrame(r2_rows).round(4), pd.DataFrame(f2_rows).round(4)


def q2_cv_from_scores(
    cog,
    scores_df: pd.DataFrame,
    path_df: pd.DataFrame,
    n_splits: Optional[int] = None,
) -> pd.DataFrame:
    """
    Cross-validated Q² (not SmartPLS blindfolding Q²).
    Uses KFold on LV scores; good for predictive diagnostics.
    """
    cfg = cog.cfg.pls
    seed = int(getattr(cfg, "PLS_SEED", 0))
    if n_splits is None:
        n_splits = int(getattr(cfg, "Q2_FOLDS", 5))

    n = scores_df.shape[0]
    k = max(2, min(int(n_splits), int(n)))
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)

    rows = []
    for to in path_df.index:
        preds = [p for p in preds_for_endogenous(path_df, to) if p in scores_df.columns]
        if len(preds) == 0 or to not in scores_df.columns:
            continue

        y = scores_df[to].values.astype(float)
        X = scores_df[preds].values.astype(float)

        sse, sso = 0.0, 0.0
        y_mean = float(np.mean(y))

        for tr, te in kf.split(X):
            beta, _ = ols_fit(y[tr], X[tr])
            Xte1 = np.column_stack([np.ones((len(te), 1)), X[te]])
            yhat = (Xte1 @ beta.reshape(-1, 1)).flatten()
            yte = y[te]
            sse += float(np.sum((yte - yhat) ** 2))
            sso += float(np.sum((yte - y_mean) ** 2))

        q2 = 1 - sse / sso if sso > 1e-12 else np.nan
        rows.append({"Construct": to, "Q2_CV": q2, "Predictors": ", ".join(preds), "folds": k})

    return pd.DataFrame(rows).round(4)


def structural_vif(scores_df: pd.DataFrame, path_df: pd.DataFrame) -> pd.DataFrame:
    """
    Inner VIF (structural VIF) based on LV scores:
      VIF_x = 1 / (1 - R2_x|others)
    """
    rows = []
    for to in path_df.index:
        preds = [p for p in preds_for_endogenous(path_df, to) if p in scores_df.columns]
        if len(preds) < 2:
            if len(preds) == 1:
                rows.append({"Endogenous": to, "Predictor": preds[0], "VIF": 1.0})
            continue

        for x in preds:
            others = [p for p in preds if p != x]
            _, r2 = ols_fit(scores_df[x].values, scores_df[others].values)
            vif = 1.0 / (1.0 - r2) if (1.0 - r2) > 1e-12 else np.inf
            rows.append({"Endogenous": to, "Predictor": x, "VIF": vif})

    out = pd.DataFrame(rows)
    if not out.empty:
        out["VIF"] = out["VIF"].round(3)
    return out


# =========================================================
# 4) Effects (direct/indirect/total)
# =========================================================
def effects_total_indirect(
    order_nodes: List[str],
    edges: List[Tuple[str, str]],
    coef_map: Dict[Tuple[str, str], float],
) -> Tuple[pd.DataFrame, List[Tuple[str, str]]]:
    """
    Compute total effects in a DAG using topological order.
    Returns:
      - effects df with direct/indirect/total for all reachable pairs
      - pairs list (from,to) reachable
    """
    out_adj = {u: [] for u in order_nodes}
    for u, v in edges:
        if u in out_adj:
            out_adj[u].append(v)

    # reachable pairs (path existence only)
    pairs = []
    for s in order_nodes:
        reach = {n: 0.0 for n in order_nodes}
        reach[s] = 1.0
        for u in order_nodes:
            for v in out_adj.get(u, []):
                reach[v] += reach[u]
        for t in order_nodes:
            if t != s and reach[t] != 0:
                pairs.append((s, t))
    pairs = list(dict.fromkeys(pairs))

    # total effects with coefficients
    total = {}
    for s in order_nodes:
        eff = {n: 0.0 for n in order_nodes}
        eff[s] = 1.0
        for u in order_nodes:
            for v in out_adj.get(u, []):
                b = float(coef_map.get((u, v), 0.0))
                eff[v] += eff[u] * b
        for t in order_nodes:
            if t != s:
                total[(s, t)] = float(eff[t])

    rows = []
    for s, t in pairs:
        direct = float(coef_map.get((s, t), 0.0))
        tot = float(total.get((s, t), 0.0))
        indir = tot - direct
        rows.append({"from": s, "to": t, "direct": direct, "indirect": indir, "total": tot})

    return pd.DataFrame(rows), pairs


# =========================================================
# 5) Bootstrap summaries
# =========================================================
def summarize_boot(effect_samples: np.ndarray, keys_df: pd.DataFrame) -> pd.DataFrame:
    est = np.nanmean(effect_samples, axis=0)
    se = np.nanstd(effect_samples, axis=0, ddof=1)
    t, p = safe_t_p(est, se)
    ci_l = np.nanquantile(effect_samples, 0.025, axis=0)
    ci_u = np.nanquantile(effect_samples, 0.975, axis=0)

    out = keys_df.copy()
    out["boot_mean"] = est
    out["boot_se"] = se
    out["t"] = t
    out["p"] = p
    out["CI2.5"] = ci_l
    out["CI97.5"] = ci_u
    return out.round(4)


def summarize_direct(keys_df: pd.DataFrame, point_est: np.ndarray, boot: np.ndarray) -> pd.DataFrame:
    se = np.nanstd(boot, axis=0, ddof=1)
    t, p = safe_t_p(point_est, se)
    ci_l = np.nanquantile(boot, 0.025, axis=0)
    ci_u = np.nanquantile(boot, 0.975, axis=0)

    out = keys_df.copy()
    out["estimate"] = point_est
    out["boot_mean"] = np.nanmean(boot, axis=0)
    out["boot_se"] = se
    out["t"] = t
    out["p"] = p
    out["CI2.5"] = ci_l
    out["CI97.5"] = ci_u
    return out.round(4)
