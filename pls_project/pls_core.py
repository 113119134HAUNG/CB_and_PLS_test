# pls_project/pls_core.py
from __future__ import annotations

import sys
import subprocess
from collections import deque
from typing import Dict, List, Tuple, Optional, Iterable, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


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

    q = deque([n for n in nodes if indeg[n] == 0])
    out: List[str] = []

    while q:
        n = q.popleft()
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
# 1) plspm dependency + runner (SmartPLS4-aligned, clean)
# =========================================================
def ensure_plspm(auto_install: bool = False):
    """
    Import plspm. Optionally pip install if auto_install=True.
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
            raise ImportError("plspm not installed (or incompatible). Please install/pin it explicitly.") from e
        subprocess.check_call([sys.executable, "-m", "pip", "-q", "install", "plspm"])
        import plspm.config as c
        from plspm.plspm import Plspm
        from plspm.mode import Mode
        from plspm.scheme import Scheme
        return c, Plspm, Mode, Scheme


def _scheme_to_plspm_scheme_strict(s: str) -> str:
    """
    SmartPLS4-aligned, clean:
      - allow only PATH / FACTORIAL
      - CENTROID not supported in SmartPLS4
      - PCA option here would be an approximation -> disallow
    """
    s = (s or "").strip().upper()
    if s in ("PATH", "PATH_WEIGHTING", "PATHWEIGHTING", ""):
        return "PATH"
    if s in ("FACTOR", "FACTORIAL", "FACTOR_WEIGHTING"):
        return "FACTORIAL"
    if s in ("CENTROID",):
        raise ValueError("PLS_SCHEME='CENTROID' is not available in SmartPLS4 (clean mode forbids it).")
    if s in ("PCA", "PRINCIPAL_COMPONENTS", "PRINCIPAL_COMPONENT_ANALYSIS"):
        raise ValueError("PLS_SCHEME='PCA' is not allowed in clean mode (would be an approximation).")
    raise ValueError(f"Unknown PLS_SCHEME: {s}")


def run_plspm_python(
    cog: Any,
    X: pd.DataFrame,
    path_df: pd.DataFrame,
    lv_blocks: Dict[str, List[str]],
    lv_modes: Dict[str, str],
    *,
    scaled: bool = True,
):
    """
    Clean PLS runner:
      - scheme from cfg.pls.PLS_SCHEME (PATH/FACTORIAL only)
      - iterations/tolerance from cfg.pls.PLSPM_MAX_ITER / cfg.pls.PLSPM_TOL
      - optional auto-install controlled by cfg.pls.AUTO_INSTALL_PLSPM (if present), else False
    """
    cfg = cog.cfg.pls

    scheme = _scheme_to_plspm_scheme_strict(getattr(cfg, "PLS_SCHEME", "PATH"))

    if not hasattr(cfg, "PLSPM_MAX_ITER") or not hasattr(cfg, "PLSPM_TOL"):
        raise AttributeError("Config.pls must define PLSPM_MAX_ITER and PLSPM_TOL (clean mode).")
    iters = int(cfg.PLSPM_MAX_ITER)
    tol = float(cfg.PLSPM_TOL)

    auto_install = bool(getattr(cfg, "AUTO_INSTALL_PLSPM", False))
    c, Plspm, Mode, Scheme = ensure_plspm(auto_install=auto_install)
    scheme_obj = getattr(Scheme, scheme, Scheme.PATH)

    config = c.Config(path_df, scaled=bool(scaled))
    for lv, inds in lv_blocks.items():
        mode = str(lv_modes.get(lv, "A")).upper()
        mode_obj = Mode.A if mode == "A" else Mode.B
        mvs = [c.MV(col) for col in inds]
        config.add_lv(lv, mode_obj, *mvs)

    last_type_err = None
    model = None
    for kw in (
        {"iterations": iters, "tolerance": tol},
        {"max_iter": iters, "tol": tol},
    ):
        try:
            model = Plspm(X, config, scheme_obj, **kw)
            break
        except TypeError as e:
            last_type_err = e
            model = None
        except Exception as e:
            raise RuntimeError(f"plspm run failed. error={e}") from e

    if model is None:
        raise TypeError(
            "plspm API does not accept iterations/tolerance kwargs in this version. "
            f"Pin a compatible plspm version. last_err={last_type_err}"
        )

    scores_raw = model.scores() if callable(getattr(model, "scores", None)) else getattr(model, "scores", None)
    if scores_raw is None:
        raise AttributeError("plspm model does not expose scores().")

    scores = scores_raw if isinstance(scores_raw, pd.DataFrame) else pd.DataFrame(scores_raw, index=X.index)
    scores = scores.apply(pd.to_numeric, errors="coerce")
    return model, scores


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


def quality_paper_table(outer_tbl: pd.DataFrame, lv_modes: dict, order: list[str]) -> pd.DataFrame:
    """
    Paper-style quality table:
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


def htmt_matrix(
    X_items: pd.DataFrame,
    group_items_dict: Dict[str, List[str]],
    groups_list: List[str],
    method: str = "pearson",
) -> pd.DataFrame:
    """
    NOTE: 不要在這裡 round；caller 決定輸出精度（推論用的 bootstrap 需要高精度）。
    Optimized: precompute within-construct mean correlations (mA/mB) once.
    """
    R = X_items.corr(method=method).abs()
    G = list(groups_list)

    within_mean: Dict[str, float] = {}
    for g in G:
        A = group_items_dict[g]
        RA = R.loc[A, A].to_numpy(dtype=float)
        triA = RA[np.triu_indices_from(RA, k=1)]
        within_mean[g] = float(np.nanmean(triA)) if triA.size else np.nan

    out = pd.DataFrame(index=G, columns=G, dtype=float)
    for i, ga in enumerate(G):
        A = group_items_dict[ga]
        mA = within_mean.get(ga, np.nan)

        for j, gb in enumerate(G):
            if i == j:
                out.loc[ga, gb] = 1.0
                continue

            B = group_items_dict[gb]
            mB = within_mean.get(gb, np.nan)

            HAB = R.loc[A, B].to_numpy(dtype=float)
            mAB = float(np.nanmean(HAB)) if HAB.size else np.nan

            denom = np.sqrt(mA * mB) if (pd.notna(mA) and pd.notna(mB) and mA > 0 and mB > 0) else np.nan
            out.loc[ga, gb] = float(mAB / denom) if pd.notna(denom) else np.nan

    return out


# =========================================================
# 3) Model extraction + sign alignment helpers
# =========================================================
def _maybe_call(x):
    return x() if callable(x) else x


def _first_existing_attr(obj, names: List[str]):
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return None


def _ensure_df(x) -> Optional[pd.DataFrame]:
    if x is None:
        return None
    if isinstance(x, pd.DataFrame):
        return x
    try:
        return pd.DataFrame(x)
    except Exception:
        return None


def apply_sign_to_scores(scores_df: pd.DataFrame, sign_map: Dict[str, int]) -> pd.DataFrame:
    """Flip LV scores using sign_map (+1/-1)."""
    if scores_df is None or scores_df.empty or not sign_map:
        return scores_df
    out = scores_df.copy()
    for lv, s in sign_map.items():
        if lv in out.columns and int(s) == -1:
            out[lv] = -out[lv]
    return out


def get_sign_map_by_anchors(X_df: pd.DataFrame, scores_df: pd.DataFrame, anchors: Dict[str, str]) -> Dict[str, int]:
    """
    Return LV sign (+1/-1) based on corr(anchor_indicator, LV_score).
    """
    sign: Dict[str, int] = {}
    for lv, a in anchors.items():
        if lv not in scores_df.columns or a not in X_df.columns:
            continue
        r = X_df[a].corr(scores_df[lv])
        sign[lv] = -1 if (pd.notna(r) and r < 0) else 1
    return sign


def apply_sign_to_paths(paths_long: pd.DataFrame, sign_map: dict) -> pd.DataFrame:
    """
    b' = b * s_to * s_from
    """
    if paths_long is None or paths_long.empty:
        return paths_long
    out = paths_long.copy()
    s_from = out["from"].map(lambda x: sign_map.get(x, 1)).astype(float)
    s_to = out["to"].map(lambda x: sign_map.get(x, 1)).astype(float)
    out["estimate"] = pd.to_numeric(out["estimate"], errors="coerce") * s_to * s_from
    return out


def apply_sign_to_outer(outer_df: pd.DataFrame, sign_map: dict) -> pd.DataFrame:
    if outer_df is None or outer_df.empty:
        return outer_df
    out = outer_df.copy()
    s = out["Construct"].map(lambda x: sign_map.get(x, 1)).astype(float)
    for col in ["OuterLoading", "OuterWeight"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce") * s
    if "Communality(h2)" in out.columns and "OuterLoading" in out.columns:
        out["Communality(h2)"] = pd.to_numeric(out["OuterLoading"], errors="coerce") ** 2
    return out


def get_path_results(
    model,
    path_df: pd.DataFrame,
    *,
    strict: bool = True,
    scores_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Prefer extracting path coefficients from plspm model.
    If strict=False, fallback to OLS(scores) when unavailable.

    Return long format: from,to,estimate
    """
    cand = _first_existing_attr(model, ["path_coefficients", "path_coefs", "inner_model", "inner_summary"])
    cand = _maybe_call(cand)
    M = _ensure_df(cand)

    if M is not None and (set(path_df.index).issubset(set(M.index)) and set(path_df.columns).issubset(set(M.columns))):
        rows = []
        for to in path_df.index:
            for fr in path_df.columns:
                if int(path_df.loc[to, fr]) == 1:
                    rows.append({"from": fr, "to": to, "estimate": float(M.loc[to, fr])})
        return pd.DataFrame(rows)

    if M is not None:
        cols = {c.lower(): c for c in M.columns}
        if ("from" in cols and "to" in cols) and ("estimate" in cols or "path" in cols or "coef" in cols):
            est_col = cols.get("estimate") or cols.get("path") or cols.get("coef")
            out = M.rename(columns={cols["from"]: "from", cols["to"]: "to", est_col: "estimate"})[["from", "to", "estimate"]]
            keep = []
            for _, r in out.iterrows():
                fr, to = r["from"], r["to"]
                keep.append((to in path_df.index) and (fr in path_df.columns) and int(path_df.loc[to, fr]) == 1)
            return out.loc[keep].reset_index(drop=True)

    if strict:
        raise AttributeError("Cannot extract path coefficients from plspm model API.")

    if scores_df is None:
        return pd.DataFrame()
    return path_estimates_from_scores(scores_df, path_df)


def get_outer_results(
    model,
    X: pd.DataFrame,
    scores_df: pd.DataFrame,
    lv_blocks: dict,
    lv_modes: dict,
    *,
    strict: bool = True,
) -> pd.DataFrame:
    """
    Robust outer extraction from plspm model API.

    Fix: ensure matrix index/columns are cast to str BEFORE lookup,
         to avoid silent NaN due to dtype mismatch (e.g., Index not str).

    Tries, in order:
      1) model.outer_model() / model.crossloadings() as "long table"
      2) same candidates as "matrix" (indicator x construct)
      3) if candidate object has .loadings/.weights attributes -> use those matrices
      4) strict=True -> raise with debug info
         strict=False -> corr fallback
    """
    expected_lvs = list(scores_df.columns) if scores_df is not None else list(lv_blocks.keys())
    expected_lvs = [str(lv) for lv in expected_lvs]
    expected_inds: List[str] = []
    for lv in expected_lvs:
        for it in lv_blocks.get(lv, []):
            it = str(it)
            if it not in expected_inds:
                expected_inds.append(it)

    def _to_df(obj) -> Optional[pd.DataFrame]:
        if obj is None:
            return None
        obj = _maybe_call(obj)

        if isinstance(obj, pd.DataFrame):
            return obj

        if isinstance(obj, (list, tuple)) and len(obj) > 0:
            first = obj[0]
            if isinstance(first, dict):
                try:
                    return pd.DataFrame(obj)
                except Exception:
                    return None
            if hasattr(first, "_asdict"):
                try:
                    return pd.DataFrame([x._asdict() for x in obj])
                except Exception:
                    return None
            if hasattr(first, "__dict__"):
                try:
                    return pd.DataFrame([x.__dict__ for x in obj])
                except Exception:
                    return None

        for attr in ["df", "dataframe", "table", "data", "frame", "values"]:
            if hasattr(obj, attr):
                v = getattr(obj, attr)
                v = _maybe_call(v)
                if isinstance(v, pd.DataFrame):
                    return v
                try:
                    return pd.DataFrame(v)
                except Exception:
                    pass

        for meth in ["to_dataframe", "to_df", "as_dataframe", "to_pandas", "to_dict"]:
            if hasattr(obj, meth) and callable(getattr(obj, meth)):
                try:
                    v = getattr(obj, meth)()
                except Exception:
                    continue
                if isinstance(v, pd.DataFrame):
                    return v
                try:
                    return pd.DataFrame(v)
                except Exception:
                    pass

        try:
            return pd.DataFrame(obj)
        except Exception:
            return None

    def _looks_like_matrix(DF: pd.DataFrame) -> bool:
        if DF is None or DF.empty:
            return False
        idx = list(map(str, DF.index))
        cols = list(map(str, DF.columns))
        ind_hit = len(set(expected_inds) & set(idx))
        lv_hit = len(set(expected_lvs) & set(cols))
        if ind_hit > 0 and lv_hit > 0:
            return True
        ind_hit2 = len(set(expected_inds) & set(cols))
        lv_hit2 = len(set(expected_lvs) & set(idx))
        return (ind_hit2 > 0 and lv_hit2 > 0)

    def _matrix_to_outer(M: pd.DataFrame) -> pd.DataFrame:
        # critical: cast to str to avoid silent mismatch
        M2 = M.copy()
        M2.index = M2.index.map(str)
        M2.columns = M2.columns.map(str)

        if set(expected_inds).issubset(set(M2.index)) and set(expected_lvs).issubset(set(M2.columns)):
            Mat = M2
        elif set(expected_lvs).issubset(set(M2.index)) and set(expected_inds).issubset(set(M2.columns)):
            Mat = M2.T.copy()
            Mat.index = Mat.index.map(str)
            Mat.columns = Mat.columns.map(str)
        else:
            A = M2.to_numpy(dtype=float, copy=False)
            if A.shape == (len(expected_inds), len(expected_lvs)):
                Mat = pd.DataFrame(A, index=expected_inds, columns=expected_lvs)
            elif A.shape == (len(expected_lvs), len(expected_inds)):
                Mat = pd.DataFrame(A.T, index=expected_inds, columns=expected_lvs)
            else:
                raise ValueError("Cannot align matrix to indicators x constructs.")

        Mat.index = Mat.index.map(str)
        Mat.columns = Mat.columns.map(str)

        rows = []
        for lv in expected_lvs:
            mode = str(lv_modes.get(lv, "A")).upper()
            for it in lv_blocks.get(lv, []):
                it = str(it)
                ol = np.nan
                if (it in Mat.index) and (lv in Mat.columns):
                    ol = pd.to_numeric(Mat.loc[it, lv], errors="coerce")
                rows.append({
                    "Construct": lv,
                    "Indicator": it,
                    "Mode": "A(reflective)" if mode == "A" else "B(formative)",
                    "OuterLoading": float(ol) if pd.notna(ol) else np.nan,
                    "OuterWeight": np.nan,
                    "Communality(h2)": (float(ol) ** 2) if (mode == "A" and pd.notna(ol)) else np.nan,
                })
        return pd.DataFrame(rows)

    def _try_long_table(DF: pd.DataFrame) -> Optional[pd.DataFrame]:
        if DF is None or DF.empty:
            return None
        cols = {str(c).lower(): c for c in DF.columns}

        c_construct = cols.get("block") or cols.get("construct") or cols.get("lv") or cols.get("latent") or cols.get("name_lv")
        c_ind = cols.get("name") or cols.get("indicator") or cols.get("mv") or cols.get("manifest") or cols.get("name_mv") or cols.get("item")
        c_loading = cols.get("loading") or cols.get("outer_loading") or cols.get("std_loading") or cols.get("standardized_loading")
        c_weight = cols.get("weight") or cols.get("outer_weight") or cols.get("outer_weights")

        if (c_construct is None) or (c_ind is None):
            best_lv = (None, -1)
            best_it = (None, -1)
            for c in DF.columns:
                s = DF[c].astype(str)
                lv_hit = int(s.isin(expected_lvs).sum())
                it_hit = int(s.isin(expected_inds).sum())
                if lv_hit > best_lv[1]:
                    best_lv = (c, lv_hit)
                if it_hit > best_it[1]:
                    best_it = (c, it_hit)
            c_construct = best_lv[0] if best_lv[1] > 0 else None
            c_ind = best_it[0] if best_it[1] > 0 else None

        if c_construct is None or c_ind is None:
            return None

        out = pd.DataFrame({
            "Construct": DF[c_construct].astype(str),
            "Indicator": DF[c_ind].astype(str),
        })
        out["OuterLoading"] = pd.to_numeric(DF[c_loading], errors="coerce") if c_loading else np.nan
        out["OuterWeight"] = pd.to_numeric(DF[c_weight], errors="coerce") if c_weight else np.nan

        out["Mode"] = out["Construct"].map(lambda lv: str(lv_modes.get(lv, "A")).upper())
        out["Mode"] = out["Mode"].map(lambda m: "A(reflective)" if m == "A" else "B(formative)")
        out["Communality(h2)"] = pd.to_numeric(out["OuterLoading"], errors="coerce") ** 2
        return out[["Construct", "Indicator", "Mode", "OuterLoading", "OuterWeight", "Communality(h2)"]]

    cand_names = ["outer_model", "crossloadings", "outer", "outer_summary", "measurement_model"]

    for nm in cand_names:
        if not hasattr(model, nm):
            continue
        raw = getattr(model, nm)

        DF = _to_df(raw)

        out = _try_long_table(DF)
        if out is not None and not out.empty:
            return out

        if DF is not None and _looks_like_matrix(DF):
            try:
                return _matrix_to_outer(DF)
            except Exception:
                pass

        obj = _maybe_call(raw)
        for sub in ["loadings", "weights", "outer_loadings", "outer_weights"]:
            if hasattr(obj, sub):
                sub_df = _to_df(getattr(obj, sub))
                if sub_df is not None and _looks_like_matrix(sub_df):
                    try:
                        tmp = _matrix_to_outer(sub_df)
                        if sub.lower().endswith("weights"):
                            tmp["OuterWeight"] = tmp["OuterLoading"]
                            tmp["OuterLoading"] = np.nan
                            tmp["Communality(h2)"] = np.nan
                        return tmp
                    except Exception:
                        pass

    if strict:
        available = [a for a in dir(model) if any(k in a.lower() for k in ["outer", "loading", "weight", "cross"])]
        raise AttributeError(
            "Cannot extract outer model from plspm model API. "
            f"Tried {cand_names}. Available outer-related attrs={available}"
        )

    cross = corr_items_vs_scores(X, scores_df, method="pearson")
    rows = []
    for lv, inds in lv_blocks.items():
        mode = str(lv_modes.get(lv, "A")).upper()
        for it in inds:
            it = str(it)
            lv = str(lv)
            if it not in cross.index or lv not in cross.columns:
                continue
            ol = float(cross.loc[it, lv])
            rows.append({
                "Construct": lv,
                "Indicator": it,
                "Mode": "A(reflective)" if mode == "A" else "B(formative)",
                "OuterLoading": ol if mode == "A" else np.nan,
                "OuterWeight": np.nan,
                "Communality(h2)": (ol ** 2) if mode == "A" else np.nan,
            })
    return pd.DataFrame(rows)


# =========================================================
# 4) Structural helpers (diagnostics)
# =========================================================
def preds_for_endogenous(path_df: pd.DataFrame, to: str) -> List[str]:
    return [c for c in path_df.columns if path_df.loc[to, c] == 1]


def ols_fit(y: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    OLS with intercept (listwise finite):
      y = b0 + b1 x1 + ... + bp xp
    Returns (beta, R2).
    If insufficient valid rows -> beta all NaN, R2 NaN.
    """
    y0 = np.asarray(y, dtype=float).ravel()
    X0 = np.asarray(X, dtype=float)

    if X0.ndim == 1:
        X0 = X0.reshape(-1, 1)

    if y0.shape[0] != X0.shape[0]:
        raise ValueError("ols_fit: y and X must have same number of rows.")

    ok = np.isfinite(y0) & np.isfinite(X0).all(axis=1)
    p = int(X0.shape[1])
    min_n = max(5, p + 2)

    if int(ok.sum()) < min_n:
        return np.full((p + 1,), np.nan, dtype=float), np.nan

    yv = y0[ok]
    Xv = X0[ok]
    X1 = np.column_stack([np.ones((Xv.shape[0], 1)), Xv])

    beta = np.linalg.lstsq(X1, yv, rcond=None)[0].astype(float)
    yhat = (X1 @ beta).astype(float)

    sse = float(np.sum((yv - yhat) ** 2))
    sst = float(np.sum((yv - np.mean(yv)) ** 2))
    r2 = 1 - sse / sst if sst > 1e-12 else np.nan
    return beta, float(r2)


def path_estimates_from_scores(scores_df: pd.DataFrame, path_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for to in path_df.index:
        preds = [p for p in preds_for_endogenous(path_df, to) if p in scores_df.columns]
        if len(preds) == 0 or to not in scores_df.columns:
            continue
        beta, _ = ols_fit(scores_df[to].values, scores_df[preds].values)
        if not np.isfinite(beta).all():
            continue
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
            f2 = (r2_full - r2_ex) / denom if (np.isfinite(r2_full) and np.isfinite(r2_ex) and denom > 1e-12) else np.nan
            f2_rows.append({"from": p, "to": to, "f2": f2, "R2_full": r2_full, "R2_excluded": r2_ex})

    return pd.DataFrame(r2_rows).round(4), pd.DataFrame(f2_rows).round(4)


def q2_cv_from_scores(
    scores_df: pd.DataFrame,
    path_df: pd.DataFrame,
    *,
    n_splits: int,
    seed: int,
) -> pd.DataFrame:
    """
    Cross-validated Q² (NOT SmartPLS blindfolding Q²).
    NaN-safe: listwise finite masking on test rows.
    """
    n = scores_df.shape[0]
    k = max(2, min(int(n_splits), int(n)))
    kf = KFold(n_splits=k, shuffle=True, random_state=int(seed))

    rows = []
    for to in path_df.index:
        preds = [p for p in preds_for_endogenous(path_df, to) if p in scores_df.columns]
        if len(preds) == 0 or to not in scores_df.columns:
            continue

        y = scores_df[to].to_numpy(dtype=float)
        X = scores_df[preds].to_numpy(dtype=float)

        sse, sso = 0.0, 0.0

        for tr, te in kf.split(X):
            beta, _ = ols_fit(y[tr], X[tr])
            if not np.isfinite(beta).all():
                continue

            Xte = X[te]
            yte = y[te]
            Xte1 = np.column_stack([np.ones((len(te), 1)), Xte])
            yhat = (Xte1 @ beta.reshape(-1, 1)).ravel()

            y_mean_tr = float(np.nanmean(y[tr]))  # train mean (NaN-safe)

            ok = np.isfinite(yte) & np.isfinite(yhat)
            if int(ok.sum()) == 0:
                continue

            sse += float(np.sum((yte[ok] - yhat[ok]) ** 2))
            sso += float(np.sum((yte[ok] - y_mean_tr) ** 2))

        q2 = 1 - sse / sso if sso > 1e-12 else np.nan
        rows.append({"Construct": to, "Q2_CV": q2, "Predictors": ", ".join(preds), "folds": k})

    return pd.DataFrame(rows).round(4)


def structural_vif(scores_df: pd.DataFrame, path_df: pd.DataFrame) -> pd.DataFrame:
    """
    Inner VIF (structural VIF) based on LV scores:
      VIF_x = 1 / (1 - R2_x|others)
    NaN-safe via ols_fit listwise.
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
            vif = 1.0 / (1.0 - r2) if (pd.notna(r2) and (1.0 - r2) > 1e-12) else np.inf
            rows.append({"Endogenous": to, "Predictor": x, "VIF": vif})

    out = pd.DataFrame(rows)
    if not out.empty:
        out["VIF"] = pd.to_numeric(out["VIF"], errors="coerce").round(3)
    return out


# =========================================================
# 5) Effects (direct/indirect/total)
# =========================================================
def effects_total_indirect(
    order_nodes: List[str],
    edges: List[Tuple[str, str]],
    coef_map: Dict[Tuple[str, str], float],
) -> Tuple[pd.DataFrame, List[Tuple[str, str]]]:
    """
    NaN-safe:
      - treat non-finite coefficients as 0, preventing NaN contagion across paths.
    """
    out_adj = {u: [] for u in order_nodes}
    for u, v in edges:
        if u in out_adj:
            out_adj[u].append(v)

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

    total = {}
    for s in order_nodes:
        eff = {n: 0.0 for n in order_nodes}
        eff[s] = 1.0
        for u in order_nodes:
            for v in out_adj.get(u, []):
                b = coef_map.get((u, v), 0.0)
                b = float(b) if (b is not None and np.isfinite(b)) else 0.0
                eff[v] += eff[u] * b
        for t in order_nodes:
            if t != s:
                total[(s, t)] = float(eff[t])

    rows = []
    for s, t in pairs:
        direct = coef_map.get((s, t), 0.0)
        direct = float(direct) if (direct is not None and np.isfinite(direct)) else 0.0
        tot = float(total.get((s, t), 0.0))
        indir = tot - direct
        rows.append({"from": s, "to": t, "direct": direct, "indirect": indir, "total": tot})

    return pd.DataFrame(rows), pairs
