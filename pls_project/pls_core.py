# pls_project/pls_core.py
from __future__ import annotations

import sys
import subprocess
import warnings
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
      - no retries with alternative hyperparams
      - no PCA approximation branch
      - optional auto-install controlled by cfg.pls.AUTO_INSTALL_PLSPM (if present), else False
    """
    cfg = cog.cfg.pls

    scheme = _scheme_to_plspm_scheme_strict(getattr(cfg, "PLS_SCHEME", "PATH"))

    # Read required numeric settings from config (no hidden defaults)
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

    # matrix form (index=to, columns=from)
    if M is not None and (set(path_df.index).issubset(set(M.index)) and set(path_df.columns).issubset(set(M.columns))):
        rows = []
        for to in path_df.index:
            for fr in path_df.columns:
                if int(path_df.loc[to, fr]) == 1:
                    rows.append({"from": fr, "to": to, "estimate": float(M.loc[to, fr])})
        return pd.DataFrame(rows)

    # long form
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

    Tries, in order:
      1) model.outer_model() / model.crossloadings() as "long table"
      2) same candidates as "matrix" (indicator x construct)
      3) if candidate object has .loadings/.weights attributes -> use those matrices
      4) strict=True -> raise with debug info
         strict=False -> corr fallback
    """

    expected_lvs = list(scores_df.columns) if scores_df is not None else list(lv_blocks.keys())
    expected_lvs = [lv for lv in expected_lvs if lv in lv_blocks or lv in expected_lvs]
    expected_inds = []
    for lv in expected_lvs:
        for it in lv_blocks.get(lv, []):
            if it not in expected_inds:
                expected_inds.append(it)

    def _to_df(obj) -> Optional[pd.DataFrame]:
        if obj is None:
            return None
        obj = _maybe_call(obj)

        if isinstance(obj, pd.DataFrame):
            return obj

        # if it's list/tuple of dicts/objects
        if isinstance(obj, (list, tuple)) and len(obj) > 0:
            first = obj[0]
            if isinstance(first, dict):
                try:
                    return pd.DataFrame(obj)
                except Exception:
                    return None
            # namedtuple / dataclass / row object
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

        # common wrappers
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

        # conversion methods
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
        # indicator x construct
        ind_hit = len(set(expected_inds) & set(map(str, DF.index)))
        lv_hit = len(set(expected_lvs) & set(map(str, DF.columns)))
        if ind_hit > 0 and lv_hit > 0:
            return True
        # maybe transposed
        ind_hit2 = len(set(expected_inds) & set(map(str, DF.columns)))
        lv_hit2 = len(set(expected_lvs) & set(map(str, DF.index)))
        return (ind_hit2 > 0 and lv_hit2 > 0)

    def _matrix_to_outer(M: pd.DataFrame) -> pd.DataFrame:
        # normalize orientation
        if set(expected_inds).issubset(set(map(str, M.index))) and set(expected_lvs).issubset(set(map(str, M.columns))):
            Mat = M.copy()
        elif set(expected_lvs).issubset(set(map(str, M.index))) and set(expected_inds).issubset(set(map(str, M.columns))):
            Mat = M.T.copy()
        else:
            # shape-based fallback
            A = M.to_numpy()
            if A.shape == (len(expected_inds), len(expected_lvs)):
                Mat = pd.DataFrame(A, index=expected_inds, columns=expected_lvs)
            elif A.shape == (len(expected_lvs), len(expected_inds)):
                Mat = pd.DataFrame(A.T, index=expected_inds, columns=expected_lvs)
            else:
                raise ValueError("Cannot align matrix to indicators x constructs.")

        rows = []
        for lv in expected_lvs:
            mode = str(lv_modes.get(lv, "A")).upper()
            for it in lv_blocks.get(lv, []):
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
            # attempt overlap-based inference
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
        return out[["Construct","Indicator","Mode","OuterLoading","OuterWeight","Communality(h2)"]]

    # candidates to try
    cand_names = ["outer_model", "crossloadings", "outer", "outer_summary", "measurement_model"]

    for nm in cand_names:
        if not hasattr(model, nm):
            continue
        raw = getattr(model, nm)

        # unwrap
        DF = _to_df(raw)

        # 1) long-table try
        out = _try_long_table(DF)
        if out is not None and not out.empty:
            return out

        # 2) matrix try
        if DF is not None and _looks_like_matrix(DF):
            try:
                return _matrix_to_outer(DF)
            except Exception:
                pass

        # 3) object might contain .loadings / .weights
        obj = _maybe_call(raw)
        for sub in ["loadings", "weights", "outer_loadings", "outer_weights"]:
            if hasattr(obj, sub):
                sub_df = _to_df(getattr(obj, sub))
                if sub_df is not None and _looks_like_matrix(sub_df):
                    try:
                        tmp = _matrix_to_outer(sub_df)
                        # treat weights if available
                        if sub.lower().endswith("weights"):
                            tmp["OuterWeight"] = tmp["OuterLoading"]
                            tmp["OuterLoading"] = np.nan
                            tmp["Communality(h2)"] = np.nan
                        return tmp
                    except Exception:
                        pass

    if strict:
        # show what we tried for faster debugging
        available = [a for a in dir(model) if any(k in a.lower() for k in ["outer","loading","weight","cross"])]
        raise AttributeError(
            "Cannot extract outer model from plspm model API. "
            f"Tried {cand_names}. Available outer-related attrs={available}"
        )

    # last resort: correlation fallback
    cross = corr_items_vs_scores(X, scores_df, method="pearson")
    rows = []
    for lv, inds in lv_blocks.items():
        mode = str(lv_modes.get(lv, "A")).upper()
        for it in inds:
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
    scores_df: pd.DataFrame,
    path_df: pd.DataFrame,
    *,
    n_splits: int,
    seed: int,
) -> pd.DataFrame:
    """
    Cross-validated Q² (NOT SmartPLS blindfolding Q²).
    """
    n = scores_df.shape[0]
    k = max(2, min(int(n_splits), int(n)))
    kf = KFold(n_splits=k, shuffle=True, random_state=int(seed))

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
# 5) Effects (direct/indirect/total)
# =========================================================
def effects_total_indirect(
    order_nodes: List[str],
    edges: List[Tuple[str, str]],
    coef_map: Dict[Tuple[str, str], float],
) -> Tuple[pd.DataFrame, List[Tuple[str, str]]]:
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
