# pls_project/pls_core.py (add helpers)

from __future__ import annotations
import numpy as np
import pandas as pd

def _maybe_call(x):
    return x() if callable(x) else x

def _first_existing_attr(obj, names: list[str]):
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return None

def _ensure_df(x) -> pd.DataFrame | None:
    if x is None:
        return None
    if isinstance(x, pd.DataFrame):
        return x
    try:
        return pd.DataFrame(x)
    except Exception:
        return None

def get_sign_map_by_anchors(X_df: pd.DataFrame, scores_df: pd.DataFrame, anchors: dict) -> dict:
    """
    回傳每個 LV 的 sign (+1/-1)，以 anchor 與 LV score 的相關為準。
    之後要把 outer/path 也一起做 sign 對齊。
    """
    sign = {}
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
    s_to   = out["to"].map(lambda x: sign_map.get(x, 1)).astype(float)
    out["estimate"] = out["estimate"].astype(float) * s_to * s_from
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

def get_path_results(model, path_df: pd.DataFrame, scores_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    優先從 plspm model 取 path coefficients；取不到才 fallback 用 OLS(scores)。
    回傳 long format: from,to,estimate
    """
    # ---- try model APIs ----
    cand = _first_existing_attr(model, ["path_coefficients", "path_coefs", "inner_model", "inner_summary"])
    cand = _maybe_call(cand)
    M = _ensure_df(cand)

    # 常見：M 是 matrix（index=to, columns=from）
    if M is not None and (set(path_df.index).issubset(set(M.index)) and set(path_df.columns).issubset(set(M.columns))):
        rows = []
        for to in path_df.index:
            for fr in path_df.columns:
                if int(path_df.loc[to, fr]) == 1:
                    rows.append({"from": fr, "to": to, "estimate": float(M.loc[to, fr])})
        return pd.DataFrame(rows)

    # 有些版本可能提供 long table
    # 嘗試找欄位名
    if M is not None:
        cols = {c.lower(): c for c in M.columns}
        if ("from" in cols and "to" in cols) and ("estimate" in cols or "path" in cols or "coef" in cols):
            est_col = cols.get("estimate") or cols.get("path") or cols.get("coef")
            out = M.rename(columns={cols["from"]:"from", cols["to"]:"to", est_col:"estimate"})[["from","to","estimate"]]
            # 只保留 path_df 真的有的邊
            keep = []
            for _, r in out.iterrows():
                fr, to = r["from"], r["to"]
                if (to in path_df.index) and (fr in path_df.columns) and int(path_df.loc[to, fr]) == 1:
                    keep.append(True)
                else:
                    keep.append(False)
            return out.loc[keep].reset_index(drop=True)

    # ---- fallback: OLS on scores ----
    if scores_df is None:
        return pd.DataFrame()
    from .pls_core import path_estimates_from_scores  # 或直接呼叫同檔案內函數
    return path_estimates_from_scores(scores_df, path_df)

def get_outer_results(
    model,
    X: pd.DataFrame,
    scores_df: pd.DataFrame,
    lv_blocks: dict,
    lv_modes: dict,
) -> pd.DataFrame:
    """
    優先從 plspm model 取 outer loadings/weights；取不到才 fallback 用 corr(indicator, LVscore)。
    回傳欄位：
      Construct, Indicator, Mode, OuterLoading, OuterWeight, Communality(h2)
    """
    # ---- try model.outer_model ----
    cand = _first_existing_attr(model, ["outer_model", "outer", "outer_summary", "measurement_model"])
    cand = _maybe_call(cand)
    OM = _ensure_df(cand)

    if OM is not None and not OM.empty:
        # 嘗試自動對齊欄名（不同 plspm 版本命名會不一樣）
        cols = {c.lower(): c for c in OM.columns}
        c_construct = cols.get("block") or cols.get("construct") or cols.get("lv") or cols.get("latent") or cols.get("name_lv")
        c_ind      = cols.get("name") or cols.get("indicator") or cols.get("mv") or cols.get("manifest") or cols.get("name_mv")
        c_loading  = cols.get("loading") or cols.get("outer_loading")
        c_weight   = cols.get("weight") or cols.get("outer_weight")

        if c_construct and c_ind:
            out = pd.DataFrame({
                "Construct": OM[c_construct].astype(str),
                "Indicator": OM[c_ind].astype(str),
            })
            if c_loading and c_loading in OM.columns:
                out["OuterLoading"] = pd.to_numeric(OM[c_loading], errors="coerce")
            else:
                out["OuterLoading"] = np.nan
            if c_weight and c_weight in OM.columns:
                out["OuterWeight"] = pd.to_numeric(OM[c_weight], errors="coerce")
            else:
                out["OuterWeight"] = np.nan

            out["Mode"] = out["Construct"].map(lambda lv: str(lv_modes.get(lv, "A")).upper())
            out["Mode"] = out["Mode"].map(lambda m: "A(reflective)" if m == "A" else "B(formative)")
            out["Communality(h2)"] = out["OuterLoading"] ** 2
            return out

    # ---- fallback: corr(indicator, LVscore) ----
    # reflective: 用 corr 作為 loading 的備援
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
