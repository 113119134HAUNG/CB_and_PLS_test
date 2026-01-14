# pls_project/paper.py
from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.decomposition import FactorAnalysis
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity


def _decimals_from_cfg(cog) -> int:
    if not hasattr(cog.cfg.pls, "PAPER_DECIMALS"):
        raise AttributeError("Config.pls must define PAPER_DECIMALS (clean mode).")
    return int(cog.cfg.pls.PAPER_DECIMALS)


def cronbach_alpha(items_df: pd.DataFrame) -> float:
    X = items_df.dropna().astype(float)
    k = X.shape[1]
    n = X.shape[0]
    if k < 2 or n < 2:
        return np.nan
    variances = X.var(axis=0, ddof=1)
    total_var = X.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return np.nan
    return float((k / (k - 1)) * (1 - variances.sum() / total_var))


def mcdonald_omega_total(items_df: pd.DataFrame, seed: int = 0) -> float:
    """
    Omega total via 1-factor FA on standardized items.
    Clean: no clipping constants. If uniqueness invalid -> NaN.
    """
    X = items_df.dropna().astype(float)
    k = X.shape[1]
    n = X.shape[0]
    if k < 3 or n < 5:
        return np.nan
    sd = X.std(ddof=1)
    if (sd == 0).any():
        return np.nan
    Z = (X - X.mean()) / sd

    try:
        fa = FactorAnalysis(n_components=1, random_state=seed)
        fa.fit(Z.values)
        loadings = fa.components_.T[:, 0]
        uniq = fa.noise_variance_.astype(float)

        if np.any(~np.isfinite(uniq)) or np.any(uniq <= 0):
            return np.nan

        lam = np.abs(loadings)
        return float((lam.sum() ** 2) / ((lam.sum() ** 2) + uniq.sum()))
    except Exception:
        return np.nan


def reliability_summary(cog, df_valid: pd.DataFrame, groups, group_items, item_cols) -> pd.DataFrame:
    seed = int(cog.cfg.pls.PLS_SEED)
    dec = _decimals_from_cfg(cog)

    rows = []
    for g in groups:
        cols = group_items[g]
        rows.append({
            "Construct": g,
            "k(items)": len(cols),
            "n(complete)": int(df_valid[cols].dropna().shape[0]),
            "Cronbach α": cronbach_alpha(df_valid[cols]),
            "McDonald ωt": mcdonald_omega_total(df_valid[cols], seed=seed)
        })
    rows.append({
        "Construct": "ALL",
        "k(items)": len(item_cols),
        "n(complete)": int(df_valid[item_cols].dropna().shape[0]),
        "Cronbach α": cronbach_alpha(df_valid[item_cols]),
        "McDonald ωt": mcdonald_omega_total(df_valid[item_cols], seed=seed)
    })
    out = pd.DataFrame(rows)
    out[["Cronbach α", "McDonald ωt"]] = out[["Cronbach α", "McDonald ωt"]].round(dec)
    return out


def item_analysis_table(cog, df_valid: pd.DataFrame, groups, group_items) -> pd.DataFrame:
    seed = int(cog.cfg.pls.PLS_SEED)
    dec = _decimals_from_cfg(cog)

    rows = []
    for g in groups:
        X = df_valid[group_items[g]].dropna().astype(float)
        if X.shape[0] < 3:
            continue
        total = X.sum(axis=1)
        for c in X.columns:
            item = X[c]
            rest_total = total - item
            citc = item.corr(rest_total)
            alpha_deleted = cronbach_alpha(X.drop(columns=[c]))
            omega_deleted = mcdonald_omega_total(X.drop(columns=[c]), seed=seed)
            rows.append({
                "Construct": g,
                "Item": c,
                "n(complete)": int(X.shape[0]),
                "Mean": float(item.mean()),
                "SD": float(item.std(ddof=1)),
                "CITC": float(citc) if pd.notna(citc) else np.nan,
                "α if deleted": alpha_deleted,
                "ωt if deleted": omega_deleted
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out[["Mean", "SD", "CITC", "α if deleted", "ωt if deleted"]] = out[
            ["Mean", "SD", "CITC", "α if deleted", "ωt if deleted"]
        ].round(dec)
    return out


def one_factor_fa_table(cog, df_valid: pd.DataFrame, groups, group_items) -> pd.DataFrame:
    seed = int(cog.cfg.pls.PLS_SEED)
    dec = _decimals_from_cfg(cog)

    rows = []
    for g in groups:
        X = df_valid[group_items[g]].dropna().astype(float)
        k = X.shape[1]
        n = X.shape[0]
        if k < 3 or n < 5:
            continue
        sd = X.std(ddof=1)
        if (sd == 0).any():
            continue
        Z = (X - X.mean()) / sd
        fa = FactorAnalysis(n_components=1, random_state=seed)
        fa.fit(Z.values)
        lam = fa.components_.T[:, 0]
        if np.nansum(lam) < 0:
            lam = -lam
        psi = fa.noise_variance_.astype(float)
        h2 = lam ** 2
        for i, it in enumerate(X.columns):
            rows.append({
                "Construct": g, "Item": it, "n(complete)": int(n),
                "Loading": float(lam[i]), "h2": float(h2[i]), "psi": float(psi[i])
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out[["Loading", "h2", "psi"]] = out[["Loading", "h2", "psi"]].round(dec)
    return out


def one_factor_efa_table(cog, df_valid: pd.DataFrame, groups, group_items):
    dec = _decimals_from_cfg(cog)

    rows, fit_rows = [], []
    for g in groups:
        X = df_valid[group_items[g]].dropna().astype(float)
        k = X.shape[1]
        n = X.shape[0]
        if k < 3 or n < 5:
            continue

        try:
            _, kmo_model = calculate_kmo(X)
            _, bart_p = calculate_bartlett_sphericity(X)
        except Exception:
            kmo_model, bart_p = np.nan, np.nan

        try:
            fa = FactorAnalyzer(n_factors=1, rotation=None, method="minres")
            fa.fit(X)
            L = fa.loadings_[:, 0]
            if np.nansum(L) < 0:
                L = -L
            h2 = fa.get_communalities()
            psi = fa.get_uniquenesses()
        except Exception:
            L = np.full(k, np.nan)
            h2 = np.full(k, np.nan)
            psi = np.full(k, np.nan)

        fit_rows.append({
            "Construct": g, "n(complete)": int(n), "k(items)": int(k),
            "KMO": float(kmo_model) if pd.notna(kmo_model) else np.nan,
            "Bartlett_p": float(bart_p) if pd.notna(bart_p) else np.nan
        })

        for i, it in enumerate(X.columns):
            rows.append({
                "Construct": g, "Item": it, "n(complete)": int(n),
                "EFA Loading": float(L[i]) if np.isfinite(L[i]) else np.nan,
                "h2": float(h2[i]) if np.isfinite(h2[i]) else np.nan,
                "psi": float(psi[i]) if np.isfinite(psi[i]) else np.nan
            })

    out = pd.DataFrame(rows)
    fit = pd.DataFrame(fit_rows)
    if not out.empty:
        out[["EFA Loading", "h2", "psi"]] = out[["EFA Loading", "h2", "psi"]].round(dec)
    if not fit.empty:
        fit[["KMO", "Bartlett_p"]] = fit[["KMO", "Bartlett_p"]].round(dec)
    return out, fit


def run_cfa(cog, df_valid, groups, group_items, item_cols):
    cfg = cog.cfg.cfa
    if not cfg.RUN_CFA:
        return (pd.DataFrame([{"RUN_CFA": False}]), pd.DataFrame(), pd.DataFrame())

    dec = _decimals_from_cfg(cog)

    try:
        from semopy import Model, calc_stats
    except Exception as e:
        return (
            pd.DataFrame([{"Info": "CFA failed (semopy not available)", "Error": str(e)}]),
            pd.DataFrame([{"CFA_error": str(e)}]),
            pd.DataFrame([{"Info": "CFA failed (semopy not available)", "Error": str(e)}]),
        )

    Xcfa = df_valid[item_cols].copy().astype(float)
    if cfg.CFA_MISSING == "mean":
        # 這會生成新值；若你要完全乾淨請在 Config 設 listwise/none
        Xcfa = Xcfa.apply(lambda s: s.fillna(s.mean()), axis=0)
    else:
        Xcfa = Xcfa.dropna()

    model_lines = []
    for g in groups:
        model_lines.append(f"{g} =~ " + " + ".join(group_items[g]))
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            model_lines.append(f"{groups[i]} ~~ {groups[j]}")
    CFA_Model = "\n".join(model_lines)

    try:
        mod = Model(CFA_Model)
        mod.fit(Xcfa, obj=cfg.CFA_OBJ)

        est = mod.inspect(std_est=True, se_robust=cfg.CFA_ROBUST_SE).copy()
        meas = est[(est["op"] == "~") & (est["rval"].isin(groups)) & (est["lval"].isin(item_cols))].copy()

        std_candidates = ["Std. Ests", "Est. Std", "Std.Est", "Std. Estimate", "Std. Est", "std_est"]
        std_col = next((c for c in std_candidates if c in meas.columns), None)
        used_std = True
        if std_col is None:
            std_col = "Estimate"
            used_std = False

        CFA_Loadings = meas.rename(columns={"rval": "Construct", "lval": "Item", std_col: "Std_Loading"})[
            ["Construct", "Item", "Std_Loading"]
        ].copy()
        CFA_Loadings["Std_Loading"] = pd.to_numeric(CFA_Loadings["Std_Loading"], errors="coerce")
        CFA_Loadings["Flag_|loading|>1"] = (CFA_Loadings["Std_Loading"].abs() > 1.0) if used_std else np.nan
        CFA_Loadings["Std_Loading"] = CFA_Loadings["Std_Loading"].round(dec)
        CFA_Loadings["Construct"] = pd.Categorical(CFA_Loadings["Construct"], categories=groups, ordered=True)
        CFA_Loadings = CFA_Loadings.sort_values(["Construct", "Item"]).reset_index(drop=True)

        stats = calc_stats(mod)
        if isinstance(stats, pd.DataFrame):
            tmp = stats.T.reset_index().rename(columns={"index": "FitIndex", "Value": "Value"})
        else:
            tmp = pd.DataFrame(list(stats.items()), columns=["FitIndex", "Value"])
        tmp["Value"] = pd.to_numeric(tmp["Value"], errors="coerce").round(4)
        CFA_Fit = tmp

        CFA_Info = pd.DataFrame([{
            "n(CFA)": int(Xcfa.shape[0]),
            "k(items)": int(Xcfa.shape[1]),
            "obj": cfg.CFA_OBJ,
            "missing": cfg.CFA_MISSING,
            "std_loading_col_used": std_col,
            "std_loading_is_standardized": used_std
        }])
        return CFA_Loadings, CFA_Info, CFA_Fit

    except Exception as e:
        return (
            pd.DataFrame([{"Info": "CFA failed", "Error": str(e)}]),
            pd.DataFrame([{"CFA_error": str(e)}]),
            pd.DataFrame([{"Info": "CFA failed", "Error": str(e)}]),
        )
