# pls_project/pls_predict.py
from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from pls_project.pls_core import run_plspm_python


def _zscore_train_apply(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize using TRAIN mean/sd; return (Z_train, Z_test, sd)
    """
    mu = np.nanmean(X_train.to_numpy(dtype=float), axis=0)
    sd = np.nanstd(X_train.to_numpy(dtype=float), axis=0, ddof=0)
    sd = np.where(sd < 1e-12, np.nan, sd)

    Ztr = (X_train.to_numpy(dtype=float) - mu) / sd
    Zte = (X_test.to_numpy(dtype=float) - mu) / sd
    return Ztr, Zte, sd


def _ols_fit_predict(y_tr: np.ndarray, X_tr: np.ndarray, X_te: np.ndarray) -> np.ndarray:
    """
    OLS with intercept, listwise on train.
    """
    y_tr = np.asarray(y_tr, dtype=float).ravel()
    X_tr = np.asarray(X_tr, dtype=float)
    X_te = np.asarray(X_te, dtype=float)

    ok = np.isfinite(y_tr) & np.isfinite(X_tr).all(axis=1)
    if ok.sum() < max(8, X_tr.shape[1] + 2):
        return np.full((X_te.shape[0],), np.nan, dtype=float)

    X1 = np.column_stack([np.ones((ok.sum(), 1)), X_tr[ok]])
    beta = np.linalg.lstsq(X1, y_tr[ok], rcond=None)[0]  # (p+1,)

    Xte1 = np.column_stack([np.ones((X_te.shape[0], 1)), X_te])
    return (Xte1 @ beta).ravel()


def _estimate_weights_from_train_scores(
    X_train: pd.DataFrame,
    scores_train: pd.DataFrame,
    lv_blocks: Dict[str, List[str]],
    order: List[str],
) -> Dict[str, pd.Series]:
    """
    Estimate outer weights by regressing LV score on its indicators (train only), on standardized indicators.
    Works as a stable proxy for applying scoring weights to test set.
    """
    W: Dict[str, pd.Series] = {}
    for lv in order:
        inds = [c for c in lv_blocks.get(lv, []) if c in X_train.columns]
        if (not inds) or (lv not in scores_train.columns):
            continue

        Xt = X_train[inds].apply(pd.to_numeric, errors="coerce")
        yt = pd.to_numeric(scores_train[lv], errors="coerce").to_numpy(dtype=float)

        Ztr, _, _ = _zscore_train_apply(Xt, Xt)
        y = yt.copy()
        y = (y - np.nanmean(y)) / (np.nanstd(y) if np.nanstd(y) > 1e-12 else np.nan)

        ok = np.isfinite(y) & np.isfinite(Ztr).all(axis=1)
        if ok.sum() < max(8, Ztr.shape[1] + 2):
            continue

        # no intercept, since both are standardized
        beta = np.linalg.lstsq(Ztr[ok], y[ok], rcond=None)[0]
        W[lv] = pd.Series(beta, index=inds, name=lv)
    return W


def _scores_from_weights_trainstats(
    X_train: pd.DataFrame,
    X_any: pd.DataFrame,
    W: Dict[str, pd.Series],
) -> pd.DataFrame:
    """
    Build LV scores for X_any using TRAIN standardization and TRAIN-estimated weights.
    """
    out = {}
    for lv, w in W.items():
        inds = [c for c in w.index if c in X_any.columns and c in X_train.columns]
        if not inds:
            continue
        Xtr = X_train[inds].apply(pd.to_numeric, errors="coerce")
        Xte = X_any[inds].apply(pd.to_numeric, errors="coerce")
        Ztr, Zte, _ = _zscore_train_apply(Xtr, Xte)
        ww = pd.to_numeric(w.loc[inds], errors="coerce").to_numpy(dtype=float)
        out[lv] = (Zte @ ww)
    return pd.DataFrame(out, index=X_any.index)


def plspredict_indicator_cv(
    cog,
    *,
    X_items: pd.DataFrame,
    path_df: pd.DataFrame,
    lv_blocks: Dict[str, List[str]],
    lv_modes: Dict[str, str],
    order: List[str],
    n_splits: int = 10,
    seed: int = 0,
    exclude_endogenous: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    PLSpredict-style OOS prediction (indicator level):
      For each fold:
        - estimate PLS on TRAIN
        - estimate scoring weights on TRAIN (proxy)
        - score TEST using TRAIN weights
        - for each endogenous LV and each of its indicators:
            y_indicator ~ predictors' LV scores (TRAIN), predict TEST
            baseline = mean(y_train)

    Returns:
      detail_df: per indicator metrics
      summary_df: per endogenous metrics (averaged)
    """
    exclude_endogenous = set(exclude_endogenous or [])

    X = X_items.reset_index(drop=True).copy()
    n = int(X.shape[0])
    k = max(2, min(int(n_splits), n))

    kf = KFold(n_splits=k, shuffle=True, random_state=int(seed))

    # define endogenous LVs (have predictors in path_df)
    endo_lvs = []
    preds_map = {}
    for to in path_df.index:
        preds = [fr for fr in path_df.columns if int(path_df.loc[to, fr]) == 1]
        if preds:
            endo_lvs.append(to)
            preds_map[to] = preds

    endo_lvs = [lv for lv in endo_lvs if lv in order and lv not in exclude_endogenous]

    # accumulate errors
    # key = (endo_lv, indicator)
    acc = {}

    cfg_pls = cog.cfg.pls
    scaled = bool(getattr(cfg_pls, "PLS_STANDARDIZED", True))

    for fold, (tr, te) in enumerate(kf.split(X), start=1):
        Xtr = X.iloc[tr].reset_index(drop=True)
        Xte = X.iloc[te].reset_index(drop=True)

        # fit pls on train
        model_tr, scores_tr = run_plspm_python(
            cog,
            Xtr,
            path_df,
            lv_blocks,
            lv_modes,
            scaled=scaled,
        )
        scores_tr = scores_tr[order].copy()

        # estimate weights from train scores, then score full (train+test) using train stats
        W = _estimate_weights_from_train_scores(Xtr, scores_tr, lv_blocks, order)
        S_te = _scores_from_weights_trainstats(Xtr, Xte, W)
        S_tr = _scores_from_weights_trainstats(Xtr, Xtr, W)

        # predict indicators for each endogenous
        for y_lv in endo_lvs:
            preds = [p for p in preds_map.get(y_lv, []) if p in S_tr.columns and p in S_te.columns]
            if not preds:
                continue

            inds = [c for c in lv_blocks.get(y_lv, []) if c in X.columns]
            if not inds:
                continue

            Xpred_tr = S_tr[preds].to_numpy(dtype=float)
            Xpred_te = S_te[preds].to_numpy(dtype=float)

            for ind in inds:
                y_tr = pd.to_numeric(Xtr[ind], errors="coerce").to_numpy(dtype=float)
                y_te = pd.to_numeric(Xte[ind], errors="coerce").to_numpy(dtype=float)

                yhat = _ols_fit_predict(y_tr, Xpred_tr, Xpred_te)
                ybar = np.nanmean(y_tr)

                # valid test rows
                ok = np.isfinite(y_te) & np.isfinite(yhat)
                if ok.sum() == 0:
                    continue

                err_model = y_te[ok] - yhat[ok]
                err_naive = y_te[ok] - ybar

                key = (y_lv, ind)
                a = acc.setdefault(key, {"se_model": 0.0, "se_naive": 0.0, "ae_model": 0.0, "ae_naive": 0.0, "n": 0})
                a["se_model"] += float(np.sum(err_model ** 2))
                a["se_naive"] += float(np.sum(err_naive ** 2))
                a["ae_model"] += float(np.sum(np.abs(err_model)))
                a["ae_naive"] += float(np.sum(np.abs(err_naive)))
                a["n"] += int(ok.sum())

    # build outputs
    rows = []
    for (lv, ind), a in acc.items():
        nobs = int(a["n"])
        if nobs <= 0:
            continue
        rmse_m = np.sqrt(a["se_model"] / nobs)
        rmse_n = np.sqrt(a["se_naive"] / nobs)
        mae_m = a["ae_model"] / nobs
        mae_n = a["ae_naive"] / nobs
        q2_pred = 1.0 - (a["se_model"] / a["se_naive"]) if a["se_naive"] > 1e-12 else np.nan
        rows.append({
            "Endogenous": lv,
            "Indicator": ind,
            "n_test_total": nobs,
            "RMSE_model": rmse_m,
            "RMSE_naive": rmse_n,
            "dRMSE(model-naive)": rmse_m - rmse_n,
            "MAE_model": mae_m,
            "MAE_naive": mae_n,
            "dMAE(model-naive)": mae_m - mae_n,
            "Q2_predict": q2_pred,
        })

    detail = pd.DataFrame(rows).sort_values(["Endogenous", "Indicator"]).reset_index(drop=True)
    if detail.empty:
        return detail, pd.DataFrame()

    summary = detail.groupby("Endogenous", as_index=False).agg({
        "n_test_total": "sum",
        "RMSE_model": "mean",
        "RMSE_naive": "mean",
        "dRMSE(model-naive)": "mean",
        "MAE_model": "mean",
        "MAE_naive": "mean",
        "dMAE(model-naive)": "mean",
        "Q2_predict": "mean",
    })
    return detail, summary
