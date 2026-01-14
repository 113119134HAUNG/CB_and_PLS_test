# pls_project/pls_bootstrap.py
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import t as tdist  # 用 t 分配算 p（SmartPLS 會顯示 p；test type 可選） :contentReference[oaicite:5]{index=5}


def _safe_t_abs(est: np.ndarray, se: np.ndarray) -> np.ndarray:
    est = np.asarray(est, dtype=float)
    se = np.asarray(se, dtype=float)
    return np.divide(np.abs(est), se, out=np.full_like(est, np.nan), where=(se > 0))


def _ci_sig(ci_l: np.ndarray, ci_u: np.ndarray) -> np.ndarray:
    ci_l = np.asarray(ci_l, dtype=float)
    ci_u = np.asarray(ci_u, dtype=float)
    ok = np.isfinite(ci_l) & np.isfinite(ci_u)
    sig = np.full_like(ci_l, False, dtype=bool)
    sig[ok] = (ci_u[ok] < 0) | (ci_l[ok] > 0)
    return sig


def summarize_direct_ci(
    cog,
    keys_df: pd.DataFrame,
    point_est: np.ndarray,
    boot: np.ndarray
) -> pd.DataFrame:
    """
    SmartPLS-like bootstrap summary (Percentile CI):
      Original Sample (O), Sample Mean (M), STDEV, T Statistics (|O/STDEV|), P Values, CI, Sig
    """
    cfg = cog.cfg.pls

    # required config (no hidden defaults)
    for k in ["PAPER_DECIMALS", "BOOT_CI_LO", "BOOT_CI_HI", "BOOT_TEST_TYPE", "BOOT_ALPHA"]:
        if not hasattr(cfg, k):
            raise AttributeError(f"Config.pls must define {k}.")

    dec = int(cfg.PAPER_DECIMALS)
    qlo = float(cfg.BOOT_CI_LO)
    qhi = float(cfg.BOOT_CI_HI)
    test_type = str(cfg.BOOT_TEST_TYPE).strip().lower()
    alpha = float(cfg.BOOT_ALPHA)

    point_est = np.asarray(point_est, dtype=float)
    boot = np.asarray(boot, dtype=float)

    # bootstrap mean / se
    boot_mean = np.nanmean(boot, axis=0)
    se = np.nanstd(boot, axis=0, ddof=1)

    # t = |O| / SE
    t_abs = _safe_t_abs(point_est, se)

    # p-values (t distribution approximation; df = valid_boot_n - 1 per coefficient)
    valid_n = np.sum(np.isfinite(boot), axis=0).astype(int)
    df = np.maximum(valid_n - 1, 1)

    if test_type == "two-tailed":
        p = 2.0 * tdist.sf(t_abs, df=df)
    elif test_type == "one-tailed":
        p = tdist.sf(t_abs, df=df)
    else:
        raise ValueError("BOOT_TEST_TYPE must be 'one-tailed' or 'two-tailed'.")

    # percentile CI
    ci_l = np.nanquantile(boot, qlo, axis=0)
    ci_u = np.nanquantile(boot, qhi, axis=0)
    sig = _ci_sig(ci_l, ci_u)

    lo_label = f"CI{qlo*100:.1f}"
    hi_label = f"CI{qhi*100:.1f}"

    out = keys_df.copy()
    out["Original Sample (O)"] = point_est
    out["Sample Mean (M)"] = boot_mean
    out["STDEV"] = se
    out["T Statistics (|O/STDEV|)"] = t_abs
    out["P Values"] = p
    out[lo_label] = ci_l
    out[hi_label] = ci_u
    out["Sig(CI)"] = sig
    out["Sig(p)"] = (p < alpha) if np.isfinite(alpha) else np.nan
    return out.round(dec)
