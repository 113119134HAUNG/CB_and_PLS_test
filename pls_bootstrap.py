# pls_bootstrap.py #
import numpy as np
import pandas as pd

def _safe_t(est: np.ndarray, se: np.ndarray) -> np.ndarray:
    est = np.asarray(est, dtype=float)
    se = np.asarray(se, dtype=float)
    return np.divide(np.abs(est), se, out=np.full_like(est, np.nan), where=(se > 0))

def _ci_sig(ci_l: np.ndarray, ci_u: np.ndarray) -> np.ndarray:
    ci_l = np.asarray(ci_l, dtype=float)
    ci_u = np.asarray(ci_u, dtype=float)
    # significant if CI does NOT include 0
    return (ci_l > 0) | (ci_u < 0)

def summarize_direct_ci(
    cog,
    keys_df: pd.DataFrame,
    point_est: np.ndarray,
    boot: np.ndarray
) -> pd.DataFrame:
    """
    Bootstrap summary (distribution-free, CI-based significance):
      estimate, boot_mean, boot_se, t, CI(lo), CI(hi), Sig
    Sig: CI does not include 0
    """
    cfg = cog.cfg.pls
    dec = int(getattr(cfg, "PAPER_DECIMALS", 3))
    qlo = float(getattr(cfg, "BOOT_CI_LO", 0.025))
    qhi = float(getattr(cfg, "BOOT_CI_HI", 0.975))

    point_est = np.asarray(point_est, dtype=float)
    se = np.nanstd(boot, axis=0, ddof=1)
    t = _safe_t(point_est, se)

    ci_l = np.nanquantile(boot, qlo, axis=0)
    ci_u = np.nanquantile(boot, qhi, axis=0)
    sig = _ci_sig(ci_l, ci_u)

    lo_label = f"CI{qlo*100:.1f}"
    hi_label = f"CI{qhi*100:.1f}"

    out = keys_df.copy()
    out["estimate"] = point_est
    out["boot_mean"] = np.nanmean(boot, axis=0)
    out["boot_se"] = se
    out["t"] = t
    out[lo_label] = ci_l
    out[hi_label] = ci_u
    out["Sig"] = sig
    return out.round(dec)
