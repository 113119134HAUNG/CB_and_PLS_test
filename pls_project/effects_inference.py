# pls_project/effects_inference.py
from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from pls_project.pls_core import effects_total_indirect


def edges_from_path_df(path_df: pd.DataFrame) -> List[Tuple[str, str]]:
    edges = []
    for to in path_df.index:
        for fr in path_df.columns:
            if int(path_df.loc[to, fr]) == 1:
                edges.append((fr, to))
    return edges


def coef_map_from_key(key_df: pd.DataFrame, est: np.ndarray) -> Dict[Tuple[str, str], float]:
    m = {}
    for (fr, to), b in zip(key_df[["from", "to"]].values.tolist(), est):
        m[(str(fr), str(to))] = float(b)
    return m


def summarize_effects_bootstrap_ci(
    *,
    order: List[str],
    edges: List[Tuple[str, str]],
    key_df: pd.DataFrame,
    point_est: np.ndarray,
    boot: np.ndarray,
    qlo: float = 0.025,
    qhi: float = 0.975,
    alpha: float = 0.05,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build:
      effects_point: from,to,direct,indirect,total,VAF
      effects_ci: CI for indirect/total + sig
      mediation: subset with indirect != 0 and (direct != 0 or meaningful), with VAF label
    """
    coef0 = coef_map_from_key(key_df, point_est)
    eff0, pairs = effects_total_indirect(order, edges, coef0)
    if eff0 is None or eff0.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    eff0 = eff0.copy()
    eff0["VAF"] = np.where(np.abs(eff0["total"]) > 1e-12, eff0["indirect"] / eff0["total"], np.nan)

    # bootstrap arrays aligned to eff0 rows
    B = boot.shape[0]
    indir_boot = np.full((B, len(eff0)), np.nan, dtype=float)
    total_boot = np.full((B, len(eff0)), np.nan, dtype=float)

    # merge template
    tmpl = eff0[["from", "to"]].copy()

    for b in range(B):
        coef_b = coef_map_from_key(key_df, boot[b, :])
        eff_b, _ = effects_total_indirect(order, edges, coef_b)
        if eff_b is None or eff_b.empty:
            continue
        mb = tmpl.merge(eff_b[["from", "to", "indirect", "total"]], on=["from", "to"], how="left")
        indir_boot[b, :] = pd.to_numeric(mb["indirect"], errors="coerce").to_numpy()
        total_boot[b, :] = pd.to_numeric(mb["total"], errors="coerce").to_numpy()

    def _ci(arr2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        lo = np.nanquantile(arr2d, qlo, axis=0)
        hi = np.nanquantile(arr2d, qhi, axis=0)
        return lo, hi

    il, iu = _ci(indir_boot)
    tl, tu = _ci(total_boot)

    sig_ind = (iu < 0) | (il > 0)
    sig_tot = (tu < 0) | (tl > 0)

    effects_ci = eff0[["from", "to"]].copy()
    effects_ci["indirect_CI_lo"] = il
    effects_ci["indirect_CI_hi"] = iu
    effects_ci["Sig_indirect(CI)"] = sig_ind
    effects_ci["total_CI_lo"] = tl
    effects_ci["total_CI_hi"] = tu
    effects_ci["Sig_total(CI)"] = sig_tot

    # mediation table (VAF)
    med = eff0.copy()
    med = med[np.isfinite(med["VAF"])].copy()
    # typical VAF heuristics (report as descriptive, not strict “test”)
    def vaf_label(v):
        if not np.isfinite(v):
            return ""
        av = abs(v)
        if av < 0.20:
            return "none/weak"
        if av < 0.80:
            return "partial"
        return "mostly/full"

    med["VAF_label"] = med["VAF"].map(vaf_label)

    # attach indirect CI sig for mediation
    med = med.merge(effects_ci[["from", "to", "indirect_CI_lo", "indirect_CI_hi", "Sig_indirect(CI)"]], on=["from", "to"], how="left")

    return eff0, effects_ci, med
