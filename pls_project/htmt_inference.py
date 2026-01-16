# pls_project/htmt_inference.py
from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from pls_project.pls_core import htmt_matrix


def htmt_inference_bootstrap(
    X_items: pd.DataFrame,
    group_items: Dict[str, List[str]],
    groups: List[str],
    *,
    B: int,
    seed: int,
    qlo: float = 0.025,
    qhi: float = 0.975,
    threshold: float = 0.90,
    corr_method: str = "pearson",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Bootstrap HTMT to get percentile CI.
    Return:
      detail_long: ConstructA, ConstructB, HTMT, CI_lo, CI_hi, Pass(CI_hi<th)
      summary: counts
    """
    X = X_items.reset_index(drop=True)
    rng = np.random.default_rng(int(seed))
    G = list(groups)

    # point estimate
    ht0 = htmt_matrix(X, group_items, G, method=corr_method)

    pairs = []
    for i in range(len(G)):
        for j in range(i + 1, len(G)):
            pairs.append((G[i], G[j]))

    boot = {p: [] for p in pairs}
    n = X.shape[0]

    for _ in range(int(B)):
        idx = rng.integers(0, n, size=n)
        Xb = X.iloc[idx].reset_index(drop=True)
        hb = htmt_matrix(Xb, group_items, G, method=corr_method)
        for a, b in pairs:
            boot[(a, b)].append(float(hb.loc[a, b]))

    rows = []
    for a, b in pairs:
        arr = np.asarray(boot[(a, b)], dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            ci_l = ci_u = np.nan
        else:
            ci_l = float(np.quantile(arr, qlo))
            ci_u = float(np.quantile(arr, qhi))
        ht = float(ht0.loc[a, b])
        rows.append({
            "ConstructA": a,
            "ConstructB": b,
            "HTMT": ht,
            "CI_lo": ci_l,
            "CI_hi": ci_u,
            "Threshold": float(threshold),
            "Pass(CI_hi<th)": (pd.notna(ci_u) and ci_u < threshold),
        })

    detail = pd.DataFrame(rows)
    summary = pd.DataFrame([{
        "B": int(B),
        "qlo": float(qlo),
        "qhi": float(qhi),
        "threshold": float(threshold),
        "n_pairs": int(len(detail)),
        "n_pass": int(detail["Pass(CI_hi<th)"].fillna(False).sum()),
        "pass_rate": float(detail["Pass(CI_hi<th)"].fillna(False).mean()),
    }])

    return detail, summary
