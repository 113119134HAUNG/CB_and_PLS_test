# pls_project/sign_fix.py
from __future__ import annotations
import pandas as pd

def choose_anchors_by_max_abs_loading(X_df: pd.DataFrame, scores_df: pd.DataFrame, lv_blocks: dict) -> dict:
    anchors = {}
    for lv, inds in lv_blocks.items():
        if lv not in scores_df.columns:
            continue
        best_it = None
        best_abs = -1
        for it in inds:
            if it not in X_df.columns:
                continue
            r = X_df[it].corr(scores_df[lv], min_periods=3)
            if pd.isna(r):
                continue
            if abs(r) > best_abs:
                best_abs = abs(r)
                best_it = it
        if best_it is not None:
            anchors[lv] = best_it
    return anchors

def sign_fix_scores_by_anchors(scores_df: pd.DataFrame, X_df: pd.DataFrame, anchors: dict) -> pd.DataFrame:
    out = scores_df.copy()
    for lv, a in anchors.items():
        if lv not in out.columns or a not in X_df.columns:
            continue
        r = X_df[a].corr(out[lv], min_periods=3)
        if pd.notna(r) and r < 0:
            out[lv] = -out[lv]
    return out
