# pls_project/schema.py
from __future__ import annotations
from dataclasses import replace
from typing import Sequence, Optional, Dict, List, Tuple
import re
import difflib
import pandas as pd

from .config import Config, ColumnConfig
from .rulebook import META_RULES, PROFILES

def _norm_cols(columns: Sequence[str]) -> List[str]:
    return [str(c).strip().replace("\u3000", " ") for c in columns]

def _first_match(cols: List[str], patterns: List[str], mode: str) -> Optional[str]:
    if mode == "exact":
        for p in patterns:
            for c in cols:
                if c == p:
                    return c
    elif mode == "contains":
        for p in patterns:
            for c in cols:
                if p in c:
                    return c
    elif mode == "regex":
        for p in patterns:
            for c in cols:
                if re.search(p, c, flags=re.I):
                    return c
    return None

def _fuzzy(cols: List[str], hint: str) -> Optional[str]:
    m = difflib.get_close_matches(hint, cols, n=1, cutoff=0.72)
    return m[0] if m else None

def resolve_columns(col_cfg: ColumnConfig, columns: Sequence[str]) -> ColumnConfig:
    cols = _norm_cols(columns)

    def pick(current: str, field: str) -> str:
        if current in cols:
            return current
        rule = META_RULES[field]
        hit = (
            _first_match(cols, rule.get("exact", []), "exact")
            or _first_match(cols, rule.get("contains", []), "contains")
            or _first_match(cols, rule.get("regex", []), "regex")
            or _fuzzy(cols, rule.get("fuzzy_hint", current))
        )
        return hit if hit else current

    return replace(
        col_cfg,
        TS_COL=pick(col_cfg.TS_COL, "TS_COL"),
        USER_COL=pick(col_cfg.USER_COL, "USER_COL"),
        EMAIL_COL=pick(col_cfg.EMAIL_COL, "EMAIL_COL"),
        EXP_COL=pick(col_cfg.EXP_COL, "EXP_COL"),
    )

def detect_profile(columns: Sequence[str]) -> str:
    cols = _norm_cols(columns)
    s = set(cols)
    best_name, best_score = "unknown", -1
    for prof in PROFILES:
        score = sum(1 for x in prof["signatures_any"] if x in s)
        if score > best_score:
            best_score, best_name = score, prof["name"]
    return best_name

def extract_item_token(colname: str, profile_name: str) -> Optional[str]:
    prof = next((p for p in PROFILES if p["name"] == profile_name), None)
    if not prof:
        m = re.match(r"^([A-Z]{1,6}\d{1,2})\b", colname.strip())
        return m.group(1) if m else None
    m = re.match(prof["item_token_regex"], colname.strip())
    return m.group(1) if m else None

def build_rename_map_for_items(columns: Sequence[str]) -> Tuple[str, Dict[str, str], List[str]]:
    cols = _norm_cols(columns)
    profile_name = detect_profile(cols)
    prof = next((p for p in PROFILES if p["name"] == profile_name), None)

    rename_map: Dict[str, str] = {}
    for c in cols:
        tok = extract_item_token(c, profile_name)
        if tok:
            rename_map[c] = tok

    scale_prefixes = prof["scale_prefixes"] if prof else []
    return profile_name, rename_map, scale_prefixes


# ==============================
# ✅ 你要的：把解析結果「填充進 cfg」
# ==============================
def apply_schema_to_config(cfg: Config, df: pd.DataFrame, *, mutate_df: bool = True) -> pd.DataFrame:
    """
    1) normalize df.columns
    2) 偵測版本、建立 rename_map，必要時把題項欄名 token 化（特別是 v2）
    3) 解析 meta 欄位並寫回 cfg.cols
    4) 寫回 cfg.runtime (profile/scale_prefixes/rename_map)
    回傳：處理後的 df（若 mutate_df=True 則會回傳已改名的 df）
    """
    if not mutate_df:
        df = df.copy()

    # normalize columns once
    df.columns = _norm_cols(df.columns)

    # profile + rename tokens (v2)
    profile_name, rename_map, scale_prefixes = build_rename_map_for_items(df.columns)
    if rename_map:
        df = df.rename(columns=rename_map)

    # resolve meta columns based on (possibly renamed) df.columns
    cfg.cols = resolve_columns(cfg.cols, df.columns)

    # write runtime info back to cfg
    cfg.runtime.profile_name = profile_name
    cfg.runtime.scale_prefixes = scale_prefixes
    cfg.runtime.rename_map = rename_map

    return df
