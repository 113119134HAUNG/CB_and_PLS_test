# pls_project/schema.py
from __future__ import annotations
from dataclasses import replace
from typing import Sequence, Optional, Dict, List, Tuple
import re
import difflib
import pandas as pd

from .config import Config, ColumnConfig
from .rulebook import META_RULES, PROFILES

# ------------------------------
# helpers
# ------------------------------
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

# ------------------------------
# meta columns resolve
# ------------------------------
def resolve_columns(col_cfg: ColumnConfig, columns: Sequence[str]) -> ColumnConfig:
    """把 ColumnConfig 對到 df.columns 的實際欄名（無副作用）。"""
    cols = _norm_cols(columns)

    def pick(current: str, field: str) -> str:
        if current in cols:
            return current
        rule = META_RULES.get(field, {})
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

# ------------------------------
# profile detection + item tokenization
# ------------------------------
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
    colname = str(colname).strip().replace("\u3000", " ")
    prof = next((p for p in PROFILES if p["name"] == profile_name), None)
    if not prof:
        m = re.match(r"^([A-Z]{1,6}\d{1,2})\b", colname)
        return m.group(1) if m else None
    m = re.match(prof["item_token_regex"], colname)
    return m.group(1) if m else None

def build_rename_map_for_items(columns: Sequence[str]) -> Tuple[str, Dict[str, str], List[str]]:
    """
    回傳：
      profile_name
      rename_map: 原欄名 -> token（只對可抽 token 的欄位）
      scale_prefixes: 該版本的 prefix 清單
    """
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

# ------------------------------
# ✅ main entry: apply to cfg
# ------------------------------
def apply_schema_to_config(cfg: Config, df: pd.DataFrame, *, mutate_df: bool = True) -> pd.DataFrame:
    """
    schema 工作流程（你要的）：
      1) normalize 欄名
      2) detect profile + tokenize items (rename to token)
      3) resolve meta columns
      4) 填充回 cfg.cols / cfg.runtime
    """
    if not mutate_df:
        df = df.copy()

    # 1) normalize columns
    df.columns = _norm_cols(df.columns)

    # 2) profile + item token rename
    profile_name, rename_map, scale_prefixes = build_rename_map_for_items(df.columns)
    if rename_map:
        df = df.rename(columns=rename_map)

    # 3) resolve meta columns (after rename)
    cfg.cols = resolve_columns(cfg.cols, df.columns)

    # 4) fill runtime
    cfg.runtime.profile_name = profile_name
    cfg.runtime.scale_prefixes = scale_prefixes
    cfg.runtime.rename_map = rename_map

    return df
