# pls_project/schema.py
from __future__ import annotations
import re
import difflib
import pandas as pd
from .rulebook import META_RULES, PROFILES
from .io_utils import _norm_text

def _norm_col(c: str) -> str:
    # 統一空白/全形空白，去頭尾
    return _norm_text(c)

def _first_real_col(columns) -> str:
    # 避免遇到 excel 匯出有 "Unnamed: 0"
    for c in columns:
        s = _norm_col(c)
        if not s:
            continue
        if s.lower().startswith("unnamed"):
            continue
        return c
    return columns[0]

def _match_meta(colnames_norm: list[str], key: str) -> int | None:
    rule = META_RULES[key]
    # exact
    exact = set(_norm_col(x) for x in rule.get("exact", []))
    for i, cn in enumerate(colnames_norm):
        if cn in exact:
            return i

    # contains
    contains = [_norm_col(x) for x in rule.get("contains", [])]
    for i, cn in enumerate(colnames_norm):
        for frag in contains:
            if frag and frag.lower() in cn.lower():
                return i

    # regex
    for pat in rule.get("regex", []):
        rx = re.compile(pat, flags=re.I)
        for i, cn in enumerate(colnames_norm):
            if rx.search(cn):
                return i

    # fuzzy（最後手段）
    hint = _norm_col(rule.get("fuzzy_hint", ""))
    if hint:
        best_i = None
        best = 0.0
        for i, cn in enumerate(colnames_norm):
            score = difflib.SequenceMatcher(None, hint, cn).ratio()
            if score > best:
                best = score
                best_i = i
        if best_i is not None and best >= 0.72:
            return best_i

    return None

def detect_profile(columns: list[str]) -> tuple[str, list[str], str]:
    # 回傳：profile_name, scale_prefixes, item_token_regex
    cols_norm = [_norm_col(c) for c in columns]
    best = ("unknown", [], "")
    best_hits = 0

    for p in PROFILES:
        sigs = set(p["signatures_any"])
        hits = sum(1 for cn in cols_norm if any(sig in cn for sig in sigs))
        if hits > best_hits:
            best_hits = hits
            best = (p["name"], p["scale_prefixes"], p["item_token_regex"])

    return best

def build_rename_map(columns: list[str], item_token_regex: str) -> dict[str, str]:
    rename = {}
    if not item_token_regex:
        return rename
    rx = re.compile(item_token_regex)
    for c in columns:
        m = rx.search(_norm_col(c))
        if m:
            rename[c] = m.group(1)
    return rename

def resolve_schema(df: pd.DataFrame, cfg, cog=None):
    """
    你只要呼叫這個：
      cols_resolved, profile_name, scale_prefixes, rename_map = resolve_schema(df, cfg, cog)
    會：
    - TS_COL：直接取 df 第一個“有效欄位”
    - 其他 meta：用 rulebook 盡量自動辨識
    - profile：自動判 v1/v2/v3
    - rename_map：把題項欄位轉成 token
    - 寫回 cfg.runtime（以及 cog，如果有給）
    """
    columns = list(df.columns)
    cols_norm = [_norm_col(c) for c in columns]

    resolved = {}
    # 你要的：第一欄直接當 TS_COL
    resolved["TS_COL"] = _first_real_col(columns)

    # 其餘 meta 欄位：用 rulebook 找得到就填，找不到就略過
    for k in ["USER_COL", "EMAIL_COL", "EXP_COL"]:
        idx = _match_meta(cols_norm, k)
        if idx is not None:
            resolved[k] = columns[idx]

    profile_name, scale_prefixes, item_token_regex = detect_profile(columns)
    rename_map = build_rename_map(columns, item_token_regex)

    # 寫回 runtime
    cfg.runtime.profile_name = profile_name
    cfg.runtime.scale_prefixes = list(scale_prefixes)
    cfg.runtime.rename_map = dict(rename_map)
    cfg.runtime.resolved_cols = dict(resolved)

    if cog is not None:
        cog.cols_resolved = dict(resolved)
        cog.profile_name = profile_name
        cog.scale_prefixes = list(scale_prefixes)
        if hasattr(cog, "log") and cog.log:
            cog.log.info(f"schema profile={profile_name}, first_col(TS)={resolved['TS_COL']}")
            cog.log.info(f"resolved meta={resolved}")
            cog.log.info(f"rename_map items={len(rename_map)}")

    return resolved, profile_name, scale_prefixes, rename_map
