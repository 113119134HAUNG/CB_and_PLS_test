# pls_project/schema.py
from __future__ import annotations

import re
import difflib
import pandas as pd

from .rulebook import META_RULES, PROFILES
from .io_utils import _norm_text


def _norm_col(c: str) -> str:
    return _norm_text(c)


def _first_real_col(columns: list[str]) -> str:
    for c in columns:
        s = _norm_col(c)
        if not s:
            continue
        if s.lower().startswith("unnamed"):
            continue
        return c
    return columns[0] if columns else ""


def _match_meta(colnames_norm: list[str], key: str) -> int | None:
    rule = META_RULES[key]

    exact = set(_norm_col(x) for x in rule.get("exact", []))
    for i, cn in enumerate(colnames_norm):
        if cn in exact:
            return i

    contains = [_norm_col(x) for x in rule.get("contains", [])]
    for i, cn in enumerate(colnames_norm):
        for frag in contains:
            if frag and frag.lower() in cn.lower():
                return i

    for pat in rule.get("regex", []):
        rx = re.compile(pat, flags=re.I)
        for i, cn in enumerate(colnames_norm):
            if rx.search(cn):
                return i

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
    """
    Detect profile by signature hit count.
    Case-insensitive: normalize to upper so 'pa1' / 'Pa1' also works.
    """
    cols_norm = [_norm_col(c) for c in columns]
    cols_u = [c.upper() for c in cols_norm]

    best_name, best_prefixes, best_regex = ("unknown", [], "")
    best_hits = 0

    for p in PROFILES:
        sigs_u = set(str(s).upper() for s in p.get("signatures_any", []))
        hits = sum(1 for cn in cols_u if any(sig in cn for sig in sigs_u))
        if hits > best_hits:
            best_hits = hits
            best_name = p["name"]
            best_prefixes = list(p.get("scale_prefixes", []))
            best_regex = str(p.get("item_token_regex", ""))

    return best_name, best_prefixes, best_regex


def build_rename_map(
    columns: list[str],
    item_token_regex: str,
    *,
    exclude_cols: set[str] | None = None
) -> dict[str, str]:
    """
    把「題項欄位」從原始欄名映射到 token（例如 "SRL3 我會..." -> "SRL3"）
    - regex 改為 case-insensitive
    - token 統一轉大寫（避免 pipeline 的 item_pat / sort_key 抓不到）
    exclude_cols：不允許被 rename 的欄位（通常放 meta 欄）
    """
    rename: dict[str, str] = {}
    if not item_token_regex:
        return rename

    exclude_cols = exclude_cols or set()
    rx = re.compile(item_token_regex, flags=re.I)

    for c in columns:
        if c in exclude_cols:
            continue
        m = rx.search(_norm_col(c))
        if m:
            token = str(m.group(1)).upper()
            rename[c] = token

    return rename


def resolve_schema(df: pd.DataFrame, cfg, cog=None):
    """
    resolved_cols, profile_name, scale_prefixes, rename_map = resolve_schema(df, cfg, cog)
    """
    columns = [str(c) for c in df.columns]
    cols_norm = [_norm_col(c) for c in columns]

    resolved: dict[str, str] = {}

    # ✅ TS_COL：先用規則找，找不到才 fallback 到第一個有效欄
    idx_ts = _match_meta(cols_norm, "TS_COL")
    if idx_ts is not None:
        resolved["TS_COL"] = columns[idx_ts]
    else:
        first_col = _first_real_col(columns)
        if not first_col:
            raise ValueError("DataFrame has no usable columns.")
        resolved["TS_COL"] = first_col

    for k in ["USER_COL", "EMAIL_COL", "EXP_COL"]:
        idx = _match_meta(cols_norm, k)
        if idx is not None:
            resolved[k] = columns[idx]

    profile_name, scale_prefixes, item_token_regex = detect_profile(columns)

    # ✅ 排除 meta 欄避免被 rename
    exclude = set(resolved.values())
    rename_map = build_rename_map(columns, item_token_regex, exclude_cols=exclude)

    if not hasattr(cfg, "runtime"):
        raise AttributeError("cfg must have .runtime to store resolved schema info.")
    cfg.runtime.profile_name = profile_name
    cfg.runtime.scale_prefixes = list(scale_prefixes)
    cfg.runtime.rename_map = dict(rename_map)
    cfg.runtime.resolved_cols = dict(resolved)

    if cog is not None:
        cog.cols_resolved = dict(resolved)
        cog.profile_name = profile_name
        cog.scale_prefixes = list(scale_prefixes)
        if hasattr(cog, "log") and cog.log:
            cog.log.info(f"schema profile={profile_name}, TS_COL={resolved.get('TS_COL','')}")
            cog.log.info(f"resolved meta={resolved}")
            cog.log.info(f"rename_map items={len(rename_map)}")

    return resolved, profile_name, scale_prefixes, rename_map
