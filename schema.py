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


def _first_real_col(columns: list[str]) -> str:
    # 避免遇到 excel 匯出有 "Unnamed: 0"
    for c in columns:
        s = _norm_col(c)
        if not s:
            continue
        if s.lower().startswith("unnamed"):
            continue
        return c
    return columns[0] if columns else ""


def _match_meta(colnames_norm: list[str], key: str) -> int | None:
    """
    回傳 meta 欄位在 df.columns 的 index；找不到回 None
    """
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
    """
    回傳：
      profile_name, scale_prefixes, item_token_regex
    """
    cols_norm = [_norm_col(c) for c in columns]
    best_name, best_prefixes, best_regex = ("unknown", [], "")
    best_hits = 0

    for p in PROFILES:
        sigs = set(p.get("signatures_any", []))
        # hits: 欄名中包含任一 signature 的次數（粗略但穩）
        hits = sum(1 for cn in cols_norm if any(sig in cn for sig in sigs))
        if hits > best_hits:
            best_hits = hits
            best_name = p["name"]
            best_prefixes = list(p.get("scale_prefixes", []))
            best_regex = str(p.get("item_token_regex", ""))

    return best_name, best_prefixes, best_regex


def build_rename_map(columns: list[str], item_token_regex: str) -> dict[str, str]:
    """
    把「題項欄位」從原始欄名映射到 token（例如 "SRL3 我會..." -> "SRL3"）
    """
    rename: dict[str, str] = {}
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
      resolved_cols, profile_name, scale_prefixes, rename_map = resolve_schema(df, cfg, cog)

    會做：
    - TS_COL：直接取 df 第一個“有效欄位”
    - USER/EMAIL/EXP：用 rulebook 盡量自動辨識；找不到就不填
    - profile：自動判 v1/v2/v3/unknown
    - rename_map：把題項欄位轉成 token（寫回 cfg.runtime.rename_map）
    - 寫回 cfg.runtime.resolved_cols / profile_name / scale_prefixes

    注意：
    - 這裡只負責 schema/欄位解析，不做資料處理、不做估計。
    """
    columns = [str(c) for c in df.columns]
    cols_norm = [_norm_col(c) for c in columns]

    resolved: dict[str, str] = {}

    # 你要的：第一欄直接當 TS_COL
    first_col = _first_real_col(columns)
    if not first_col:
        raise ValueError("DataFrame has no usable columns.")
    resolved["TS_COL"] = first_col

    # 其餘 meta 欄位：用 rulebook 找得到就填，找不到就略過
    for k in ["USER_COL", "EMAIL_COL", "EXP_COL"]:
        idx = _match_meta(cols_norm, k)
        if idx is not None:
            resolved[k] = columns[idx]

    profile_name, scale_prefixes, item_token_regex = detect_profile(columns)
    rename_map = build_rename_map(columns, item_token_regex)

    # 寫回 runtime（cfg.runtime 必須存在）
    if not hasattr(cfg, "runtime"):
        raise AttributeError("cfg must have .runtime to store resolved schema info.")
    cfg.runtime.profile_name = profile_name
    cfg.runtime.scale_prefixes = list(scale_prefixes)
    cfg.runtime.rename_map = dict(rename_map)
    cfg.runtime.resolved_cols = dict(resolved)

    # optional: also write into cog for convenience
    if cog is not None:
        cog.cols_resolved = dict(resolved)
        cog.profile_name = profile_name
        cog.scale_prefixes = list(scale_prefixes)
        if hasattr(cog, "log") and cog.log:
            cog.log.info(f"schema profile={profile_name}, first_col(TS)={resolved['TS_COL']}")
            cog.log.info(f"resolved meta={resolved}")
            cog.log.info(f"rename_map items={len(rename_map)}")

    return resolved, profile_name, scale_prefixes, rename_map
