# pls_project/schema.py
from __future__ import annotations

import re
import difflib
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .rulebook import META_RULES, PROFILES
from .io_utils import _norm_text


_EMAIL_RX = re.compile(r"^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}$", re.I)


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


# ------------------------------
# New: infer meta columns by VALUES (not fixed column names)
# ------------------------------
def _email_ratio(s: pd.Series) -> float:
    x = s.dropna().astype(str).str.strip()
    if len(x) == 0:
        return 0.0
    return float(x.apply(lambda v: bool(_EMAIL_RX.match(v))).mean())


def _datetime_ratio(s: pd.Series) -> float:
    dt = pd.to_datetime(s, errors="coerce")
    return float(dt.notna().mean())


def _binary_ratio_01(s: pd.Series) -> float:
    x = pd.to_numeric(s, errors="coerce").dropna()
    if len(x) == 0:
        return 0.0
    u = set(np.unique(x))
    return 1.0 if u.issubset({0, 1}) else 0.0


def _yesno_ratio(s: pd.Series) -> float:
    x = s.dropna().astype(str).str.strip().str.lower()
    if len(x) == 0:
        return 0.0
    yes = {"yes", "y", "true", "t", "1", "有", "是"}
    no = {"no", "n", "false", "f", "0", "沒有", "否", "未"}
    return float(x.apply(lambda v: (v in yes) or (v in no)).mean())


def _exp_like_score(colname: str, s: pd.Series) -> float:
    """
    Experience-like scoring (content-first, name as tie-breaker).
    Higher is more likely to be "experience/use_experience".
    """
    name = str(colname).lower()
    name_kw = ("exp", "experience", "use", "usage", "ai", "tool", "freq", "frequency", "proficiency")
    name_score = sum(k in name for k in name_kw)

    # content signals
    bin01 = _binary_ratio_01(s)
    yn = _yesno_ratio(s)

    x = pd.to_numeric(s, errors="coerce")
    nonna = x.notna().sum()
    if nonna > 0:
        xnum = x.dropna().astype(float)
        prop_zero = float(np.mean(np.isclose(xnum, 0.0))) if len(xnum) else 0.0
        prop_pos = float(np.mean(xnum > 0.0)) if len(xnum) else 0.0
        # experience-like if has some zeros and some positives
        num_mix = 1.0 if (prop_zero >= 0.05 and prop_pos >= 0.20) else 0.0
    else:
        num_mix = 0.0

    # weights: binary/yesno strongest, then numeric mix, then name hints
    return 3.0 * max(bin01, yn) + 1.5 * num_mix + 0.5 * float(name_score)


def infer_meta_by_values(df: pd.DataFrame) -> Dict[str, str]:
    """
    Infer meta columns without relying on fixed names.
    Returns possibly empty dict with keys among: TS_COL, EMAIL_COL, EXP_COL, USER_COL
    """
    cols = list(map(str, df.columns))
    out: Dict[str, str] = {}

    # EMAIL_COL
    best_email = (None, 0.0)
    for c in cols:
        r = _email_ratio(df[c])
        if r > best_email[1]:
            best_email = (c, r)
    if best_email[0] is not None and best_email[1] >= 0.80:
        out["EMAIL_COL"] = best_email[0]

    # TS_COL
    best_ts = (None, 0.0)
    for c in cols:
        r = _datetime_ratio(df[c])
        if r > best_ts[1]:
            best_ts = (c, r)
    if best_ts[0] is not None and best_ts[1] >= 0.80:
        out["TS_COL"] = best_ts[0]

    # EXP_COL
    best_exp = (None, -1.0)
    for c in cols:
        sc = _exp_like_score(c, df[c])
        if sc > best_exp[1]:
            best_exp = (c, sc)
    # threshold: must have some signal; otherwise leave unset
    if best_exp[0] is not None and best_exp[1] >= 2.5:
        out["EXP_COL"] = best_exp[0]

    # USER_COL (optional, conservative)
    # Only infer if there's a non-email string column with moderate uniqueness.
    email_col = out.get("EMAIL_COL")
    best_user = (None, 0.0)
    for c in cols:
        if c == email_col:
            continue
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            continue
        x = s.dropna().astype(str).str.strip()
        if len(x) < 10:
            continue
        if _email_ratio(x) >= 0.50:
            continue
        uniq_ratio = x.nunique() / max(1, len(x))
        # prefer something like names: not constant, not purely unique ID
        if 0.10 <= uniq_ratio <= 0.95:
            score = 1.0 - abs(uniq_ratio - 0.60)  # peak around 0.6
            if score > best_user[1]:
                best_user = (c, score)
    if best_user[0] is not None and best_user[1] >= 0.25:
        out["USER_COL"] = best_user[0]

    return out


def resolve_schema(df: pd.DataFrame, cfg, cog=None):
    """
    resolved_cols, profile_name, scale_prefixes, rename_map = resolve_schema(df, cfg, cog)

    New behavior:
      - Do NOT force TS_COL fallback to first column
      - Use name-based META_RULES first, then infer by values
      - If still not found, leave key missing (pipeline will skip related features)
    """
    columns = [str(c) for c in df.columns]
    cols_norm = [_norm_col(c) for c in columns]

    resolved: dict[str, str] = {}

    # 1) name/rule-based meta matching
    for k in ["TS_COL", "USER_COL", "EMAIL_COL", "EXP_COL"]:
        idx = _match_meta(cols_norm, k)
        if idx is not None:
            resolved[k] = columns[idx]

    # 2) profile detect
    profile_name, scale_prefixes, item_token_regex = detect_profile(columns)

    # 3) value-based inference (fill missing only)
    inferred = infer_meta_by_values(df)
    for k, v in inferred.items():
        if k not in resolved and v in columns:
            resolved[k] = v

    # 4) build rename map (exclude inferred meta cols too)
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
            cog.log.info(f"schema profile={profile_name}")
            cog.log.info(f"resolved meta={resolved}")
            cog.log.info(f"rename_map items={len(rename_map)}")

    return resolved, profile_name, scale_prefixes, rename_map
