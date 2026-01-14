# pls_project/pipeline.py
from __future__ import annotations

import re
from itertools import groupby

import numpy as np
import pandas as pd

from pls_project.schema import resolve_schema
from pls_project.io_utils import to_score, reverse_1to5, get_or_create_ws, write_block
from pls_project.paper import (
    reliability_summary,
    item_analysis_table,
    one_factor_fa_table,
    one_factor_efa_table,
    run_cfa,
)
from pls_project.pls_core import (
    topo_sort,
    make_plspm_path,
    htmt_matrix,
    r2_f2_from_scores,
    q2_cv_from_scores,
    structural_vif,
)
from pls_project.pls_estimate import estimate_pls_basic_paper

# 若你之後有做 bootstrap loop，再把 summarize_direct_ci 用起來
# from pls_project.pls_bootstrap import summarize_direct_ci


def run_pipeline(cog, reverse_target: bool, tag: str):
    cfg = cog.cfg
    OUT_XLSX = f"{cfg.io.OUT_XLSX_BASE}_{tag}.xlsx"
    OUT_CSV = f"{cfg.io.OUT_CSV_BASE}_{tag}.csv"

    print("\n" + "=" * 80)
    print(f"🚀 Running scenario: {tag} | reverse_{cfg.scenario.SCENARIO_TARGET}={reverse_target}")
    print("=" * 80)

    # ==============================
    # Load
    # ==============================
    df = pd.read_excel(cfg.io.XLSX_PATH, sheet_name=cfg.io.SHEET_NAME)
    df.columns = df.columns.astype(str).str.strip()
    print("Data shape:", df.shape)

    # ==============================
    # Schema resolve (profile/meta/rename_map)
    # ==============================
    resolved_cols, profile_name, scale_prefixes, rename_map = resolve_schema(df, cfg, cog)

    # Rename item columns -> token (e.g., "SRL3 我會..." -> "SRL3")
    # 注意：只 rename 題項欄位；meta 欄位通常不會被匹配到
    if rename_map:
        df = df.rename(columns=rename_map)

    # Meta columns (prefer resolved; fallback to cfg.cols.*)
    TS_COL = resolved_cols.get("TS_COL", cfg.cols.TS_COL)
    USER_COL = resolved_cols.get("USER_COL", cfg.cols.USER_COL)
    EMAIL_COL = resolved_cols.get("EMAIL_COL", cfg.cols.EMAIL_COL)
    EXP_COL = resolved_cols.get("EXP_COL", cfg.cols.EXP_COL)

    # Scales (prefer detected profile; fallback to cfg.scales.SCALES)
    SCALES = list(scale_prefixes) if scale_prefixes else list(cfg.scales.SCALES)

    # ==============================
    # Find item columns (after rename)
    # ==============================
    # token pattern: (PREFIX)(digits 1~2) is enough for v1/v2; v3 is A/B/C + 2 digits (still matches \d{1,2})
    scale_alt = "|".join(map(re.escape, SCALES))
    item_pat = re.compile(rf"^(?:{scale_alt})\d{{1,2}}$")

    item_cols = [c for c in df.columns if item_pat.fullmatch(str(c))]
    if not item_cols:
        raise ValueError(
            f"找不到題項欄位！"
            f"目前 profile={profile_name}, prefixes={SCALES}. "
            f"請確認欄名是否已被 schema rename 成 token（如 PA1 / SRL3 / A11）。"
        )

    def sort_key(c: str):
        m = re.match(r"^([A-Z]+)(\d+)$", str(c))
        if not m:
            return (999, 999)
        prefix, num = m.group(1), int(m.group(2))
        return (SCALES.index(prefix) if prefix in SCALES else 999, num)

    item_cols = sorted(item_cols, key=sort_key)

    groups = [g for g in SCALES if any(col.startswith(g) for col in item_cols)]
    group_items = {g: [c for c in item_cols if c.startswith(g)] for g in groups}
    print("Profile:", profile_name)
    print("Groups:", groups)

    # ==============================
    # Reverse list
    # ==============================
    REVERSE_ITEMS = list(cfg.scales.BASE_REVERSE_ITEMS)
    if reverse_target and (cfg.scenario.SCENARIO_TARGET in group_items):
        REVERSE_ITEMS += list(group_items[cfg.scenario.SCENARIO_TARGET])

    # ==============================
    # Flags (experience / duplication / careless)
    # ==============================
    df["_flag_noexp"] = False
    df["_flag_dup"] = False

    def has_experience(x) -> bool:
        if pd.isna(x):
            return False
        if isinstance(x, (int, float, np.integer, np.floating)):
            return float(x) > 0
        s = str(x).strip()
        no_kw = ["沒有", "否", "不曾", "從未", "未", "無"]
        yes_kw = ["有", "是", "曾", "使用過"]
        if any(k in s for k in no_kw):
            return False
        if any(k in s for k in yes_kw):
            return True
        return False

    if EXP_COL in df.columns:
        df["_flag_noexp"] = ~df[EXP_COL].apply(has_experience)

    # timestamp for duplicate resolution
    df["_ts"] = pd.to_datetime(df[TS_COL], errors="coerce") if TS_COL in df.columns else pd.NaT
    df["_orig_order"] = np.arange(len(df))

    dup_key = [c for c in [USER_COL, EMAIL_COL] if c in df.columns]
    if dup_key:
        df_sorted = df.sort_values(["_ts", "_orig_order"], na_position="last").copy()
        df_sorted["_flag_dup"] = df_sorted.duplicated(subset=dup_key, keep=cfg.filt.KEEP_DUP)
        df = df_sorted.sort_values("_orig_order")

    # ==============================
    # Likert -> numeric
    # ==============================
    for c in item_cols:
        df[c] = df[c].apply(to_score).astype(float)

    # reverse items
    for c in REVERSE_ITEMS:
        if c in df.columns:
            df[c] = df[c].apply(reverse_1to5).astype(float)

    # careless flags
    k_all = len(item_cols)
    df["_missing_rate"] = df[item_cols].isna().mean(axis=1)
    df["_sd_items"] = df[item_cols].std(axis=1, ddof=0)

    def max_run_length(arr) -> int:
        v = [int(x) for x in arr if pd.notna(x)]
        if len(v) == 0:
            return 0
        return max(sum(1 for _ in g) for _, g in groupby(v))

    df["_longstring"] = df[item_cols].apply(lambda r: max_run_length(r.values), axis=1)
    df["_flag_careless"] = (
        (df["_missing_rate"] > cfg.filt.MAX_MISSING_RATE)
        | (df["_sd_items"] <= cfg.filt.MIN_SD_ITEMS)
        | (df["_longstring"] >= int(np.ceil(cfg.filt.LONGSTRING_PCT * k_all)))
    )

    def join_reasons(r):
        reasons = []
        if bool(r.get("_flag_noexp", False)):
            reasons.append("無GenAI學習經驗")
        if bool(r.get("_flag_dup", False)):
            reasons.append("重複填答")
        if bool(r.get("_flag_careless", False)):
            reasons.append("疑似隨意/草率(缺漏/直線/長串)")
        return ";".join(reasons)

    df["_exclude_reason"] = df.apply(join_reasons, axis=1)
    excluded_df = df[df["_exclude_reason"] != ""].copy()
    df_valid = df[df["_exclude_reason"] == ""].copy()

    print("🧹 無效問卷剔除完成")
    print("原始樣本數:", len(df_valid) + len(excluded_df))
    print("剔除樣本數:", len(excluded_df))
    print("有效樣本數:", len(df_valid))

    if cfg.io.DROP_EMAIL_IN_VALID_DF and (EMAIL_COL in df_valid.columns):
        df_valid = df_valid.drop(columns=[EMAIL_COL])

    df_valid = df_valid.drop(columns=[c for c in df_valid.columns if c.startswith("_")], errors="ignore")

    # ==============================
    # Paper tables
    # ==============================
    RelTable = reliability_summary(cog, df_valid, groups, group_items, item_cols)
    ItemTable = item_analysis_table(cog, df_valid, groups, group_items)
    FA1Table = one_factor_fa_table(cog, df_valid, groups, group_items)
    EFA1Table, EFA1Fit = one_factor_efa_table(cog, df_valid, groups, group_items)

    # CFA
    CFA_Loadings, CFA_Info, CFA_Fit = run_cfa(cog, df_valid, groups, group_items, item_cols)

    # ==============================
    # PLS main (one-shot estimation)
    # ==============================
    PLS1_info = PLS1_outer = PLS1_quality = PLS1_htmt = PLS1_cross = pd.DataFrame()
    PLS2_info = PLS2_outer = PLS2_quality = PLS2_htmt = PLS2_cross = PLS2_commitment = pd.DataFrame()
    PLS1_R2 = PLS1_f2 = PLS1_Q2 = PLS1_VIF = pd.DataFrame()
    PLS2_R2 = PLS2_f2 = PLS2_Q2 = PLS2_VIF = pd.DataFrame()
    PLS1_BOOTPATH = PLS2_BOOTPATH = pd.DataFrame()

    if cfg.pls.RUN_PLS:
        # clean-mode guards
        scheme_up = str(getattr(cfg.pls, "PLS_SCHEME", "PATH")).strip().upper()
        if getattr(cfg.pls, "CLEAN_MODE", True):
            if scheme_up in ("CENTROID",):
                raise ValueError("CLEAN_MODE=True forbids PLS_SCHEME='CENTROID' (SmartPLS4 removed it).")
            if (scheme_up == "PCA") and (not bool(getattr(cfg.pls, "ALLOW_PCA", False))):
                raise ValueError("CLEAN_MODE=True and ALLOW_PCA=False forbids PLS_SCHEME='PCA' approximation.")
            if str(getattr(cfg.pls, "PLS_MISSING", "listwise")).lower() == "mean":
                raise ValueError("CLEAN_MODE=True forbids PLS_MISSING='mean' (generates new values).")

        Xpls = df_valid[item_cols].copy().astype(float)

        # missing handling (config-driven)
        miss = str(getattr(cfg.pls, "PLS_MISSING", "listwise")).lower()
        if miss == "none":
            if Xpls.isna().any().any():
                raise ValueError("PLS_MISSING='none' but missing values exist. Handle missing before import.")
        elif miss == "listwise":
            Xpls = Xpls.dropna()
        elif miss == "mean":
            Xpls = Xpls.apply(lambda s: s.fillna(s.mean()), axis=0)
        else:
            raise ValueError(f"Unknown PLS_MISSING: {cfg.pls.PLS_MISSING}")

        Xpls = Xpls.reset_index(drop=True)

        # ---- Model1 ----
        edges1 = [
            ("SA", "MIND"), ("SA", "ACO"), ("SA", "CCO"),
            ("PA", "BB"), ("PA", "BS"),
            ("BS", "MIND"), ("BS", "ACO"), ("BS", "CCO"), ("BS", "CI"),
            ("MIND", "CI"), ("BB", "CI"), ("ACO", "CI"), ("CCO", "CI"),
            ("CI", "LO"),
        ]
        lv_set1 = [g for g in ["PA", "SA", "BB", "BS", "MIND", "ACO", "CCO", "CI", "LO"] if g in groups]
        order1 = topo_sort(lv_set1, edges1)
        path1 = make_plspm_path(order1, edges1)
        lv_blocks1 = {lv: group_items[lv] for lv in order1}
        lv_modes1 = {lv: "A" for lv in order1}

        res1 = estimate_pls_basic_paper(
            cog,
            Xpls=Xpls,
            item_cols=item_cols,
            path_df=path1,
            lv_blocks=lv_blocks1,
            lv_modes=lv_modes1,
            order=order1,
        )
        PLS1_cross = res1["PLS_cross"]
        PLS1_outer = res1["PLS_outer"]
        PLS1_quality = res1["PLS_quality"]
        scores1 = res1["scores"]

        PLS1_htmt = htmt_matrix(
            Xpls[item_cols],
            {g: lv_blocks1[g] for g in order1},
            order1,
            method=str(getattr(cfg.pls, "HTMT_CORR_METHOD", "pearson")),
        )

        PLS1_R2, PLS1_f2 = r2_f2_from_scores(scores1[order1], path1)
        PLS1_Q2 = q2_cv_from_scores(scores1[order1], path1, n_splits=int(cfg.pls.Q2_FOLDS), seed=int(cfg.pls.PLS_SEED))
        PLS1_VIF = structural_vif(scores1[order1], path1)

        PLS1_info = pd.DataFrame([{
            "Model": f"Model1 baseline (ACO/CCO separate) [{tag}]",
            "profile": profile_name,
            "order": " > ".join(order1),
            "n(PLS)": int(Xpls.shape[0]),
            "scheme": cfg.pls.PLS_SCHEME,
            "missing": cfg.pls.PLS_MISSING,
            "sign_fix": bool(getattr(cfg.pls, "SIGN_FIX", True)),
            "estimates": "outer/path from plspm model API (strict)",
        }])

        # ---- Model2 (two-stage Commitment formative) ----
        edges2 = [
            ("SA", "MIND"), ("SA", "Commitment"),
            ("PA", "BB"), ("PA", "BS"),
            ("BS", "MIND"), ("BS", "Commitment"), ("BS", "CI"),
            ("MIND", "CI"), ("BB", "CI"), ("Commitment", "CI"),
            ("CI", "LO"),
        ]
        lv_set2 = [g for g in ["PA", "SA", "BB", "BS", "MIND", "CI", "LO"] if g in groups] + ["Commitment"]
        order2 = topo_sort(lv_set2, edges2)
        path2 = make_plspm_path(order2, edges2)

        if ("ACO" not in scores1.columns) or ("CCO" not in scores1.columns):
            raise KeyError("Model2 requires ACO and CCO scores from Model1.")
        if scores1[["ACO", "CCO"]].isna().any().any():
            raise ValueError("Stage1 ACO/CCO scores contain NaN. Clean mode forbids fillna(mean).")

        Xpls2 = Xpls.copy()
        Xpls2["ACO_score"] = scores1["ACO"].to_numpy()
        Xpls2["CCO_score"] = scores1["CCO"].to_numpy()

        lv_blocks2 = {}
        for lv in order2:
            if lv == "Commitment":
                lv_blocks2[lv] = ["ACO_score", "CCO_score"]
            else:
                lv_blocks2[lv] = group_items[lv]
        lv_modes2 = {lv: ("B" if lv == "Commitment" else "A") for lv in order2}

        stage2_indicators = []
        for lv in order2:
            stage2_indicators += lv_blocks2[lv]
        stage2_indicators = list(dict.fromkeys(stage2_indicators))

        res2 = estimate_pls_basic_paper(
            cog,
            Xpls=Xpls2,
            item_cols=stage2_indicators,
            path_df=path2,
            lv_blocks=lv_blocks2,
            lv_modes=lv_modes2,
            order=order2,
        )
        PLS2_cross = res2["PLS_cross"]
        PLS2_outer = res2["PLS_outer"]
        PLS2_quality = res2["PLS_quality"]
        scores2 = res2["scores"]

        refl2 = [lv for lv in order2 if lv != "Commitment"]
        PLS2_htmt = htmt_matrix(
            Xpls[item_cols],
            {g: group_items[g] for g in refl2},
            refl2,
            method=str(getattr(cfg.pls, "HTMT_CORR_METHOD", "pearson")),
        )

        PLS2_R2, PLS2_f2 = r2_f2_from_scores(scores2[order2], path2)
        PLS2_Q2 = q2_cv_from_scores(scores2[order2], path2, n_splits=int(cfg.pls.Q2_FOLDS), seed=int(cfg.pls.PLS_SEED))
        PLS2_VIF = structural_vif(scores2[order2], path2)

        PLS2_commitment = PLS2_outer[PLS2_outer["Construct"] == "Commitment"].copy()

        PLS2_info = pd.DataFrame([{
            "Model": f"Model2 two-stage (Commitment formative) [{tag}]",
            "profile": profile_name,
            "order": " > ".join(order2),
            "n(PLS)": int(Xpls.shape[0]),
            "scheme": cfg.pls.PLS_SCHEME,
            "missing": cfg.pls.PLS_MISSING,
            "sign_fix": bool(getattr(cfg.pls, "SIGN_FIX", True)),
            "estimates": "outer/path from plspm model API (strict)",
        }])

        # Bootstrap direct paths (目前留空；你若之後做 bootstrap loop 再填)
        PLS1_BOOTPATH = pd.DataFrame()
        PLS2_BOOTPATH = pd.DataFrame()

    # ==============================
    # Export Excel + CSV
    # ==============================
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        if cfg.io.EXPORT_EXCLUDED_SHEET and (excluded_df is not None) and (not excluded_df.empty):
            excluded_df.to_excel(writer, sheet_name="排除樣本", index=False)

        ws = get_or_create_ws(writer, cfg.io.PAPER_SHEET)
        r = 0
        r = write_block(writer, cfg.io.PAPER_SHEET, ws, r, f"A. Reliability summary (α / ωt) [{tag}]", RelTable, index=False)
        r = write_block(
            writer, cfg.io.PAPER_SHEET, ws, r,
            f"B. Item analysis (all constructs stacked) [{tag}]",
            ItemTable[["Construct", "Item", "n(complete)", "Mean", "SD", "CITC", "α if deleted", "ωt if deleted"]],
            index=False,
        )
        r = write_block(
            writer, cfg.io.PAPER_SHEET, ws, r,
            f"C. One-factor loadings (sklearn FA, all constructs stacked) [{tag}]",
            FA1Table[["Construct", "Item", "n(complete)", "Loading", "h2", "psi"]],
            index=False,
        )
        r = write_block(
            writer, cfg.io.PAPER_SHEET, ws, r,
            f"D. One-factor EFA per construct (factor_analyzer, all constructs stacked) [{tag}]",
            EFA1Table[["Construct", "Item", "n(complete)", "EFA Loading", "h2", "psi"]],
            index=False,
        )
        if EFA1Fit is not None and (not EFA1Fit.empty):
            r = write_block(
                writer, cfg.io.PAPER_SHEET, ws, r,
                f"D-2. EFA suitability (KMO / Bartlett) per construct [{tag}]",
                EFA1Fit[["Construct", "n(complete)", "k(items)", "KMO", "Bartlett_p"]],
                index=False,
            )

        if cfg.cfa.RUN_CFA:
            r = write_block(writer, cfg.io.PAPER_SHEET, ws, r, f"E. CFA standardized loadings [{tag}]", CFA_Loadings, index=False)
            r = write_block(writer, cfg.io.PAPER_SHEET, ws, r, f"F. CFA model info [{tag}]", CFA_Info, index=False)
            r = write_block(writer, cfg.io.PAPER_SHEET, ws, r, f"G. CFA fit indices [{tag}]", CFA_Fit, index=False)

        ws.freeze_panes = "A2"

        if cfg.pls.RUN_PLS:
            ws_pls = get_or_create_ws(writer, cfg.io.PLS_SHEET)
            rp = 0

            rp = write_block(writer, cfg.io.PLS_SHEET, ws_pls, rp, "MODEL 1 (Baseline): ACO / CCO separate", PLS1_info, index=False)
            rp = write_block(writer, cfg.io.PLS_SHEET, ws_pls, rp, "M1-A. Outer model (model API)", PLS1_outer, index=False)
            rp = write_block(writer, cfg.io.PLS_SHEET, ws_pls, rp, "M1-B. CR / AVE", PLS1_quality, index=False)
            rp = write_block(writer, cfg.io.PLS_SHEET, ws_pls, rp, "M1-C. HTMT", PLS1_htmt, index=True)
            rp = write_block(writer, cfg.io.PLS_SHEET, ws_pls, rp, "M1-D. Cross-loadings (LV scores corr)", PLS1_cross, index=True)
            rp = write_block(writer, cfg.io.PLS_SHEET, ws_pls, rp, "M1-X. R²", PLS1_R2, index=False)
            rp = write_block(writer, cfg.io.PLS_SHEET, ws_pls, rp, "M1-Y. f² (per path)", PLS1_f2, index=False)
            rp = write_block(writer, cfg.io.PLS_SHEET, ws_pls, rp, "M1-Z. Q²(CV)", PLS1_Q2, index=False)
            rp = write_block(writer, cfg.io.PLS_SHEET, ws_pls, rp, "M1-V. Structural VIF", PLS1_VIF, index=False)
            rp = write_block(writer, cfg.io.PLS_SHEET, ws_pls, rp, "M1-ZZ. Bootstrap DIRECT paths (t/CI/Sig)", PLS1_BOOTPATH, index=False)

            rp += 2

            rp = write_block(writer, cfg.io.PLS_SHEET, ws_pls, rp, "MODEL 2 (Main): Commitment(formative) two-stage", PLS2_info, index=False)
            rp = write_block(writer, cfg.io.PLS_SHEET, ws_pls, rp, "M2-A. Commitment outer (model API)", PLS2_commitment, index=False)
            rp = write_block(writer, cfg.io.PLS_SHEET, ws_pls, rp, "M2-B. Outer model (model API)", PLS2_outer, index=False)
            rp = write_block(writer, cfg.io.PLS_SHEET, ws_pls, rp, "M2-C. CR / AVE", PLS2_quality, index=False)
            rp = write_block(writer, cfg.io.PLS_SHEET, ws_pls, rp, "M2-D. HTMT", PLS2_htmt, index=True)
            rp = write_block(writer, cfg.io.PLS_SHEET, ws_pls, rp, "M2-E. Cross-loadings (LV scores corr)", PLS2_cross, index=True)
            rp = write_block(writer, cfg.io.PLS_SHEET, ws_pls, rp, "M2-X. R²", PLS2_R2, index=False)
            rp = write_block(writer, cfg.io.PLS_SHEET, ws_pls, rp, "M2-Y. f² (per path)", PLS2_f2, index=False)
            rp = write_block(writer, cfg.io.PLS_SHEET, ws_pls, rp, "M2-Z. Q²(CV)", PLS2_Q2, index=False)
            rp = write_block(writer, cfg.io.PLS_SHEET, ws_pls, rp, "M2-V. Structural VIF", PLS2_VIF, index=False)
            rp = write_block(writer, cfg.io.PLS_SHEET, ws_pls, rp, "M2-ZZ. Bootstrap DIRECT paths (t/CI/Sig)", PLS2_BOOTPATH, index=False)

            ws_pls.freeze_panes = "A2"

    df_valid.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print("✅ 已輸出：", OUT_XLSX)
    print("✅ 已輸出：", OUT_CSV)
    return OUT_XLSX, OUT_CSV
