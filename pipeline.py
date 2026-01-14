# pls_project/pipeline.py
from __future__ import annotations

import re
from itertools import groupby

import numpy as np
import pandas as pd

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
from pls_project.pls_bootstrap import summarize_direct_ci


def run_pipeline(cog, reverse_target: bool, tag: str):
    cfg = cog.cfg
    OUT_XLSX = f"{cfg.io.OUT_XLSX_BASE}_{tag}.xlsx"
    OUT_CSV  = f"{cfg.io.OUT_CSV_BASE}_{tag}.csv"

    print("\n" + "=" * 80)
    print(f"🚀 Running scenario: {tag} | reverse_{cfg.scenario.SCENARIO_TARGET}={reverse_target}")
    print("=" * 80)

    df = pd.read_excel(cfg.io.XLSX_PATH, sheet_name=cfg.io.SHEET_NAME)
    df.columns = df.columns.astype(str).str.strip()
    print("Data shape:", df.shape)

    # find item columns
    SCALES = list(cfg.scales.SCALES)
    scale_alt = "|".join(SCALES)
    item_pat = re.compile(rf"^(?:{scale_alt})\d{{1,2}}$")
    item_cols = [c for c in df.columns if item_pat.fullmatch(str(c))]

    def sort_key(c):
        m = re.match(r"^([A-Z]+)(\d+)$", str(c))
        return (SCALES.index(m.group(1)) if m and m.group(1) in SCALES else 999, int(m.group(2)) if m else 999)

    item_cols = sorted(item_cols, key=sort_key)
    if not item_cols:
        raise ValueError("找不到題項欄位！請確認欄名是否為 PA1/BS1/.../LO6。")

    groups = [g for g in SCALES if any(col.startswith(g) for col in item_cols)]
    group_items = {g: [c for c in item_cols if c.startswith(g)] for g in groups}
    print("Groups:", groups)

    # reverse list
    REVERSE_ITEMS = list(cfg.scales.BASE_REVERSE_ITEMS)
    if reverse_target and cfg.scenario.SCENARIO_TARGET in group_items:
        REVERSE_ITEMS += list(group_items[cfg.scenario.SCENARIO_TARGET])

    # flags
    df["_flag_noexp"] = False
    df["_flag_dup"] = False

    # experience column
    EXP_COL = cfg.cols.EXP_COL
    TS_COL = cfg.cols.TS_COL
    USER_COL = cfg.cols.USER_COL
    EMAIL_COL = cfg.cols.EMAIL_COL

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

    df["_ts"] = pd.to_datetime(df[TS_COL], errors="coerce") if TS_COL in df.columns else pd.NaT
    df["_orig_order"] = np.arange(len(df))

    dup_key = [c for c in [USER_COL, EMAIL_COL] if c in df.columns]
    if dup_key:
        df_sorted = df.sort_values(["_ts", "_orig_order"], na_position="last").copy()
        df_sorted["_flag_dup"] = df_sorted.duplicated(subset=dup_key, keep=cfg.filt.KEEP_DUP)
        df = df_sorted.sort_values("_orig_order")

    # likert -> numeric
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
        (df["_missing_rate"] > cfg.filt.MAX_MISSING_RATE) |
        (df["_sd_items"] <= cfg.filt.MIN_SD_ITEMS) |
        (df["_longstring"] >= int(np.ceil(cfg.filt.LONGSTRING_PCT * k_all)))
    )

    # build exclusion reason
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

    # Paper tables
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
        Xpls = df_valid[item_cols].copy().astype(float)

        # missing handling (config-driven)
        if cfg.pls.PLS_MISSING == "none":
            if Xpls.isna().any().any():
                raise ValueError("PLS_MISSING='none' but missing values exist. Handle missing before import.")
        elif cfg.pls.PLS_MISSING == "listwise":
            Xpls = Xpls.dropna()
        elif cfg.pls.PLS_MISSING == "mean":
            # 這會生成新值；若你要完全乾淨請改成 none/listwise
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

        PLS1_htmt = htmt_matrix(Xpls[item_cols], {g: lv_blocks1[g] for g in order1}, order1, method=cfg.pls.HTMT_CORR_METHOD)

        PLS1_R2, PLS1_f2 = r2_f2_from_scores(scores1[order1], path1)
        PLS1_Q2 = q2_cv_from_scores(scores1[order1], path1, n_splits=int(cfg.pls.Q2_FOLDS), seed=int(cfg.pls.PLS_SEED))
        PLS1_VIF = structural_vif(scores1[order1], path1)

        PLS1_info = pd.DataFrame([{
            "Model": f"Model1 baseline (ACO/CCO separate) [{tag}]",
            "order": " > ".join(order1),
            "n(PLS)": int(Xpls.shape[0]),
            "scheme": cfg.pls.PLS_SCHEME,
            "missing": cfg.pls.PLS_MISSING,
            "sign_fix": bool(getattr(cfg.pls, "SIGN_FIX", True)),
            "estimates": "outer/path from plspm model API (strict)"
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
        PLS2_htmt = htmt_matrix(Xpls[item_cols], {g: group_items[g] for g in refl2}, refl2, method=cfg.pls.HTMT_CORR_METHOD)

        PLS2_R2, PLS2_f2 = r2_f2_from_scores(scores2[order2], path2)
        PLS2_Q2 = q2_cv_from_scores(scores2[order2], path2, n_splits=int(cfg.pls.Q2_FOLDS), seed=int(cfg.pls.PLS_SEED))
        PLS2_VIF = structural_vif(scores2[order2], path2)

        PLS2_commitment = PLS2_outer[PLS2_outer["Construct"] == "Commitment"].copy()

        PLS2_info = pd.DataFrame([{
            "Model": f"Model2 two-stage (Commitment formative) [{tag}]",
            "order": " > ".join(order2),
            "n(PLS)": int(Xpls.shape[0]),
            "scheme": cfg.pls.PLS_SCHEME,
            "missing": cfg.pls.PLS_MISSING,
            "sign_fix": bool(getattr(cfg.pls, "SIGN_FIX", True)),
            "estimates": "outer/path from plspm model API (strict)"
        }])

        # Optional: Bootstrap direct paths (if you already generate boot arrays elsewhere)
        # Here left empty unless you implement full bootstrap loop.
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
        r = write_block(writer, cfg.io.PAPER_SHEET, ws, r, f"B. Item analysis (all constructs stacked) [{tag}]",
                        ItemTable[["Construct","Item","n(complete)","Mean","SD","CITC","α if deleted","ωt if deleted"]], index=False)
        r = write_block(writer, cfg.io.PAPER_SHEET, ws, r, f"C. One-factor loadings (sklearn FA, all constructs stacked) [{tag}]",
                        FA1Table[["Construct","Item","n(complete)","Loading","h2","psi"]], index=False)
        r = write_block(writer, cfg.io.PAPER_SHEET, ws, r, f"D. One-factor EFA per construct (factor_analyzer, all constructs stacked) [{tag}]",
                        EFA1Table[["Construct","Item","n(complete)","EFA Loading","h2","psi"]], index=False)
        if EFA1Fit is not None and (not EFA1Fit.empty):
            r = write_block(writer, cfg.io.PAPER_SHEET, ws, r, f"D-2. EFA suitability (KMO / Bartlett) per construct [{tag}]",
                            EFA1Fit[["Construct","n(complete)","k(items)","KMO","Bartlett_p"]], index=False)

        if cfg.cfa.RUN_CFA:
            r = write_block(writer, c_
