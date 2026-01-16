# pls_project/pipeline.py
from __future__ import annotations

import re
from itertools import groupby

import numpy as np
import pandas as pd

from scipy.stats import f as fdist
from scipy.stats import ncf

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
    ols_fit,
    run_plspm_python,
    get_path_results,
    apply_sign_to_paths,
    get_sign_map_by_anchors,
    apply_sign_to_scores, 
)


from pls_project.pls_estimate import estimate_pls_basic_paper
from pls_project.pls_bootstrap import summarize_direct_ci

from pls_project.micom import micom_two_group, MICOMSettings

from pls_project.cbsem_wlsmv import run_cbsem_esem_then_cfa_sem_wlsmv
from pls_project.measureq_mlr import run_measureq
from pls_project.cmv_clf_mlr import run_cmv_clf_mlr

# inference modules
from pls_project.effects_inference import edges_from_path_df, summarize_effects_bootstrap_ci
from pls_project.htmt_inference import htmt_inference_bootstrap
from pls_project.pls_predict import plspredict_indicator_cv


# =========================================================
# CMV: Full collinearity VIF (Kock-style diagnostic) on LV scores
# =========================================================
def full_collinearity_vif(
    scores_df: pd.DataFrame,
    *,
    threshold: float = 3.3,
    decimals: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if scores_df is None or scores_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    cols = list(scores_df.columns)
    rows = []

    for y in cols:
        Xcols = [c for c in cols if c != y]
        if not Xcols:
            continue

        yv = pd.to_numeric(scores_df[y], errors="coerce").to_numpy()
        Xv = scores_df[Xcols].apply(pd.to_numeric, errors="coerce").to_numpy()

        ok = np.isfinite(yv) & np.isfinite(Xv).all(axis=1)
        if ok.sum() < max(5, len(Xcols) + 2):
            r2 = np.nan
            vif = np.nan
        else:
            _, r2 = ols_fit(yv[ok], Xv[ok])
            vif = (1.0 / (1.0 - r2)) if (pd.notna(r2) and (1.0 - r2) > 1e-12) else np.inf

        rows.append({
            "Construct": y,
            "R2_on_others": r2,
            "VIF_full": vif,
            "Threshold": float(threshold),
            "Pass(VIF<=TH)": (pd.notna(vif) and np.isfinite(vif) and (vif <= threshold)),
        })

    detail = pd.DataFrame(rows)
    if detail.empty:
        return pd.DataFrame(), pd.DataFrame()

    max_vif = pd.to_numeric(detail["VIF_full"], errors="coerce").max()
    pass_all = bool((pd.to_numeric(detail["VIF_full"], errors="coerce") <= threshold).fillna(False).all())

    summary = pd.DataFrame([{
        "Max_VIF_full": max_vif,
        "Threshold": float(threshold),
        "Pass_all(VIF<=TH)": pass_all,
        "Note": "Full collinearity VIF is commonly used as a CMV diagnostic on LV scores.",
    }])

    return summary.round(decimals), detail.round(decimals)


# =========================================================
# G*Power (equivalent) - multiple regression R^2 deviation from zero
# =========================================================
def gpower_required_n_multiple_regression(
    *,
    u_predictors: int,
    f2: float,
    alpha: float,
    power: float,
    n_max: int = 20000,
) -> int:
    u = int(u_predictors)
    if u < 1:
        raise ValueError("u_predictors must be >= 1")
    if f2 <= 0:
        raise ValueError("f2 must be > 0")
    if not (0 < alpha < 1) or not (0 < power < 1):
        raise ValueError("alpha and power must be in (0,1)")

    def _power_at_n(N: int) -> float:
        v = N - u - 1
        if v <= 1:
            return 0.0
        lam = float(f2) * float(N)
        fcrit = fdist.isf(alpha, u, v)
        return float(1.0 - ncf.cdf(fcrit, u, v, lam))

    N_lo = u + 3
    if _power_at_n(N_lo) >= power:
        return N_lo

    N = N_lo
    while N < n_max and _power_at_n(N) < power:
        N = int(N * 1.2) + 1

    if N >= n_max:
        return n_max

    hi = N
    lo = max(N_lo, int(hi / 1.2) - 5)
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if _power_at_n(mid) >= power:
            hi = mid
        else:
            lo = mid
    return hi


def gpower_table_for_path_model(
    *,
    path_df: pd.DataFrame,
    n_actual: int,
    f2: float,
    alpha: float,
    power: float,
) -> pd.DataFrame:
    if path_df is None or path_df.empty:
        return pd.DataFrame([{"Info": "path_df empty; G*Power not applicable."}])

    rows = []
    for to in path_df.index:
        preds = [c for c in path_df.columns if int(path_df.loc[to, c]) == 1]
        if preds:
            rows.append({"Endogenous": to, "k_predictors": len(preds), "Predictors": ", ".join(preds)})

    if not rows:
        return pd.DataFrame([{"Info": "No endogenous equations detected in path_df; G*Power not applicable."}])

    tmp = pd.DataFrame(rows)
    u_max = int(tmp["k_predictors"].max())

    n_req = gpower_required_n_multiple_regression(
        u_predictors=u_max,
        f2=float(f2),
        alpha=float(alpha),
        power=float(power),
    )

    summary = pd.DataFrame([{
        "Rule": "Multiple regression (R² deviation from zero), conservative on max predictors",
        "u_max_predictors": u_max,
        "f2": float(f2),
        "alpha": float(alpha),
        "power_target": float(power),
        "N_required": int(n_req),
        "N_actual": int(n_actual),
        "Pass(N_actual>=N_required)": bool(int(n_actual) >= int(n_req)),
    }])

    return pd.concat([summary, pd.DataFrame([{}]), tmp], ignore_index=True)


# =========================================================
# helpers
# =========================================================
def _apply_sign_to_scores(scores_df: pd.DataFrame, sign_map: dict) -> pd.DataFrame:
    out = scores_df.copy()
    for lv, s in sign_map.items():
        if lv in out.columns and int(s) == -1:
            out[lv] = -out[lv]
    return out


def _bootstrap_paths_single_model(
    cog,
    *,
    X: pd.DataFrame,
    path_df: pd.DataFrame,
    lv_blocks: dict,
    lv_modes: dict,
    order: list[str],
    key_df: pd.DataFrame,
    anchors: dict,
) -> np.ndarray:
    cfg = cog.cfg.pls
    B = int(cfg.PLS_BOOT)
    seed = int(cfg.PLS_SEED)
    retry = int(getattr(cfg, "BOOT_RETRY", 0))
    sign_fix_on = bool(getattr(cfg, "SIGN_FIX", True))

    rng = np.random.default_rng(seed)
    n = int(X.shape[0])
    boot = np.full((B, len(key_df)), np.nan, dtype=float)

    for b in range(B):
        idx = rng.integers(0, n, size=n)
        Xb = X.iloc[idx].reset_index(drop=True)

        ok = False
        for _ in range(max(1, retry + 1)):
            try:
                mb, sb = run_plspm_python(
                    cog,
                    Xb,
                    path_df,
                    lv_blocks,
                    lv_modes,
                    scaled=bool(getattr(cfg, "PLS_STANDARDIZED", True)),
                )

                sb = sb[order].copy()

                if sign_fix_on and anchors:
                    sign_map_b = get_sign_map_by_anchors(Xb, sb, anchors)
                else:
                    sign_map_b = {}

                pe_b = get_path_results(mb, path_df, strict=True)
                if sign_fix_on and anchors:
                    pe_b = apply_sign_to_paths(pe_b, sign_map_b)

                pe_b = key_df.merge(pe_b, on=["from", "to"], how="left")
                if pe_b["estimate"].isna().any():
                    raise RuntimeError("Missing path estimate in a bootstrap replicate (single model).")

                boot[b, :] = pe_b["estimate"].astype(float).values
                ok = True
                break
            except Exception:
                ok = False
                continue

        if (b % 200) == 0 and hasattr(cog, "log") and cog.log:
            cog.log.info(f"Bootstrap(single) {b}/{B} ok={ok}")

    return boot


def _bootstrap_paths_two_stage_model2(
    cog,
    *,
    X_stage1: pd.DataFrame,
    path1: pd.DataFrame,
    lv_blocks1: dict,
    lv_modes1: dict,
    order1: list[str],
    anchors1: dict,
    commitment_name: str,
    comp_lvs: list[str],
    comp_cols: list[str],
    path2: pd.DataFrame,
    lv_blocks2: dict,
    lv_modes2: dict,
    order2: list[str],
    anchors2: dict,
    key2: pd.DataFrame,
) -> np.ndarray:
    cfg = cog.cfg.pls
    B = int(cfg.PLS_BOOT)
    seed = int(cfg.PLS_SEED)
    retry = int(getattr(cfg, "BOOT_RETRY", 0))
    sign_fix_on = bool(getattr(cfg, "SIGN_FIX", True))

    rng = np.random.default_rng(seed)
    n = int(X_stage1.shape[0])
    boot2 = np.full((B, len(key2)), np.nan, dtype=float)

    for b in range(B):
        idx = rng.integers(0, n, size=n)
        Xb = X_stage1.iloc[idx].reset_index(drop=True)

        ok = False
        for _ in range(max(1, retry + 1)):
            try:
                # ---- stage1 ----
                m1b, s1b = run_plspm_python(
                    cog,
                    Xb,
                    path1,
                    lv_blocks1,
                    lv_modes1,
                    scaled=bool(getattr(cfg, "PLS_STANDARDIZED", True)),
                )
                s1b = s1b[order1].copy()

                if sign_fix_on and anchors1:
                    sign_map1b = get_sign_map_by_anchors(Xb, s1b, anchors1)
                    s1b = _apply_sign_to_scores(s1b, sign_map1b)

                for lv in comp_lvs:
                    if lv not in s1b.columns:
                        raise RuntimeError(f"Stage1 bootstrap missing component LV score: {lv}")
                if s1b[comp_lvs].isna().any().any():
                    raise RuntimeError("Stage1 bootstrap component scores contain NaN (clean mode forbids fillna).")

                Xb2 = Xb.copy()
                for lv, col in zip(comp_lvs, comp_cols):
                    Xb2[col] = s1b[lv].to_numpy()

                # ---- stage2 ----
                m2b, s2b = run_plspm_python(
                    cog,
                    Xb2,
                    path2,
                    lv_blocks2,
                    lv_modes2,
                    scaled=bool(getattr(cfg, "PLS_STANDARDIZED", True)),
                )
                s2b = s2b[order2].copy()

                if sign_fix_on and anchors2:
                    sign_map2b = get_sign_map_by_anchors(Xb2, s2b, anchors2)
                else:
                    sign_map2b = {}

                pe2b = get_path_results(m2b, path2, strict=True)
                if sign_fix_on and anchors2:
                    pe2b = apply_sign_to_paths(pe2b, sign_map2b)

                pe2b = key2.merge(pe2b, on=["from", "to"], how="left")
                if pe2b["estimate"].isna().any():
                    raise RuntimeError("Missing path estimate in a bootstrap replicate (model2).")

                boot2[b, :] = pe2b["estimate"].astype(float).values
                ok = True
                break

            except Exception:
                ok = False
                continue

        if (b % 200) == 0 and hasattr(cog, "log") and cog.log:
            cog.log.info(f"Bootstrap(two-stage) {b}/{B} ok={ok}")

    return boot2


# =========================================================
# main
# =========================================================
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

    if rename_map:
        df = df.rename(columns=rename_map)
        if df.columns.duplicated().any():
            dups = df.columns[df.columns.duplicated()].tolist()
            raise ValueError(
                f"Duplicate columns after rename_map. Duplicates={dups}. "
                "This usually means multiple original columns mapped to the same token."
            )

    TS_COL = resolved_cols.get("TS_COL", cfg.cols.TS_COL)
    USER_COL = resolved_cols.get("USER_COL", cfg.cols.USER_COL)
    EMAIL_COL = resolved_cols.get("EMAIL_COL", cfg.cols.EMAIL_COL)
    EXP_COL = resolved_cols.get("EXP_COL", cfg.cols.EXP_COL)

    EXP_COL_EXPER = "使用AI工具來學習之經驗"
    EXP_COL_FREQ = "使用AI工具來學習之頻率"
    if EXP_COL in df.columns and ("頻率" in str(EXP_COL)) and (EXP_COL_EXPER in df.columns):
        EXP_COL = EXP_COL_EXPER

    if rename_map:
        TS_COL = rename_map.get(TS_COL, TS_COL)
        USER_COL = rename_map.get(USER_COL, USER_COL)
        EMAIL_COL = rename_map.get(EMAIL_COL, EMAIL_COL)
        EXP_COL = rename_map.get(EXP_COL, EXP_COL)

    SCALES = list(scale_prefixes) if scale_prefixes else list(cfg.scales.SCALES)

    # ==============================
    # Find item columns (after rename)
    # ==============================
    scale_alt = "|".join(map(re.escape, SCALES))
    item_pat = re.compile(rf"^(?:{scale_alt})\d{{1,2}}$")
    item_cols = [c for c in df.columns if item_pat.fullmatch(str(c))]

    if not item_cols:
        raise ValueError(
            f"找不到題項欄位！profile={profile_name}, prefixes={SCALES}. "
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
    # Flags (experience / duplication / careless) with config switches
    # ==============================
    fcfg = cfg.filt
    FILTER_NOEXP = bool(getattr(fcfg, "FILTER_NOEXP", True))
    FILTER_DUP = bool(getattr(fcfg, "FILTER_DUPLICATE", True))
    FILTER_CARELESS = bool(getattr(fcfg, "FILTER_CARELESS", True))

    df["_flag_noexp"] = False
    df["_flag_dup"] = False
    df["_flag_careless"] = False

    def has_experience(x) -> bool:
        if pd.isna(x):
            return False
        if isinstance(x, (int, float, np.integer, np.floating)):
            return float(x) > 0

        s = str(x).strip()
        no_kw = ["沒有", "否", "不曾", "從未", "未", "無"]
        if any(k in s for k in no_kw):
            return False
        if re.search(r"\d", s) and any(u in s for u in ["小時", "天", "週", "月", "年"]):
            return True
        yes_kw = ["有", "是", "曾", "使用過"]
        if any(k in s for k in yes_kw):
            return True
        return False

    if FILTER_NOEXP and (EXP_COL in df.columns):
        df["_flag_noexp"] = ~df[EXP_COL].apply(has_experience)

    if FILTER_DUP:
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

    for c in REVERSE_ITEMS:
        if c in df.columns:
            df[c] = df[c].apply(reverse_1to5).astype(float)

    # ==============================
    # Careless detection (missing / SD / longstring / jump)
    # ==============================
    if FILTER_CARELESS:
        k_all = len(item_cols)
        miss_rate = df[item_cols].isna().mean(axis=1)
        sd_items = df[item_cols].std(axis=1, ddof=0)

        def max_run_length(arr) -> int:
            v = [int(x) for x in arr if pd.notna(x)]
            if len(v) == 0:
                return 0
            return max(sum(1 for _ in g) for _, g in groupby(v))

        longstring = df[item_cols].apply(lambda r: max_run_length(r.values), axis=1)

        use_miss = bool(getattr(fcfg, "CARELESS_USE_MISSING", True))
        use_sd = bool(getattr(fcfg, "CARELESS_USE_SD", True))
        use_long = bool(getattr(fcfg, "CARELESS_USE_LONGSTRING", True))

        use_jump = bool(getattr(fcfg, "USE_JUMP_FILTER", False))
        jump_diff = float(getattr(fcfg, "JUMP_DIFF", 3))
        jump_rate_th = float(getattr(fcfg, "JUMP_RATE_TH", 0.80))
        min_items_for_jump = int(getattr(fcfg, "MIN_ITEMS_FOR_JUMP", 4))

        cond = pd.Series(False, index=df.index)
        if use_miss:
            cond = cond | (miss_rate > cfg.filt.MAX_MISSING_RATE)
        if use_sd:
            cond = cond | (sd_items <= cfg.filt.MIN_SD_ITEMS)
        if use_long:
            cond = cond | (longstring >= int(np.ceil(cfg.filt.LONGSTRING_PCT * k_all)))

        if use_jump:
            def jump_rate_row(arr) -> float:
                v = [float(x) for x in arr if pd.notna(x)]
                if len(v) < min_items_for_jump:
                    return 0.0
                d = np.abs(np.diff(np.asarray(v, dtype=float)))
                if d.size == 0:
                    return 0.0
                return float(np.mean(d >= jump_diff))

            jump_rate = df[item_cols].apply(lambda r: jump_rate_row(r.values), axis=1)
            cond = cond | (jump_rate >= jump_rate_th)

        df["_flag_careless"] = cond

    def join_reasons(r):
        reasons = []
        if FILTER_NOEXP and bool(r.get("_flag_noexp", False)):
            reasons.append("無GenAI學習經驗")
        if FILTER_DUP and bool(r.get("_flag_dup", False)):
            reasons.append("重複填答")
        if FILTER_CARELESS and bool(r.get("_flag_careless", False)):
            reasons.append("疑似隨意/草率(缺漏/直線/長串/跳動)")
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
    CFA_Loadings, CFA_Info, CFA_Fit = run_cfa(cog, df_valid, groups, group_items, item_cols)

    # ==============================
    # CB-SEM line 1: ESEM -> CFA/SEM (ordered/WLSMV)
    # ==============================
    CBSEM = {
        "info": pd.DataFrame(),
        "ESEM_fit": pd.DataFrame(),
        "ESEM_loadings": pd.DataFrame(),
        "CFA_fit": pd.DataFrame(),
        "CFA_loadings": pd.DataFrame(),
        "SEM_fit": pd.DataFrame(),
        "SEM_paths": pd.DataFrame(),
    }
    if bool(getattr(cfg.cfa, "RUN_CBSEM_WLSMV", False)):
        try:
            df_cb = df_valid[item_cols].copy()
            nf = int(getattr(cfg.cfa, "ESEM_NFACTORS", 0))
            if nf <= 0:
                nf = len(groups)
            rotation = str(getattr(cfg.cfa, "ESEM_ROTATION", "geomin"))
            missing = str(getattr(cfg.cfa, "CBSEM_MISSING", "listwise"))

            sem_edges = list(getattr(cfg.cfa, "SEM_EDGES", []))
            if not sem_edges:
                sem_edges = list(getattr(cfg.pls, "MODEL1_EDGES", []))
            sem_edges = [(a, b) for (a, b) in sem_edges if (a in groups and b in groups)]
            run_sem = bool(getattr(cfg.cfa, "RUN_SEM_WLSMV", True))

            CBSEM = run_cbsem_esem_then_cfa_sem_wlsmv(
                cog,
                df_cb,
                items=item_cols,
                groups=groups,
                group_items=group_items,
                esem_nfactors=nf,
                rotation=rotation,
                missing=missing,
                run_sem=run_sem,
                sem_edges=sem_edges,
                rscript=str(getattr(cfg.cfa, "RSCRIPT_BIN", "Rscript")),
            )
        except Exception as e:
            CBSEM = {"info": pd.DataFrame([{"Error": f"CBSEM_WLSMV failed: {e}"}])}

    # ==============================
    # CB-SEM line 2: measureQ best-practice (MLR robustness)
    # ==============================
    MQ = {"info": pd.DataFrame(), "measureQ_log": pd.DataFrame()}
    if bool(getattr(cfg.cfa, "RUN_MEASUREQ", False)):
        try:
            model_lines = []
            for g in groups:
                its = group_items.get(g, [])
                if its:
                    model_lines.append(f"{g} =~ " + " + ".join(its))
            model_syntax = "\n".join(model_lines)

            MQ = run_measureq(
                cog,
                df_valid[item_cols].copy(),
                model_syntax=model_syntax,
                items=item_cols,
                b_no=int(getattr(cfg.cfa, "MEASUREQ_B_NO", 1000)),
                htmt=bool(getattr(cfg.cfa, "MEASUREQ_HTMT", True)),
                cluster=getattr(cfg.cfa, "MEASUREQ_CLUSTER", None),
                rscript=str(getattr(cfg.cfa, "RSCRIPT_BIN", "Rscript")),
            )
        except Exception as e:
            MQ = {"info": pd.DataFrame([{"Error": f"measureQ failed: {e}"}]), "measureQ_log": pd.DataFrame()}

    # ==============================
    # CB-SEM line 3: CMV robustness via CLF/ULMC-like model (MLR)
    # ==============================
    CMV3 = {
        "info": pd.DataFrame(),
        "BASE_fit": pd.DataFrame(),
        "CLF_fit": pd.DataFrame(),
        "DELTA_fit": pd.DataFrame(),
        "BASE_loadings": pd.DataFrame(),
        "CLF_loadings_sub": pd.DataFrame(),
        "CLF_loadings_method": pd.DataFrame(),
        "DELTA_loadings": pd.DataFrame(),
        "DELTA_summary": pd.DataFrame(),
        "console_log": pd.DataFrame(),
    }
    if bool(getattr(cfg.cfa, "RUN_CMV_CLF_MLR", False)):
        try:
            CMV3 = run_cmv_clf_mlr(
                cog,
                df_valid[item_cols].copy(),
                items=item_cols,
                groups=groups,
                group_items=group_items,
                estimator=str(getattr(cfg.cfa, "CLF_ESTIMATOR", "MLR")),
                missing=str(getattr(cfg.cfa, "CLF_MISSING", "ML")),
                method_name=str(getattr(cfg.cfa, "CLF_METHOD_NAME", "CMV")),
                orthogonal=bool(getattr(cfg.cfa, "CLF_ORTHOGONAL", True)),
                equal_loadings=bool(getattr(cfg.cfa, "CLF_EQUAL_LOADINGS", True)),
                delta_loading_flag=float(getattr(cfg.cfa, "CLF_DELTA_LOADING_FLAG", 0.10)),
                rscript=str(getattr(cfg.cfa, "RSCRIPT_BIN", "Rscript")),
            )
        except Exception as e:
            CMV3 = {"info": pd.DataFrame([{"Error": f"CMV_CLF_MLR failed: {e}"}])}

    # =========================================================
    # PLS + MICOM + Inference
    # =========================================================
    MICOM_out: dict[str, dict[str, pd.DataFrame]] = {}

    PLS1_info = PLS1_outer = PLS1_quality = PLS1_htmt = PLS1_cross = pd.DataFrame()
    PLS2_info = PLS2_outer = PLS2_quality = PLS2_htmt = PLS2_cross = PLS2_commitment = pd.DataFrame()

    PLS1_R2 = PLS1_f2 = PLS1_Q2 = PLS1_VIF = pd.DataFrame()
    PLS2_R2 = PLS2_f2 = PLS2_Q2 = PLS2_VIF = pd.DataFrame()

    PLS1_CMV_SUM = PLS1_VIF_FULL = pd.DataFrame()
    PLS2_CMV_SUM = PLS2_VIF_FULL = pd.DataFrame()

    PLS1_GPOWER = pd.DataFrame()
    PLS2_GPOWER = pd.DataFrame()

    PLS1_BOOTPATH = PLS2_BOOTPATH = pd.DataFrame()

    PLS1_PATHS = pd.DataFrame()
    PLS2_PATHS = pd.DataFrame()

    PLS1_EFF_POINT = PLS1_EFF_CI = PLS1_MED = pd.DataFrame()
    PLS2_EFF_POINT = PLS2_EFF_CI = PLS2_MED = pd.DataFrame()

    PLS1_HTMTINF_DETAIL = PLS1_HTMTINF_SUM = pd.DataFrame()
    PLS2_HTMTINF_DETAIL = PLS2_HTMTINF_SUM = pd.DataFrame()

    PLS1_PRED_DETAIL = PLS1_PRED_SUM = pd.DataFrame()
    PLS2_PRED_DETAIL = PLS2_PRED_SUM = pd.DataFrame()

    if cfg.pls.RUN_PLS:
        scheme_up = str(getattr(cfg.pls, "PLS_SCHEME", "PATH")).strip().upper()
        if getattr(cfg.pls, "CLEAN_MODE", True):
            if scheme_up in ("CENTROID",):
                raise ValueError("CLEAN_MODE=True forbids PLS_SCHEME='CENTROID' (SmartPLS4 removed it).")
            if (scheme_up == "PCA") and (not bool(getattr(cfg.pls, "ALLOW_PCA", False))):
                raise ValueError("CLEAN_MODE=True and ALLOW_PCA=False forbids PLS_SCHEME='PCA' approximation.")
            if str(getattr(cfg.pls, "PLS_MISSING", "listwise")).lower() == "mean":
                raise ValueError("CLEAN_MODE=True forbids PLS_MISSING='mean' (generates new values).")

        # ---- build Xpls + aligned meta index ----
        Xpls_raw = df_valid[item_cols].copy().astype(float)
        miss = str(getattr(cfg.pls, "PLS_MISSING", "listwise")).lower()

        if miss == "none":
            if Xpls_raw.isna().any().any():
                raise ValueError("PLS_MISSING='none' but missing values exist. Handle missing before import.")
            idx_keep = Xpls_raw.index
            Xpls = Xpls_raw.copy()
        elif miss == "listwise":
            idx_keep = Xpls_raw.dropna().index
            Xpls = Xpls_raw.loc[idx_keep].copy()
        elif miss == "mean":
            idx_keep = Xpls_raw.index
            Xpls = Xpls_raw.copy()
            Xpls = Xpls.apply(lambda s: s.fillna(s.mean()), axis=0)
        else:
            raise ValueError(f"Unknown PLS_MISSING: {cfg.pls.PLS_MISSING}")

        Xpls = Xpls.reset_index(drop=True)
        df_meta = df_valid.loc[idx_keep].reset_index(drop=True)

        RUN_FULL_VIF = bool(getattr(cfg.pls, "RUN_FULL_COLLINEARITY_VIF", True))
        FULL_VIF_TH = float(getattr(cfg.pls, "FULL_VIF_THRESHOLD", 3.3))
        DEC = int(getattr(cfg.pls, "PAPER_DECIMALS", 3))

        RUN_GPOWER = bool(getattr(cfg.pls, "RUN_GPOWER", False))
        GP_F2 = float(getattr(cfg.pls, "GPOWER_F2", 0.02))
        GP_ALPHA = float(getattr(cfg.pls, "GPOWER_ALPHA", 0.05))
        GP_POWER = float(getattr(cfg.pls, "GPOWER_POWER", 0.80))

        RUN_M1 = bool(getattr(cfg.pls, "RUN_MODEL1", True))
        RUN_M2 = bool(getattr(cfg.pls, "RUN_MODEL2", True))

        RUN_EFFECTS = bool(getattr(cfg.pls, "RUN_EFFECTS_INFERENCE", True))
        RUN_HTMT_INF = bool(getattr(cfg.pls, "RUN_HTMT_INFERENCE", True))
        RUN_PREDICT = bool(getattr(cfg.pls, "RUN_PLS_PREDICT", True))

        HTMT_BOOT = int(getattr(cfg.pls, "HTMT_BOOT", 200))
        HTMT_TH = float(getattr(cfg.pls, "HTMT_THRESHOLD", 0.90))
        PRED_FOLDS = int(getattr(cfg.pls, "PREDICT_FOLDS", 10))

        res1 = None
        order1 = None
        path1 = None
        scores1 = None
        lv_blocks1 = None
        lv_modes1 = None

        # --------------------------
        # Model1
        # --------------------------
        if RUN_M1:
            edges1_all = list(getattr(cfg.pls, "MODEL1_EDGES", []))
            edges1 = [(a, b) for (a, b) in edges1_all if (a in groups) and (b in groups)]
            nodes1 = sorted({x for e in edges1 for x in e}, key=lambda x: groups.index(x) if x in groups else 999)

            if not edges1 or not nodes1:
                PLS1_info = pd.DataFrame([{
                    "Info": "Model1 skipped: no valid edges after intersecting with detected groups.",
                    "profile": profile_name,
                    "tag": tag,
                }])
            else:
                order1 = topo_sort(nodes1, edges1)
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
                PLS1_PATHS = res1.get("paths_long", pd.DataFrame()).copy()

                PLS1_htmt = htmt_matrix(
                    Xpls[item_cols],
                    {g: lv_blocks1[g] for g in order1},
                    order1,
                    method=str(getattr(cfg.pls, "HTMT_CORR_METHOD", "pearson")),
                ).round(DEC)

                PLS1_R2, PLS1_f2 = r2_f2_from_scores(scores1[order1], path1)
                PLS1_Q2 = q2_cv_from_scores(scores1[order1], path1, n_splits=int(cfg.pls.Q2_FOLDS), seed=int(cfg.pls.PLS_SEED))
                PLS1_VIF = structural_vif(scores1[order1], path1)

                if RUN_FULL_VIF:
                    PLS1_CMV_SUM, PLS1_VIF_FULL = full_collinearity_vif(scores1[order1], threshold=FULL_VIF_TH, decimals=DEC)

                if RUN_GPOWER:
                    PLS1_GPOWER = gpower_table_for_path_model(
                        path_df=path1,
                        n_actual=int(Xpls.shape[0]),
                        f2=GP_F2,
                        alpha=GP_ALPHA,
                        power=GP_POWER,
                    )

                # MICOM
                RUN_MICOM = bool(getattr(cfg.mga, "RUN_MICOM", False))
                if RUN_MICOM:
                    split_vars = list(getattr(cfg.mga, "MICOM_SPLITS", getattr(cfg.mga, "MGA_SPLITS", [])))
                    Bm = int(getattr(cfg.mga, "MICOM_BOOT", getattr(cfg.mga, "MGA_BOOT", 200)))
                    alpha_m = float(getattr(cfg.mga, "MICOM_ALPHA", 0.05))
                    min_n = int(getattr(cfg.mga, "MGA_MIN_N_PER_GROUP", 30))
                    seed_m = int(getattr(cfg.pls, "PLS_SEED", 0))

                    settings = MICOMSettings(
                        B=Bm,
                        seed=seed_m,
                        alpha=alpha_m,
                        min_n_per_group=min_n,
                        standardized=True,
                    )

                    num_ratio_th = float(getattr(cfg.mga, "MICOM_NUMERIC_RATIO_TH", 0.80))

                    for sv in split_vars:
                        sv = str(sv)

                        if (scores1 is not None) and (sv in scores1.columns):
                            v = pd.to_numeric(scores1[sv], errors="coerce")
                            src = "LV_score"
                        elif sv in df_meta.columns:
                            v = df_meta[sv]
                            src = "meta"
                        else:
                            MICOM_out[sv] = {"info": pd.DataFrame([{"Error": f"Split var not found: {sv}"}])}
                            continue

                        vv = pd.to_numeric(v, errors="coerce")
                        num_ratio = float(vv.notna().mean())

                        if num_ratio >= num_ratio_th:
                            thr = float(np.nanmedian(vv))
                            if not np.isfinite(thr):
                                MICOM_out[sv] = {"info": pd.DataFrame([{"Error": f"Numeric split failed (median is NaN): {sv}"}])}
                                continue
                            gmask = (vv >= thr).fillna(False).to_numpy()
                            X_use = Xpls
                            note = f"{src}; numeric_ratio={num_ratio:.2f}; median split @ {thr:.4f}"
                        else:
                            s = v.astype(str).fillna("NA")
                            levels = s.value_counts().index.tolist()
                            if len(levels) < 2:
                                MICOM_out[sv] = {"info": pd.DataFrame([{"Error": f"Split var has <2 levels: {sv}"}])}
                                continue
                            top2 = levels[:2]
                            keep = s.isin(top2).to_numpy()
                            X_use = Xpls.loc[keep].reset_index(drop=True)
                            s2 = s.loc[keep].reset_index(drop=True)
                            gmask = (s2 == top2[0]).to_numpy()
                            note = f"{src}; numeric_ratio={num_ratio:.2f}; top2 levels {top2}"

                        try:
                            out = micom_two_group(
                                cog,
                                X_full=X_use,
                                path_df=path1,
                                lv_blocks=lv_blocks1,
                                lv_modes=lv_modes1,
                                order=order1,
                                group_mask=gmask,
                                anchors=res1.get("anchors", {}) if isinstance(res1, dict) else {},
                                settings=settings,
                            )
                            out["info"] = pd.concat(
                                [pd.DataFrame([{"split_var": sv, "split_note": note}]), out.get("info", pd.DataFrame())],
                                ignore_index=True,
                            )
                            MICOM_out[sv] = out
                        except Exception as e:
                            MICOM_out[sv] = {"info": pd.DataFrame([{"Error": f"MICOM failed ({sv}): {e}"}])}

                # Inference (Model1): HTMT CI + PLSpredict
                if RUN_HTMT_INF:
                    try:
                        PLS1_HTMTINF_DETAIL, PLS1_HTMTINF_SUM = htmt_inference_bootstrap(
                            X_items=Xpls[item_cols],
                            group_items={g: lv_blocks1[g] for g in order1},
                            groups=order1,
                            B=HTMT_BOOT,
                            seed=int(cfg.pls.PLS_SEED),
                            qlo=float(getattr(cfg.pls, "BOOT_CI_LO", 0.025)),
                            qhi=float(getattr(cfg.pls, "BOOT_CI_HI", 0.975)),
                            threshold=HTMT_TH,
                            corr_method=str(getattr(cfg.pls, "HTMT_CORR_METHOD", "pearson")),
                        )
                        PLS1_HTMTINF_DETAIL = PLS1_HTMTINF_DETAIL.round(DEC)
                        PLS1_HTMTINF_SUM = PLS1_HTMTINF_SUM.round(DEC)
                    except Exception as e:
                        PLS1_HTMTINF_SUM = pd.DataFrame([{"Error": f"HTMT inference failed: {e}"}])

                if RUN_PREDICT:
                    try:
                        PLS1_PRED_DETAIL, PLS1_PRED_SUM = plspredict_indicator_cv(
                            cog,
                            X_items=Xpls[item_cols],
                            path_df=path1,
                            lv_blocks=lv_blocks1,
                            lv_modes=lv_modes1,
                            order=order1,
                            n_splits=PRED_FOLDS,
                            seed=int(cfg.pls.PLS_SEED),
                            exclude_endogenous=None,
                        )
                        PLS1_PRED_DETAIL = PLS1_PRED_DETAIL.round(DEC)
                        PLS1_PRED_SUM = PLS1_PRED_SUM.round(DEC)
                    except Exception as e:
                        PLS1_PRED_SUM = pd.DataFrame([{"Error": f"PLSpredict failed: {e}"}])

                PLS1_info = pd.DataFrame([{
                    "Model": f"Model1 [{tag}]",
                    "profile": profile_name,
                    "order": " > ".join(order1),
                    "n(PLS)": int(Xpls.shape[0]),
                    "scheme": cfg.pls.PLS_SCHEME,
                    "missing": cfg.pls.PLS_MISSING,
                    "sign_fix": bool(getattr(cfg.pls, "SIGN_FIX", True)),
                    "B(bootstrap)": int(cfg.pls.PLS_BOOT),
                    "CMV_fullVIF_th": FULL_VIF_TH if RUN_FULL_VIF else "",
                    "GPower(f2,alpha,power)": f"{GP_F2},{GP_ALPHA},{GP_POWER}" if RUN_GPOWER else "",
                    "MICOM": bool(getattr(cfg.mga, "RUN_MICOM", False)),
                    "Inference(Effects/HTMT/PLSpredict)": f"{RUN_EFFECTS}/{RUN_HTMT_INF}/{RUN_PREDICT}",
                    "estimates": "outer/path from plspm model API (strict)",
                }])

                if int(cfg.pls.PLS_BOOT) > 0:
                    boot1 = _bootstrap_paths_single_model(
                        cog,
                        X=Xpls,
                        path_df=path1,
                        lv_blocks=lv_blocks1,
                        lv_modes=lv_modes1,
                        order=order1,
                        key_df=res1["key"],
                        anchors=res1["anchors"],
                    )
                    PLS1_BOOTPATH = summarize_direct_ci(cog, res1["key"], res1["est"], boot1)

                    if RUN_EFFECTS:
                        try:
                            edges_for_eff = edges_from_path_df(path1)
                            PLS1_EFF_POINT, PLS1_EFF_CI, PLS1_MED = summarize_effects_bootstrap_ci(
                                order=order1,
                                edges=edges_for_eff,
                                key_df=res1["key"],
                                point_est=res1["est"],
                                boot=boot1,
                                qlo=float(getattr(cfg.pls, "BOOT_CI_LO", 0.025)),
                                qhi=float(getattr(cfg.pls, "BOOT_CI_HI", 0.975)),
                                alpha=float(getattr(cfg.pls, "BOOT_ALPHA", 0.05)),
                            )
                            PLS1_EFF_POINT = PLS1_EFF_POINT.round(DEC)
                            PLS1_EFF_CI = PLS1_EFF_CI.round(DEC)
                            PLS1_MED = PLS1_MED.round(DEC)
                        except Exception as e:
                            PLS1_EFF_CI = pd.DataFrame([{"Error": f"Effects inference failed: {e}"}])

        else:
            PLS1_info = pd.DataFrame([{
                "Info": "Model1 disabled by config (RUN_MODEL1=False).",
                "profile": profile_name,
                "tag": tag,
            }])

        # --------------------------
        # Model2 (two-stage)
        # --------------------------
        if RUN_M2:
            if res1 is None or scores1 is None or order1 is None or path1 is None:
                PLS2_info = pd.DataFrame([{
                    "Info": "Model2 skipped: requires Model1 results but Model1 was not run / not available.",
                    "profile": profile_name,
                    "tag": tag,
                }])
            else:
                commitment = str(getattr(cfg.pls, "MODEL2_COMMITMENT_NAME", "Commitment"))
                comp_lvs = list(getattr(cfg.pls, "MODEL2_COMPONENT_LVS", ["ACO", "CCO"]))
                comp_cols = list(getattr(cfg.pls, "MODEL2_COMPONENT_COLS", ["ACO_score", "CCO_score"]))
                if len(comp_lvs) != len(comp_cols):
                    raise ValueError("MODEL2_COMPONENT_LVS and MODEL2_COMPONENT_COLS must have the same length.")
                commitment_anchor = getattr(cfg.pls, "MODEL2_COMMITMENT_ANCHOR", None)

                missing_comp = [lv for lv in comp_lvs if lv not in scores1.columns]
                if missing_comp:
                    PLS2_info = pd.DataFrame([{
                        "Info": f"Model2 skipped: missing stage1 LV scores {missing_comp}.",
                        "profile": profile_name,
                        "tag": tag,
                    }])
                elif scores1[comp_lvs].isna().any().any():
                    raise ValueError("Model2 blocked: stage1 component scores contain NaN (clean mode forbids fillna).")
                else:
                    edges2_all = list(getattr(cfg.pls, "MODEL2_EDGES", []))

                    def ok_node(x: str) -> bool:
                        return (x in groups) or (x == commitment)

                    edges2 = [(a, b) for (a, b) in edges2_all if ok_node(a) and ok_node(b)]
                    nodes2 = sorted({x for e in edges2 for x in e if x != commitment}, key=lambda x: groups.index(x) if x in groups else 999)
                    if commitment not in nodes2:
                        nodes2.append(commitment)

                    if not edges2 or not nodes2:
                        PLS2_info = pd.DataFrame([{
                            "Info": "Model2 skipped: no valid edges after intersecting with detected groups + commitment.",
                            "profile": profile_name,
                            "tag": tag,
                        }])
                    else:
                        order2 = topo_sort(nodes2, edges2)
                        path2 = make_plspm_path(order2, edges2)

                        Xpls2 = Xpls.copy()
                        for lv, col in zip(comp_lvs, comp_cols):
                            Xpls2[col] = pd.to_numeric(scores1[lv], errors="coerce").to_numpy()

                        lv_blocks2 = {}
                        for lv in order2:
                            if lv == commitment:
                                lv_blocks2[lv] = list(comp_cols)
                            else:
                                lv_blocks2[lv] = group_items[lv]

                        lv_modes2 = {lv: ("B" if lv == commitment else "A") for lv in order2}
                        lv_modes2[commitment] = str(getattr(cfg.pls, "MODEL2_COMMITMENT_MODE", "B")).upper()

                        stage2_indicators = []
                        for lv in order2:
                            stage2_indicators += lv_blocks2[lv]
                        stage2_indicators = list(dict.fromkeys(stage2_indicators))

                        anchor_override = {commitment: commitment_anchor} if commitment_anchor else None

                        res2 = estimate_pls_basic_paper(
                            cog,
                            Xpls=Xpls2,
                            item_cols=stage2_indicators,
                            path_df=path2,
                            lv_blocks=lv_blocks2,
                            lv_modes=lv_modes2,
                            order=order2,
                            anchor_overrides=anchor_override,
                        )

                        PLS2_cross = res2["PLS_cross"]
                        PLS2_outer = res2["PLS_outer"]
                        PLS2_quality = res2["PLS_quality"]
                        scores2 = res2["scores"]
                        PLS2_PATHS = res2.get("paths_long", pd.DataFrame()).copy()

                        refl2 = [lv for lv in order2 if lv != commitment]
                        if refl2:
                            PLS2_htmt = htmt_matrix(
                                Xpls[item_cols],
                                {g: group_items[g] for g in refl2},
                                refl2,
                                method=str(getattr(cfg.pls, "HTMT_CORR_METHOD", "pearson")),
                            ).round(DEC)

                        PLS2_R2, PLS2_f2 = r2_f2_from_scores(scores2[order2], path2)
                        PLS2_Q2 = q2_cv_from_scores(scores2[order2], path2, n_splits=int(cfg.pls.Q2_FOLDS), seed=int(cfg.pls.PLS_SEED))
                        PLS2_VIF = structural_vif(scores2[order2], path2)

                        if RUN_FULL_VIF:
                            PLS2_CMV_SUM, PLS2_VIF_FULL = full_collinearity_vif(scores2[order2], threshold=FULL_VIF_TH, decimals=DEC)

                        if RUN_GPOWER:
                            PLS2_GPOWER = gpower_table_for_path_model(
                                path_df=path2,
                                n_actual=int(Xpls.shape[0]),
                                f2=GP_F2,
                                alpha=GP_ALPHA,
                                power=GP_POWER,
                            )

                        PLS2_commitment = PLS2_outer[PLS2_outer["Construct"] == commitment].copy()

                        # Inference (Model2): HTMT CI + PLSpredict
                        if RUN_HTMT_INF and refl2:
                            try:
                                PLS2_HTMTINF_DETAIL, PLS2_HTMTINF_SUM = htmt_inference_bootstrap(
                                    X_items=Xpls[item_cols],
                                    group_items={g: group_items[g] for g in refl2},
                                    groups=refl2,
                                    B=HTMT_BOOT,
                                    seed=int(cfg.pls.PLS_SEED),
                                    qlo=float(getattr(cfg.pls, "BOOT_CI_LO", 0.025)),
                                    qhi=float(getattr(cfg.pls, "BOOT_CI_HI", 0.975)),
                                    threshold=HTMT_TH,
                                    corr_method=str(getattr(cfg.pls, "HTMT_CORR_METHOD", "pearson")),
                                )
                                PLS2_HTMTINF_DETAIL = PLS2_HTMTINF_DETAIL.round(DEC)
                                PLS2_HTMTINF_SUM = PLS2_HTMTINF_SUM.round(DEC)
                            except Exception as e:
                                PLS2_HTMTINF_SUM = pd.DataFrame([{"Error": f"HTMT inference failed: {e}"}])

                        if RUN_PREDICT:
                            try:
                                PLS2_PRED_DETAIL, PLS2_PRED_SUM = plspredict_indicator_cv(
                                    cog,
                                    X_items=Xpls2[stage2_indicators],
                                    path_df=path2,
                                    lv_blocks=lv_blocks2,
                                    lv_modes=lv_modes2,
                                    order=order2,
                                    n_splits=PRED_FOLDS,
                                    seed=int(cfg.pls.PLS_SEED),
                                    exclude_endogenous=None,
                                )
                                PLS2_PRED_DETAIL = PLS2_PRED_DETAIL.round(DEC)
                                PLS2_PRED_SUM = PLS2_PRED_SUM.round(DEC)
                            except Exception as e:
                                PLS2_PRED_SUM = pd.DataFrame([{"Error": f"PLSpredict failed: {e}"}])

                        PLS2_info = pd.DataFrame([{
                            "Model": f"Model2 [{tag}]",
                            "profile": profile_name,
                            "order": " > ".join(order2),
                            "n(PLS)": int(Xpls.shape[0]),
                            "scheme": cfg.pls.PLS_SCHEME,
                            "missing": cfg.pls.PLS_MISSING,
                            "sign_fix": bool(getattr(cfg.pls, "SIGN_FIX", True)),
                            "B(bootstrap)": int(cfg.pls.PLS_BOOT),
                            "commitment": commitment,
                            "components": ",".join(comp_lvs),
                            "CMV_fullVIF_th": FULL_VIF_TH if RUN_FULL_VIF else "",
                            "GPower(f2,alpha,power)": f"{GP_F2},{GP_ALPHA},{GP_POWER}" if RUN_GPOWER else "",
                            "Inference(Effects/HTMT/PLSpredict)": f"{RUN_EFFECTS}/{RUN_HTMT_INF}/{RUN_PREDICT}",
                            "estimates": "outer/path from plspm model API (strict)",
                        }])

                        if int(cfg.pls.PLS_BOOT) > 0:
                            boot2 = _bootstrap_paths_two_stage_model2(
                                cog,
                                X_stage1=Xpls,
                                path1=path1,
                                lv_blocks1={lv: group_items[lv] for lv in order1},
                                lv_modes1={lv: "A" for lv in order1},
                                order1=order1,
                                anchors1=res1["anchors"],
                                commitment_name=commitment,
                                comp_lvs=comp_lvs,
                                comp_cols=comp_cols,
                                path2=path2,
                                lv_blocks2=lv_blocks2,
                                lv_modes2=lv_modes2,
                                order2=order2,
                                anchors2=res2["anchors"],
                                key2=res2["key"],
                            )
                            PLS2_BOOTPATH = summarize_direct_ci(cog, res2["key"], res2["est"], boot2)

                            if RUN_EFFECTS:
                                try:
                                    edges_for_eff2 = edges_from_path_df(path2)
                                    PLS2_EFF_POINT, PLS2_EFF_CI, PLS2_MED = summarize_effects_bootstrap_ci(
                                        order=order2,
                                        edges=edges_for_eff2,
                                        key_df=res2["key"],
                                        point_est=res2["est"],
                                        boot=boot2,
                                        qlo=float(getattr(cfg.pls, "BOOT_CI_LO", 0.025)),
                                        qhi=float(getattr(cfg.pls, "BOOT_CI_HI", 0.975)),
                                        alpha=float(getattr(cfg.pls, "BOOT_ALPHA", 0.05)),
                                    )
                                    PLS2_EFF_POINT = PLS2_EFF_POINT.round(DEC)
                                    PLS2_EFF_CI = PLS2_EFF_CI.round(DEC)
                                    PLS2_MED = PLS2_MED.round(DEC)
                                except Exception as e:
                                    PLS2_EFF_CI = pd.DataFrame([{"Error": f"Effects inference failed: {e}"}])

        else:
            PLS2_info = pd.DataFrame([{
                "Info": "Model2 disabled by config (RUN_MODEL2=False).",
                "profile": profile_name,
                "tag": tag,
            }])

    # ==============================
    # Export Excel + CSV
    # ==============================
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        if cfg.io.EXPORT_EXCLUDED_SHEET and (excluded_df is not None) and (not excluded_df.empty):
            excluded_df.to_excel(writer, sheet_name="排除樣本", index=False)

        # ---- Paper ----
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

        # ---- MICOM ----
        if bool(getattr(cfg.mga, "RUN_MICOM", False)):
            micom_sheet = str(getattr(cfg.io, "MICOM_SHEET", "MICOM"))
            ws_micom = get_or_create_ws(writer, micom_sheet)
            rr0 = 0

            if not MICOM_out:
                rr0 = write_block(
                    writer, micom_sheet, ws_micom, rr0,
                    f"MICOM [{tag}]",
                    pd.DataFrame([{"Info": "MICOM enabled but no results (Model1 not run or failed)."}]),
                    index=False,
                )
            else:
                for sv, res in MICOM_out.items():
                    rr0 = write_block(writer, micom_sheet, ws_micom, rr0, f"MICOM - {sv} - INFO [{tag}]", res.get("info", pd.DataFrame()), index=False)
                    rr0 = write_block(writer, micom_sheet, ws_micom, rr0, f"MICOM - {sv} - Step1 Configural [{tag}]", res.get("step1_configural", pd.DataFrame()), index=False)
                    rr0 = write_block(writer, micom_sheet, ws_micom, rr0, f"MICOM - {sv} - Step2 Compositional [{tag}]", res.get("step2_compositional", pd.DataFrame()), index=False)
                    rr0 = write_block(writer, micom_sheet, ws_micom, rr0, f"MICOM - {sv} - Step3 Mean/Var [{tag}]", res.get("step3_means_vars", pd.DataFrame()), index=False)
                    rr0 = write_block(writer, micom_sheet, ws_micom, rr0, f"MICOM - {sv} - Summary [{tag}]", res.get("summary", pd.DataFrame()), index=False)
                    rr0 += 1

            ws_micom.freeze_panes = "A2"

        # ---- CBSEM_WLSMV ----
        if bool(getattr(cfg.cfa, "RUN_CBSEM_WLSMV", False)):
            cb_sheet = str(getattr(cfg.io, "CBSEM_SHEET", "CBSEM_WLSMV"))
            ws_cb = get_or_create_ws(writer, cb_sheet)
            rc = 0
            rc = write_block(writer, cb_sheet, ws_cb, rc, f"CBSEM-WLSMV INFO [{tag}]", CBSEM.get("info", pd.DataFrame()), index=False)
            rc = write_block(writer, cb_sheet, ws_cb, rc, f"A. ESEM fit (scaled) [{tag}]", CBSEM.get("ESEM_fit", pd.DataFrame()), index=False)
            rc = write_block(writer, cb_sheet, ws_cb, rc, f"B. ESEM standardized loadings [{tag}]", CBSEM.get("ESEM_loadings", pd.DataFrame()), index=False)
            rc = write_block(writer, cb_sheet, ws_cb, rc, f"C. CFA fit (scaled) [{tag}]", CBSEM.get("CFA_fit", pd.DataFrame()), index=False)
            rc = write_block(writer, cb_sheet, ws_cb, rc, f"D. CFA standardized loadings [{tag}]", CBSEM.get("CFA_loadings", pd.DataFrame()), index=False)
            rc = write_block(writer, cb_sheet, ws_cb, rc, f"E. SEM fit (scaled) [{tag}]", CBSEM.get("SEM_fit", pd.DataFrame()), index=False)
            rc = write_block(writer, cb_sheet, ws_cb, rc, f"F. SEM standardized paths [{tag}]", CBSEM.get("SEM_paths", pd.DataFrame()), index=False)
            ws_cb.freeze_panes = "A2"

        # ---- MEASUREQ ----
        if bool(getattr(cfg.cfa, "RUN_MEASUREQ", False)):
            mq_sheet = str(getattr(cfg.io, "MEASUREQ_SHEET", "MEASUREQ"))
            ws_mq = get_or_create_ws(writer, mq_sheet)
            rm = 0
            rm = write_block(writer, mq_sheet, ws_mq, rm, f"measureQ INFO [{tag}]", MQ.get("info", pd.DataFrame()), index=False)
            keys = [k for k in MQ.keys() if k not in ("info",)]
            for k in keys:
                rm = write_block(writer, mq_sheet, ws_mq, rm, f"{k} [{tag}]", MQ.get(k, pd.DataFrame()), index=False)
            ws_mq.freeze_panes = "A2"

        # ---- CMV_CLF_MLR ----
        if bool(getattr(cfg.cfa, "RUN_CMV_CLF_MLR", False)):
            cmv_sheet = str(getattr(cfg.io, "CMV_CLF_SHEET", "CMV_CLF_MLR"))
            ws_cmv = get_or_create_ws(writer, cmv_sheet)
            rr = 0
            rr = write_block(writer, cmv_sheet, ws_cmv, rr, f"CMV-CLF INFO [{tag}]", CMV3.get("info", pd.DataFrame()), index=False)
            rr = write_block(writer, cmv_sheet, ws_cmv, rr, f"A. Baseline fit [{tag}]", CMV3.get("BASE_fit", pd.DataFrame()), index=False)
            rr = write_block(writer, cmv_sheet, ws_cmv, rr, f"B. CLF fit [{tag}]", CMV3.get("CLF_fit", pd.DataFrame()), index=False)
            rr = write_block(writer, cmv_sheet, ws_cmv, rr, f"C. Fit delta (CLF-BASE) [{tag}]", CMV3.get("DELTA_fit", pd.DataFrame()), index=False)
            rr = write_block(writer, cmv_sheet, ws_cmv, rr, f"D. Baseline standardized loadings [{tag}]", CMV3.get("BASE_loadings", pd.DataFrame()), index=False)
            rr = write_block(writer, cmv_sheet, ws_cmv, rr, f"E. CLF substantive loadings [{tag}]", CMV3.get("CLF_loadings_sub", pd.DataFrame()), index=False)
            rr = write_block(writer, cmv_sheet, ws_cmv, rr, f"F. CLF method loadings [{tag}]", CMV3.get("CLF_loadings_method", pd.DataFrame()), index=False)
            rr = write_block(writer, cmv_sheet, ws_cmv, rr, f"G. Loading delta (CLF-BASE) [{tag}]", CMV3.get("DELTA_loadings", pd.DataFrame()), index=False)
            rr = write_block(writer, cmv_sheet, ws_cmv, rr, f"H. Delta summary [{tag}]", CMV3.get("DELTA_summary", pd.DataFrame()), index=False)
            rr = write_block(writer, cmv_sheet, ws_cmv, rr, f"Z. Console log [{tag}]", CMV3.get("console_log", pd.DataFrame()), index=False)
            ws_cmv.freeze_panes = "A2"

        # ---- PLS ----
        if cfg.pls.RUN_PLS:
            ws_pls = get_or_create_ws(writer, cfg.io.PLS_SHEET)
            rp = 0

            # ===== Model 1 =====
            rp = write_block(writer, cfg.io.PLS_SHEET, ws_pls, rp, "MODEL 1", PLS1_info, index=False)
            rp = write_block(writer, cfg.io.PLS_SHEET, ws_pls, rp, "M1-A. Outer model (model API)", PLS1_outer, index=False)
            rp = write_block(writer, cfg.io.PLS_SHEET, ws_pls, rp, "M1-B. CR / AVE", PLS1_quality, index=False)
            rp = write_block(writer, cfg.io.PLS_SHEET, ws_pls, rp, "M1-C. HTMT", PLS1_htmt, index=True)
            rp = write_block(writer, cfg.io.PLS_SHEET, ws_pls, rp, "M1-CI. HTMT inference summary", PLS1_HTMTINF_SUM, index=False)
            rp = write_block(writer, cfg.io.PLS_SHEET, ws_pls, rp, "M1-CI. HTMT inference detail", PLS1_HTMTINF_DETAIL, index=False)

            rp = write_block(writer, cfg.io.PLS_SHEET, ws_pls, rp, "M1-D. Cross-loadings (LV scores corr)", PLS1_cross, index=True)
            rp = write_block(writer, cfg.io.PLS_SHEET, ws_pls, rp, "M1-Paths. Path coefficients (model API)", PLS1_PATHS, index=False)

            rp = write_block(writer, cfg.io.PLS_SHEET, ws_pls, rp, "M1-X. R²", PLS1_R2, index=False)
            rp = write_block(writer, cfg.io.PLS_SHEET, ws_pls, rp, "M1-Y. f² (per path)", PLS1_f2, index=False)
            rp = write_block(writer, cfg.io.PLS_SHEET, ws_pls, rp, "M1-Z. Q²(CV)", PLS1_Q2, index=False)
            rp = write_block(writer, cfg.io.PLS_SHEET, ws_pls, rp, "M1-V. Structural VIF", PLS1_VIF, index=False)

            rp = write_block(writer, cfg.io.PLS_SHEET, ws_pls, rp, "M1-CMV. Full collinearity VIF summary", PLS1_CMV_SUM, index=False)
            rp = write_block(writer, cfg.io.PLS_SHEET, ws_pls, rp, "M1-CMV. Full collinearity VIF detail", PLS1_VIF_FULL, index=False)

            rp = write_block(writer, cfg.io.PLS_SHEET, ws_pls, rp, "M1-Power. G*Power (equivalent) table", PLS1_GPOWER, index=False)

            rp = write_block(writer, cfg.io.PLS_SHEET, ws_pls, rp, "M1-Predict. PLSpredict summary", PLS1_PRED_SUM, index=False)
            rp = write_block(writer, cfg.io.PLS_SHEET, ws_pls, rp, "M1-Predict. PLSpredict detail", PLS1_PRED_DETAIL, index=False)

            rp = write_block(writer, cfg.io.PLS_SHEET, ws_pls, rp, "M1-ZZ. Bootstrap DIRECT paths (t/CI/Sig)", PLS1_BOOTPATH, index=False)

            rp = write_block(writer, cfg.io.PLS_SHEET, ws_pls, rp, "M1-Effects. Point (direct/indirect/total/VAF)", PLS1_EFF_POINT, index=False)
            rp = write_block(writer, cfg.io.PLS_SHEET, ws_pls, rp, "M1-Effects. Bootstrap CI (indirect/total)", PLS1_EFF_CI, index=False)
            rp = write_block(writer, cfg.io.PLS_SHEET, ws_pls, rp, "M1-Effects. Mediation (VAF label)", PLS1_MED, index=False)

            rp += 2

            # ===== Model 2 =====
            rp = write_block(writer, cfg.io.PLS_SHEET, ws_pls, rp, "MODEL 2", PLS2_info, index=False)
            rp = write_block(writer, cfg.io.PLS_SHEET, ws_pls, rp, "M2-A. Commitment outer (model API)", PLS2_commitment, index=False)
            rp = write_block(writer, cfg.io.PLS_SHEET, ws_pls, rp, "M2-B. Outer model (model API)", PLS2_outer, index=False)
            rp = write_block(writer, cfg.io.PLS_SHEET, ws_pls, rp, "M2-C. CR / AVE", PLS2_quality, index=False)
            rp = write_block(writer, cfg.io.PLS_SHEET, ws_pls, rp, "M2-D. HTMT", PLS2_htmt, index=True)
            rp = write_block(writer, cfg.io.PLS_SHEET, ws_pls, rp, "M2-CI. HTMT inference summary", PLS2_HTMTINF_SUM, index=False)
            rp = write_block(writer, cfg.io.PLS_SHEET, ws_pls, rp, "M2-CI. HTMT inference detail", PLS2_HTMTINF_DETAIL, index=False)

            rp = write_block(writer, cfg.io.PLS_SHEET, ws_pls, rp, "M2-E. Cross-loadings (LV scores corr)", PLS2_cross, index=True)
            rp = write_block(writer, cfg.io.PLS_SHEET, ws_pls, rp, "M2-Paths. Path coefficients (model API)", PLS2_PATHS, index=False)

            rp = write_block(writer, cfg.io.PLS_SHEET, ws_pls, rp, "M2-X. R²", PLS2_R2, index=False)
            rp = write_block(writer, cfg.io.PLS_SHEET, ws_pls, rp, "M2-Y. f² (per path)", PLS2_f2, index=False)
            rp = write_block(writer, cfg.io.PLS_SHEET, ws_pls, rp, "M2-Z. Q²(CV)", PLS2_Q2, index=False)
            rp = write_block(writer, cfg.io.PLS_SHEET, ws_pls, rp, "M2-V. Structural VIF", PLS2_VIF, index=False)

            rp = write_block(writer, cfg.io.PLS_SHEET, ws_pls, rp, "M2-CMV. Full collinearity VIF summary", PLS2_CMV_SUM, index=False)
            rp = write_block(writer, cfg.io.PLS_SHEET, ws_pls, rp, "M2-CMV. Full collinearity VIF detail", PLS2_VIF_FULL, index=False)

            rp = write_block(writer, cfg.io.PLS_SHEET, ws_pls, rp, "M2-Power. G*Power (equivalent) table", PLS2_GPOWER, index=False)

            rp = write_block(writer, cfg.io.PLS_SHEET, ws_pls, rp, "M2-Predict. PLSpredict summary", PLS2_PRED_SUM, index=False)
            rp = write_block(writer, cfg.io.PLS_SHEET, ws_pls, rp, "M2-Predict. PLSpredict detail", PLS2_PRED_DETAIL, index=False)

            rp = write_block(writer, cfg.io.PLS_SHEET, ws_pls, rp, "M2-ZZ. Bootstrap DIRECT paths (t/CI/Sig)", PLS2_BOOTPATH, index=False)

            rp = write_block(writer, cfg.io.PLS_SHEET, ws_pls, rp, "M2-Effects. Point (direct/indirect/total/VAF)", PLS2_EFF_POINT, index=False)
            rp = write_block(writer, cfg.io.PLS_SHEET, ws_pls, rp, "M2-Effects. Bootstrap CI (indirect/total)", PLS2_EFF_CI, index=False)
            rp = write_block(writer, cfg.io.PLS_SHEET, ws_pls, rp, "M2-Effects. Mediation (VAF label)", PLS2_MED, index=False)

            ws_pls.freeze_panes = "A2"

    df_valid.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print("✅ 已輸出：", OUT_XLSX)
    print("✅ 已輸出：", OUT_CSV)
    return OUT_XLSX, OUT_CSV
