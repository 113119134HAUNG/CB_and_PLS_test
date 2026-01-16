# pls_project/cbsem_wlsmv.py
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any

import numpy as np
import pandas as pd


# =========================================================
# Model builders
# =========================================================
def _build_cfa_model(groups: List[str], group_items: Dict[str, List[str]]) -> str:
    lines = []
    for g in groups:
        items = [x for x in group_items.get(g, []) if isinstance(x, str)]
        if not items:
            continue
        rhs = " + ".join(items)
        lines.append(f"{g} =~ {rhs}")
    return "\n".join(lines)


def _build_sem_model(
    cfa_model: str,
    edges: List[Tuple[str, str]],
    *,
    regress_op: str = "~",
) -> str:
    """
    edges: list of (from, to), meaning: to ~ from
    """
    to_map: Dict[str, List[str]] = {}
    for fr, to in edges:
        to_map.setdefault(to, []).append(fr)

    lines = [cfa_model.strip(), ""]
    for to, frs in to_map.items():
        rhs = " + ".join(frs)
        lines.append(f"{to} {regress_op} {rhs}")
    return "\n".join(lines).strip()


# =========================================================
# Existing: ESEM -> CFA -> SEM (ordered/WLSMV)
# =========================================================
def run_cbsem_esem_then_cfa_sem_wlsmv(
    cog,
    df_items: pd.DataFrame,
    *,
    items: List[str],
    groups: List[str],
    group_items: Dict[str, List[str]],
    esem_nfactors: int,
    rotation: str = "geomin",
    missing: str = "listwise",
    run_sem: bool = True,
    sem_edges: Optional[List[Tuple[str, str]]] = None,
    rscript: str = "Rscript",
) -> Dict[str, pd.DataFrame]:
    """
    Run:
      - ESEM (lavaan efa-block) with ordered/WLSMV
      - CFA (simple structure from group_items)
      - optional SEM (structural paths from sem_edges)

    Returns dict of DataFrames (empty if not produced):
      info, ESEM_fit, ESEM_loadings, CFA_fit, CFA_loadings, SEM_fit, SEM_paths
    """
    cfg = getattr(cog, "cfg", None)
    dec = int(getattr(getattr(cfg, "cfa", object()), "PAPER_DECIMALS", 4)) if cfg else 4

    items = [str(x) for x in items]
    groups = [str(x) for x in groups]

    cfa_model = _build_cfa_model(groups, group_items)
    if not cfa_model.strip():
        raise ValueError("CFA model is empty: check groups/group_items.")

    if run_sem:
        if sem_edges is None:
            sem_edges = []
        sem_model = _build_sem_model(cfa_model, sem_edges)
    else:
        sem_model = ""

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        data_csv = td / "data.csv"
        out_dir = td / "out"
        out_dir.mkdir(parents=True, exist_ok=True)

        df_items.to_csv(data_csv, index=False, encoding="utf-8-sig")

        # ---- write models to files ----
        cfa_file = td / "model_cfa.txt"
        cfa_file.write_text(cfa_model, encoding="utf-8")

        sem_file = td / "model_sem.txt"
        if run_sem and sem_model.strip():
            sem_file.write_text(sem_model, encoding="utf-8")

        # ---- R script ----
        r_code = f"""
        suppressPackageStartupMessages(library(lavaan))

        df <- read.csv("{data_csv.as_posix()}", check.names=FALSE, fileEncoding="UTF-8-BOM")
        items <- c({",".join([f'"{x}"' for x in items])})

        # coerce to ordered factors (works for numeric 1..5 or strings)
        for (v in items) {{
          if (! (v %in% names(df))) stop(paste0("Missing item column: ", v))
          if (is.numeric(df[[v]])) {{
            df[[v]] <- as.integer(round(df[[v]]))
          }}
          df[[v]] <- ordered(df[[v]])
        }}

        write_fit <- function(fit, file) {{
          idx <- c("chisq.scaled","df.scaled","pvalue.scaled","cfi.scaled","tli.scaled","rmsea.scaled","srmr")
          fm <- tryCatch(fitMeasures(fit, idx), error=function(e) rep(NA, length(idx)))
          out <- data.frame(Index=idx, Value=as.numeric(fm))
          write.csv(out, file=file, row.names=FALSE)
        }}

        write_loadings <- function(fit, file) {{
          sol <- standardizedSolution(fit)
          L <- subset(sol, op=="=~")[, c("lhs","rhs","est.std")]
          names(L) <- c("Factor","Item","Loading_std")
          write.csv(L, file=file, row.names=FALSE)
        }}

        write_paths <- function(fit, file) {{
          sol <- standardizedSolution(fit)
          P <- subset(sol, op=="~")[, c("lhs","rhs","est.std")]
          names(P) <- c("Endogenous","Predictor","Beta_std")
          write.csv(P, file=file, row.names=FALSE)
        }}

        # ---- ESEM ----
        nf <- {int(esem_nfactors)}
        Fs <- paste0("F", 1:nf)
        efa_lhs <- paste(Fs, collapse=" + ")
        efa_rhs <- paste(items, collapse=" + ")
        model_esem <- paste0('efa("blk1")*', efa_lhs, ' =~ ', efa_rhs)

        fit_esem <- sem(
          model_esem,
          data=df,
          ordered=items,
          estimator="WLSMV",
          rotation="{rotation}",
          missing="{missing}"
        )
        write_fit(fit_esem, file.path("{out_dir.as_posix()}", "ESEM_fit.csv"))
        write_loadings(fit_esem, file.path("{out_dir.as_posix()}", "ESEM_loadings.csv"))

        # ---- CFA ----
        model_cfa <- paste(readLines("{cfa_file.as_posix()}", warn=FALSE), collapse="\\n")
        fit_cfa <- cfa(
          model_cfa,
          data=df,
          ordered=items,
          estimator="WLSMV",
          missing="{missing}"
        )
        write_fit(fit_cfa, file.path("{out_dir.as_posix()}", "CFA_fit.csv"))
        write_loadings(fit_cfa, file.path("{out_dir.as_posix()}", "CFA_loadings.csv"))

        # ---- SEM (optional) ----
        if ({'TRUE' if run_sem and sem_model.strip() else 'FALSE'}) {{
          model_sem <- paste(readLines("{sem_file.as_posix()}", warn=FALSE), collapse="\\n")
          fit_sem <- sem(
            model_sem,
            data=df,
            ordered=items,
            estimator="WLSMV",
            missing="{missing}"
          )
          write_fit(fit_sem, file.path("{out_dir.as_posix()}", "SEM_fit.csv"))
          write_paths(fit_sem, file.path("{out_dir.as_posix()}", "SEM_paths.csv"))
        }}
        """

        r_file = td / "run_cbsem_wlsmv.R"
        r_file.write_text(r_code, encoding="utf-8")

        proc = subprocess.run([rscript, str(r_file)], capture_output=True, text=True)
        (out_dir / "r_stdout.txt").write_text(proc.stdout or "", encoding="utf-8", errors="replace")
        (out_dir / "r_stderr.txt").write_text(proc.stderr or "", encoding="utf-8", errors="replace")
        if proc.returncode != 0:
            raise RuntimeError(f"R failed with code={proc.returncode}. See r_stderr.txt in {out_dir}")

        def read_csv(name: str) -> pd.DataFrame:
            p = out_dir / name
            if not p.exists():
                return pd.DataFrame()
            return pd.read_csv(p)

        out = {
            "info": pd.DataFrame([{
                "engine": "lavaan",
                "ordered": True,
                "estimator": "WLSMV",
                "missing": missing,
                "rotation": rotation,
                "ESEM_nfactors": int(esem_nfactors),
                "run_sem": bool(run_sem),
                "decimals": int(dec),
            }]),
            "ESEM_fit": read_csv("ESEM_fit.csv").round(dec),
            "ESEM_loadings": read_csv("ESEM_loadings.csv").round(dec),
            "CFA_fit": read_csv("CFA_fit.csv").round(dec),
            "CFA_loadings": read_csv("CFA_loadings.csv").round(dec),
            "SEM_fit": read_csv("SEM_fit.csv").round(dec),
            "SEM_paths": read_csv("SEM_paths.csv").round(dec),
        }
        return out


# =========================================================
# NEW: MG-CFA invariance (ordered/WLSMV)
# =========================================================
def _auto_pick_group_var(
    df: pd.DataFrame,
    *,
    exclude_cols: List[str],
    numeric_ratio_th: float = 0.80,
    max_levels: int = 8,
) -> Optional[str]:
    """
    Pick a reasonable grouping variable from df columns excluding items.
    Preference:
      1) categorical-like (object/category/bool) with 2..max_levels levels
      2) numeric with high numeric ratio (>= numeric_ratio_th) and enough unique -> median split later
    """
    cand = [c for c in df.columns if c not in set(exclude_cols)]
    if not cand:
        return None

    # 1) categorical-like
    best = None
    best_levels = None
    for c in cand:
        s = df[c]
        # treat as categorical if dtype looks categorical-ish OR non-numeric
        if pd.api.types.is_bool_dtype(s) or pd.api.types.is_object_dtype(s) or pd.api.types.is_categorical_dtype(s):
            lv = s.dropna().astype(str).value_counts()
            k = int(lv.shape[0])
            if 2 <= k <= max_levels:
                # prefer closer to 2 (simpler), then larger N
                score = (k, -int(lv.sum()))
                if best is None or score < best_levels:
                    best = c
                    best_levels = score

    if best is not None:
        return str(best)

    # 2) numeric-ish
    for c in cand:
        v = pd.to_numeric(df[c], errors="coerce")
        if float(v.notna().mean()) >= float(numeric_ratio_th):
            nunq = int(v.nunique(dropna=True))
            if nunq >= 2:
                return str(c)

    return None


def _make_two_group_column(
    s: pd.Series,
    *,
    numeric_ratio_th: float = 0.80,
) -> Tuple[pd.Series, str]:
    """
    Return (group_series, note). Group series values are strings with 2 levels.
    - If numeric-ish: median split
    - Else: take top-2 levels (by freq)
    """
    vnum = pd.to_numeric(s, errors="coerce")
    num_ratio = float(vnum.notna().mean())
    if num_ratio >= float(numeric_ratio_th) and int(vnum.nunique(dropna=True)) >= 2:
        thr = float(np.nanmedian(vnum.to_numpy(dtype=float)))
        g = pd.Series(np.where(vnum >= thr, "High", "Low"), index=s.index)
        note = f"numeric median split @ {thr:.6g} (numeric_ratio={num_ratio:.2f})"
        return g, note

    # categorical: top2 levels
    s2 = s.astype(str).where(s.notna(), other="NA")
    vc = s2.value_counts()
    if vc.shape[0] < 2:
        # fallback: everything NA -> single group
        g = pd.Series(["G1"] * len(s2), index=s2.index)
        return g, "categorical fallback (single level)"
    top2 = vc.index[:2].tolist()
    g = s2.where(s2.isin(top2), other=np.nan)
    # map to fixed labels (stable)
    g = g.map({top2[0]: "G1", top2[1]: "G2"})
    note = f"categorical top2 levels: {top2}"
    return g, note


def run_cbsem_cfa_invariance_wlsmv(
    cog,
    df_items_plus_meta: pd.DataFrame,
    *,
    items: List[str],
    groups: List[str],
    group_items: Dict[str, List[str]],
    group_var: Optional[str] = None,
    missing: str = "listwise",
    numeric_ratio_th: float = 0.80,
    min_n_per_group: int = 30,
    include_strict: bool = False,
    rscript: str = "Rscript",
) -> Dict[str, pd.DataFrame]:
    """
    MG-CFA invariance (ordered/WLSMV) using lavaan:
      - configural
      - metric: group.equal = c("loadings")
      - scalar (ordered): group.equal = c("loadings","thresholds")
      - optional strict: add "residuals"

    Auto-grouping:
      - if group_var is None, auto-pick a non-item column from df_items_plus_meta
      - always reduce to TWO groups (median split for numeric; top2 for categorical)
        (this matches your MICOM two-group design and keeps WLSMV stable)

    Returns:
      info, INV_group_counts, INV_fit_long, INV_fit_delta, INV_pass
    """
    cfg = getattr(cog, "cfg", None)
    dec = int(getattr(getattr(cfg, "cfa", object()), "PAPER_DECIMALS", 4)) if cfg else 4

    items = [str(x) for x in items]
    groups = [str(x) for x in groups]

    cfa_model = _build_cfa_model(groups, group_items)
    if not cfa_model.strip():
        raise ValueError("CFA model is empty: check groups/group_items.")

    df = df_items_plus_meta.copy()
    miss_items = [c for c in items if c not in df.columns]
    if miss_items:
        raise ValueError(f"Invariance needs all item columns in df. Missing: {miss_items}")

    # pick group var if not provided
    picked = None
    if group_var is not None:
        if str(group_var) not in df.columns:
            raise ValueError(f"group_var not found in df: {group_var}")
        picked = str(group_var)
    else:
        picked = _auto_pick_group_var(df, exclude_cols=items)
        if picked is None:
            return {
                "info": pd.DataFrame([{"Error": "No usable group_var found (df has only items or no suitable meta column)."}]),
                "INV_group_counts": pd.DataFrame(),
                "INV_fit_long": pd.DataFrame(),
                "INV_fit_delta": pd.DataFrame(),
                "INV_pass": pd.DataFrame(),
            }

    # build 2-group column
    gcol, gnote = _make_two_group_column(df[picked], numeric_ratio_th=numeric_ratio_th)
    df["_MGROUP"] = gcol

    # drop rows without group assignment or missing items (lavaan handles missing per 'missing', but group can't be NA)
    df_use = df.loc[df["_MGROUP"].notna(), ["_MGROUP"] + items].copy()

    # quick group counts
    counts = df_use["_MGROUP"].value_counts(dropna=False)
    if counts.shape[0] < 2:
        return {
            "info": pd.DataFrame([{"Error": f"Grouping failed: <2 groups after processing. group_var={picked} note={gnote}"}]),
            "INV_group_counts": pd.DataFrame({"Group": counts.index.astype(str), "n": counts.values}),
            "INV_fit_long": pd.DataFrame(),
            "INV_fit_delta": pd.DataFrame(),
            "INV_pass": pd.DataFrame(),
        }
    if int(counts.min()) < int(min_n_per_group):
        return {
            "info": pd.DataFrame([{
                "Error": "Grouping blocked: group size too small.",
                "group_var": picked,
                "group_note": gnote,
                "min_n_per_group": int(min_n_per_group),
                "counts": counts.to_dict(),
            }]),
            "INV_group_counts": pd.DataFrame({"Group": counts.index.astype(str), "n": counts.values}),
            "INV_fit_long": pd.DataFrame(),
            "INV_fit_delta": pd.DataFrame(),
            "INV_pass": pd.DataFrame(),
        }

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        data_csv = td / "data.csv"
        out_dir = td / "out"
        out_dir.mkdir(parents=True, exist_ok=True)

        df_use.to_csv(data_csv, index=False, encoding="utf-8-sig")

        cfa_file = td / "model_cfa.txt"
        cfa_file.write_text(cfa_model, encoding="utf-8")

        strict_flag = "TRUE" if include_strict else "FALSE"

        r_code = f"""
        suppressPackageStartupMessages(library(lavaan))

        df <- read.csv("{data_csv.as_posix()}", check.names=FALSE, fileEncoding="UTF-8-BOM")
        items <- c({",".join([f'"{x}"' for x in items])})
        gvar  <- "_MGROUP"

        if (! (gvar %in% names(df))) stop("Missing _MGROUP column.")
        df[[gvar]] <- as.factor(df[[gvar]])
        if (nlevels(df[[gvar]]) < 2) stop("Need at least 2 groups.")

        # items -> ordered
        for (v in items) {{
          if (! (v %in% names(df))) stop(paste0("Missing item column: ", v))
          if (is.numeric(df[[v]])) df[[v]] <- as.integer(round(df[[v]]))
          df[[v]] <- ordered(df[[v]])
        }}

        # group counts
        gtab <- as.data.frame(table(df[[gvar]]))
        names(gtab) <- c("Group","n")
        write.csv(gtab, file.path("{out_dir.as_posix()}", "INV_group_counts.csv"), row.names=FALSE)

        model_cfa <- paste(readLines("{cfa_file.as_posix()}", warn=FALSE), collapse="\\n")

        get_fit <- function(fit, step_name) {{
          idx <- c("chisq.scaled","df.scaled","pvalue.scaled",
                   "cfi.scaled","tli.scaled","rmsea.scaled","srmr")
          fm <- tryCatch(fitMeasures(fit, idx), error=function(e) rep(NA, length(idx)))
          data.frame(Step=step_name, Index=idx, Value=as.numeric(fm))
        }}

        # ---- configural ----
        fit_config <- cfa(
          model_cfa,
          data=df,
          group=gvar,
          ordered=items,
          estimator="WLSMV",
          missing="{missing}"
        )

        # ---- metric (loadings) ----
        fit_metric <- cfa(
          model_cfa,
          data=df,
          group=gvar,
          group.equal=c("loadings"),
          ordered=items,
          estimator="WLSMV",
          missing="{missing}"
        )

        # ---- scalar for ordered (thresholds) ----
        fit_scalar <- cfa(
          model_cfa,
          data=df,
          group=gvar,
          group.equal=c("loadings","thresholds"),
          ordered=items,
          estimator="WLSMV",
          missing="{missing}"
        )

        rows <- rbind(
          get_fit(fit_config, "configural"),
          get_fit(fit_metric, "metric"),
          get_fit(fit_scalar, "scalar")
        )

        if ({strict_flag}) {{
          fit_strict <- cfa(
            model_cfa,
            data=df,
            group=gvar,
            group.equal=c("loadings","thresholds","residuals"),
            ordered=items,
            estimator="WLSMV",
            missing="{missing}"
          )
          rows <- rbind(rows, get_fit(fit_strict, "strict"))
        }}

        write.csv(rows, file.path("{out_dir.as_posix()}", "INV_fit_long.csv"), row.names=FALSE)

        # ---- delta table ----
        wide <- reshape(rows, idvar="Index", timevar="Step", direction="wide")
        num <- function(x) suppressWarnings(as.numeric(x))

        if ("Value.metric" %in% names(wide) && "Value.configural" %in% names(wide)) {{
          wide$Delta_metric_minus_configural <- num(wide$Value.metric) - num(wide$Value.configural)
        }} else {{
          wide$Delta_metric_minus_configural <- NA
        }}

        if ("Value.scalar" %in% names(wide) && "Value.metric" %in% names(wide)) {{
          wide$Delta_scalar_minus_metric <- num(wide$Value.scalar) - num(wide$Value.metric)
        }} else {{
          wide$Delta_scalar_minus_metric <- NA
        }}

        if ("Value.strict" %in% names(wide) && "Value.scalar" %in% names(wide)) {{
          wide$Delta_strict_minus_scalar <- num(wide$Value.strict) - num(wide$Value.scalar)
        }} else {{
          wide$Delta_strict_minus_scalar <- NA
        }}

        write.csv(wide, file.path("{out_dir.as_posix()}", "INV_fit_delta.csv"), row.names=FALSE)
        """

        r_file = td / "run_invariance_wlsmv.R"
        r_file.write_text(r_code, encoding="utf-8")

        proc = subprocess.run([rscript, str(r_file)], capture_output=True, text=True)
        (out_dir / "r_stdout.txt").write_text(proc.stdout or "", encoding="utf-8", errors="replace")
        (out_dir / "r_stderr.txt").write_text(proc.stderr or "", encoding="utf-8", errors="replace")
        if proc.returncode != 0:
            raise RuntimeError(f"R failed with code={proc.returncode}. See r_stderr.txt in {out_dir}")

        def read_csv(name: str) -> pd.DataFrame:
            p = out_dir / name
            return pd.read_csv(p) if p.exists() else pd.DataFrame()

        fit_long = read_csv("INV_fit_long.csv")
        fit_delta = read_csv("INV_fit_delta.csv")
        group_counts = read_csv("INV_group_counts.csv")

        # ---- pass flags (common heuristics) ----
        # Metric: |ΔCFI|<=.01, ΔRMSEA<=.015, ΔSRMR<=.01
        # Scalar: |ΔCFI|<=.01, ΔRMSEA<=.015, ΔSRMR<=.015
        def _get_delta(index_name: str, col: str) -> Optional[float]:
            if fit_delta.empty or "Index" not in fit_delta.columns or col not in fit_delta.columns:
                return None
            m = fit_delta["Index"].astype(str) == str(index_name)
            if not m.any():
                return None
            v = pd.to_numeric(fit_delta.loc[m, col], errors="coerce").iloc[0]
            return float(v) if pd.notna(v) else None

        d_cfi_metric = _get_delta("cfi.scaled", "Delta_metric_minus_configural")
        d_rmsea_metric = _get_delta("rmsea.scaled", "Delta_metric_minus_configural")
        d_srmr_metric = _get_delta("srmr", "Delta_metric_minus_configural")

        d_cfi_scalar = _get_delta("cfi.scaled", "Delta_scalar_minus_metric")
        d_rmsea_scalar = _get_delta("rmsea.scaled", "Delta_scalar_minus_metric")
        d_srmr_scalar = _get_delta("srmr", "Delta_scalar_minus_metric")

        pass_tbl = pd.DataFrame([{
            "Rule_metric": "|ΔCFI|<=.01 & ΔRMSEA<=.015 & ΔSRMR<=.01",
            "ΔCFI_metric": d_cfi_metric,
            "ΔRMSEA_metric": d_rmsea_metric,
            "ΔSRMR_metric": d_srmr_metric,
            "Pass_metric": (
                (d_cfi_metric is not None and abs(d_cfi_metric) <= 0.01) and
                (d_rmsea_metric is not None and d_rmsea_metric <= 0.015) and
                (d_srmr_metric is not None and d_srmr_metric <= 0.01)
            ) if (d_cfi_metric is not None and d_rmsea_metric is not None and d_srmr_metric is not None) else False,
            "Rule_scalar": "|ΔCFI|<=.01 & ΔRMSEA<=.015 & ΔSRMR<=.015",
            "ΔCFI_scalar": d_cfi_scalar,
            "ΔRMSEA_scalar": d_rmsea_scalar,
            "ΔSRMR_scalar": d_srmr_scalar,
            "Pass_scalar": (
                (d_cfi_scalar is not None and abs(d_cfi_scalar) <= 0.01) and
                (d_rmsea_scalar is not None and d_rmsea_scalar <= 0.015) and
                (d_srmr_scalar is not None and d_srmr_scalar <= 0.015)
            ) if (d_cfi_scalar is not None and d_rmsea_scalar is not None and d_srmr_scalar is not None) else False,
        }])

        out = {
            "info": pd.DataFrame([{
                "engine": "lavaan",
                "purpose": "MG-CFA invariance (ordered/WLSMV) via two-group design",
                "picked_group_var": picked,
                "group_note": gnote,
                "missing": missing,
                "min_n_per_group": int(min_n_per_group),
                "steps": "configural, metric(loadings), scalar(loadings+thresholds)" + (", strict(+residuals)" if include_strict else ""),
                "decimals": int(dec),
            }]),
            "INV_group_counts": group_counts.round(dec),
            "INV_fit_long": fit_long.round(dec),
            "INV_fit_delta": fit_delta.round(dec),
            "INV_pass": pass_tbl.round(dec),
        }
        return out