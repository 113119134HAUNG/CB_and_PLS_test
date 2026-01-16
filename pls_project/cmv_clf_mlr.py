# pls_project/cmv_clf_mlr.py
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


def _build_measurement_model(groups: List[str], group_items: Dict[str, List[str]]) -> str:
    lines = []
    for g in groups:
        its = [x for x in group_items.get(g, []) if isinstance(x, str)]
        if its:
            lines.append(f"{g} =~ " + " + ".join(its))
    return "\n".join(lines)


def _build_clf_model(
    meas_model: str,
    items: List[str],
    groups: List[str],
    *,
    method_name: str = "CMV",
    orthogonal: bool = True,
    equal_loadings: bool = True,
) -> str:
    """
    Common Latent Factor (CLF / ULMC-like) model:
      - add CMV factor loading on all items
      - optionally constrain all method loadings equal (recommended for stability)
      - fix method variance to 1 for identification
      - optionally force CMV orthogonal to substantive factors

    Returns lavaan model syntax string.
    """
    items = [str(x) for x in items]
    groups = [str(x) for x in groups]

    # method factor loading line
    if equal_loadings:
        # all method loadings share the same label m
        cmv_rhs = " + ".join([f"m*{it}" for it in items])
    else:
        cmv_rhs = " + ".join(items)

    lines = [meas_model.strip(), ""]
    lines.append(f"{method_name} =~ {cmv_rhs}")

    # fix method factor variance for identification
    lines.append(f"{method_name} ~~ 1*{method_name}")

    # orthogonal to substantive factors
    if orthogonal:
        for g in groups:
            lines.append(f"{method_name} ~~ 0*{g}")

    return "\n".join(lines).strip()


def run_cmv_clf_mlr(
    cog,
    df_items: pd.DataFrame,
    *,
    items: List[str],
    groups: List[str],
    group_items: Dict[str, List[str]],
    estimator: str = "MLR",
    missing: str = "ML",
    method_name: str = "CMV",
    orthogonal: bool = True,
    equal_loadings: bool = True,
    delta_loading_flag: float = 0.10,
    rscript: str = "Rscript",
) -> Dict[str, pd.DataFrame]:
    """
    Third line (MLR robustness):
      - Fit baseline measurement model (MLR + FIML)
      - Fit CLF/ULMC-like model (baseline + CMV factor)
      - Output fit indices + loading changes

    Returns dict of DataFrames:
      info,
      BASE_fit, CLF_fit, DELTA_fit,
      BASE_loadings, CLF_loadings_sub, CLF_loadings_method,
      DELTA_loadings, DELTA_summary,
      console_log
    """
    cfg = getattr(cog, "cfg", None)
    dec = int(getattr(getattr(cfg, "cfa", object()), "PAPER_DECIMALS", 3)) if cfg else 3

    items = [str(x) for x in items]
    groups = [str(x) for x in groups]

    meas_model = _build_measurement_model(groups, group_items)
    if not meas_model.strip():
        raise ValueError("Baseline measurement model is empty. Check groups/group_items.")

    clf_model = _build_clf_model(
        meas_model,
        items,
        groups,
        method_name=method_name,
        orthogonal=bool(orthogonal),
        equal_loadings=bool(equal_loadings),
    )

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        data_csv = td / "data.csv"
        out_dir = td / "out"
        out_dir.mkdir(parents=True, exist_ok=True)

        df_items.to_csv(data_csv, index=False, encoding="utf-8-sig")

        r_code = f"""
        suppressPackageStartupMessages(library(lavaan))

        Data <- read.csv("{data_csv.as_posix()}", check.names=FALSE, fileEncoding="UTF-8-BOM")

        items <- c({",".join([f'"{x}"' for x in items])})

        # quick check
        for (v in items) {{
          if (! (v %in% names(Data))) stop(paste0("Missing item column: ", v))
        }}

        write_fit <- function(fit, file) {{
          idx <- c("chisq","df","pvalue","cfi","tli","rmsea","srmr",
                   "cfi.robust","tli.robust","rmsea.robust")
          fm <- tryCatch(fitMeasures(fit, idx), error=function(e) rep(NA, length(idx)))
          out <- data.frame(Index=idx, Value=as.numeric(fm))
          write.csv(out, file=file, row.names=FALSE)
        }}

        write_loadings <- function(fit, file, keep_lhs=NULL, drop_lhs=NULL) {{
          sol <- standardizedSolution(fit)
          L <- subset(sol, op=="=~")[, c("lhs","rhs","est.std")]
          names(L) <- c("Factor","Item","Loading_std")
          if (!is.null(keep_lhs)) {{
            L <- subset(L, Factor %in% keep_lhs)
          }}
          if (!is.null(drop_lhs)) {{
            L <- subset(L, !(Factor %in% drop_lhs))
          }}
          write.csv(L, file=file, row.names=FALSE)
        }}

        # capture console output (warnings, messages)
        sink(file.path("{out_dir.as_posix()}", "console.txt"))
        cat("=== Baseline model ===\\n")
        cat('{meas_model.replace("'", '"')}')
        cat("\\n\\n=== CLF model ===\\n")
        cat('{clf_model.replace("'", '"')}')
        cat("\\n\\n")

        # ---- baseline ----
        fit_base <- cfa(
          model = '{meas_model.replace("'", '"')}',
          data = Data,
          estimator = "{estimator}",
          missing = "{missing}"
        )
        cat("\\nBASE converged:", lavInspect(fit_base, "converged"), "\\n")
        write_fit(fit_base, file.path("{out_dir.as_posix()}", "BASE_fit.csv"))
        write_loadings(fit_base, file.path("{out_dir.as_posix()}", "BASE_loadings.csv"))

        # ---- CLF ----
        fit_clf <- cfa(
          model = '{clf_model.replace("'", '"')}',
          data = Data,
          estimator = "{estimator}",
          missing = "{missing}"
        )
        cat("\\nCLF converged:", lavInspect(fit_clf, "converged"), "\\n")
        write_fit(fit_clf, file.path("{out_dir.as_posix()}", "CLF_fit.csv"))

        # substantive loadings: drop CMV
        write_loadings(
          fit_clf,
          file.path("{out_dir.as_posix()}", "CLF_loadings_sub.csv"),
          drop_lhs=c("{method_name}")
        )
        # method loadings: keep CMV only
        write_loadings(
          fit_clf,
          file.path("{out_dir.as_posix()}", "CLF_loadings_method.csv"),
          keep_lhs=c("{method_name}")
        )

        sink()
        """

        r_file = td / "run_cmv_clf_mlr.R"
        r_file.write_text(r_code, encoding="utf-8")

        subprocess.run([rscript, str(r_file)], check=True)

        def read_csv(name: str) -> pd.DataFrame:
            p = out_dir / name
            return pd.read_csv(p) if p.exists() else pd.DataFrame()

        base_fit = read_csv("BASE_fit.csv")
        clf_fit = read_csv("CLF_fit.csv")
        base_L = read_csv("BASE_loadings.csv")
        clf_L_sub = read_csv("CLF_loadings_sub.csv")
        clf_L_m = read_csv("CLF_loadings_method.csv")

        # ---- delta fit ----
        delta_fit = pd.DataFrame()
        if (not base_fit.empty) and (not clf_fit.empty):
            tmp = base_fit.merge(clf_fit, on="Index", how="outer", suffixes=("_BASE", "_CLF"))
            tmp["Delta(CLF-BASE)"] = pd.to_numeric(tmp["Value_CLF"], errors="coerce") - pd.to_numeric(tmp["Value_BASE"], errors="coerce")
            delta_fit = tmp

        # ---- delta loadings (substantive) ----
        delta_L = pd.DataFrame()
        delta_summary = pd.DataFrame()
        if (not base_L.empty) and (not clf_L_sub.empty):
            tmp = base_L.merge(clf_L_sub, on=["Factor", "Item"], how="outer", suffixes=("_BASE", "_CLF"))
            tmp["Delta(CLF-BASE)"] = pd.to_numeric(tmp["Loading_std_CLF"], errors="coerce") - pd.to_numeric(tmp["Loading_std_BASE"], errors="coerce")
            tmp["Abs_Delta"] = tmp["Delta(CLF-BASE)"].abs()
            tmp["Flag(|Δ|>=th)"] = tmp["Abs_Delta"] >= float(delta_loading_flag)
            delta_L = tmp

            delta_summary = pd.DataFrame([{
                "delta_loading_flag": float(delta_loading_flag),
                "mean_abs_delta": float(pd.to_numeric(tmp["Abs_Delta"], errors="coerce").mean()),
                "max_abs_delta": float(pd.to_numeric(tmp["Abs_Delta"], errors="coerce").max()),
                "n_flagged(|Δ|>=th)": int(tmp["Flag(|Δ|>=th)"].fillna(False).sum()),
                "n_pairs": int(len(tmp)),
            }])

        # console log -> one column
        console_path = out_dir / "console.txt"
        if console_path.exists():
            lines = console_path.read_text(encoding="utf-8", errors="replace").splitlines()
            console_df = pd.DataFrame({"console": lines})
        else:
            console_df = pd.DataFrame()

        out = {
            "info": pd.DataFrame([{
                "engine": "lavaan",
                "purpose": "CMV robustness via CLF/ULMC-like model (MLR line)",
                "estimator": estimator,
                "missing": missing,
                "method_factor": method_name,
                "orthogonal": bool(orthogonal),
                "equal_loadings": bool(equal_loadings),
                "delta_loading_flag": float(delta_loading_flag),
                "decimals": int(dec),
            }]),
            "BASE_fit": base_fit.round(dec),
            "CLF_fit": clf_fit.round(dec),
            "DELTA_fit": delta_fit.round(dec),
            "BASE_loadings": base_L.round(dec),
            "CLF_loadings_sub": clf_L_sub.round(dec),
            "CLF_loadings_method": clf_L_m.round(dec),
            "DELTA_loadings": delta_L.round(dec),
            "DELTA_summary": delta_summary.round(dec),
            "console_log": console_df,
        }
        return out
