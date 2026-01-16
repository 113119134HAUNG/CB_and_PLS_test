# pls_project/cbsem_wlsmv.py
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import pandas as pd


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
    # group by endogenous
    to_map: Dict[str, List[str]] = {}
    for fr, to in edges:
        to_map.setdefault(to, []).append(fr)

    lines = [cfa_model.strip(), ""]
    for to, frs in to_map.items():
        rhs = " + ".join(frs)
        lines.append(f"{to} {regress_op} {rhs}")
    return "\n".join(lines).strip()


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

        # ---- R script ----
        # Notes:
        # - ordered=items triggers WLSMV in lavaan for categorical indicators
        # - we coerce to ordered factors in R defensively
        r_code = f"""
        suppressPackageStartupMessages(library(lavaan))

        df <- read.csv("{data_csv.as_posix()}", check.names=FALSE, fileEncoding="UTF-8-BOM")
        items <- c({",".join([f'"{x}"' for x in items])})

        # coerce to ordered factors (works for numeric 1..5 or strings)
        for (v in items) {{
          if (! (v %in% names(df))) stop(paste0("Missing item column: ", v))
          # keep NA; coerce values to integer-ish if numeric
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
        model_cfa <- '{cfa_model.replace("'", '"')}'
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
          model_sem <- '{sem_model.replace("'", '"')}'
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

        # ---- run ----
        subprocess.run([rscript, str(r_file)], check=True)

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
