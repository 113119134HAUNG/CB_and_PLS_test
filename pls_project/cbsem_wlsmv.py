# pls_project/cbsem_wlsmv.py
from __future__ import annotations
import subprocess, sys, tempfile
from pathlib import Path
import pandas as pd

def run_cbsem_esem_then_cfa_sem_wlsmv(
    df: pd.DataFrame,
    items: list[str],
    *,
    esem_nfactors: int,
    cfa_model: str,
    sem_model: str | None = None,
    rotation: str = "geomin",
    rscript: str = "Rscript",
) -> dict[str, pd.DataFrame]:
    """
    Return dict of tables:
      ESEM_fit, ESEM_loadings, CFA_fit, CFA_loadings, (SEM_fit, SEM_paths)
    """
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        data_csv = td / "data.csv"
        out_dir = td / "out"
        out_dir.mkdir(parents=True, exist_ok=True)

        df.to_csv(data_csv, index=False, encoding="utf-8-sig")

        r_code = f"""
        suppressPackageStartupMessages(library(lavaan))

        df <- read.csv("{data_csv.as_posix()}", check.names=FALSE)
        items <- c({",".join([f'"{x}"' for x in items])})

        # ordered factors for Likert
        for (v in items) {{
          df[[v]] <- ordered(df[[v]])
        }}

        # ---- ESEM (lavaan EFA block) ----
        nf <- {int(esem_nfactors)}
        # build efa block syntax: efa("blk1")*F1+F2+... =~ item1+item2+...
        Fs <- paste0("F", 1:nf)
        efa_lhs <- paste(Fs, collapse=" + ")
        efa_rhs <- paste(items, collapse=" + ")
        model_esem <- paste0('efa("blk1")*', efa_lhs, ' =~ ', efa_rhs)

        fit_esem <- sem(
          model_esem,
          data=df,
          ordered=items,
          estimator="WLSMV",
          rotation="{rotation}"
        )

        # standardized solution -> loadings only
        sol_esem <- standardizedSolution(fit_esem)
        load_esem <- subset(sol_esem, op=="=~")[, c("lhs","rhs","est.std")]
        names(load_esem) <- c("Factor","Item","Loading_std")
        write.csv(load_esem, file=file.path("{out_dir.as_posix()}", "ESEM_loadings.csv"), row.names=FALSE)

        fm_esem <- fitMeasures(fit_esem, c("cfi.scaled","tli.scaled","rmsea.scaled","srmr"))
        fit_esem_tbl <- data.frame(Index=names(fm_esem), Value=as.numeric(fm_esem))
        write.csv(fit_esem_tbl, file=file.path("{out_dir.as_posix()}", "ESEM_fit.csv"), row.names=FALSE)

        # ---- CFA ----
        model_cfa <- '{cfa_model.replace("'", '"')}'
        fit_cfa <- cfa(model_cfa, data=df, ordered=items, estimator="WLSMV")

        sol_cfa <- standardizedSolution(fit_cfa)
        load_cfa <- subset(sol_cfa, op=="=~")[, c("lhs","rhs","est.std")]
        names(load_cfa) <- c("Factor","Item","Loading_std")
        write.csv(load_cfa, file=file.path("{out_dir.as_posix()}", "CFA_loadings.csv"), row.names=FALSE)

        fm_cfa <- fitMeasures(fit_cfa, c("cfi.scaled","tli.scaled","rmsea.scaled","srmr"))
        fit_cfa_tbl <- data.frame(Index=names(fm_cfa), Value=as.numeric(fm_cfa))
        write.csv(fit_cfa_tbl, file=file.path("{out_dir.as_posix()}", "CFA_fit.csv"), row.names=FALSE)

        # ---- SEM (optional) ----
        if (!is.null({ 'TRUE' if sem_model else 'NULL' })) {{
          model_sem <- '{(sem_model or "").replace("'", '"')}'
          fit_sem <- sem(model_sem, data=df, ordered=items, estimator="WLSMV")

          sol_sem <- standardizedSolution(fit_sem)
          path_sem <- subset(sol_sem, op=="~")[, c("lhs","rhs","est.std")]
          names(path_sem) <- c("Endogenous","Predictor","Beta_std")
          write.csv(path_sem, file=file.path("{out_dir.as_posix()}", "SEM_paths.csv"), row.names=FALSE)

          fm_sem <- fitMeasures(fit_sem, c("cfi.scaled","tli.scaled","rmsea.scaled","srmr"))
          fit_sem_tbl <- data.frame(Index=names(fm_sem), Value=as.numeric(fm_sem))
          write.csv(fit_sem_tbl, file=file.path("{out_dir.as_posix()}", "SEM_fit.csv"), row.names=FALSE)
        }}
        """

        r_file = td / "run_cbsem.R"
        r_file.write_text(r_code, encoding="utf-8")

        subprocess.run([rscript, str(r_file)], check=True)

        def read(name: str) -> pd.DataFrame:
            p = out_dir / name
            return pd.read_csv(p) if p.exists() else pd.DataFrame()

        out = {
            "ESEM_fit": read("ESEM_fit.csv"),
            "ESEM_loadings": read("ESEM_loadings.csv"),
            "CFA_fit": read("CFA_fit.csv"),
            "CFA_loadings": read("CFA_loadings.csv"),
            "SEM_fit": read("SEM_fit.csv"),
            "SEM_paths": read("SEM_paths.csv"),
        }
        return out
