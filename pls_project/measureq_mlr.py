# pls_project/measureq_mlr.py
from __future__ import annotations
import subprocess, tempfile
from pathlib import Path
import pandas as pd

def run_measureq_mlr(
    df: pd.DataFrame,
    items: list[str],
    *,
    cfa_model: str,
    bootstrap: int,
    rscript: str = "Rscript",
) -> dict[str, pd.DataFrame]:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        data_csv = td / "data.csv"
        out_dir = td / "out"
        out_dir.mkdir(parents=True, exist_ok=True)

        df.to_csv(data_csv, index=False, encoding="utf-8-sig")

        r_code = f"""
        suppressPackageStartupMessages(library(lavaan))

        # 你需要先確保 measureQ 已安裝/可 library()
        suppressPackageStartupMessages(library(measureQ))

        df <- read.csv("{data_csv.as_posix()}", check.names=FALSE)
        items <- c({",".join([f'"{x}"' for x in items])})

        model <- '{cfa_model.replace("'", '"')}'

        fit <- cfa(
          model,
          data=df,
          estimator="MLR",
          missing="ML",
          se="bootstrap",
          bootstrap={int(bootstrap)}
        )

        # ---- 下面這段要依 measureQ 版本調整：輸出你要的表 ----
        # 例：假設 measureQ 有一個主函數 measureQ(fit, ...) 回傳 list of tables
        res <- measureQ(fit)

        # 假設 res$table1, res$table2...
        write.csv(res$table1, file=file.path("{out_dir.as_posix()}", "measureQ_table1.csv"), row.names=FALSE)
        write.csv(res$table2, file=file.path("{out_dir.as_posix()}", "measureQ_table2.csv"), row.names=FALSE)
        """

        r_file = td / "run_measureQ.R"
        r_file.write_text(r_code, encoding="utf-8")
        subprocess.run([rscript, str(r_file)], check=True)

        out = {}
        for nm in ["measureQ_table1.csv", "measureQ_table2.csv"]:
            p = out_dir / nm
            out[nm.replace(".csv","")] = pd.read_csv(p) if p.exists() else pd.DataFrame()
        return out
