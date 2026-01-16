# pls_project/measureq_mlr.py
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


def run_measureq(
    cog,
    df_items: pd.DataFrame,
    *,
    model_syntax: str,
    items: List[str],
    b_no: int = 1000,
    htmt: bool = True,
    cluster: Optional[str] = None,
    rscript: str = "Rscript",
) -> Dict[str, pd.DataFrame]:
    """
    Runs measureQ(Model, Data, b.no=..., HTMT="TRUE"/"FALSE", cluster=...)
    and exports any returned tables (if available) + a console log.

    Returns dict of DataFrames:
      info, measureQ_log, plus any tables discovered as CSVs.
    """
    cfg = getattr(cog, "cfg", None)
    dec = int(getattr(getattr(cfg, "cfa", object()), "PAPER_DECIMALS", 4)) if cfg else 4

    items = [str(x) for x in items]
    model_syntax = (model_syntax or "").strip()
    if not model_syntax:
        raise ValueError("measureQ requires non-empty model_syntax (measurement model).")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        data_csv = td / "data.csv"
        out_dir = td / "out"
        out_dir.mkdir(parents=True, exist_ok=True)

        df_items.to_csv(data_csv, index=False, encoding="utf-8-sig")

        cluster_arg = f', cluster="{cluster}"' if cluster else ""
        htmt_arg = "TRUE" if htmt else "FALSE"

        r_code = f"""
        suppressPackageStartupMessages(library(measureQ))

        Data <- read.csv("{data_csv.as_posix()}", check.names=FALSE)
        items <- c({",".join([f'"{x}"' for x in items])})

        # (optional) ensure item columns exist
        for (v in items) {{
          if (! (v %in% names(Data))) stop(paste0("Missing item column: ", v))
        }}

        Model <- '{model_syntax.replace("'", '"')}'

        # capture console output
        sink(file.path("{out_dir.as_posix()}", "measureQ_console.txt"))
        res <- measureQ(Model, Data, b.no={int(b_no)}, HTMT="{htmt_arg}"{cluster_arg})
        print(res)
        sink()

        # if res is a list, export any table-like elements
        if (is.list(res)) {{
          nms <- names(res)
          if (!is.null(nms)) {{
            for (nm in nms) {{
              obj <- res[[nm]]
              if (is.data.frame(obj) || is.matrix(obj)) {{
                write.csv(as.data.frame(obj),
                          file=file.path("{out_dir.as_posix()}", paste0("measureQ_", nm, ".csv")),
                          row.names=FALSE)
              }}
            }}
          }}
        }}
        """

        r_file = td / "run_measureQ.R"
        r_file.write_text(r_code, encoding="utf-8")

        subprocess.run([rscript, str(r_file)], check=True)

        # read any exported CSV tables
        tables: Dict[str, pd.DataFrame] = {}
        for p in sorted(out_dir.glob("measureQ_*.csv")):
            name = p.stem  # measureQ_xxx
            tables[name] = pd.read_csv(p).round(dec)

        # read console log as a "one-column table" so you can write it into Excel
        log_path = out_dir / "measureQ_console.txt"
        if log_path.exists():
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            log_df = pd.DataFrame({"measureQ_console": lines})
        else:
            log_df = pd.DataFrame()

        out = {
            "info": pd.DataFrame([{
                "tool": "measureQ",
                "b.no": int(b_no),
                "HTMT": bool(htmt),
                "cluster": cluster or "",
                "decimals": int(dec),
            }]),
            "measureQ_log": log_df,
        }
        out.update(tables)
        return out
