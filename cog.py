# pls_project/cog.py
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional
import numpy as np
import logging

if TYPE_CHECKING:
    from .config import Config
    from .config import ColumnConfig

@dataclass
class Cog:
    cfg: "Config"
    rng: np.random.Generator = field(init=False)
    log: logging.Logger = field(init=False)

    # 讓 pipeline resolve 後塞進來（共用）
    cols_resolved: Optional["ColumnConfig"] = None
    profile_name: Optional[str] = None
    scale_prefixes: Optional[list[str]] = None

    def __post_init__(self):
        # RNG
        self.rng = np.random.default_rng(self.cfg.pls.PLS_SEED)

        # Logger
        self.log = logging.getLogger("pls_pipeline")
        self.log.propagate = False  # 避免重複印出

        # 避免 notebook / 重跑 cell 時重複加 handler
        if not any(isinstance(h, logging.StreamHandler) for h in self.log.handlers):
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
            self.log.addHandler(h)

        # level（你也可以之後放到 cfg 裡）
        self.log.setLevel(logging.INFO)
