# pls_project/cog.py
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional
import numpy as np
import logging

if TYPE_CHECKING:
    from .config import Config

@dataclass
class Cog:
    cfg: "Config"
    rng: np.random.Generator = field(init=False)
    log: logging.Logger = field(init=False)

    cols_resolved: Optional[dict] = None
    profile_name: Optional[str] = None
    scale_prefixes: Optional[list[str]] = None

    def __post_init__(self):
        self.rng = np.random.default_rng(self.cfg.pls.PLS_SEED)

        self.log = logging.getLogger("pls_pipeline")
        self.log.propagate = False

        # 用 name 避免重複加 handler
        if not any(getattr(h, "name", "") == "pls_pipeline_stream" for h in self.log.handlers):
            h = logging.StreamHandler()
            h.name = "pls_pipeline_stream"
            h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
            self.log.addHandler(h)

        self.log.setLevel(logging.INFO)
