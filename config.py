# config.py (只貼需要替換/更新的兩個 dataclass)

from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class FilterConfig:
    # ---- NEW: 三個前處理開關 ----
    FILTER_NOEXP: bool = True
    FILTER_DUPLICATE: bool = True
    FILTER_CARELESS: bool = True

    # 原本的 duplicate / careless 參數
    KEEP_DUP: str = "last"
    MAX_MISSING_RATE: float = 0.20
    MIN_SD_ITEMS: float = 0.01
    LONGSTRING_PCT: float = 0.90

    # 可選：若你只想關掉其中一種草率指標，也可再加開關（不加也行）
    CARELESS_USE_MISSING: bool = True
    CARELESS_USE_SD: bool = True
    CARELESS_USE_LONGSTRING: bool = True

    USE_JUMP_FILTER: bool = False
    JUMP_DIFF: int = 3
    JUMP_RATE_TH: float = 0.80
    MIN_ITEMS_FOR_JUMP: int = 4

@dataclass
class PLSConfig:
    RUN_PLS: bool = True

    # ---- Clean rules ----
    CLEAN_MODE: bool = True
    AUTO_INSTALL_PLSPM: bool = False
    ALLOW_FALLBACK: bool = False
    ALLOW_PCA: bool = False

    # ---- SmartPLS4-like algorithm settings ----
    PLS_SCHEME: str = "PATH"          # PATH / FACTORIAL
    PLS_STANDARDIZED: bool = True
    PLSPM_MAX_ITER: int = 3000
    PLSPM_TOL: float = 1e-7

    # ---- Correlation methods ----
    HTMT_CORR_METHOD: str = "pearson"
    PLS_CROSS_CORR_METHOD: str = "pearson"

    # ---- Sign orientation ----
    SIGN_FIX: bool = True

    # ---- Paper output ----
    PAPER_DECIMALS: int = 3

    # ---- Bootstrap ----
    PLS_BOOT: int = 200
    PLS_SEED: int = 0
    Q2_FOLDS: int = 5
    BOOT_RETRY: int = 0

    # ---- NEW: Model1 / Model2 控制 ----
    RUN_MODEL1: bool = True
    RUN_MODEL2: bool = True

    # Model1 的邊（用 token/LV 名稱；會自動與 schema 抓到的 groups 取交集）
    MODEL1_EDGES: List[Tuple[str, str]] = field(default_factory=lambda: [
        ("SA","MIND"), ("SA","ACO"), ("SA","CCO"),
        ("PA","BB"), ("PA","BS"),
        ("BS","MIND"), ("BS","ACO"), ("BS","CCO"), ("BS","CI"),
        ("MIND","CI"), ("BB","CI"), ("ACO","CI"), ("CCO","CI"),
        ("CI","LO"),
    ])

    # Model2（two-stage）的設定
    MODEL2_COMMITMENT_NAME: str = "Commitment"
    MODEL2_COMPONENT_LVS: List[str] = field(default_factory=lambda: ["ACO", "CCO"])     # stage1 scores 的來源 LV
    MODEL2_COMPONENT_COLS: List[str] = field(default_factory=lambda: ["ACO_score", "CCO_score"])  # stage2 X 的欄名
    MODEL2_COMMITMENT_MODE: str = "B"  # formative

    MODEL2_EDGES: List[Tuple[str, str]] = field(default_factory=lambda: [
        ("SA","MIND"), ("SA","Commitment"),
        ("PA","BB"), ("PA","BS"),
        ("BS","MIND"), ("BS","Commitment"), ("BS","CI"),
        ("MIND","CI"), ("BB","CI"), ("Commitment","CI"),
        ("CI","LO"),
    ])

    # 可選：指定 Commitment 的 anchor（避免 anchors 選到另一個 component）
    MODEL2_COMMITMENT_ANCHOR: str = "CCO_score"
