# pls_project/config.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Union, Tuple, Optional


# ==============================
# Column config (meta columns)
# ==============================
@dataclass
class ColumnConfig:
    TS_COL: str = "時間戳記"
    USER_COL: str = "使用者名稱"
    EXP_COL: str = "請問您是否曾使用 GenAI 進行學習的經驗？"
    EMAIL_COL: str = "電子郵件地址"


# ==============================
# Runtime filled by schema
# ==============================
@dataclass
class RuntimeConfig:
    profile_name: str = "unknown"
    scale_prefixes: List[str] = field(default_factory=list)
    rename_map: Dict[str, str] = field(default_factory=dict)
    resolved_cols: Dict[str, str] = field(default_factory=dict)


# ==============================
# IO config
# ==============================
@dataclass
class IOConfig:
    XLSX_PATH: str = "/content/drive/MyDrive/BACK.xlsx"
    SHEET_NAME: Union[int, str] = 0
    OUT_XLSX_BASE: str = "/content/drive/MyDrive/前測_論文表格輸出"
    OUT_CSV_BASE: str = "/content/drive/MyDrive/BACK_scored"

    # sheets
    PAPER_SHEET: str = "Paper_OnePage"
    PLS_SHEET: str = "PLS_SmartPLS"
    MICOM_SHEET: str = "MICOM"
    CBSEM_SHEET: str = "CBSEM_WLSMV"
    MEASUREQ_SHEET: str = "MEASUREQ"
    CMV_CLF_SHEET: str = "CMV_CLF_MLR"

    EXPORT_EXCLUDED_SHEET: bool = True
    DROP_EMAIL_IN_VALID_DF: bool = True


# ==============================
# Scale config
# ==============================
@dataclass
class ScaleConfig:
    SCALES: List[str] = field(default_factory=lambda: ["PA", "BS", "BB", "SA", "ACO", "CCO", "MIND", "CI", "LO"])
    BASE_REVERSE_ITEMS: List[str] = field(default_factory=list)


# ==============================
# Filter config (pre-processing controls)
# ==============================
@dataclass
class FilterConfig:
    FILTER_NOEXP: bool = True
    FILTER_DUPLICATE: bool = False
    FILTER_CARELESS: bool = False

    # duplicate setting
    KEEP_DUP: str = "last"

    # careless thresholds
    MAX_MISSING_RATE: float = 0.20
    MIN_SD_ITEMS: float = 0.01
    LONGSTRING_PCT: float = 0.90

    # careless sub-switches
    CARELESS_USE_MISSING: bool = True
    CARELESS_USE_SD: bool = True
    CARELESS_USE_LONGSTRING: bool = True

    # jump filter (NOW implemented in pipeline.py)
    USE_JUMP_FILTER: bool = False
    JUMP_DIFF: int = 3
    JUMP_RATE_TH: float = 0.80
    MIN_ITEMS_FOR_JUMP: int = 4


# ==============================
# CFA / CB-SEM config
# ==============================
@dataclass
class CFAConfig:
    # ---- existing CFA (paper.py run_cfa) ----
    RUN_CFA: bool = True
    CFA_MISSING: str = "listwise"
    CFA_OBJ: str = "MLW"
    CFA_ROBUST_SE: bool = False

    # decimals for CB-SEM/measureQ exports
    PAPER_DECIMALS: int = 3

    # ---- R runtime ----
    RSCRIPT_BIN: str = "Rscript"  # path or command name

    # ---- line 1: lavaan ESEM -> CFA/SEM with ordered/WLSMV ----
    RUN_CBSEM_WLSMV: bool = False
    RUN_SEM_WLSMV: bool = True
    ESEM_NFACTORS: int = 0
    ESEM_ROTATION: str = "geomin"
    CBSEM_MISSING: str = "listwise"
    SEM_EDGES: List[Tuple[str, str]] = field(default_factory=list)

    # ---- line 2: measureQ best-practice ----
    RUN_MEASUREQ: bool = False
    MEASUREQ_B_NO: int = 1000
    MEASUREQ_HTMT: bool = True
    MEASUREQ_CLUSTER: Optional[str] = None

    # ---- line 3: CMV robustness via CLF/ULMC-like model (MLR line) ----
    RUN_CMV_CLF_MLR: bool = False
    CLF_ESTIMATOR: str = "MLR"
    CLF_MISSING: str = "ML"
    CLF_METHOD_NAME: str = "CMV"
    CLF_ORTHOGONAL: bool = True
    CLF_EQUAL_LOADINGS: bool = True
    CLF_DELTA_LOADING_FLAG: float = 0.10


# ==============================
# PLS config
# ==============================
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
    PLS_MISSING: str = "listwise"     # "none" | "listwise" | "mean"

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
    BOOT_CI_LO: float = 0.025
    BOOT_CI_HI: float = 0.975
    BOOT_ALPHA: float = 0.05
    BOOT_TEST_TYPE: str = "two-tailed"

    # ---- Model control ----
    RUN_MODEL1: bool = True
    RUN_MODEL2: bool = True

    # ---- CMV diagnostic (Full collinearity VIF) ----
    RUN_FULL_COLLINEARITY_VIF: bool = True
    FULL_VIF_THRESHOLD: float = 3.3

    # ---- G*Power-equivalent (multiple regression) ----
    RUN_GPOWER: bool = False
    GPOWER_F2: float = 0.02
    GPOWER_ALPHA: float = 0.05
    GPOWER_POWER: float = 0.80

    # ---- NEW: inference switches (used by pipeline.py) ----
    RUN_EFFECTS_INFERENCE: bool = True
    RUN_HTMT_INFERENCE: bool = True
    RUN_PLS_PREDICT: bool = True

    HTMT_BOOT: int = 200
    HTMT_THRESHOLD: float = 0.90

    PREDICT_FOLDS: int = 10

    MODEL1_EDGES: List[Tuple[str, str]] = field(default_factory=lambda: [
        ("SA", "MIND"), ("SA", "ACO"), ("SA", "CCO"),
        ("PA", "BB"), ("PA", "BS"),
        ("BS", "MIND"), ("BS", "ACO"), ("BS", "CCO"), ("BS", "CI"),
        ("MIND", "CI"), ("BB", "CI"), ("ACO", "CI"), ("CCO", "CI"),
        ("CI", "LO"),
    ])

    MODEL2_COMMITMENT_NAME: str = "Commitment"
    MODEL2_COMPONENT_LVS: List[str] = field(default_factory=lambda: ["ACO", "CCO"])
    MODEL2_COMPONENT_COLS: List[str] = field(default_factory=lambda: ["ACO_score", "CCO_score"])
    MODEL2_COMMITMENT_MODE: str = "B"

    MODEL2_EDGES: List[Tuple[str, str]] = field(default_factory=lambda: [
        ("SA", "MIND"), ("SA", "Commitment"),
        ("PA", "BB"), ("PA", "BS"),
        ("BS", "MIND"), ("BS", "Commitment"), ("BS", "CI"),
        ("MIND", "CI"), ("BB", "CI"), ("Commitment", "CI"),
        ("CI", "LO"),
    ])

    MODEL2_COMMITMENT_ANCHOR: str = "CCO_score"


# ==============================
# MGA config
# ==============================
@dataclass
class MGAConfig:
    RUN_MGA: bool = False
    MGA_BOOT: int = 200
    MGA_MIN_N_PER_GROUP: int = 30
    MGA_SPLITS: List[str] = field(default_factory=lambda: ["CCO", "BS"])

    # MICOM
    RUN_MICOM: bool = True
    MICOM_BOOT: int = 200
    MICOM_ALPHA: float = 0.05
    MICOM_SPLITS: List[str] = field(default_factory=lambda: ["CCO", "BS"])

    # NEW: numeric detection threshold for MICOM split var
    MICOM_NUMERIC_RATIO_TH: float = 0.80


# ==============================
# Scenario config
# ==============================
@dataclass
class ScenarioConfig:
    SCENARIO_TARGET: str = "PA"
    RUN_REVERSE_SCENARIO: bool = False


# ==============================
# Main Config container
# ==============================
@dataclass
class Config:
    cols: ColumnConfig = field(default_factory=ColumnConfig)
    io: IOConfig = field(default_factory=IOConfig)
    scales: ScaleConfig = field(default_factory=ScaleConfig)
    filt: FilterConfig = field(default_factory=FilterConfig)
    cfa: CFAConfig = field(default_factory=CFAConfig)
    pls: PLSConfig = field(default_factory=PLSConfig)
    mga: MGAConfig = field(default_factory=MGAConfig)
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
