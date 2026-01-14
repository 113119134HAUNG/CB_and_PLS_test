# pls_project/config.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict

# ==============================
# Column config (meta columns)
# ==============================
@dataclass
class ColumnConfig:
    TS_COL: str    = "時間戳記"
    USER_COL: str  = "使用者名稱"
    EXP_COL: str   = "請問您是否曾使用 GenAI 進行學習的經驗？"
    EMAIL_COL: str = "電子郵件地址"

# ==============================
# Runtime filled by schema
# ==============================
@dataclass
class RuntimeConfig:
    """
    schema.py 解析 df.columns 後，會把結果寫回這裡（runtime in-memory）。
    - profile_name: 問卷版本（v1/v2/v3/unknown）
    - scale_prefixes: 該版本的題項 prefix 清單
    - rename_map: 原欄名 -> token（題項欄位）
    """
    profile_name: str = "unknown"
    scale_prefixes: List[str] = field(default_factory=list)
    rename_map: Dict[str, str] = field(default_factory=dict)

# ==============================
# 其他 configs（你原本的）
# ==============================
@dataclass
class IOConfig:
    XLSX_PATH: str = "/content/drive/MyDrive/BACK.xlsx"
    SHEET_NAME: int | str = 0
    OUT_XLSX_BASE: str = "/content/drive/MyDrive/前測_論文表格輸出"
    OUT_CSV_BASE: str = "/content/drive/MyDrive/BACK_scored"
    PAPER_SHEET: str = "Paper_OnePage"
    PLS_SHEET: str = "PLS_SmartPLS"
    EXPORT_EXCLUDED_SHEET: bool = True
    DROP_EMAIL_IN_VALID_DF: bool = True

@dataclass
class ScaleConfig:
    SCALES: List[str] = field(default_factory=lambda: ["PA","BS","BB","SA","ACO","CCO","MIND","CI","LO"])
    BASE_REVERSE_ITEMS: List[str] = field(default_factory=list)

@dataclass
class FilterConfig:
    KEEP_DUP: str = "last"
    MAX_MISSING_RATE: float = 0.20
    MIN_SD_ITEMS: float = 0.01
    LONGSTRING_PCT: float = 0.90
    USE_JUMP_FILTER: bool = False
    JUMP_DIFF: int = 3
    JUMP_RATE_TH: float = 0.80
    MIN_ITEMS_FOR_JUMP: int = 4

@dataclass
class CFAConfig:
    RUN_CFA: bool = True
    CFA_MISSING: str = "listwise"
    CFA_OBJ: str = "MLW"
    CFA_ROBUST_SE: bool = False

@dataclass
class PLSConfig:
    RUN_PLS: bool = True
    PLS_MISSING: str = "mean"
    PLS_SCHEME: str = "PATH"          # SmartPLS4 對齊：PATH / FACTORIAL / PCA
    HTMT_CORR_METHOD: str = "pearson"
    PLS_BOOT: int = 200
    PLS_SEED: int = 0
    Q2_FOLDS: int = 5
    PLSPM_MAX_ITER: int = 3000        # SmartPLS4 對齊
    PLSPM_TOL: float = 1e-7           # SmartPLS4 對齊
    BOOT_RETRY: int = 5

@dataclass
class MGAConfig:
    RUN_MGA: bool = False
    MGA_BOOT: int = 200
    MGA_MIN_N_PER_GROUP: int = 30
    MGA_SPLITS: List[str] = field(default_factory=lambda: ["CCO","BS"])

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

    # schema.py 會填充這裡
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
