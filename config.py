# pls_project/config.py
from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class ColumnConfig:
    TS_COL: str    = "時間戳記"
    USER_COL: str  = "使用者名稱"
    EXP_COL: str   = "請問您是否曾使用 GenAI 進行學習的經驗？"
    EMAIL_COL: str = "電子郵件地址"

@dataclass
class RuntimeConfig:
    """由 schema 在 runtime 填充，不需要你手動改。"""
    profile_name: str = "unknown"
    scale_prefixes: List[str] = field(default_factory=list)
    rename_map: Dict[str, str] = field(default_factory=dict)

# ...其餘 IOConfig / ScaleConfig / FilterConfig / CFAConfig / PLSConfig / MGAConfig / ScenarioConfig

@dataclass
class Config:
    cols: ColumnConfig = field(default_factory=ColumnConfig)
    # io: IOConfig = ...
    # scales: ScaleConfig = ...
    # ...
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
