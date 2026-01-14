# pls_project/config.py
from dataclasses import dataclass, field
from typing import List

@dataclass
class ColumnConfig:
    TS_COL: str    = "時間戳記"
    USER_COL: str  = "使用者名稱"
    EXP_COL: str   = "請問您是否曾使用 GenAI 進行學習的經驗？"
    EMAIL_COL: str = "電子郵件地址"

# ...其餘 IOConfig / ScaleConfig / FilterConfig / CFAConfig / PLSConfig / MGAConfig / ScenarioConfig / Config
