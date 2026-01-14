# pls_project/rulebook.py
from __future__ import annotations

# =========================
# Meta columns rulebook
# =========================
META_RULES = {
    "TS_COL": {
        "exact": ["時間戳記"],
        "contains": ["時間戳", "timestamp", "Timestamp", "提交時間", "填寫時間"],
        "regex": [r"時間.*戳", r"^timestamp$"],
        "fuzzy_hint": "時間戳記",
    },
    "EMAIL_COL": {
        "exact": ["電子郵件地址"],
        "contains": ["Email", "E-mail", "email", "信箱", "郵件"],
        "regex": [r"mail", r"email", r"電子.*郵件"],
        "fuzzy_hint": "電子郵件地址",
    },
    "USER_COL": {
        "exact": ["使用者名稱"],
        "contains": ["姓名", "name", "暱稱", "使用者"],
        "regex": [r"姓名", r"name"],
        "fuzzy_hint": "使用者名稱",
    },
    # v1: 是否曾使用GenAI(是/否)
    # v2: 使用AI工具來學習之經驗 / 頻率
    # v3: 沒有是/否題（只有開始時間/近3個月頻率等），所以解析不到也要允許
    "EXP_COL": {
        "exact": [
            "請問您是否曾使用 GenAI 進行學習的經驗？",  # v1
            "使用AI工具來學習之經驗",                  # v2
            "使用AI工具來學習之頻率",                  # v2
        ],
        "contains": [
            "是否曾使用", "GenAI", "AI工具來學習之經驗", "AI工具來學習之頻率",
            "使用 AI 工具來學習之經驗", "使用 AI 工具來學習之頻率",
        ],
        "regex": [
            r"是否.*使用.*AI", r"GenAI", r"AI.*學習.*經驗", r"AI.*學習.*頻率",
        ],
        "fuzzy_hint": "使用AI工具來學習之經驗",
    },
}

# =========================
# Questionnaire profiles rulebook
# =========================
PROFILES = [
    {
        "name": "v1",
        "signatures_any": ["PA1", "BS1", "BB1", "SA1", "ACO1", "CCO1", "MIND1", "CI1", "LO1"],
        "scale_prefixes": ["PA","BS","BB","SA","ACO","CCO","MIND","CI","LO"],
        # v1 欄名通常就是 token 本身：PA1, BS2...
        "item_token_regex": r"^([A-Z]{1,6}\d{1,2})\b",
    },
    {
        "name": "v2",
        "signatures_any": ["SRL2","SRL3","AM1","CN1","RN1","AN1","EM1","KS1","LO1"],
        "scale_prefixes": ["SRL","AM","CN","RN","AN","EM","KS","LO"],
        # v2 可能是 "SRL3 我會將..."：取最前面的 token
        "item_token_regex": r"^([A-Z]{1,6}\d{1,2})\b",
    },
    {
        "name": "v3",
        "signatures_any": ["A11","A12","B11","B12","C11","C12"],
        "scale_prefixes": ["A","B","C"],
        # v3 token: A11/B12/C24 這種
        "item_token_regex": r"^([ABC]\d{2})\b",
    },
]
