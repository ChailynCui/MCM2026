"""特殊情况处理模块"""

from __future__ import annotations

import re
from typing import List

import numpy as np
import pandas as pd


WEEK_JUDGE_PATTERN = re.compile(r"^week(\d+)_judge(\d+)_score$")


def handle_missing_values(data: pd.DataFrame, strategy: str = "drop") -> pd.DataFrame:
    """处理缺失值

    Args:
        data: DataFrame
        strategy: 'drop' 或 'fill'
    """
    if strategy == "drop":
        return data.dropna()
    if strategy == "fill":
        return data.fillna(data.mean(numeric_only=True))
    raise ValueError("Unknown strategy")

#没有用上
def normalize_categorical(data: pd.DataFrame, categorical_columns: List[str]) -> pd.DataFrame:
    """标准化分类数据"""
    for col in categorical_columns:
        if col in data.columns:
            data[col] = data[col].astype(str).str.lower().str.strip()

    return data

#没有用上
def handle_duplicates(data: pd.DataFrame, subset: List[str] | None = None) -> pd.DataFrame:
    """处理重复值"""
    return data.drop_duplicates(subset=subset)


def mark_withdrawal_and_elimination(data: pd.DataFrame) -> pd.DataFrame:
    """识别退赛选手"""
    data = data.copy()
    if "results" not in data.columns:
        data["exit_type"] = "unknown"
        return data

    results = data["results"].astype(str).str.lower()
    data["is_withdrawn"] = results.str.contains("withdrawn|withdrew|retired|disqualified")
    data["exit_type"] = np.where(data["is_withdrawn"], "withdrawn", "eliminated")
    return data
