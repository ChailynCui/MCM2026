"""错误检测模块"""

from __future__ import annotations

import re
from typing import Dict, List

import pandas as pd


WEEK_JUDGE_PATTERN = re.compile(r"^week(\d+)_judge(\d+)_score$")

#没有用上
def detect_outliers(data: pd.DataFrame, columns: List[str] | None = None) -> Dict[str, pd.DataFrame]:
    """检测异常值"""
    if columns is None:
        columns = data.select_dtypes(include=["number"]).columns

    outliers = {}
    for col in columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers[col] = data[(data[col] < lower_bound) | (data[col] > upper_bound)]

    return outliers


def validate_ranges(data: pd.DataFrame, ranges_dict: Dict[str, tuple]) -> Dict[str, pd.DataFrame]:
    """验证数据范围"""
    errors = {}
    for col, (min_val, max_val) in ranges_dict.items():
        if col in data.columns:
            out_of_range = data[(data[col] < min_val) | (data[col] > max_val)]
            if len(out_of_range) > 0:
                errors[col] = out_of_range

    return errors

#这个不算数据预处理，后面统计时再用
def detect_score_overflow(data: pd.DataFrame, max_score: float = 10.0) -> pd.DataFrame:
    """检测单项评委分数超过上限的记录（可能是特殊赛段加分）"""
    score_cols = [col for col in data.columns if WEEK_JUDGE_PATTERN.match(col)]
    overflow_mask = (data[score_cols] > max_score).any(axis=1)
    return data[overflow_mask]


def detect_exit_week_mismatch(data: pd.DataFrame) -> pd.DataFrame:
    """检测 results 和分数序列推断的退出周不一致"""
    required_cols = {"exit_week_from_results", "exit_week_from_scores"}
    if not required_cols.issubset(set(data.columns)):
        return pd.DataFrame()
    mismatch = (
        data["exit_week_from_results"].notna()
        & data["exit_week_from_scores"].notna()
        & (data["exit_week_from_results"] != data["exit_week_from_scores"])
    )
    return data[mismatch]

#检测重复名次，最终数据出现重复名次的均为并列名次，无错误
def detect_placement_inconsistency(data: pd.DataFrame) -> pd.DataFrame:
    """检测 placement 与 season 内唯一性异常"""
    if "season" not in data.columns or "placement" not in data.columns:
        return pd.DataFrame()
    placement_counts = data.groupby(["season", "placement"]).size().reset_index(name="count")
    return placement_counts[placement_counts["count"] > 1]
