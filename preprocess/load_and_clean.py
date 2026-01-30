"""数据加载和清理模块"""

from __future__ import annotations

import re
from typing import List, Tuple

import numpy as np
import pandas as pd


WEEK_JUDGE_PATTERN = re.compile(r"^week(\d+)_judge(\d+)_score$")


def load_data(filepath: str) -> pd.DataFrame | None:
    """加载 CSV 数据文件"""
    try:
        data = pd.read_csv(filepath)
        print(f"成功加载数据: {filepath}")
        return data
    except Exception as e:
        print(f"数据加载失败: {e}")
        return None


def list_week_score_columns(columns: List[str]) -> List[str]:
    """返回所有 weekX_judgeY_score 列名"""
    return [col for col in columns if WEEK_JUDGE_PATTERN.match(col)]


def get_week_numbers(columns: List[str]) -> List[int]:
    """返回所有出现的周次编号"""
    weeks = sorted({int(WEEK_JUDGE_PATTERN.match(col).group(1)) for col in columns if WEEK_JUDGE_PATTERN.match(col)})
    return weeks


def coerce_score_columns(data: pd.DataFrame) -> pd.DataFrame:
    """将评分列转换为数值，N/A -> NaN"""
    score_cols = list_week_score_columns(list(data.columns))
    data[score_cols] = data[score_cols].replace("N/A", np.nan)
    data[score_cols] = data[score_cols].apply(pd.to_numeric, errors="coerce")
    return data


def compute_week_totals(data: pd.DataFrame) -> pd.DataFrame:
    """计算每位选手每周的评委总分（忽略缺失评委）"""
    data = data.copy()
    weeks = get_week_numbers(list(data.columns))
    for week in weeks:
        cols = [f"week{week}_judge{j}_score" for j in range(1, 5) if f"week{week}_judge{j}_score" in data.columns]
        data[f"week{week}_total_score"] = data[cols].sum(axis=1, skipna=True)
    return data


def parse_results_exit_week(results: str) -> int | None:
    """从 results 字段解析退出周次"""
    if not isinstance(results, str):
        return None
    match = re.search(r"Eliminated Week (\d+)", results)
    if match:
        return int(match.group(1))
    return None


def infer_exit_week_from_scores(row: pd.Series, weeks: List[int]) -> int | None:
    """根据周总分推断最后有分的周次"""
    last_positive = None
    for week in weeks:
        total_col = f"week{week}_total_score"
        if total_col in row and pd.notna(row[total_col]) and row[total_col] > 0:
            last_positive = week
    return last_positive


def derive_exit_week(data: pd.DataFrame) -> pd.DataFrame:
    """生成 exit_week 字段，优先使用 results，若不一致则以分数序列为准"""
    data = data.copy()
    weeks = get_week_numbers(list(data.columns))
    data["exit_week_from_results"] = data["results"].apply(parse_results_exit_week)
    data["exit_week_from_scores"] = data.apply(lambda r: infer_exit_week_from_scores(r, weeks), axis=1)
    data["exit_week"] = data["exit_week_from_results"].fillna(data["exit_week_from_scores"])
    mismatch = (
        data["exit_week_from_results"].notna()
        & data["exit_week_from_scores"].notna()
        & (data["exit_week_from_results"] != data["exit_week_from_scores"])
    )
    data.loc[mismatch, "exit_week"] = data.loc[mismatch, "exit_week_from_scores"]
    return data


def clean_data(data: pd.DataFrame) -> pd.DataFrame | None:
    """清理数据（重复值、评分列、退出周等）"""
    if data is None:
        return None

    data = data.copy()
    data = data.drop_duplicates()
    data = coerce_score_columns(data)
    data = compute_week_totals(data)
    data = derive_exit_week(data)

    print(f"缺失值统计:\n{data.isnull().sum()}")

    return data


def reshape_to_long_weeks(data: pd.DataFrame) -> pd.DataFrame:
    """将宽表转为选手-周粒度的长表"""
    data = data.copy()
    weeks = get_week_numbers(list(data.columns))
    records = []
    for week in weeks:
        total_col = f"week{week}_total_score"
        if total_col not in data.columns:
            continue
        
        # 统计每周每人有多少个评委给分，以及不同评委给分的标准差
        judge_cols = [f"week{week}_judge{j}_score" for j in range(1, 5) if f"week{week}_judge{j}_score" in data.columns]
        judge_count = data[judge_cols].notna().sum(axis=1)
        # judge_std: 同一选手在该周，不同评委给分的标准差（衡量评委意见分歧程度）
        judge_std = data[judge_cols].std(axis=1, skipna=True).fillna(0)
        
        temp = data[[
            "celebrity_name",
            "ballroom_partner",
            "celebrity_industry",
            "celebrity_homestate",
            "celebrity_homecountry/region",
            "celebrity_age_during_season",
            "season",
            "results",
            "placement",
            "exit_week",
            total_col,
        ]].copy()
        temp = temp.rename(columns={total_col: "judge_total_score"})
        temp["judge_count"] = judge_count
        temp["judge_std"] = judge_std
        temp["week"] = week
        records.append(temp)

    long_df = pd.concat(records, ignore_index=True)
    
    print("\n长表结构信息:")
    print(f"形状: {long_df.shape}")
    print(f"列名: {list(long_df.columns)}")
    print(f"数据类型:\n{long_df.dtypes}")
    
    return long_df
