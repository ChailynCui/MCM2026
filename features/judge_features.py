"""评委特征工程模块"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_judge_features(long_df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """构建评委评分特征（静态/动态）

    静态特征：
    - judge_mean: 该选手在该周的评委评分平均值
    - judge_std: 该选手在该周，不同评委给分的标准差（评委意见分歧程度）
    
    动态特征：
    - judge_mean_roll_std: 该选手评分均值的波动性（前k周标准差）
    - judge_std_roll_std: 该选手评委分歧程度的波动性（前k周标准差）
    
    输入 long_df 需包含：season, week, celebrity_name, judge_total_score, judge_count, judge_std
    """
    df = long_df.copy()

    # 静态特征：评委评分均分（选手自己的评分），评委评分分歧（不同评委的标准差）
    df["judge_mean"] = df["judge_total_score"] / df["judge_count"].replace(0, np.nan)
    df["judge_mean"] = df["judge_mean"].fillna(0)
    # judge_std 已经在 reshape_to_long_weeks 中计算，这里直接使用

    # 动态特征：需要按时间排序
    df = df.sort_values(["season", "celebrity_name", "week"])
    
    # 评委评分波动性（该选手评分均值在前k周的标准差）
    df["judge_mean_roll_std"] = (
        df.groupby(["season", "celebrity_name"])["judge_mean"]
        .rolling(window, min_periods=1).std()
        .reset_index(level=[0, 1], drop=True)
        .fillna(0)
    )
    
    # 评委评分分歧波动性（该选手评委分歧程度在前k周的标准差）
    df["judge_std_roll_std"] = (
        df.groupby(["season", "celebrity_name"])["judge_std"]
        .rolling(window, min_periods=1).std()
        .reset_index(level=[0, 1], drop=True)
        .fillna(0)
    )

    return df
