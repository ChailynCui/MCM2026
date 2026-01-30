"""环境特征工程模块"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_environment_features(long_df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """构建赛季-周层面的环境特征并合并到选手-周粒度
    
    静态特征（该周的特征）：
    - env_mean: 评委评分总体均分
    - env_std: 评委评分总体差异性（标准差）
    - env_count: 此赛段参赛人数
    - env_week: 赛段时间（第k赛段）
    - env_intensity: 赛段竞争激烈程度（最高分-最低分）
    - env_boundary_margin: 淘汰边缘分数密集程度（最低分和次低分的差距）
    - env_ties: 并列数量
    - env_special: 赛制特征（是否是特殊赛段，如最后一周）
    
    动态特征（与前k周的变化）：
    - env_mean_roll_std: 评委评分总体均分波动性
    - env_std_roll_std: 评委评分标准差波动性
    - env_count_delta: 参赛人数变化
    - env_intensity_delta: 激烈程度变化
    - env_ties_delta: 并列数量变化
    - env_boundary_margin_delta: 淘汰边缘密集程度变化
    """
    df = long_df.copy()

    # ===== 静态特征：赛季-周层面的统计 =====
    season_week = df.groupby(["season", "week"])["judge_total_score"].agg(
        env_mean="mean",
        env_std="std",
        env_min="min",
        env_max="max",
        env_count="count",
    ).reset_index()

    # 赛段时间（第k赛段）
    season_week["env_week"] = season_week.groupby("season")["week"].rank(method="first")
    
    # 赛段竞争激烈程度（最高分-最低分）
    season_week["env_intensity"] = season_week["env_max"] - season_week["env_min"]

    # 淘汰边缘分数密集程度（最低分和次低分的差距）
    def _margin(group: pd.Series) -> float:
        values = group.sort_values().values
        if len(values) < 2:
            return np.nan
        return values[1] - values[0]

    margin = df.groupby(["season", "week"])["judge_total_score"].apply(_margin).reset_index(name="env_boundary_margin")
    season_week = season_week.merge(margin, on=["season", "week"], how="left")

    # 并列数量
    ties = df.groupby(["season", "week"])["judge_total_score"].apply(lambda s: s.duplicated().sum()).reset_index(name="env_ties")
    season_week = season_week.merge(ties, on=["season", "week"], how="left")

    # 赛制特征：是否是特殊赛段（最后一周）
    max_week_by_season = df.groupby("season")["week"].max().reset_index(name="max_week")
    season_week = season_week.merge(max_week_by_season, on="season", how="left")
    season_week["env_special"] = (season_week["week"] == season_week["max_week"]).astype(int)
    season_week = season_week.drop(columns=["max_week"])

    # ===== 动态特征：前k周的波动性 =====
    season_week = season_week.sort_values(["season", "week"])
    
    # 评委评分总体均分波动性（前k周的标准差）
    season_week["env_mean_roll_std"] = (
        season_week.groupby("season")["env_mean"]
        .rolling(window, min_periods=1).std()
        .reset_index(level=0, drop=True)
        .fillna(0)
    )
    
    # 评委评分标准差波动性（前k周的标准差）
    season_week["env_std_roll_std"] = (
        season_week.groupby("season")["env_std"]
        .rolling(window, min_periods=1).std()
        .reset_index(level=0, drop=True)
        .fillna(0)
    )
    
    # 与前一周的变化
    season_week["env_count_delta"] = season_week.groupby("season")["env_count"].diff().fillna(0)
    season_week["env_intensity_delta"] = season_week.groupby("season")["env_intensity"].diff().fillna(0)
    season_week["env_ties_delta"] = season_week.groupby("season")["env_ties"].diff().fillna(0)
    season_week["env_boundary_margin_delta"] = season_week.groupby("season")["env_boundary_margin"].diff().fillna(0)

    df = df.merge(season_week, on=["season", "week"], how="left")
    return df
