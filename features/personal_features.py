"""个人特征工程模块"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_personal_features(long_df: pd.DataFrame) -> pd.DataFrame:
    """构建个人静态/动态特征
    
    静态特征：
    - age: 角色年龄
    - industry: 角色行业
    - homestate_count: 该州参赛人数
    - homecountry_count: 该国/地区参赛人数
    - partner: 角色搭档
    - partner_history_count: 搭档历史参赛次数
    - partner_history_best: 搭档历史最佳名次
    
    动态特征：
    - weeks_in_competition: 累计参赛周数
    - rank_max_change: 表演轨迹-从最开始到当前的排名最大变化
    - rank_mean: 表演轨迹-从最开始到当前的排名均值
    - rank_std: 表演轨迹-从最开始到当前的排名方差
    - rank_spike_count: 异常提升次数（提升幅度超过K/2个人）
    - bottom_k_count: 危险区经历次数（在后K/4出现的次数）
    """
    df = long_df.copy()

    # ===== 静态特征 =====
    # 1. 角色年龄
    df["age"] = pd.to_numeric(df["celebrity_age_during_season"], errors="coerce")
    
    # 2. 角色行业
    df["industry"] = df["celebrity_industry"].astype(str)
    
    # 3. 角色地区（地区人数）
    df["homestate"] = df["celebrity_homestate"].fillna("unknown").astype(str)
    df["homecountry"] = df["celebrity_homecountry/region"].fillna("unknown").astype(str)
    df["homestate_count"] = df.groupby(["season", "homestate"])["celebrity_name"].transform("count")
    df["homecountry_count"] = df.groupby(["season", "homecountry"])["celebrity_name"].transform("count")

    # 4. 角色搭档
    df["partner"] = df["ballroom_partner"].fillna("unknown").astype(str)
    
    # 5. 搭档历史参赛次数和最佳名次
    partner_history = df.groupby("partner").agg(
        partner_history_count=("season", "nunique"),
        partner_history_best=("placement", "min")
    ).reset_index()
    df = df.merge(partner_history, on="partner", how="left")
    df["partner_history_count"] = df["partner_history_count"].fillna(1)
    df["partner_history_best"] = df["partner_history_best"].fillna(99)

    # ===== 动态特征 =====
    df = df.sort_values(["season", "celebrity_name", "week"])
    
    # 1. 累计参赛周数
    df["weeks_in_competition"] = df.groupby(["season", "celebrity_name"]).cumcount() + 1

    # 2. 计算排名
    df["judge_rank"] = df.groupby(["season", "week"])["judge_total_score"].rank(ascending=False, method="min")
    
    # 3. 表演轨迹：从最开始到当前的排名统计
    # 3.1 排名最大变化（当前排名 - 第一周排名）
    df["first_rank"] = df.groupby(["season", "celebrity_name"])["judge_rank"].transform("first")
    df["rank_max_change"] = df["judge_rank"] - df["first_rank"]
    
    # 3.2 排名均值和方差（从第一周到当前周）
    df["rank_mean"] = df.groupby(["season", "celebrity_name"])["judge_rank"].expanding().mean().reset_index(level=[0, 1], drop=True)
    df["rank_std"] = df.groupby(["season", "celebrity_name"])["judge_rank"].expanding().std().reset_index(level=[0, 1], drop=True).fillna(0)
    
    # 4. 异常提升次数：提升幅度超过K/2个人（K为当前比赛人数）
    df["rank_delta"] = df.groupby(["season", "celebrity_name"])["judge_rank"].diff().fillna(0)
    df["current_k"] = df.groupby(["season", "week"])["judge_rank"].transform("count")
    df["rank_spike"] = (df["rank_delta"] <= -(df["current_k"] / 2)).astype(int)  # 排名上升超过K/2
    df["rank_spike_count"] = df.groupby(["season", "celebrity_name"])["rank_spike"].cumsum()
    
    # 5. 危险区经历次数：在后K/4出现的次数
    df["danger_threshold"] = df["current_k"] - (df["current_k"] / 4)  # 前75%的安全线
    df["bottom_k_flag"] = (df["judge_rank"] > df["danger_threshold"]).astype(int)
    df["bottom_k_count"] = df.groupby(["season", "celebrity_name"])["bottom_k_flag"].cumsum()

    return df
