"""公平性评估模块"""

import pandas as pd
import numpy as np


def calculate_fairness_score(predictions, sensitive_attribute, groups):
    """计算公平性分数
    
    Args:
        predictions: 预测值
        sensitive_attribute: 敏感属性（如性别、种族等）
        groups: 敏感属性的不同组别
    """
    fairness_scores = {}
    
    for group in groups:
        group_mask = sensitive_attribute == group
        group_predictions = predictions[group_mask]
        fairness_scores[group] = {
            'count': np.sum(group_mask),
            'mean': np.mean(group_predictions),
            'std': np.std(group_predictions)
        }
    
    return fairness_scores


def calculate_disparate_impact(outcomes, protected_group_mask, control_group_mask):
    """计算差异影响比率 (Disparate Impact Ratio)"""
    protected_rate = np.mean(outcomes[protected_group_mask])
    control_rate = np.mean(outcomes[control_group_mask])
    
    if control_rate == 0:
        return np.inf if protected_rate > 0 else 1.0
    
    disparate_impact_ratio = protected_rate / control_rate
    
    return disparate_impact_ratio


def evaluate_bias(predictions, actual, sensitive_attribute):
    """评估预测偏差"""
    errors = predictions - actual
    
    bias_analysis = {}
    for unique_val in sensitive_attribute.unique():
        mask = sensitive_attribute == unique_val
        bias_analysis[unique_val] = {
            'mean_error': np.mean(errors[mask]),
            'std_error': np.std(errors[mask]),
            'count': np.sum(mask)
        }
    
    return bias_analysis


def reversal_ratios(long_df: pd.DataFrame, judge_rank_col: str = "judge_rank", audience_rank_col: str = "elimination_prob") -> dict:
    """计算裁判/粉丝逆转比例

    - 裁判被粉丝逆转：裁判倒数 k，但粉丝前 k（粉丝给了他高投票，但裁判给了低分）
    - 粉丝被裁判逆转：粉丝倒数 k，但裁判前 k（粉丝给了他低投票，但裁判给了高分）
    
    衡量：裁判判分与粉丝投票的一致性程度
    """
    df = long_df.copy()

    # 按赛季-周计算排名
    df["judge_rank"] = df.groupby(["season", "week"])[judge_rank_col].rank(ascending=True, method="min")
    
    # 将 audience_rank 从概率转换为排名（高概率 = 低排名）
    df["audience_rank"] = df.groupby(["season", "week"])[audience_rank_col].rank(ascending=False, method="min")

    # 倒数k（这里k=2，即倒数2名）
    max_rank = df.groupby(["season", "week"])["judge_rank"].transform("max")
    judge_bottom = df["judge_rank"] >= (max_rank - 1)  # 倒数2名
    judge_top = df["judge_rank"] <= 2                   # 前2名

    audience_bottom = df["audience_rank"] >= (df.groupby(["season", "week"])["audience_rank"].transform("max") - 1)
    audience_top = df["audience_rank"] <= 2

    # 逆转计算
    judge_overturned = (judge_bottom & audience_top).mean()    # 裁判给低分，粉丝给高投票
    audience_overturned = (audience_bottom & judge_top).mean()  # 粉丝给低投票，裁判给高分

    return {
        "judge_overturned_ratio": float(judge_overturned),      # 裁判被粉丝逆转
        "audience_overturned_ratio": float(audience_overturned),  # 粉丝被裁判逆转
    }
