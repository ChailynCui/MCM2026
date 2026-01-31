"""一致性评估模块"""

from __future__ import annotations

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, kendalltau


def calculate_consistency_score(predictions, actual):
    """计算预测的一致性分数"""
    if len(predictions) != len(actual):
        raise ValueError("预测值和实际值长度不一致")
    
    # 计算相关系数
    pearson_corr = np.corrcoef(predictions, actual)[0, 1]
    spearman_corr, _ = spearmanr(predictions, actual)
    kendall_tau, _ = kendalltau(predictions, actual)
    
    consistency = {
        'pearson': pearson_corr,
        'spearman': spearman_corr,
        'kendall_tau': kendall_tau,
        'average': (pearson_corr + spearman_corr + kendall_tau) / 3
    }
    
    return consistency


def evaluate_temporal_consistency(data, time_column, target_column):
    """评估时间序列一致性"""
    if time_column not in data.columns or target_column not in data.columns:
        raise ValueError("指定的列不存在")
    
    sorted_data = data.sort_values(time_column)
    autocorr = sorted_data[target_column].autocorr()
    
    return {
        'autocorrelation': autocorr,
        'trend': 'increasing' if sorted_data[target_column].diff().mean() > 0 else 'decreasing'
    }


def hit_at_k(
    predicted_scores: pd.Series,
    actual_eliminated_names: list[str],
    k: int = 2,
    higher_is_worse: bool = False
) -> float:
    """按周计算 Hit@k 分数

    规则：
    - 若 k < 淘汰人数 n，则分数 = 命中数 / k（上限 1）
    - 若 k >= n，则分数 = 命中数 / n
    """
    if not actual_eliminated_names:
        return np.nan
    if higher_is_worse:
        worst_k = predicted_scores.nlargest(k).index
    else:
        worst_k = predicted_scores.nsmallest(k).index
    hit_count = sum(name in worst_k for name in actual_eliminated_names)
    n = len(actual_eliminated_names)
    denom = min(k, n)
    if denom == 0:
        return np.nan
    return float(hit_count / denom)


def elimination_boundary_margin(predicted_scores: pd.Series, n_eliminated: int = 1, higher_is_worse: bool = False) -> float:
    """淘汰边界间隔
    
    计算被淘汰的最后一人与未被淘汰的第一人之间的间隔
    
    Parameters
    ----------
    predicted_scores : pd.Series
        该周所有人的预测分数（以celebrity_name为index）
    n_eliminated : int
        该周实际淘汰人数
    higher_is_worse : bool
        是否分数越高越危险（True表示排名法，False表示百分比法）
    
    Returns
    -------
    float
        边界间隔值
    """
    if len(predicted_scores) < n_eliminated + 1:
        return np.nan
    
    sorted_scores = predicted_scores.sort_values(ascending=(not higher_is_worse))
    # 按危险程度排序：
    # - higher_is_worse=False (百分比法): 分数低的在前（危险）
    # - higher_is_worse=True (排名法): 分数高的在前（危险）
    
    # 边界是第n_eliminated个和第n_eliminated+1个之间
    last_eliminated_score = sorted_scores.iloc[n_eliminated - 1]
    first_safe_score = sorted_scores.iloc[n_eliminated]
    
    # 返回绝对间隔
    return abs(first_safe_score - last_eliminated_score)


def calculate_hit_at_k(
    long_df: pd.DataFrame,
    predicted_col: str = "combined_share",
    k: int = 2,
    higher_is_worse: bool = False
) -> dict:
    """计算 Hit@k - 能否定位淘汰者
    
    对于每一周，检查是否在倒数k名中包含实际淘汰者
    """
    results = []
    
    for (season, week), group in long_df.groupby(["season", "week"]):
        eliminated = group[group["is_eliminated"] == 1]["celebrity_name"].tolist()
        
        if len(eliminated) == 0:
            # 无淘汰周
            continue
        
        # 将predicted_col转换为以celebrity_name为index的Series
        scores_series = group.set_index("celebrity_name")[predicted_col]
        
        hit = hit_at_k(scores_series, eliminated, k=k, higher_is_worse=higher_is_worse)
        results.append({
            'season': season,
            'week': week,
            'n_eliminated': len(eliminated),
            f'hit_at_{k}': hit
        })
    
    if len(results) == 0:
        return {'hit_rate': np.nan, 'details': pd.DataFrame()}
    
    results_df = pd.DataFrame(results)
    hit_rate = results_df[f'hit_at_{k}'].mean()
    
    return {
        'hit_rate': float(hit_rate),
        'hit_count': float(results_df[f'hit_at_{k}'].sum()),
        'total_eliminations': int(results_df["n_eliminated"].sum()),
        'details': results_df
    }


def analyze_boundary_margin_distribution(long_df: pd.DataFrame, predicted_col: str = "audience_share", higher_is_worse: bool = False, by_elimination_count: bool = False) -> dict:
    """分析淘汰边界间隔分布
    
    Parameters
    ----------
    long_df : pd.DataFrame
        包含淘汰标签和预测分数的长表
    predicted_col : str
        预测分数列名（如 "combined_share", "combined_rank"）
    higher_is_worse : bool
        是否分数越高越危险（用于排名法vs百分比法）
    by_elimination_count : bool
        是否按淘汰人数分组统计，默认为 False（全局统计）
    
    Returns
    -------
    dict
        包含统计信息和详细分布
    """
    margins = []
    
    for (season, week), group in long_df.groupby(["season", "week"]):
        n_elim = int(group["is_eliminated"].sum())
        if n_elim == 0:
            # 无淘汰周跳过
            continue
        
        scores_series = group.set_index("celebrity_name")[predicted_col]
        margin = elimination_boundary_margin(scores_series, n_eliminated=n_elim, higher_is_worse=higher_is_worse)
        
        if not np.isnan(margin):
            margins.append({
                'season': season,
                'week': week,
                'n_eliminated': n_elim,
                'boundary_margin': margin
            })
    
    if len(margins) == 0:
        return {
            'mean': np.nan,
            'std': np.nan,
            'min': np.nan,
            'max': np.nan,
            'median': np.nan,
            'q25': np.nan,
            'q75': np.nan,
            'distribution': np.array([])
        }
    
    margins_df = pd.DataFrame(margins)
    
    result = {
        'mean': float(margins_df["boundary_margin"].mean()),
        'std': float(margins_df["boundary_margin"].std()),
        'min': float(margins_df["boundary_margin"].min()),
        'max': float(margins_df["boundary_margin"].max()),
        'median': float(margins_df["boundary_margin"].median()),
        'q25': float(margins_df["boundary_margin"].quantile(0.25)),
        'q75': float(margins_df["boundary_margin"].quantile(0.75)),
        'distribution': margins_df["boundary_margin"].values
    }
    
    # 按淘汰人数分组
    if by_elimination_count:
        result['by_elimination_count'] = {}
        for n_elim in sorted(margins_df["n_eliminated"].unique()):
            subset = margins_df[margins_df["n_eliminated"] == n_elim]
            if len(subset) > 0:
                result['by_elimination_count'][int(n_elim)] = {
                    'mean': float(subset["boundary_margin"].mean()),
                    'std': float(subset["boundary_margin"].std()),
                    'median': float(subset["boundary_margin"].median()),
                    'count': len(subset),
                    'min': float(subset["boundary_margin"].min()),
                    'max': float(subset["boundary_margin"].max()),
                }
    
    return result

