"""一致性评估模块"""

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


def hit_at_k(predicted_scores: pd.Series, actual_eliminated_name: str, k: int = 2) -> int:
    """是否在倒数 k 名中命中淘汰者"""
    if actual_eliminated_name is None:
        return np.nan
    lowest_k = predicted_scores.nsmallest(k).index
    return int(actual_eliminated_name in lowest_k)


def elimination_boundary_margin(predicted_scores: pd.Series) -> float:
    """淘汰边界间隔（倒数第二 - 倒数第一）"""
    sorted_scores = predicted_scores.sort_values()
    if len(sorted_scores) < 2:
        return np.nan
    return sorted_scores.iloc[1] - sorted_scores.iloc[0]


def calculate_hit_at_k(long_df: pd.DataFrame, predicted_col: str = "audience_share", k: int = 2) -> dict:
    """计算 Hit@k - 能否定位淘汰者
    
    对于每一周，检查是否在倒数k名中包含实际淘汰者
    """
    results = []
    
    for (season, week), group in long_df.groupby(["season", "week"]):
        eliminated = group[group["is_eliminated"] == 1]["celebrity_name"].values
        
        if len(eliminated) == 0:
            # 无淘汰周
            continue
        
        # 将predicted_col转换为以celebrity_name为index的Series
        scores_series = group.set_index("celebrity_name")[predicted_col]
        
        for elim_name in eliminated:
            hit = hit_at_k(scores_series, elim_name, k=k)
            results.append({
                'season': season,
                'week': week,
                'celebrity_name': elim_name,
                f'hit_at_{k}': hit
            })
    
    if len(results) == 0:
        return {'hit_rate': np.nan, 'details': pd.DataFrame()}
    
    results_df = pd.DataFrame(results)
    hit_rate = results_df[f'hit_at_{k}'].mean()
    
    return {
        'hit_rate': float(hit_rate),
        'hit_count': int(results_df[f'hit_at_{k}'].sum()),
        'total_eliminations': len(results_df),
        'details': results_df
    }


def analyze_boundary_margin_distribution(long_df: pd.DataFrame, predicted_col: str = "audience_share") -> dict:
    """分析淘汰边界间隔分布"""
    margins = long_df.groupby(["season", "week"]).apply(
        lambda g: elimination_boundary_margin(g.set_index("celebrity_name")[predicted_col])
    ).reset_index(name="boundary_margin")
    
    margins_clean = margins[margins["boundary_margin"].notna()]
    
    return {
        'mean': float(margins_clean["boundary_margin"].mean()),
        'std': float(margins_clean["boundary_margin"].std()),
        'min': float(margins_clean["boundary_margin"].min()),
        'max': float(margins_clean["boundary_margin"].max()),
        'median': float(margins_clean["boundary_margin"].median()),
        'q25': float(margins_clean["boundary_margin"].quantile(0.25)),
        'q75': float(margins_clean["boundary_margin"].quantile(0.75)),
        'distribution': margins_clean["boundary_margin"].values
    }

