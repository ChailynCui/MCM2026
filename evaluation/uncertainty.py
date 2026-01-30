"""不确定性评估模块"""

import pandas as pd
import numpy as np


def calculate_prediction_intervals(predictions, errors, confidence=0.95):
    """计算预测区间"""
    from scipy import stats
    
    std_error = np.std(errors)
    z_score = stats.norm.ppf((1 + confidence) / 2)
    
    margin_of_error = z_score * std_error
    
    intervals = {
        'predictions': predictions,
        'lower_bound': predictions - margin_of_error,
        'upper_bound': predictions + margin_of_error,
        'margin_of_error': margin_of_error,
        'confidence_level': confidence
    }
    
    return intervals


def calculate_uncertainty_metrics(residuals):
    """计算不确定性指标"""
    metrics = {
        'mse': np.mean(residuals ** 2),
        'rmse': np.sqrt(np.mean(residuals ** 2)),
        'mae': np.mean(np.abs(residuals)),
        'std': np.std(residuals),
        'variance': np.var(residuals)
    }
    
    return metrics


def quantify_parameter_uncertainty(model, X_test):
    """量化参数不确定性"""
    # 这里可以实现 Bootstrap 或 Bayesian 方法
    # 示例实现
    
    return {
        'method': 'placeholder',
        'uncertainty': None
    }


def entropy_uncertainty(vote_share: pd.Series) -> float:
    """用熵衡量不确定性（越大越不确定）"""
    share = vote_share.clip(lower=1e-12)
    return float(-(share * np.log(share)).sum())


def weekly_uncertainty(long_df: pd.DataFrame, vote_share_col: str = "vote_share") -> pd.DataFrame:
    """按赛季-周计算不确定性"""
    grouped = long_df.groupby(["season", "week"])[vote_share_col].apply(entropy_uncertainty).reset_index(name="entropy")
    return grouped


def analyze_vote_share_intervals(long_df: pd.DataFrame, vote_share_col: str = "audience_share", confidence: float = 0.95) -> dict:
    """分析每个选手每周的投票比例区间和方差
    
    返回：
    - 选手级别：平均区间宽度、后验方差
    - 周级别：该周的不确定性分析
    """
    from scipy import stats
    
    # 选手级别分析
    celeb_analysis = long_df.groupby(["season", "celebrity_name"])[vote_share_col].agg([
        'mean',
        'std',
        'count',
        'min',
        'max'
    ]).reset_index()
    
    # 计算95%置信区间宽度
    z_score = stats.norm.ppf((1 + confidence) / 2)
    celeb_analysis['interval_width'] = (z_score * celeb_analysis['std'] / np.sqrt(celeb_analysis['count'])).fillna(0)
    celeb_analysis['variance'] = celeb_analysis['std'] ** 2
    
    # 周级别分析
    week_analysis = long_df.groupby(["season", "week"])[vote_share_col].agg([
        'mean',
        'std',
        'count',
        'min',
        'max'
    ]).reset_index()
    
    week_analysis['interval_width'] = (z_score * week_analysis['std'] / np.sqrt(week_analysis['count'])).fillna(0)
    week_analysis['variance'] = week_analysis['std'] ** 2
    
    # 识别高不确定性周（方差大的周）
    high_uncertainty_weeks = week_analysis[week_analysis['variance'] > week_analysis['variance'].quantile(0.75)]
    
    return {
        'celeb_level': celeb_analysis,
        'week_level': week_analysis,
        'high_uncertainty_weeks': high_uncertainty_weeks,
        'mean_interval_width': float(celeb_analysis['interval_width'].mean()),
        'mean_variance': float(celeb_analysis['variance'].mean()),
        'high_uncertainty_reason': _analyze_uncertainty_reasons(long_df, high_uncertainty_weeks)
    }


def _analyze_uncertainty_reasons(long_df: pd.DataFrame, high_uncertainty_weeks: pd.DataFrame) -> dict:
    """分析高不确定性周的原因"""
    reasons = {}
    
    for _, row in high_uncertainty_weeks.iterrows():
        season, week = int(row['season']), int(row['week'])
        week_data = long_df[(long_df['season'] == season) & (long_df['week'] == week)]
        
        # 可能的原因：
        # 1. 参赛人数少
        # 2. 评分分布均匀（竞争激烈）
        # 3. 并列现象多
        
        n_contestants = len(week_data)
        score_std = week_data['judge_total_score'].std()
        n_ties = week_data['judge_total_score'].duplicated().sum()
        
        reasons[f'Season {season}, Week {week}'] = {
            'n_contestants': int(n_contestants),
            'score_std': float(score_std),
            'n_ties': int(n_ties),
            'reason': _determine_reason(n_contestants, score_std, n_ties)
        }
    
    return reasons


def _determine_reason(n_contestants: int, score_std: float, n_ties: int) -> str:
    """判断不确定性的主要原因"""
    if n_contestants <= 2:
        return "参赛人数过少"
    elif score_std < 2.0:
        return "评分分布均匀（竞争激烈）"
    elif n_ties > n_contestants * 0.3:
        return "并列现象较多"
    else:
        return "综合因素"

