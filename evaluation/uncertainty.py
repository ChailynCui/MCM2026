"""不确定性评估模块"""

from __future__ import annotations

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


def analyze_vote_share_intervals(
    long_df: pd.DataFrame,
    feature_cols: list[str],
    model_class,
    train_mask: pd.Series | None = None,
    vote_share_col: str = "audience_share",
    confidence: float = 0.95,
    n_bootstrap: int = 500,
    random_state: int = 42
) -> dict:
    """使用模型Bootstrap分析每个选手每周的投票比例不确定性
    
    方法：
    1. 对训练样本有放回重采样
    2. 重新训练观众投票预测模型
    3. 对全体样本预测投票比例
    4. 统计每个选手-周的Bootstrap分布区间与方差
    
    Parameters
    ----------
    long_df : pd.DataFrame
        包含特征与标签的长表
    feature_cols : list[str]
        模型特征列
    model_class : type
        可实例化的模型类（例如 VoteShareModel）
    train_mask : pd.Series | None
        训练样本掩码（与 long_df 对齐），若为 None 则使用全量样本
    vote_share_col : str
        原始投票比例列名，用于输出对比
    confidence : float
        置信度，默认0.95
    n_bootstrap : int
        Bootstrap重采样次数，默认500
    random_state : int
        随机种子
    
    Returns
    -------
    dict
        包含每个选手-周的区间和不确定性指标
    """
    if train_mask is None:
        train_mask = pd.Series(True, index=long_df.index)

    train_df = long_df.loc[train_mask]
    if train_df.empty:
        return {
            'celeb_week_level': pd.DataFrame(),
            'mean_interval_width': np.nan,
            'mean_variance': np.nan,
            'summary': {}
        }

    n_rows = len(long_df)
    rng = np.random.default_rng(random_state)

    bootstrap_preds = np.zeros((n_bootstrap, n_rows), dtype=float)

    for b in range(n_bootstrap):
        sample_idx = rng.choice(train_df.index.to_numpy(), size=len(train_df), replace=True)
        boot_df = long_df.loc[sample_idx]

        model = model_class()
        model.fit(boot_df[feature_cols], boot_df["is_eliminated"])

        elimination_prob = model.predict(long_df[feature_cols])
        survival_prob = pd.Series(1 - elimination_prob, index=long_df.index)
        audience_share = survival_prob.groupby([long_df["season"], long_df["week"]]).transform(
            lambda s: s / (s.sum() + 1e-10)
        )
        bootstrap_preds[b, :] = audience_share.to_numpy()

    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    ci_lower = np.percentile(bootstrap_preds, lower_percentile, axis=0)
    ci_upper = np.percentile(bootstrap_preds, upper_percentile, axis=0)
    interval_width = ci_upper - ci_lower

    mean_bootstrap = bootstrap_preds.mean(axis=0)
    std_bootstrap = bootstrap_preds.std(axis=0)

    celeb_week_df = long_df[["season", "week", "celebrity_name", vote_share_col]].copy()
    celeb_week_df = celeb_week_df.rename(columns={vote_share_col: "vote_share"})
    celeb_week_df["bootstrap_mean"] = mean_bootstrap
    celeb_week_df["bootstrap_std"] = std_bootstrap
    celeb_week_df["bootstrap_variance"] = std_bootstrap ** 2
    celeb_week_df["interval_lower"] = ci_lower
    celeb_week_df["interval_upper"] = ci_upper
    celeb_week_df["interval_width"] = interval_width
    celeb_week_df["n_bootstrap"] = n_bootstrap
    celeb_week_df["confidence_level"] = confidence

    summary = {
        'mean_interval_width': float(np.mean(interval_width)),
        'std_interval_width': float(np.std(interval_width)),
        'mean_variance': float(np.mean(std_bootstrap ** 2)),
        'mean_std': float(np.mean(std_bootstrap)),
        'total_observations': int(n_rows),
        'total_weeks': int(long_df.groupby(["season", "week"]).ngroups)
    }

    return {
        'celeb_week_level': celeb_week_df,
        'bootstrap_samples': bootstrap_preds,  # (n_bootstrap, n_rows)
        'mean_interval_width': float(np.mean(interval_width)),
        'mean_variance': float(np.mean(std_bootstrap ** 2)),
        'summary': summary
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


def analyze_vote_count_uncertainty(
    bootstrap_samples: np.ndarray,
    long_df: pd.DataFrame,
    base_total_votes: float = 5e6,
    votes_variation_by_week: bool = True,
    random_state: int = 42
) -> dict:
    """分析投票总数和个人票数的不确定性
    
    子问题3：观众投票总数确定性分析
    
    核心设计：
    1. 直接使用 Bootstrap 原始样本（保证周内比例和=1）
    2. 每周有不同的总票数（基于选手数、赛季进度）
    3. 可复现（固定随机种子）
    
    Parameters
    ----------
    bootstrap_samples : np.ndarray
        Bootstrap 原始样本矩阵 (n_bootstrap, n_rows)，来自 analyze_vote_share_intervals
    long_df : pd.DataFrame
        原始长表，包含 season, week, celebrity_name, audience_share 列
    base_total_votes : float
        基准投票总数，默认 500万
    votes_variation_by_week : bool
        是否允许每周投票总数不同，默认 True
    random_state : int
        随机种子，默认 42
        
    Returns
    -------
    dict
        包含：
        - 'weekly_summary': 每周投票总数的统计
        - 'individual_summary': 每位选手票数的不确定性指标
        - 'overall_stats': 全局统计
    """
    rng = np.random.default_rng(random_state)
    n_bootstrap = bootstrap_samples.shape[0]
    
    # 为每周计算不同的总票数（基于选手数和赛季进度）
    week_info = long_df.groupby(['season', 'week']).agg(
        n_contestants=('celebrity_name', 'count'),
        week_num=('week', 'first'),
        season_num=('season', 'first')
    ).reset_index()
    
    weeks_data = []
    individual_data = []
    
    for idx, week_row in week_info.iterrows():
        season = week_row['season']
        week = week_row['week']
        n_contestants = week_row['n_contestants']
        
        # 每周的总票数根据选手数和赛季进度调整
        if votes_variation_by_week:
            # 选手越少（赛季后期），投票数略减少
            # 赛季越新，投票数略增加（节目热度提升）
            contestant_factor = 0.8 + 0.2 * (n_contestants / 12)  # 假设最多12人
            season_factor = 0.9 + 0.1 * (season / 34)  # 第34季热度最高
            week_total_votes = base_total_votes * contestant_factor * season_factor
        else:
            week_total_votes = base_total_votes
        
        weeks_data.append({
            'season': season,
            'week': week,
            'total_votes': week_total_votes,
            'n_contestants': n_contestants
        })
        
        # 获取该周所有选手的索引
        week_mask = (long_df['season'] == season) & (long_df['week'] == week)
        week_indices = long_df.index[week_mask].tolist()
        week_group = long_df.loc[week_mask]
        
        # 从 Bootstrap 样本中提取该周所有选手的比例（保证和=1）
        # bootstrap_samples[:, week_indices] 形状: (n_bootstrap, n_contestants_in_week)
        week_shares_bootstrap = bootstrap_samples[:, week_indices]  # (n_bootstrap, n_contestants)
        
        # 票数 = 比例 × 总票数
        # week_votes_bootstrap: (n_bootstrap, n_contestants)
        week_votes_bootstrap = week_shares_bootstrap * week_total_votes
        
        # 为每位选手计算统计指标
        for i, (_, row) in enumerate(week_group.iterrows()):
            celebrity = row['celebrity_name']
            
            # 该选手的票数样本
            vote_samples = week_votes_bootstrap[:, i]
            
            vote_mean = np.mean(vote_samples)
            vote_std = np.std(vote_samples)
            vote_ci_lower = np.percentile(vote_samples, 2.5)
            vote_ci_upper = np.percentile(vote_samples, 97.5)
            
            individual_data.append({
                'season': season,
                'week': week,
                'celebrity_name': celebrity,
                'vote_mean': vote_mean,
                'vote_std': vote_std,
                'vote_ci_lower': vote_ci_lower,
                'vote_ci_upper': vote_ci_upper,
                'vote_ci_width': vote_ci_upper - vote_ci_lower,
                'coefficient_of_variation': vote_std / vote_mean if vote_mean > 0 else np.nan,
                'relative_ci_width': (vote_ci_upper - vote_ci_lower) / vote_mean if vote_mean > 0 else np.nan
            })
    
    weeks_df = pd.DataFrame(weeks_data)
    individual_df = pd.DataFrame(individual_data)
    
    return {
        'weekly_summary': weeks_df,
        'individual_summary': individual_df,
        'overall_stats': {
            'base_total_votes': base_total_votes,
            'mean_weekly_votes': float(weeks_df['total_votes'].mean()),
            'std_weekly_votes': float(weeks_df['total_votes'].std()),
            'mean_individual_cv': float(individual_df['coefficient_of_variation'].mean()),
            'mean_individual_ci_width': float(individual_df['vote_ci_width'].mean()),
            'mean_individual_std': float(individual_df['vote_std'].mean()),
            'mean_relative_ci_width': float(individual_df['relative_ci_width'].mean())
        }
    }

