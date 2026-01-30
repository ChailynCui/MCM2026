import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '.')

from preprocess.load_and_clean import load_data, clean_data, reshape_to_long_weeks
from features.judge_features import build_judge_features
from features.environment_features import build_environment_features
from features.personal_features import build_personal_features
from models.vote_share_model import VoteShareModel
from evaluation.consistency import calculate_hit_at_k, hit_at_k

# 加载并构建数据
raw_data = pd.read_csv('data/2026_MCM_Problem_C_Data.csv')
cleaned_data = clean_data(raw_data)
long_df = reshape_to_long_weeks(cleaned_data)

# 创建淘汰标签
max_week_by_season = long_df.groupby('season')['week'].max().reset_index(name='max_week')
long_df = long_df.merge(max_week_by_season, on='season', how='left')
is_champion = (long_df['placement'] == 1)
long_df.loc[is_champion, 'exit_week'] = 0
long_df['is_eliminated'] = (long_df['week'] == long_df['exit_week']).astype(int)

# 添加特征
long_df = build_judge_features(long_df)
long_df = build_environment_features(long_df)
long_df = build_personal_features(long_df)

# 训练模型
feature_cols = [
    'judge_mean', 'judge_std', 'judge_mean_roll_std', 'judge_std_roll_std',
    'env_mean', 'env_std', 'env_count', 'env_special', 'env_week',
    'env_intensity', 'env_boundary_margin', 'env_ties', 'env_mean_roll_std',
    'env_std_roll_std', 'env_count_delta', 'env_intensity_delta', 'env_ties_delta',
    'env_boundary_margin_delta', 'weeks_in_competition', 'rank_max_change',
    'rank_mean', 'rank_std', 'rank_spike_count', 'bottom_k_count'
]

long_df[feature_cols] = long_df[feature_cols].fillna(0)

print('=== 训练模型 ===')
elimination_model = VoteShareModel()
valid_mask = long_df['judge_total_score'] > 0
valid_df = long_df[valid_mask].copy()
elimination_model.fit(valid_df[feature_cols].values, valid_df['is_eliminated'].values)

# 预测
long_df['elimination_prob'] = elimination_model.predict(long_df[feature_cols].values)
long_df['elimination_prob'] = long_df['elimination_prob'].clip(0, 1)  # 限制在[0,1]
print(f'预测完成')

# 计算观众投票比例
long_df['survival_prob'] = 1 - long_df['elimination_prob']
long_df['audience_share'] = long_df.groupby(['season', 'week'])['survival_prob'].transform(
    lambda s: s / (s.sum() + 1e-10)
)

# 检查 Season 2, Week 1
print('\n=== Season 2, Week 1 的预测结果 ===')
week_data = long_df[(long_df['season'] == 2) & (long_df['week'] == 1)].copy()
display_cols = ['celebrity_name', 'judge_total_score', 'elimination_prob', 'audience_share', 'is_eliminated']
result = week_data[display_cols].sort_values('audience_share')
print(result)

print('\n=== 手动测试 hit_at_k ===')
# 获取淘汰者
eliminated_in_week = week_data[week_data['is_eliminated'] == 1]['celebrity_name'].values
print(f'淘汰者: {eliminated_in_week}')

# 测试hit_at_k
audience_share_series = week_data.set_index('celebrity_name')['audience_share']
print(f'\naudience_share Series (sorted):')
print(audience_share_series.sort_values())

print(f'\naudience_share.nsmallest(2):')
print(audience_share_series.nsmallest(2))

# 尝试手动查看是否命中
for elim_name in eliminated_in_week:
    result_hit = hit_at_k(audience_share_series, elim_name, k=2)
    print(f'\nhit_at_k({elim_name}, k=2) = {result_hit}')
    
    # Debug: 看看index是什么类型
    lowest_2 = audience_share_series.nsmallest(2).index
    print(f'  nsmallest(2).index = {lowest_2.tolist()}')
    print(f'  {elim_name} in index? {elim_name in lowest_2}')

# 使用完整函数
print('\n=== 使用 calculate_hit_at_k 函数 ===')
results = calculate_hit_at_k(long_df, predicted_col='audience_share', k=2)
print(f'Hit@2 结果: hit_rate={results["hit_rate"]}, details样本:')
print(results['details'].head(10))
