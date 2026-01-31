# MCM 2026 问题 C（DWTS）项目说明

本项目基于《与星共舞》（DWTS）公开数据，构建观众投票估算模型，并对不同投票结合方法进行比较分析。

## 项目目标
- 估算每位选手在每周的观众投票比例（不可观测量）。
- 评估估算结果对淘汰结果的解释能力（一致性指标）。
- 比较“百分比结合法”与“排名结合法”的淘汰预测效果。
- 分析不确定性与公平性指标。

## 目录结构
- main.py：主流程入口（数据加载、清洗、特征、训练、预测与评估）。
- data/：数据文件目录（包含 2026_MCM_Problem_C_Data.csv）。
- preprocess/：数据清洗与重塑。
- features/：特征工程（评委、环境、个人特征、交互特征）。
- models/：模型与规则判定逻辑。
- evaluation/：一致性、不确定性、公平性评估。
- problem.md：题目说明与背景。

## 运行方式
- 确保 data/2026_MCM_Problem_C_Data.csv 存在。
- 直接运行 main.py。

## 核心流程说明（main.py）
1. 数据加载
  - 读取 data/2026_MCM_Problem_C_Data.csv。
  - 若文件不存在直接退出。
2. 数据清理
  - 评分列转数值、计算每周总分、推断退出周。
  - 标记退赛选手与淘汰类型。
3. 异常检测与范围校验
  - 年龄、赛季、名次范围检查。
  - 统计异常值、分数超限、退出周不一致、名次重复等。
4. 构建周粒度长表
  - 宽表转长表（选手-周）。
  - 生成 `judge_total_score`、`judge_count`、`judge_std`。
  - 生成 `is_eliminated`（本周是否淘汰）。
  - 统计每周实际淘汰人数 `n_eliminated`。
5. 特征工程
  - 评委特征：均分、分歧、滚动波动。
  - 环境特征：周均分、标准差、参赛人数、边界差等。
  - 个人特征：年龄、地域、搭档历史、排名轨迹。
6. 模型训练
  - 使用 `share_train_mask` 过滤有效周数据。
  - 训练 `VoteShareModel` 预测淘汰概率 `elimination_prob`。
7. 观众投票估算
  - `survival_prob = 1 - elimination_prob`。
  - 赛季-周内归一化得到 `audience_share`。
  - 基于 `audience_share` 生成 `audience_rank`。
8. 组合规则指标
  - `judge_share`：评委得分百分比。
  - `combined_share = judge_share + audience_share`。
  - `judge_rank`：评委排名。
  - `combined_rank = judge_rank + audience_rank`。
9. 两种规则的淘汰预测
  - 百分比法：每周取 `combined_share` 最低的 N 人。
  - 排名法：每周取 `combined_rank` 最大的 N 人。
  - 输出 `pred_eliminated_share` 与 `pred_eliminated_rank`。
10. 规则判定（Season 28+）
   - 基于 `Season28PlusRules` 评估 rank/share/plus 三种规则一致性。
11. 模型评估
   - 淘汰预测准确率（整体、按周、按淘汰人数分组）。
   - 一致性指标（Hit@k、边界间隔）。
   - 不确定性指标（区间宽度、熵）。
   - 公平性指标（裁判/观众逆转比例）。

## 投票结合方法
- 百分比法：评委得分百分比 + 观众投票百分比。
- 排名法：评委排名 + 观众排名。

当前实现会对两种方法分别生成淘汰预测：
- pred_eliminated_share（百分比法）
- pred_eliminated_rank（排名法）

## 主要评估指标
- 一致性：Hit@k、淘汰边界间隔
- 不确定性：置信区间宽度、熵
- 公平性：评委与观众逆转比例

## 详细函数文档

### main.py
- `main()`：主程序入口，包含完整数据流（加载→清洗→特征→训练→预测→评估）。
  - 关键输出列：
    - `elimination_prob`：淘汰概率预测
    - `audience_share`：观众投票比例
    - `audience_rank`：观众投票排名
    - `combined_share`：百分比结合法综合分
    - `combined_rank`：排名法综合排名
    - `pred_eliminated_share`：百分比法预测淘汰
    - `pred_eliminated_rank`：排名法预测淘汰

### preprocess/load_and_clean.py
- `load_data(filepath)`：读取 CSV 并返回 DataFrame，失败返回 None。
- `list_week_score_columns(columns)`：筛选所有 weekX_judgeY_score 列名。
- `get_week_numbers(columns)`：解析所有周次编号。
- `coerce_score_columns(data)`：评分列转换为数值，N/A→NaN。
- `compute_week_totals(data)`：按周汇总评委总分（weekX_total_score）。
- `parse_results_exit_week(results)`：从 results 文本解析退出周。
- `infer_exit_week_from_scores(row, weeks)`：根据分数序列推断退出周。
- `derive_exit_week(data)`：生成 exit_week，优先 results，冲突时以分数序列为准。
- `clean_data(data)`：去重、转换评分列、计算总分、推断退出周。
- `reshape_to_long_weeks(data)`：宽表转为选手-周粒度长表，计算 `judge_count` 与 `judge_std`。

### preprocess/handle_special_cases.py
- `handle_missing_values(data, strategy)`：缺失值处理（drop/fill）。
- `normalize_categorical(data, categorical_columns)`：分类字段标准化（小写、去空格）。
- `handle_duplicates(data, subset)`：去重。
- `mark_withdrawal_and_elimination(data)`：标记退赛与淘汰（`is_withdrawn`、`exit_type`）。

### preprocess/detect_errors.py
- `detect_outliers(data, columns)`：IQR 异常值检测。
- `validate_ranges(data, ranges_dict)`：验证字段范围。
- `detect_score_overflow(data, max_score)`：检测单项评委打分超限。
- `detect_exit_week_mismatch(data)`：results 与分数推断退出周不一致。
- `detect_placement_inconsistency(data)`：检测同赛季 placement 重复。

### features/judge_features.py
- `build_judge_features(long_df, window=3)`：评委特征（均分、分歧、波动性）。
  - `judge_mean`：该选手当周评委均分，计算为 $\text{judge_total_score} / \text{judge_count}$，`judge_count=0` 时置 0。
  - `judge_std`：该选手当周不同评委评分的标准差（在 `reshape_to_long_weeks()` 中计算）。
  - `judge_mean_roll_std`：该选手在同一赛季内，前 $k$ 周 `judge_mean` 的滚动标准差（波动性）。
  - `judge_std_roll_std`：该选手在同一赛季内，前 $k$ 周 `judge_std` 的滚动标准差。

### features/environment_features.py
- `build_environment_features(long_df, window=3)`：赛季-周环境特征。
  - `env_mean`：该周所有选手 `judge_total_score` 的均值。
  - `env_std`：该周所有选手 `judge_total_score` 的标准差。
  - `env_count`：该周参赛人数。
  - `env_week`：赛季内周次序号（按周次排名）。
  - `env_intensity`：竞争激烈程度，$\max(\text{score})-\min(\text{score})$。
  - `env_boundary_margin`：淘汰边缘差距，为该周最低分与次低分的差值。
  - `env_ties`：该周评分并列次数（重复分数的数量）。
  - `env_special`：是否为赛季最后一周（最后一周记为 1）。
  - 动态特征（前 $k$ 周滚动）：
    - `env_mean_roll_std`：`env_mean` 的滚动标准差。
    - `env_std_roll_std`：`env_std` 的滚动标准差。
  - 与上周差分：
    - `env_count_delta`、`env_intensity_delta`、`env_ties_delta`、`env_boundary_margin_delta`。

### features/personal_features.py
- `build_personal_features(long_df)`：个人静态/动态特征。
  - 静态特征：
    - `age`：`celebrity_age_during_season` 转数值。
    - `industry`：`celebrity_industry` 的文本类别。
    - `homestate_count`：该赛季同州参赛人数（按 `season, homestate` 计数）。
    - `homecountry_count`：该赛季同国家/地区参赛人数（按 `season, homecountry` 计数）。
    - `partner_history_count`：该舞伴历史参与赛季数（按 `partner` 聚合）。
    - `partner_history_best`：该舞伴历史最佳名次（`placement` 最小值）。
  - 动态特征：
    - `weeks_in_competition`：该选手在本赛季的累计参赛周数。
    - `judge_rank`：该周评委总分排名（分高排名靠前）。
    - `rank_max_change`：当前排名与首周排名之差（轨迹变化）。
    - `rank_mean`：从首周到当前周的排名均值。
    - `rank_std`：从首周到当前周的排名标准差。
    - `rank_spike_count`：排名异常提升次数（单周提升幅度超过 $K/2$，$K$ 为当周人数）。
    - `bottom_k_count`：处于“危险区”的累计次数（排名落在后 $K/4$）。

### features/interaction_features.py
- `create_interaction_features(data, feature1, feature2)`：交互项，计算 $\text{feature1} \times \text{feature2}$。
- `create_polynomial_features(data, column, degree)`：多项式特征，如 $x^2, x^3$。
- `create_ratio_features(data, numerator, denominator)`：比率特征，计算 $\frac{\text{numerator}}{\text{denominator}+10^{-10}}$。
- `create_interaction_pairs(data, pairs)`：批量生成交互项。

### models/vote_share_model.py
- `VoteShareModel.fit(X, y)`：岭回归拟合（带标准化）。
- 计算过程：
  - 标准化：$X'=(X-\mu)/\sigma$，$\sigma=0$ 的特征置 1。
  - 加偏置项：$[1, X']$。
  - 岭回归解：$\hat{\beta}=(X^T X + \lambda I)^{-1} X^T y$，$\lambda=10^{-6}$。
- `VoteShareModel.predict(X)`：输出连续预测值。
- `VoteShareModel.predict_weekly_share(X, groups)`：按赛季-周归一化为投票比例。
- 归一化：$\text{softmax}(z_i)=\frac{e^{z_i}}{\sum_j e^{z_j}}$。
- `VoteShareModel.get_feature_importance()`：基于系数的特征重要性。
- `_softmax(values)`：内部归一化函数。

### models/vote_rank_model.py
- `VoteRankModel.fit(X, y)`：岭回归拟合（带标准化）。
- 与 `VoteShareModel.fit()` 相同的标准化与闭式解。
- `VoteRankModel.predict(X)`：输出连续预测值。
- `VoteRankModel.predict_rank(X, groups)`：按赛季-周输出排名（1 为最高）。
- `VoteRankModel.get_feature_importance()`：基于系数的特征重要性。

### models/season28_plus.py
- `Season28PlusRules.apply_plus_rule(df, score_col)`：先取倒数两位，再由评委决定淘汰。
- `Season28PlusRules.infer_method_by_elimination(long_df, audience_share, audience_rank)`：
  - 生成 `combined_rank`、`combined_share`，并预测 `rank/share/plus` 三种规则下淘汰者。
- `Season28PlusRules.validate_rules(results)`：比较预测与实际淘汰一致性。
- `Season28PlusRules.get_rule_description()`：规则描述文本。

### evaluation/consistency.py
- `calculate_consistency_score(predictions, actual)`：Pearson/Spearman/Kendall 一致性。
- `evaluate_temporal_consistency(data, time_column, target_column)`：时间序列自相关。
- `hit_at_k(predicted_scores, actual_eliminated_name, k)`：淘汰者是否在倒数 k 名。
- 计算方式：在 `predicted_scores` 中取最小的 $k$ 个索引，判断是否包含实际淘汰者。
- `elimination_boundary_margin(predicted_scores)`：淘汰边界间隔。
- 计算方式：倒数第 2 与倒数第 1 的差值。
- `calculate_hit_at_k(long_df, predicted_col, k)`：按周计算 Hit@k。
- `analyze_boundary_margin_distribution(long_df, predicted_col)`：淘汰边界间隔分布。

### evaluation/uncertainty.py
- `calculate_prediction_intervals(predictions, errors, confidence)`：预测区间。
- 计算方式：$\hat{y} \pm z_{\alpha/2}\,\sigma$。
- `calculate_uncertainty_metrics(residuals)`：MSE/MAE/方差等。
- `quantify_parameter_uncertainty(model, X_test)`：参数不确定性占位函数。
- `entropy_uncertainty(vote_share)`：基于熵的不确定性。
- 计算方式：$-\sum_i p_i \log p_i$。
- `weekly_uncertainty(long_df, vote_share_col)`：按周熵。
- `analyze_vote_share_intervals(long_df, vote_share_col, confidence)`：选手/周区间与方差。
- 选手级：按选手汇总 `mean/std/count` 并给出区间宽度。
- 周级：按周汇总 `mean/std/count` 并给出区间宽度。
- `_analyze_uncertainty_reasons(long_df, high_uncertainty_weeks)`：高不确定性原因解析。
- `_determine_reason(n_contestants, score_std, n_ties)`：原因判定。

### evaluation/fairness.py
- `calculate_fairness_score(predictions, sensitive_attribute, groups)`：分组公平性。
- `calculate_disparate_impact(outcomes, protected_group_mask, control_group_mask)`：差异影响比。
- `evaluate_bias(predictions, actual, sensitive_attribute)`：偏差分析。
- `reversal_ratios(long_df, judge_rank_col, audience_rank_col)`：评委与观众逆转比例。

### debug_hit_at_k.py
- 独立调试脚本：复现 Hit@k 计算过程、检查索引与命中逻辑。

## 备注
- 当前训练使用 `share_train_mask` 过滤有效周（`judge_total_score > 0`）。
- 百分比法与排名法并行生成淘汰预测，便于对比。
- 如需调整规则或训练范围，请修改 main.py 中的训练掩码与淘汰预测逻辑。
