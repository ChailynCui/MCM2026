"""第 28 季及以后规则判定模块"""

from __future__ import annotations

import pandas as pd


class Season28PlusRules:
    """第 28 季及以后的规则判定"""

    def __init__(self):
        self.rules = {}

    def apply_plus_rule(self, df: pd.DataFrame, score_col: str) -> pd.Series:
        """第28季规则：先取综合排名最后两位，再由评委决定淘汰"""
        bottom2 = df.nsmallest(2, score_col)
        eliminated = bottom2.sort_values("judge_rank", ascending=False).iloc[0]["celebrity_name"]
        return eliminated

    def infer_method_by_elimination(self, long_df: pd.DataFrame, audience_share: pd.Series, audience_rank: pd.Series) -> pd.DataFrame:
        """对赛季方法进行判定：排名法 / 百分比法 / plus 规则

        Args:
            long_df: 选手-周粒度数据，需含 judge_total_score、judge_rank、exit_week
            audience_share: 观众投票比例预测
            audience_rank: 观众投票排名预测
        """
        df = long_df.copy()
        df["audience_share"] = audience_share.values
        df["audience_rank"] = audience_rank.values
        df["judge_share"] = df.groupby(["season", "week"])["judge_total_score"].transform(lambda s: s / s.sum())

        df["combined_rank"] = df["judge_rank"] + df["audience_rank"]
        df["combined_share"] = df["judge_share"] + df["audience_share"]

        def _elimination_week(group: pd.DataFrame) -> pd.Series:
            season = group["season"].iloc[0]
            week = group["week"].iloc[0]
            eliminated_actual = group[group["exit_week"] == week]["celebrity_name"].tolist()
            if len(eliminated_actual) == 0:
                return pd.Series({"season": season, "week": week, "actual": None, "rank": None, "share": None, "plus": None})

            actual = eliminated_actual[0]
            rank_pred = group.loc[group["combined_rank"].idxmin(), "celebrity_name"]
            share_pred = group.loc[group["combined_share"].idxmin(), "celebrity_name"]
            plus_pred = self.apply_plus_rule(group, "combined_share")
            return pd.Series({"season": season, "week": week, "actual": actual, "rank": rank_pred, "share": share_pred, "plus": plus_pred})

        results = df.groupby(["season", "week"]).apply(_elimination_week).reset_index(drop=True)
        return results

    def validate_rules(self, results: pd.DataFrame) -> pd.DataFrame:
        """验证规则的合法性"""
        if results.empty:
            return pd.DataFrame()
        summary = {}
        for method in ["rank", "share", "plus"]:
            valid = results["actual"].notna()
            summary[method] = (results.loc[valid, method] == results.loc[valid, "actual"]).mean()
        return pd.DataFrame([summary])

    def get_rule_description(self) -> str:
        """获取规则描述"""
        return "第 28 季及以后的规则判定规则"
