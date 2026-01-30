"""投票排名模型（所有季）"""

import pandas as pd
import numpy as np


class VoteRankModel:
    """投票排名预测模型"""
    
    def __init__(self):
        self.coef_ = None
        self.bias_ = None
        self.mean_ = None
        self.std_ = None
        self.features = None
    
    def fit(self, X, y):
        """训练模型
        
        Args:
            X: 特征矩阵
            y: 排名目标值
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        X_scaled = (X - self.mean_) / self.std_
        X_bias = np.c_[np.ones(X_scaled.shape[0]), X_scaled]
        ridge = 1e-6 * np.eye(X_bias.shape[1])
        params = np.linalg.inv(X_bias.T @ X_bias + ridge) @ X_bias.T @ y
        self.bias_ = params[0]
        self.coef_ = params[1:]
        self.features = X.columns if hasattr(X, 'columns') else None
        return self
    
    def predict(self, X):
        """预测排名"""
        if self.coef_ is None:
            raise ValueError("模型未训练")
        X = np.asarray(X, dtype=float)
        X_scaled = (X - self.mean_) / self.std_
        return X_scaled @ self.coef_ + self.bias_

    def predict_rank(self, X, groups):
        """按赛季-周输出投票排名（1 为最高）"""
        raw = self.predict(X)
        df = pd.DataFrame({"raw": raw, "season": groups[:, 0], "week": groups[:, 1]})
        df["vote_rank"] = df.groupby(["season", "week"])["raw"].rank(ascending=False, method="min")
        return df["vote_rank"].values
    
    def get_feature_importance(self):
        """获取特征重要性"""
        if self.coef_ is None:
            raise ValueError("模型未训练")
        
        importance = pd.DataFrame({
            'feature': self.features,
            'importance': np.abs(self.coef_)
        }).sort_values('importance', ascending=False)
        
        return importance
