"""交互特征工程模块"""

import pandas as pd
import numpy as np


def create_interaction_features(data, feature1, feature2):
    """创建特征交互项"""
    if feature1 in data.columns and feature2 in data.columns:
        interaction_name = f'{feature1}_x_{feature2}'
        data[interaction_name] = data[feature1] * data[feature2]
        return data, interaction_name
    return data, None


def create_polynomial_features(data, column, degree=2):
    """创建多项式特征"""
    if column in data.columns:
        for d in range(2, degree + 1):
            poly_name = f'{column}_pow{d}'
            data[poly_name] = data[column] ** d
    return data


def create_ratio_features(data, numerator, denominator):
    """创建比率特征"""
    if numerator in data.columns and denominator in data.columns:
        ratio_name = f'{numerator}_ratio_{denominator}'
        data[ratio_name] = data[numerator] / (data[denominator] + 1e-10)  # 避免除零
        return data, ratio_name
    return data, None


def create_interaction_pairs(data, pairs):
    """批量创建交互特征

    Args:
        data: DataFrame
        pairs: [(feature1, feature2), ...]
    """
    created = []
    for feature1, feature2 in pairs:
        data, name = create_interaction_features(data, feature1, feature2)
        if name:
            created.append(name)
    return data, created
