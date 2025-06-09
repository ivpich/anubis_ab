"""Data preprocessing utilities for Anubis.

This module contains helper functions for cleaning and transforming raw
experiment data.
"""
from typing import Tuple
import numpy as np
import pandas as pd
import scipy.stats as scs
from scipy.special import inv_boxcox


def remove_outlier(df: pd.DataFrame, column: str, low: float = 0.05, high: float = 0.95) -> pd.Index:
    """Return index of rows without outliers.

    Parameters
    ----------
    df : DataFrame
        Source dataset.
    column : str
        Column to analyse for outliers.
    low : float, optional
        Lower percentile, by default 0.05.
    high : float, optional
        Upper percentile, by default 0.95.
    """
    quant_df = df.quantile([low, high])
    mask = (df[column] > quant_df.loc[low, column]) & (df[column] < quant_df.loc[high, column])
    return df[column].dropna()[mask].index


def remove_outlier_interquartil(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Remove outliers based on IQR rule."""
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    fence_low = q1 - 1.5 * iqr
    fence_high = q3 + 1.5 * iqr
    return df[(df[column] > fence_low) & (df[column] < fence_high)]


def box_cox_transform(x: pd.Series, lmbda: float | None = None) -> Tuple[pd.Series, float]:
    """Apply Box-Cox transformation.

    Parameters
    ----------
    x : Series
        Data to transform.
    lmbda : float, optional
        Lambda value. If ``None`` the value will be estimated.
    """
    if lmbda is None:
        transformed, lmbda = scs.boxcox(x)
    else:
        transformed = inv_boxcox(x, lmbda)
    return pd.Series(transformed, index=x.index), lmbda
