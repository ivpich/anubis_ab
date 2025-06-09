"""Data preprocessing utilities.

Cleaning extreme values or transforming metrics can stabilise variance and help
statistical tests meet their assumptions. Typical use cases are preparing
revenue or engagement metrics before an A/B experiment.
"""
from typing import Tuple
import numpy as np
import pandas as pd
import scipy.stats as scs
from scipy.special import inv_boxcox


def remove_outlier(df: pd.DataFrame, column: str, low: float = 0.05, high: float = 0.95) -> pd.Index:
    """Return index of observations within the ``[low, high]`` percentile range.

    Cutting off both tails removes extreme values that could skew the mean. The
    function returns the index so you can subset the original DataFrame.
    """
    quant_df = df.quantile([low, high])
    mask = (df[column] > quant_df.loc[low, column]) & (df[column] < quant_df.loc[high, column])
    return df[column].dropna()[mask].index


def remove_outlier_interquartil(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Drop rows outside ``[Q1-1.5*IQR, Q3+1.5*IQR]``.

    This classic rule is robust to skewed distributions and works well for most
    business metrics.
    """
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    fence_low = q1 - 1.5 * iqr
    fence_high = q3 + 1.5 * iqr
    return df[(df[column] > fence_low) & (df[column] < fence_high)]


def box_cox_transform(x: pd.Series, lmbda: float | None = None) -> Tuple[pd.Series, float]:
    """Apply a Box–Cox transformation.

    The transform ``y = (x^lambda - 1) / lambda`` stabilises variance and can
    make the distribution more normal. The fitted ``lambda`` is returned so the
    inverse transform can later be applied.
    """
    if lmbda is None:
        transformed, lmbda = scs.boxcox(x)
    else:
        transformed = inv_boxcox(x, lmbda)
    return pd.Series(transformed, index=x.index), lmbda
