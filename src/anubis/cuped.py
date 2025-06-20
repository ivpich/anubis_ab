"""Implementation of the CUPED variance reduction technique."""
import numpy as np
import pandas as pd


def get_cuped_adjusted(A_before: list[float], B_before: list[float], A_after: list[float], B_after: list[float]):
    """Return CUPED adjusted arrays for A/B samples."""
    cv = np.cov([A_after + B_after, A_before + B_before])
    theta = cv[0, 1] / cv[1, 1]
    mean_before = np.mean(A_before + B_before)
    A_adj = [after - (before - mean_before) * theta for after, before in zip(A_after, A_before)]
    B_adj = [after - (before - mean_before) * theta for after, before in zip(B_after, B_before)]
    return pd.Series(A_adj), pd.Series(B_adj)
