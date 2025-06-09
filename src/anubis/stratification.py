"""Sampling and stratification helpers."""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from sklearn.utils import shuffle


def get_quantiles(data: pd.DataFrame, cols: list[str]):
    """Add quantile columns for given fields."""
    temp = data.copy()
    quantiles = temp.quantile(q=[0.2, 0.4, 0.6, 0.8]).to_dict()
    cols_q = []

    def quantile(x, p, d):
        if x <= d[p][0.2]:
            return 1
        elif x <= d[p][0.4]:
            return 2
        elif x <= d[p][0.6]:
            return 3
        elif x <= d[p][0.8]:
            return 4
        else:
            return 5

    for c in cols:
        cq = c + "_q"
        temp[cq] = temp[c].apply(quantile, args=(c, quantiles))
        cols_q.append(cq)
    return temp, cols_q


def stratified_sample_rows(data: pd.DataFrame, cols: list[str], id_col: str, n_cohorts: int, n_rows: int, rs: int | None = None):
    """Return stratified cohorts with roughly equal representation."""
    experiment = pd.DataFrame(columns=list(data.columns))
    unallocated = pd.DataFrame()
    data = data.set_index(cols)
    l_data = len(data)
    for _, temp in data.groupby(level=cols):
        temp.reset_index(inplace=True)
        l_temp = len(temp)
        m = math.ceil(n_rows * (l_temp / l_data))
        if m * n_cohorts > l_temp:
            continue
        unall = temp.copy()
        temp = temp.sample(n=m * n_cohorts, random_state=rs)
        unall = unall[~unall[id_col].isin(temp[id_col])]
        unallocated = pd.concat([unallocated, unall])
        temp = shuffle(temp, random_state=rs)
        df_split = np.array_split(temp, n_cohorts)
        for i in range(n_cohorts):
            df_split[i]["cohort"] = i + 1
        experiment = pd.concat([experiment] + df_split)
    return experiment, unallocated
