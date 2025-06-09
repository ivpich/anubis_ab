"""Collection of statistical tests for analysing experiment results.

The wrappers expose consistent return values and hide some of the parameter
choices required when using the underlying scipy/statsmodels functions. They are
not meant to replace full-fledged libraries but to capture common patterns used
in A/B testing.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import scipy.stats as scs
import statsmodels.stats.api as sms
import statsmodels.stats.proportion as ssp
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats
import bootstrapped.compare_functions as bs_compare


def ab_test_bootstrap(a: pd.Series, b: pd.Series, stat: str = "mean", compare_func: str = "difference", n_bootstraps: int = 1000, alpha: float = 0.05) -> tuple:
    """Bootstrap comparison of two samples.

    Useful when the metric distribution is heavy tailed or the standard
    parametric assumptions do not hold. Returns the effect estimate, confidence
    interval and a boolean flag of significance.
    """
    if stat not in ("median", "std", "sum", "mean"):
        raise ValueError("stat must be one of 'median', 'std', 'sum', 'mean'")
    if compare_func not in ("percent_change", "difference"):
        raise ValueError("compare_func must be 'percent_change' or 'difference'")

    stat_func = {
        "median": bs_stats.median,
        "std": bs_stats.std,
        "sum": bs_stats.sum,
        "mean": bs_stats.mean,
    }[stat]

    cmp = bs_compare.percent_change if compare_func == "percent_change" else bs_compare.difference

    results = bs.bootstrap_ab(a.values, b.values, stat_func, cmp, alpha=alpha, iteration_batch_size=10, num_iterations=n_bootstraps)
    is_sign = np.sign(results.lower_bound) == np.sign(results.upper_bound)
    effect = results.value
    ci = (results.lower_bound, results.upper_bound)
    return effect, ci, is_sign


def ab_test_nonparametric(a: pd.Series, b: pd.Series, alternative: str = "two-sided", alpha: float = 0.05) -> tuple:
    """Mann–Whitney U test for independent samples."""
    st, pval = scs.mannwhitneyu(a, b, alternative=alternative)
    diff = a.median() - b.median()
    is_sign = pval <= alpha
    return st, pval, diff, is_sign


def ab_test_parametric_continuous(a: pd.Series, b: pd.Series, alternative: str = "two-sided", alpha: float = 0.05) -> tuple:
    """Two sample t-test with automatic variance check."""
    stat_l, pval_l = scs.levene(a, b)
    usevar = "pooled" if pval_l > alpha else "unequal"
    st, pval, _ = sms.ttest_ind(a, b, alternative=alternative, usevar=usevar)
    diff = a.mean() - b.mean()
    ci = sms.CompareMeans.from_data(a, b).tconfint_diff(alternative=alternative, usevar=usevar)
    is_sign = pval <= alpha
    return st, pval, diff, ci, is_sign


def ab_test_parametric_nominal(a: pd.Series, b: pd.Series, alternative: str = "two-sided", alpha: float = 0.05) -> tuple:
    """Two sample z-test for proportions."""
    size_a, size_b = a.count(), b.count()
    success_a, success_b = a.sum(), b.sum()
    z_stat, pval = ssp.proportions_ztest([success_a, success_b], nobs=[size_a, size_b], alternative=alternative)
    ci = ssp.confint_proportions_2indep(success_a, size_a, success_b, size_b, alpha=alpha)
    prop_diff = success_a / size_a - success_b / size_b
    is_sign = pval <= alpha
    return z_stat, pval, prop_diff, ci, is_sign
