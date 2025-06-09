"""Helpers for validating experiment methodology via simulation.

Running many random splits gives intuition about expected p-value distributions
and empirical error rates.
"""
import numpy as np
from scipy import stats
from .power import estimate_sample_size


def run_synthetic_experiments(values, sample_size, effect=0, n_iter=10000):
    """Run repeated random splits and collect p-values."""
    pvalues = []
    for _ in range(n_iter):
        a, b = np.random.choice(values, size=(2, sample_size,), replace=False)
        b = b + effect
        pval = stats.ttest_ind(a, b).pvalue
        pvalues.append(pval)
    return np.array(pvalues)


def estimate_ci_bernoulli(p: float, n: int, alpha: float = 0.05):
    """Confidence interval for a Bernoulli proportion."""
    t = stats.norm.ppf(1 - alpha / 2, loc=0, scale=1)
    std_n = np.sqrt(p * (1 - p) / n)
    return p - t * std_n, p + t * std_n


def print_estimated_errors(pvalues_aa, pvalues_ab, alpha: float):
    """Compute and display empirical type I and type II error rates."""
    first = np.mean(pvalues_aa < alpha)
    second = np.mean(pvalues_ab >= alpha)
    ci_first = estimate_ci_bernoulli(first, len(pvalues_aa))
    ci_second = estimate_ci_bernoulli(second, len(pvalues_ab))
    print(f"error I = {first:0.4f}  CI=({ci_first[0]:0.4f}, {ci_first[1]:0.4f})")
    print(f"error II = {second:0.4f} CI=({ci_second[0]:0.4f}, {ci_second[1]:0.4f})")
