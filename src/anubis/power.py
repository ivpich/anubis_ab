"""Power analysis utilities used to plan experiments."""
from typing import Iterable
import numpy as np
import pandas as pd
from scipy.stats import norm
import scipy.stats as scs
import statsmodels.stats.api as sms


def get_mde_mean(data: pd.Series, test_size: float = 0.5, alpha: float = 0.05, power: float = 0.8) -> float:
    """Estimate minimum detectable effect for a mean metric."""
    n_total = data.count()
    std = data.std()
    z_alpha = scs.norm.ppf(1 - alpha / 2)
    z_beta = scs.norm.ppf(power)
    q0, q1 = test_size, 1 - test_size
    c = (z_alpha + z_beta) ** 2 * (1 / q0 + 1 / q1)
    es = np.sqrt(c / n_total)
    return std * es


def get_mde_proportion(data: pd.Series, alpha: float = 0.05, power: float = 0.8) -> float:
    """Minimum detectable effect for a proportion assuming 50/50 split."""
    n = data.count()
    sd = data.std()
    s = np.sqrt((sd ** 2 / n) + (sd ** 2 / n))
    m = norm.ppf(q=1 - alpha / 2) + norm.ppf(q=power)
    return m * s


def estimate_sample_size(effect_size: float, data: pd.Series, alpha: float = 0.05, power: float = 0.8) -> float:
    """Sample size for detecting ``effect_size`` in a mean metric."""
    sd = data.std()
    m = (norm.ppf(q=1 - alpha / 2) + norm.ppf(q=power)) ** 2
    return 2 * m * sd ** 2 / effect_size ** 2


def min_sample_size_nominal(bcr: float, mde: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Sample size for two-proportion z-test."""
    standard_norm = scs.norm(0, 1)
    z_beta = standard_norm.ppf(power)
    z_alpha = standard_norm.ppf(1 - alpha / 2)
    pooled_prob = (bcr + bcr + mde) / 2
    n = 2 * pooled_prob * (1 - pooled_prob) * (z_beta + z_alpha) ** 2 / mde ** 2
    return int(round(n))


def min_sample_size_nominal_in_r(bcr: float, mde: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Replicates ``pwr.2p.test`` from R for convenience."""
    prop = bcr + mde
    effect_size = sms.proportion_effectsize(bcr, prop)
    sample_size = sms.NormalIndPower().solve_power(effect_size, power=power, alpha=alpha, ratio=1)
    return int(round(sample_size))


def sample_power_difftest(d: float, s: float, power: float = 0.8, sig: float = 0.05) -> int:
    """Size calculation based on difference in means."""
    z = norm.isf([sig / 2])
    zp = -1 * norm.isf([power])
    n = s * ((zp + z) ** 2) / (d ** 2)
    return int(round(n[0]))


def min_sample_size_continuous(X: Iterable[float], mde: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """Sample size estimate for t-test of a continuous metric."""
    def get_c(alpha: float, power: float) -> float:
        data = [{"0.05": 7.85, "0.01": 11.68}, {"0.05": 10.51, "0.01": 14.88}]
        df = pd.DataFrame(data, index=["0.8", "0.9"])
        return df.loc[str(power), str(alpha)]

    sd = np.std(X, ddof=1)
    c = get_c(alpha, power)
    return 1 + (2 * c) * ((sd / mde) ** 2)


def min_sample_size_continuous_nonparametric(X: Iterable[float], mde: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """Sample size for a non-parametric alternative to the t-test."""
    sd = np.std(X, ddof=1)
    n = 1.15 * 2 * (sd ** 2) * ((scs.norm.ppf(1 - alpha / 2) + scs.norm.ppf(power)) ** 2) / (mde ** 2)
    return n
