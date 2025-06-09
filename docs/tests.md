# Statistical tests

Provides wrappers around common hypothesis tests.

* `ab_test_bootstrap` — non-parametric bootstrap comparison. Works well with heavy-tailed metrics.
* `ab_test_nonparametric` — Mann–Whitney U test for ordinal or non-normal data.
* `ab_test_parametric_continuous` — two-sample t-test with automatic variance check.
* `ab_test_parametric_nominal` — z-test for binary metrics (proportions).

Each function returns the statistic, p-value and effect estimate so you can interpret experiment results.
