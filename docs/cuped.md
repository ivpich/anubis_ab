# CUPED

CUPED (Controlled Pre-experiment Data) is a variance reduction method. It uses a pre-experiment metric correlated with the target metric to adjust the post-experiment measurements.

* `get_cuped_adjusted` — returns adjusted series for the test and control groups using the classic formula:
  `theta = cov(post, pre) / var(pre)`
  `adjusted = post - theta * (pre - mean_pre)`

Applying CUPED can significantly improve the sensitivity of an experiment when the correlation is high.
