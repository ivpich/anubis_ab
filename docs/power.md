# Power analysis

Functions used to plan experiment sizes.

Key formulas:

* `get_mde_mean` calculates minimal detectable effect for a mean: `MDE = std * sqrt((z_alpha+z_beta)^2 * (1/q0 + 1/q1) / N)`.
* `estimate_sample_size` solves the reverse problem — how many observations are required for the chosen effect size.
* `min_sample_size_nominal` and `min_sample_size_nominal_in_r` compute the required sample for a z-test on proportions.

Use these helpers before running an experiment to ensure the test is adequately powered.
