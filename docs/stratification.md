# Stratification

Tools for controlled sampling. Stratified samples help reduce variance when user behaviour differs across segments.

* `get_quantiles` — creates quantile buckets used as strata.
* `stratified_sample_rows` — allocate rows into a fixed number of cohorts preserving the share of each stratum.
