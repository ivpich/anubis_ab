# Preprocessing

Utilities that help clean raw experiment data before running tests.

* `remove_outlier` — trims observations outside selected percentiles. Useful when extreme values can distort the mean.
* `remove_outlier_interquartil` — applies the 1.5×IQR rule to drop outliers.
* `box_cox_transform` — Box–Cox transformation to approximate normality; return the transformed series and fitted lambda.

These steps reduce noise and can make statistical tests more reliable.
