Operators API
=============

Complete parameter documentation for all operator functions. Usage: ``from phandas import *``

Cross-sectional Operators
-------------------------

.. autofunction:: phandas.rank

.. autofunction:: phandas.mean

.. autofunction:: phandas.median

.. autofunction:: phandas.normalize

.. autofunction:: phandas.zscore

.. autofunction:: phandas.quantile

.. autofunction:: phandas.scale

.. autofunction:: phandas.spread

.. autofunction:: phandas.signal

Time Series Operators
---------------------

Basic Statistics
~~~~~~~~~~~~~~~~

.. autofunction:: phandas.ts_delay

.. autofunction:: phandas.ts_delta

.. autofunction:: phandas.ts_mean

.. autofunction:: phandas.ts_median

.. autofunction:: phandas.ts_sum

.. autofunction:: phandas.ts_product

.. autofunction:: phandas.ts_std_dev

Ranking and Extrema
~~~~~~~~~~~~~~~~~~~

.. autofunction:: phandas.ts_rank

.. autofunction:: phandas.ts_max

.. autofunction:: phandas.ts_min

.. autofunction:: phandas.ts_arg_max

.. autofunction:: phandas.ts_arg_min

Higher-order Statistics
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: phandas.ts_skewness

.. autofunction:: phandas.ts_kurtosis

.. autofunction:: phandas.ts_cv

.. autofunction:: phandas.ts_jumpiness

.. autofunction:: phandas.ts_trend_strength

.. autofunction:: phandas.ts_vr

.. autofunction:: phandas.ts_autocorr

.. autofunction:: phandas.ts_reversal_count

Standardization
~~~~~~~~~~~~~~~

.. autofunction:: phandas.ts_zscore

.. autofunction:: phandas.ts_scale

.. autofunction:: phandas.ts_quantile

.. autofunction:: phandas.ts_av_diff

Decay Weighting
~~~~~~~~~~~~~~~

.. autofunction:: phandas.ts_decay_linear

.. autofunction:: phandas.ts_decay_exp_window

Correlation and Regression
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: phandas.ts_corr

.. autofunction:: phandas.ts_covariance

.. autofunction:: phandas.ts_regression

Other
~~~~~

.. autofunction:: phandas.ts_step

.. autofunction:: phandas.ts_count_nans

.. autofunction:: phandas.ts_backfill

Neutralization Operators
------------------------

.. autofunction:: phandas.vector_neut

.. autofunction:: phandas.regression_neut

Group Operators
---------------

.. autofunction:: phandas.group

.. autofunction:: phandas.group_neutralize

.. autofunction:: phandas.group_mean

.. autofunction:: phandas.group_median

.. autofunction:: phandas.group_rank

.. autofunction:: phandas.group_scale

.. autofunction:: phandas.group_zscore

.. autofunction:: phandas.group_normalize

Math Operators
--------------

Elementary Functions
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: phandas.log

.. autofunction:: phandas.ln

.. autofunction:: phandas.sqrt

.. autofunction:: phandas.s_log_1p

.. autofunction:: phandas.sign

.. autofunction:: phandas.inverse

Power Functions
~~~~~~~~~~~~~~~

.. autofunction:: phandas.power

.. autofunction:: phandas.signed_power

Comparison and Conditional
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: phandas.maximum

.. autofunction:: phandas.minimum

.. autofunction:: phandas.where

Arithmetic Operations
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: phandas.add

.. autofunction:: phandas.subtract

.. autofunction:: phandas.multiply

.. autofunction:: phandas.divide

.. autofunction:: phandas.reverse
