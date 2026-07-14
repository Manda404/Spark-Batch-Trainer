Examples
========

This section links the runnable recipes for each supported classification
task. It does not introduce new API surface — see :doc:`quickstart` and
:doc:`../configuration` for that.

Each recipe shows the three configuration dictionaries in context:

* ``model_config`` — backend-specific hyperparameters (XGBoost, CatBoost, or
  LightGBM).
* ``training_config`` — shared options such as ``num_batches``,
  ``max_patience``, and ``metric_mode``; see :doc:`../configuration` for the
  full list.
* ``learning_rate_config`` — optional exponential-decay scheduling, supported
  by XGBoost and LightGBM only.

Available scenarios
--------------------

* :doc:`examples_binary` — backend recipes for binary classification.
* :doc:`examples_multiclass` — backend recipes for multiclass classification.
* :doc:`dataset_overview` — the splits and input contract the recipes assume.

Supported scope
----------------

The public trainers currently target classification. Regression and ranking
are not documented as supported workflows.
