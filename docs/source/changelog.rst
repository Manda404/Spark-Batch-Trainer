Changelog
=========

Unreleased
----------

* Updated the supported runtime, formatting target, type checking, continuous
  integration, and documentation to Python 3.14.
* Added a backend factory and a stable, English configuration vocabulary.
* Separated backend, configuration, data preparation, evaluation,
  visualization, and observability responsibilities.
* Added typed training history and validated shared training options.
* Reorganized the Sphinx documentation around concepts, architecture, user
  workflows, and a focused API reference.
* Switched public docstrings and the Sphinx Napoleon configuration from
  NumPy style to Google style.
* Fixed an ``UnboundLocalError`` that could mask the real exception when a
  backend failed before its first batch was collected.
* Sample weights are computed and normalized independently for each collected
  training batch and once for the complete validation dataset.
* ``metric_mode="auto"`` now logs a warning instead of silently defaulting
  to ``"min"`` when it cannot recognize a custom metric name.
* Unified categorical-feature detection across backends; CatBoost and
  LightGBM no longer redetect features independently from the validation set.
* Added strict configuration, schema, label, batch-size, and finite-metric
  validation.
* Added ``prepare_features()`` so bounded inference reuses the fitted feature
  order and categorical schema.
* Moved the duplicated collection and evaluation loop into ``BatchTrainer``.

1.0.0
-----

* Added incremental batch training for XGBoost, CatBoost, and LightGBM
  classifiers using Spark DataFrame inputs.
* Added validation monitoring, global batch-level stopping, optional sample
  weighting, and learning-rate scheduling where supported.
