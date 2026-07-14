Changelog
=========

Unreleased
----------

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
* Sample weights are now computed against the global target label space
  learned once from train and validation data, instead of per batch — this
  makes weight scale consistent across batches and enables normalization.
* ``metric_mode="auto"`` now logs a warning instead of silently defaulting
  to ``"min"`` when it cannot recognize a custom metric name.
* Unified categorical-feature detection across backends; CatBoost and
  LightGBM no longer redetect features independently from the validation set.

1.0.0
-----

* Added incremental batch training for XGBoost, CatBoost, and LightGBM
  classifiers using Spark DataFrame inputs.
* Added validation monitoring, global batch-level stopping, optional sample
  weighting, and learning-rate scheduling where supported.
