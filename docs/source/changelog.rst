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

1.0.0
-----

* Added incremental batch training for XGBoost, CatBoost, and LightGBM
  classifiers using Spark DataFrame inputs.
* Added validation monitoring, global batch-level stopping, optional sample
  weighting, and learning-rate scheduling where supported.
