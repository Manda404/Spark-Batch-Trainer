Configuration Reference
=======================

The recommended API accepts three dictionaries.

``model_config``
----------------

Passed to the selected model SDK. Use the parameter names supported by your
installed XGBoost, CatBoost, or LightGBM version.

``training_config``
-------------------

.. list-table:: Shared training options
   :header-rows: 1

   * - Name
     - Default
     - Meaning
   * - ``num_batches``
     - ``10``
     - Number of target-stratified Spark fragments.
   * - ``max_patience``
     - ``5``
     - Consecutive non-improving batches before stopping.
   * - ``use_sample_weight``
     - ``False``
     - Apply balanced sample weights.
   * - ``show_learning_curve``
     - ``False``
     - Display blocking Matplotlib figures after training.
   * - ``metric_mode``
     - ``"auto"``
     - ``"auto"``, ``"min"``, or ``"max"``.
   * - ``min_delta``
     - ``0.0``
     - Minimum score change considered an improvement.

``metric_mode="auto"`` recognizes common score metrics (``auc``, ``accuracy``,
``precision``, ``recall``, ``ndcg@k``, ``map@k``, …) as maximized and defaults
every other metric name to minimized. A custom metric that should be
maximized but is not recognized logs a warning and is still treated as
minimized — pass ``metric_mode="max"`` explicitly for those metrics instead
of relying on ``"auto"``.

``learning_rate_config``
------------------------

Available for XGBoost and LightGBM.

.. code-block:: python

   learning_rate_config = {
       "initial_lr": 0.05,
       "decay_rate": 0.9,
       "min_lr": 1e-4,
   }

Use only ``model_config``, ``training_config``, and
``learning_rate_config``. The legacy keyword names are no longer supported.
