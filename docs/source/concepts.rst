Core Concepts
=============

Spark and driver responsibilities
---------------------------------

Spark performs the window, stratification, persistence, and filtering. The
driver owns pandas conversion, sample weights, model training, evaluation, and
model snapshots.

.. warning::

   ``num_batches`` controls the number of fragments, not their byte size. A
   wide row, skewed data, a large validation set, or model copies can still
   exhaust driver memory.

Stratified batches
------------------

Rows are partitioned by target and assigned with ``ntile`` after seeded random
ordering. Class counts are approximately balanced across batches; exact ratios
are not guaranteed when class counts are small.

Global early stopping
---------------------

The library compares one validation score after each completed batch. Known
score metrics such as AUC are maximized; loss metrics are minimized. Use
``metric_mode="min"`` or ``metric_mode="max"`` for custom metrics and
``min_delta`` to ignore insignificant changes.

Learning-rate scheduling
------------------------

XGBoost and LightGBM support exponential decay between batches. Without a
scheduler, the backend preserves the learning rate from ``model_config``.
CatBoost currently uses the learning rate supplied directly to its model.

Training history
----------------

``get_training_history()`` returns immutable tuples containing train curves,
validation curves, processed batch numbers, and learning rates.
