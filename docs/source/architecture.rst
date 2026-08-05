Architecture
============

Package map
-----------

.. code-block:: text

   spark_batch_trainer/
   ├── __init__.py         small public API with lazy backend imports
   ├── factory.py          create_trainer() backend selection
   ├── backends/           small XGBoost, CatBoost, and LightGBM adapters
   ├── data/               internal Spark and pandas preparation functions
   └── training/           shared lifecycle, configuration, and history

Dependency direction
--------------------

.. code-block:: text

   package facade -> factory -> selected backend -> model SDK
                                  |
                                  +-> BatchTrainer -> data functions
                                                  -> run state
                                                  -> metric functions

Backends are loaded lazily. Importing training or data components
does not import XGBoost, CatBoost, or LightGBM. Matplotlib is imported only when
a plot is requested.

Extension guidance
------------------

A new backend should isolate SDK calls for model creation, continuation,
evaluation history, and serialization. Shared Spark, pandas, metric, and early
stopping behavior must not be copied into the backend. Avoid compatibility
folders and one-file packages: add a module only when it owns a clear,
independent responsibility.

Shared workflow
---------------

The :class:`~spark_batch_trainer.training.base.BatchTrainer` base class owns
input validation, pandas preparation, learning-rate scheduling, metric
recording, global early stopping, and final model selection. Small internal
dataclasses make the values passed through that lifecycle explicit.

Each backend therefore owns only the SDK boundary: constructing its native
model, supplying the correct continuation object, and calling ``fit``. This
keeps backend differences visible without copying the orchestration logic.
