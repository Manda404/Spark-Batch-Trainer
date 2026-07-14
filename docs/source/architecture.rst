Architecture
============

Package map
-----------

.. code-block:: text

   spark_batch_trainer/
   ├── __init__.py         small public API with lazy backend imports
   ├── factory.py          create_trainer() backend selection
   ├── logging.py          console and optional file logging
   ├── backends/           XGBoost, CatBoost, and LightGBM adapters
   ├── data/               Spark batching and pandas preparation
   └── training/           configuration, lifecycle, metrics, and history

Dependency direction
--------------------

.. code-block:: text

   package facade -> factory -> selected backend -> model SDK
                                  |-> data
                                  |-> training

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

.. note::

   The three existing backends currently duplicate a handful of small,
   backend-agnostic methods (input validation, runtime-state initialization,
   batch preparation) almost verbatim rather than sharing them through
   :class:`~spark_batch_trainer.training.base.BatchTrainer`. This is known
   debt, not the intended pattern — a future refactor should move that logic
   into the shared base class rather than a fourth backend repeating it again.
