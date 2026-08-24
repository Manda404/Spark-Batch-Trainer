Installation
============

Requirements
------------

* Python 3.14
* Java supported by the installed PySpark release
* Poetry for development from source

Install from source
-------------------

.. code-block:: bash

   git clone https://github.com/Manda404/SparkBatchTrainer.git
   cd SparkBatchTrainer
   poetry install

The package is not documented here as available from a public package index.
Use the source installation until a release is published and verified.

Verify the public API
---------------------

.. code-block:: bash

   poetry run python -c "from spark_batch_trainer import create_trainer; print(create_trainer('xgboost'))"

Run tests
---------

.. code-block:: bash

   poetry run pytest -q tests/core tests/utils
   poetry run pytest -q tests/fit

The integration suite starts a local Spark JVM and trains every backend in
binary and multiclass modes.
