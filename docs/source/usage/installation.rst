Installation
============

Prerequisites
-------------
- Python >= 3.11
- Apache Spark (compatible version: >= 4.0.0)
- pip or poetry

Installation with pip
---------------------
.. code-block:: bash

   pip install spark-batch-trainer

Installation with poetry
------------------------
.. code-block:: bash

   poetry add spark-batch-trainer

From source (GitHub)
--------------------
.. code-block:: bash

   git clone https://github.com/Manda404/SparkBatchTrainer.git
   cd SparkBatchTrainer
   poetry install

Verification
------------
.. code-block:: python

   import spark_batch_trainer
   print("Installation successful!")
