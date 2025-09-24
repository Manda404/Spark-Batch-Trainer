Installation
============

Prérequis
---------
- Python >= 3.11
- Apache Spark (version compatible : >= 4.0.0)
- pip ou poetry

Installation avec pip
---------------------
.. code-block:: bash

   pip install spark-batch-trainer

Installation avec poetry
------------------------
.. code-block:: bash

   poetry add spark-batch-trainer

Depuis la source (GitHub)
-------------------------
.. code-block:: bash

   git clone https://github.com/Manda404/SparkBatchTrainer.git
   cd SparkBatchTrainer
   poetry install

Vérification
------------
.. code-block:: python

   import spark_batch_trainer
   print("Installation réussie !")
