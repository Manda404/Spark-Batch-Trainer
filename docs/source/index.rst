Spark Batch Trainer
===================

Train XGBoost, CatBoost, and LightGBM models sequentially from stratified
Spark DataFrame batches.

.. important::

   Spark distributes batch assignment and filtering. Model training runs on
   the Python driver. The validation dataset and each pandas batch must fit in
   driver memory.

Start here
----------

* :doc:`usage/installation` — install the package and verify dependencies.
* :doc:`usage/quickstart` — run the smallest valid training example.
* :doc:`concepts` — understand continuation training, memory, and stopping.
* :doc:`architecture` — learn how the packages fit together.
* :doc:`api/index` — browse the supported public API.

.. toctree::
   :maxdepth: 2
   :caption: User guide

   introduction
   usage/installation
   usage/quickstart
   usage/tutorials
   concepts
   configuration
   usage/examples
   usage/dataset_overview
   usage/examples_binary
   usage/examples_multiclass

.. toctree::
   :maxdepth: 2
   :caption: Design and API

   architecture
   api/index

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   changelog

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
