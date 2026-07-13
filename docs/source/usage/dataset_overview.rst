Preparing datasets
==================

Spark Batch Trainer does not own feature engineering or dataset splitting.
Prepare those inputs before training and keep their purpose explicit.

Required splits
---------------

``train``
   Used to construct batches and update the model.

``validation``
   Collected once on the driver and used to monitor every batch. It must be
   representative and small enough to fit in memory.

``test``
   Kept outside the trainer and evaluated only after model selection.

Input contract
--------------

Training and validation inputs must be Spark DataFrames with matching feature
columns and the same target column. Before calling ``fit``:

* handle missing values;
* encode the target for the selected backend;
* verify compatible feature types;
* prevent entity or time leakage across splits; and
* estimate how much memory a collected batch will require.

Example split
-------------

For random, independent observations, a deterministic Spark split is enough
for a first experiment:

.. code-block:: python

   train_df, validation_df, test_df = source_df.randomSplit(
       [0.7, 0.15, 0.15], seed=42
   )

For temporal or grouped data, use a domain-appropriate split instead. A random
split can leak future information or repeated entities across datasets.

Memory boundary
---------------

Spark performs partitioning and filtering, but backend models consume pandas
objects on the driver. The validation set and one training batch must fit in
driver memory, including conversion overhead and model allocations.
