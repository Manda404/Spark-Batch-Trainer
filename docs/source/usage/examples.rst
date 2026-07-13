Examples
========

This section provides a guided introduction to the **practical examples** included in
**Spark Batch Trainer**. The goal is to help you quickly understand how the framework
can be applied to real-world machine learning tasks.

What you will learn
-------------------
The examples are designed to show you step by step how to:

1. **Configure a model**
   Define framework-specific hyperparameters (*model_config*) for
   **XGBoost**, **CatBoost**, or **LightGBM**.

2. **Set up batch-wise training**
   Control the training process with *training_config* parameters such as
   number of batches, batch size, and early stopping patience.

3. **Apply learning rate scheduling**
   Use *learning_rate_config* to dynamically adjust the learning rate
   (e.g., with **exponential decay**) during training.

4. **Retrieve and evaluate the final model**
   Once training is complete, the trained model is ready for predictions
   and evaluation.

Available scenarios
---------------------------

Currently, Spark Batch Trainer provides complete examples for two major
classification tasks:

- :doc:`examples_binary`: backend recipes for binary classification
- :doc:`examples_multiclass`: backend recipes for multiclass classification

Additionally, the :doc:`dataset_overview` page explains the datasets used
throughout the examples.

Supported scope
---------------

The public trainers currently target classification. Regression and ranking
are not documented as supported workflows.

How to use this section
-----------------------

- **New users**: Start with the :doc:`dataset_overview` to understand the data.
- **Hands-on learners**: Go directly to the :doc:`examples_binary` example.
- **Advanced users**: Explore the :doc:`examples_multiclass` to see how batch-wise
  training scales to more complex problems.
