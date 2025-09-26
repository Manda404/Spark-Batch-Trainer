Overview
=================

This section provides a guided introduction to the **practical examples** included in
**Spark Batch Trainer**. The goal is to help you quickly understand how the framework
can be applied to real-world machine learning tasks.

What You Will Learn
-------------------
The examples are designed to show you step by step how to:

1. **Configure a model**  
   Define framework-specific hyperparameters (*config_model*) for  
   **XGBoost**, **CatBoost**, or **LightGBM**.

2. **Set up batch-wise training**  
   Control the training process with *config_training* parameters such as  
   number of batches, batch size, and early stopping patience.

3. **Apply learning rate scheduling**  
   Use *config_lr_scheduler* to dynamically adjust the learning rate  
   (e.g., with **exponential decay**) during training.

4. **Retrieve and evaluate the final model**  
   Once training is complete, the trained model is ready for predictions  
   and evaluation.

Available Example Scenarios
---------------------------

Currently, Spark Batch Trainer provides complete examples for two major
classification tasks:

- :doc:`examples_binary` → Full workflow for **binary classification**  
- :doc:`examples_multiclass` → Full workflow for **multiclass classification**

Additionally, the :doc:`dataset_overview` page explains the datasets used
throughout the examples.

Planned Extensions
------------------

.. note::

   The current release (**v1.0.0**) of **Spark Batch Trainer** supports only
   **classification tasks** (binary and multiclass), on both **balanced**
   and **imbalanced** datasets.

   Support for additional machine learning tasks, such as **regression** and
   **ranking**, is planned in future releases.

How to Use This Section
-----------------------

- **New users**: Start with the :doc:`dataset_overview` to understand the data.  
- **Hands-on learners**: Go directly to the :doc:`examples_binary` example.  
- **Advanced users**: Explore the :doc:`examples_multiclass` to see how batch-wise
  training scales to more complex problems.  
