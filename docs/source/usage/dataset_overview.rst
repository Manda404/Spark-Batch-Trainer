Dataset
================

This section introduces the two datasets used in the **Spark Batch Trainer** examples:  

- The **Diabetes Dataset** for **binary classification** tasks  
- The **Obesity Dataset** for **multiclass classification** tasks  

.. note::

   These datasets are provided for illustrative purposes only, to demonstrate
   how the framework works.

---

1. Diabetes Dataset (Binary Classification)
-------------------------------------------

📌 **Goal**: Predict whether a patient has **diabetes** based on medical and demographic features.  

- **Source**: *National Institute of Diabetes and Digestive and Kidney Diseases* (Diabetes.csv)  
- **Shape**: `shape = (100000, 9)`  
- **Available columns**:  

  - `gender` → Patient’s gender  
  - `age` → Age  
  - `hypertension` → Hypertension history (0 = No, 1 = Yes)  
  - `heart_disease` → Heart disease history (0 = No, 1 = Yes)  
  - `smoking_history` → Smoking habits  
  - `bmi` → Body mass index  
  - `HbA1c_level` → Average blood sugar level over 3 months  
  - `blood_glucose_level` → Current blood glucose level  
  - `diabetes` → **Target variable (binary: 0 = Non-diabetic, 1 = Diabetic)**  

.. note::

   This dataset is used to illustrate **binary classification**
   in training examples with **XGBoost**, **CatBoost**, and **LightGBM**.

---

2. Obesity Dataset (Multiclass Classification)
----------------------------------------------

📌 **Goal**: Predict an individual’s **body weight category** based on eating habits, lifestyle, and anthropometric measures.  

- **Source**: Academic dataset on obesity (ObesityDataset.csv)  
- **Shape**: `shape = (2111, 17)`  
- **Available columns**:  

  - `Age` → Age  
  - `Gender` → Gender  
  - `Height` → Height (meters)  
  - `Weight` → Weight (kg)  
  - `CALC` → Alcohol consumption  
  - `FAVC` → Frequent consumption of high-calorie foods  
  - `FCVC` → Vegetable consumption  
  - `NCP` → Number of main meals per day  
  - `SCC` → Calorie monitoring  
  - `SMOKE` → Smoking habit  
  - `CH2O` → Daily water intake  
  - `family_history_with_overweight` → Family history of overweight  
  - `FAF` → Weekly physical activity  
  - `TUE` → Time spent using electronic devices  
  - `CAEC` → Snacking between meals  
  - `MTRANS` → Main mode of transportation  
  - `NObeyesdad` → **Target variable (multiclass, 7 categories)**:  

    - `Insufficient_Weight`  
    - `Normal_Weight`  
    - `Overweight_Level_I`  
    - `Overweight_Level_II`  
    - `Obesity_Type_I`  
    - `Obesity_Type_II`  
    - `Obesity_Type_III`  

.. note::

   This dataset is used to illustrate **multiclass classification**
   in training examples with **XGBoost**, **CatBoost**, and **LightGBM**.

---

3. Preparation and Usage
------------------------

Before using **Spark Batch Trainer**, it is recommended to follow these steps:  

1. **Load the dataset**  
   For example, with `pandas.read_csv("file.csv")`.  

2. **Preprocess the data (if necessary)**  

   - Encode categorical variables  
   - Handle missing values  
   - Normalize / standardize numerical features  

3. **Split the dataset into subsets**  

   - **data_train** (60%): the portion of the dataset used to **teach the model** how to make predictions.  
   - **data_valid** (20%): a separate portion used to **check the model’s performance while training** and decide when to stop (early stopping).  
   - **data_test** (20%): a final portion kept aside to **simulate real-world unseen data** and measure how well the model generalizes.

4. **Convert the train / validation subsets** into **Spark DataFrames**  
   These DataFrames can be passed directly to the framework’s `.fit()` methods.  

---

4. Example Code Snippet
-----------------------

Below is a minimal example of how to load, split, and convert a dataset for **Spark Batch Trainer**:

.. code-block:: python

   import pandas as pd
   from sklearn.model_selection import train_test_split
   from pyspark.sql import SparkSession

   # Load CSV (example with Diabetes dataset)
   df = pd.read_csv("Diabetes.csv")

   # Split into train (60%) / validation (20%) / test (20%)
   train, temp = train_test_split(df, test_size=0.4, random_state=42)
   valid, test = train_test_split(temp, test_size=0.5, random_state=42)

   # Initialize Spark
   spark = SparkSession.builder.appName("CreateSparkDataFrame").getOrCreate()

   # Convert to Spark DataFrames (schema is inferred automatically)
   spark_train_df = spark.createDataFrame(train)
   spark_valid_df = spark.createDataFrame(valid)

   # These Spark DataFrames can now be used with:
   # spark_train_df : input for .fit()
   # spark_valid_df : input for validation in .fit()

.. note::

   These two datasets are used throughout the tutorials to illustrate different tasks:

   - :doc:`examples_binary` for binary classification
   - :doc:`examples_multiclass` for multiclass classification
