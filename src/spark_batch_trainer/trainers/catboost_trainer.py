from copy import deepcopy
from catboost import CatBoostClassifier, Pool
from pandas import DataFrame as PandasDataFrame
from typing import Optional, List, Dict, Any
from pyspark.sql import DataFrame as SparkDataFrame
from spark_batch_trainer.core.base_trainer import BatchTrainer
from spark_batch_trainer.core.memory_optimizer import MemoryOptimizer
from spark_batch_trainer.core.class_weight_optimizer import OptimizedWeightCalculator


class CatBoostTrainer(BatchTrainer):
    """
    Batch trainer for CatBoost classifiers.

    This class supports incremental training in mini-batches on Spark DataFrames,
    with validation support, global early stopping, and categorical feature handling.
    """

    def __init__(self):
        """
        Initialize the CatBoost batch trainer.

        Attributes
        ----------
        _global_train_loss : list of list of float
            Training loss history across all batches.
        _global_valid_loss : list of list of float
            Validation loss history across all batches.
        _global_iterations : list of int
            Indices of processed batches.
        _model : CatBoostClassifier or None
            Final trained CatBoost model after training.
        _categorical_features : list of str or None
            Names of categorical features if any were detected.
        _optimizer : MemoryOptimizer
            Utility for reducing memory usage of validation DataFrames
            (optimizes size in MB before training).
        _weight_calculator : OptimizedWeightCalculator
            Utility for computing and caching sample/class weights
            across training batches.
        """
        super().__init__()
        self._global_train_loss: List[List[float]] = []
        self._global_valid_loss: List[List[float]] = []
        self._global_iterations: List[int] = []
        self._model: Optional[CatBoostClassifier] = None
        self._categorical_features: Optional[List[str]] = None
        self._optimizer = MemoryOptimizer(use_logger=True)
        self._weight_calculator = OptimizedWeightCalculator()


    def fit(
        self,
        train_dataframe: Optional[SparkDataFrame],
        valid_dataframe: Optional[SparkDataFrame],
        target_column: str,
        **kwargs,
    ) -> None:
        """
        Train a CatBoost classifier incrementally on Spark DataFrames in batches.

        Parameters
        ----------
        train_dataframe : SparkDataFrame or None
            Training dataset in Spark DataFrame format.
        valid_dataframe : SparkDataFrame or None
            Validation dataset in Spark DataFrame format. Required for metric tracking and early stopping.
        target_column : str
            Target column name in both train and validation DataFrames.
        **kwargs : dict
            Additional configuration dictionaries.

            - config_model : dict
                CatBoost hyperparameters.
            - config_training : dict
                Training options such as ``num_batches`` (int), ``eval_metric`` (str),
                ``use_sample_weight`` (bool), ``max_patience`` (int).

        Returns
        -------
        None
            The trained model is stored internally in ``self.model``.

        Raises
        ------
        ValueError
            If input DataFrames or target column are invalid.
        RuntimeError
            If no model is successfully trained.

        Notes
        -----
        - Supports global early stopping across batches.
        - Handles categorical features automatically if detected.
        """
        # Extract configurations (model, training)
        config_model: Dict[str, Any] = deepcopy(kwargs.get("config_model", {}))
        config_training: Dict[str, Any] = deepcopy(kwargs.get("config_training", {}))
        num_batches: int = config_training.get("num_batches", 10)

        # Validate input parameters before starting training
        self._logger.info("Validating input parameters")
        self._validate_input_parameters(
            train_dataframe, valid_dataframe, target_column, num_batches
        )

        # Preprocess training data and split into batches (generator)
        self._logger.info("Preparing training data")
        dataframe_generator = self._apply_pandas_processing_to_generator(
            train_dataframe, target_column, num_batches
        )

        # Prepare validation dataset (converted to pandas + preprocessing)
        self._logger.info("Preparing validation data")
        # Prepare validation dataset (converted to pandas + preprocessing) and return dict (features, target, sample weights)
        valid_data = self._prepare_validation_data(valid_dataframe, target_column)

        # Initialize training state (best model, patience, eval metric, etc.)
        self._logger.info("Setup training state for CatBoost")
        self._setup_training_state(config_training, num_batches)

        # Log the start of training
        self._logger.info(f"🚀 Starting CatBoost training with {num_batches} batches")

        try:
            # --- Main training loop ---
            for batch_id, processed_batch in enumerate(dataframe_generator):
                self._logger.info(
                    f"\n--- 📦 Processing batch {batch_id + 1}/{num_batches} ---"
                )

                # Prepare the batch data (features, target)
                current_batch_data = self._prepare_batch_data(
                    processed_batch, target_column, batch_id + 1
                )

                # Train CatBoost on the current batch
                current_model = self._train_batch(
                    batch_data=current_batch_data,
                    valid_data=valid_data,
                    batch_num=batch_id + 1,
                    config_model=config_model,
                    config_training=config_training,
                )

                # Evaluate model performance and update best model if improved
                should_stop = self._evaluate_trained_model(
                    current_model=current_model,
                    config_training=config_training,
                    batch_num=batch_id + 1,
                )

                # Stop training early if patience threshold is reached
                if should_stop:
                    self._logger.info("Early stopping triggered - Training completed")
                    break

        except Exception as e:
            # Log and re-raise any exception for debugging
            self._logger.error(f"Training failed at batch {batch_id + 1}: {str(e)}")
            raise

        # Finalize training: select best model, generate diagnostics, and log results
        self._wrap_up_training(
            config_training=config_training,
            config_model=config_model,
        )


    def get_trained_model(self) -> Any:
        """
        Retrieve the final trained model instance.

        Returns
        -------
        Any
            Trained model object (e.g. ``CatBoostClassifier``).
        """
        return self.model

    def _train_batch(
        self,
        batch_data: Dict[str, Any],
        valid_data: Dict[str, Any],
        batch_num: int,
        config_model: Dict[str, Any],
        config_training: Dict[str, Any],
    ) -> CatBoostClassifier:
        """
        Train a CatBoost model on a single batch with optional warm restart.

        Parameters
        ----------
        batch_data : dict
            Dictionary containing the batch data:
            - ``"features"`` : pandas.DataFrame
                Features for the current batch.
            - ``"target"`` : pandas.Series
                Target values for the current batch.
            - ``"sample_weight"`` : array-like, optional
                Sample weights for the current batch.
        valid_data : dict
            Dictionary containing the validation data:
            - ``"features"`` : pandas.DataFrame
                Validation features.
            - ``"target"`` : pandas.Series
                Validation target values.
            - ``"sample_weight"`` : array-like, optional
                Validation sample weights.
        batch_num : int
            Index of the current batch (1-indexed).
        config_model : dict
            Hyperparameters for the CatBoost model.
        config_training : dict
            Training configuration including:
            - ``"previous_model"`` : CatBoostClassifier or None
                Model from the previous batch for warm restart.
            - ``"num_batches"`` : int
                Total number of batches.
            - ``"use_sample_weight"`` : bool, optional (default=True)
                Whether to apply sample weights during training.

        Returns
        -------
        CatBoostClassifier
            Trained CatBoost model for the current batch.

        Notes
        -----
        - Warm restart is enabled if ``config_training["previous_model"]`` is provided.
        - Validation performance is monitored with ``eval_set``.
        - ``use_best_model`` is disabled so that early stopping is managed globally
        across batches.
        """
        # Log the beginning of training
        self._logger.info(
            f"🏋️ Training CatBoost model on batch {batch_num}/{config_training['num_batches']}"
        )

        # Initialize CatBoost model with given hyperparameters
        model = CatBoostClassifier(**config_model)

        # Build CatBoost Pools with optional sample weights
        train_pool = Pool(
            batch_data["features"],
            batch_data["target"],
            weight=batch_data.get("sample_weight") if config_training.get("use_sample_weight") else None,
            cat_features=self._categorical_features if self._categorical_features else None,
        )
        valid_pool = Pool(
            valid_data["features"],
            valid_data["target"],
            weight=valid_data.get("sample_weight") if config_training.get("use_sample_weight") else None,
            cat_features=self._categorical_features if self._categorical_features else None,
        )

        # Train the model on the current batch
        model.fit(
            train_pool,
            eval_set=valid_pool,
            init_model=config_training.get("previous_model"),
            use_best_model=False,  # handled globally outside
            verbose=config_model.get("verbose"),
        )

        # Log the completion
        self._logger.info(
            f"Model trained on batch {batch_num}/{config_training['num_batches']}"
        )

        return model

    def _evaluate_trained_model(
        self,
        current_model: CatBoostClassifier,
        config_training: Dict[str, Any],
        batch_num: int,
    ) -> bool:
        """
        Evaluate the current CatBoost model after a batch and update the training state.

        Parameters
        ----------
        current_model : CatBoostClassifier
            Model trained on the current batch.
        config_training : dict
            Training state dictionary that tracks progress and early stopping, with keys:
            - ``"previous_model"`` : CatBoostClassifier or None
                Last trained model for warm restart.
            - ``"best_model"`` : CatBoostClassifier or None
                Best model found so far based on validation loss.
            - ``"best_valid_loss"`` : float
                Best validation loss observed so far.
            - ``"patience_counter"`` : int
                Number of consecutive batches without improvement.
            - ``"max_patience"`` : int
                Maximum patience allowed before triggering early stopping.
        batch_num : int
            Index of the current batch (1-indexed).

        Returns
        -------
        bool
            True if early stopping should be triggered, False otherwise.

        Notes
        -----
        - Updates ``best_model`` if the validation loss improves.
        - Global early stopping is controlled with ``patience_counter`` and ``max_patience``.
        """
        # --- Start evaluation for the current batch ---
        self._logger.info(f"Evaluating model performance on batch {batch_num}.")

        # Save the current model as the previous model before it gets replaced
        config_training["previous_model"] = deepcopy(current_model)

        # Extract evaluation metrics from the model's training history
        # (requires .fit() with eval_set and eval_metric configured)
        train_scores, valid_scores, eval_metric = self._get_training_metrics(
            current_model, "catboost"
        )

        # Get the final training and validation losses for this batch
        train_loss = train_scores[-1]
        valid_loss = valid_scores[-1]

        # Store metrics for global tracking
        self._global_train_loss.append(train_scores)
        self._global_valid_loss.append(valid_scores)
        self._global_iterations.append(batch_num)

        # Log performance metrics for this batch
        self._logger.info(
            f"Batch {batch_num} - {eval_metric} | Train: {train_loss:.5f} | Valid: {valid_loss:.5f}"
        )

        # --- Early stopping logic ---
        if valid_loss < config_training["best_valid_loss"]:
            # Improvement detected → update best model and reset patience
            improvement = config_training["best_valid_loss"] - valid_loss
            config_training["best_valid_loss"] = valid_loss
            config_training["patience_counter"] = 0
            config_training["best_model"] = deepcopy(current_model)

            self._logger.info(
                f"🎉 New best model found - "
                f"{eval_metric}: {valid_loss:.5f} (improvement: {improvement:.5f})"
            )
            return False  # Do not stop training

        # No improvement → increment patience counter
        config_training["patience_counter"] += 1
        self._logger.info(
            f"⏳ No improvement - Patience: "
            f"{config_training['patience_counter']}/{config_training['max_patience']}"
        )

        # Trigger early stopping if patience limit is reached
        if config_training["patience_counter"] >= config_training["max_patience"]:
            self._logger.warning("Early stopping triggered.")
            return True  # Stop training

        # Continue training
        return False

    def _prepare_batch_data(
        self,
        processed_batch: PandasDataFrame,
        target_column: str,
        batch_num: int,
    ) -> Dict[str, Any]:
        """
        Prepare a training batch by extracting features, targets, and sample weights.

        This method:
        - Computes sample weights for the batch based on the target distribution.
        - Converts object-type features to categorical dtype when applicable.
        - Optimizes memory usage of the feature matrix using the class-level optimizer.

        Parameters
        ----------
        processed_batch : PandasDataFrame
            Batch of training data after preprocessing.
            Must include both input features and the target column.
        target_column : str
            Name of the target column in `processed_batch`.
        batch_num : int
            Index of the current batch (used for logging or tracking).

        Returns
        -------
        dict
            Dictionary containing:

            - "features" : pandas.DataFrame
                Optimized feature matrix (target column excluded).
            - "target" : pandas.Series
                Target vector aligned with the feature matrix.
            - "sample_weight" : numpy.ndarray
                Computed weights for each sample, based on class distribution.
        """

        # Log the beginning of sample weight calculation for this batch
        self._logger.info(f"Calculating sample weights for batch {batch_num}")

        # Compute sample weights based on the distribution of the target values
        # (important for handling class imbalance)
        sample_weight = self._weight_calculator.calculate_sample_weights(
            processed_batch[target_column]
        )

        # Log the categorical conversion process
        self._logger.info("Converting all features of type 'object' to 'categorical'")

        # Convert columns with dtype "object" into pandas "category"
        # This is useful for models like XGBoost/CatBoost/lightgbm that can leverage categorical features
        processed_batch, _ = self._convert_object_to_category_dtype(
            processed_batch, target_column
        )

        # Return the batch data in a standardized dictionary format:
        # - Features (optimized using the instance-level MemoryOptimizer)
        # - Target vector
        # - Sample weights
        return {
            "features": self._optimizer.optimize(
                processed_batch.drop(
                    columns=[target_column]
                )  # Optimized feature matrix
            ),
            "target": processed_batch[target_column],  # Target vector
            "sample_weight": sample_weight,  # Sample weights for balancing
        }

    def _prepare_validation_data(
        self,
        valid_dataframe: Optional[SparkDataFrame],
        target_column: str,
    ) -> Dict[str, Any]:
        """
        Prepare and preprocess the validation dataset.

        This method converts the Spark DataFrame to pandas, applies preprocessing,
        detects categorical features, converts them to categorical dtype if needed,
        and computes class weights for the target column.

        Parameters
        ----------
        valid_dataframe : SparkDataFrame or None
            Validation dataset in Spark format. If None, no validation data is prepared.
        target_column : str
            Name of the target column in the dataset.

        Returns
        -------
        dict
            A dictionary containing:
            - "features" (PandasDataFrame): Optimized feature matrix.
            - "target" (PandasSeries): Target vector.
            - "sample_weight" (ndarray): Computed class/sample weights.
        """

        self._logger.info("Preparing validation data...")
        # Convert Spark DataFrame to pandas and apply preprocessing if needed
        valid_data_processed: PandasDataFrame = self._apply_pandas_processing(
            valid_dataframe
        )

        # Convert object-type columns to categorical dtype if applicable
        valid_data, cat_is_present = self._convert_object_to_category_dtype(
            valid_data_processed, target_column
        )

        # Detect categorical features in the validation set
        if cat_is_present:
            self._logger.info("Categorical features detected in validation set")
            self._categorical_features = (
                valid_data.drop(columns=[target_column])
                .select_dtypes(include=["category"])
                .columns.tolist()
            )
            self._logger.info(
                f"Number of categorical features identified: {len(self._categorical_features)}"
            )

        # Compute sample weights based on the target distribution
        sample_weights_valid = self._weight_calculator.calculate_sample_weights(
            valid_data[target_column]
        )

        # Log that sample weights have been successfully computed
        self._logger.info("Sample weights for validation set computed successfully")

        # Return the processed validation dataset
        return {
            "features": self._optimizer.optimize(
                valid_data.drop(columns=[target_column])
            ),  # Feature matrix optimized
            "target": valid_data[target_column],  # Target vector
            "sample_weight": sample_weights_valid,  # Class weights
        }

    def _wrap_up_training(
        self,
        config_training: Dict[str, Any],
        config_model: Dict[str, Any],
    ) -> None:
        """
        Finalize the CatBoost training process by cleaning resources,
        selecting the final model, and generating optional visualizations.

        Parameters
        ----------
        config_training : dict
            Training state and configuration dictionary with keys:
            - ``"best_model"`` : CatBoostClassifier or None
                Best model found during training (based on validation loss).
            - ``"previous_model"`` : CatBoostClassifier or None
                Last trained model if no best model was found.
            - ``"best_valid_loss"`` : float
                Best validation loss observed.
            - ``"show_learning_curve"`` : bool
                Whether to generate learning curve visualizations.
        config_model : dict
            Model hyperparameters, including:
            - ``"eval_metric"`` : str
                The evaluation metric used during training.

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If no model was successfully trained.

        Notes
        -----
        - Clears cached values from the weight calculator to free memory.
        - Stores the final model in ``self.model``.
        - Generates learning curve plots if requested.
        """
        # --- Resource cleanup ---
        self._logger.info(f"Clearing cache of {type(self._weight_calculator).__name__}")
        self._weight_calculator.clear_cache()

        # --- Select the final model ---
        final_model = (
            config_training["best_model"]
            if config_training["best_model"] is not None
            else config_training["previous_model"]
        )

        if final_model is None:
            raise RuntimeError("No model was successfully trained")

        self.model = final_model

        # --- Optional visualization ---
        if config_training.get("show_learning_curve", False):
            self._plot_learning_curve(
                self._global_train_loss,    # Training losses across batches
                self._global_valid_loss,    # Validation losses across batches
                self._global_iterations,    # Batch indices
                "CatBoost",                # Model name for labeling
                config_model["eval_metric"]  # Evaluation metric
            )

        # --- Training summary ---
        total_batches = len(self._global_iterations)
        final_valid_loss = config_training["best_valid_loss"]
        model_type = "best" if config_training["best_model"] is not None else "last"

        self._logger.info("CatBoost training completed successfully")
        self._logger.info(f"Total batches processed: {total_batches}")
        self._logger.info(f"Best validation loss: {final_valid_loss:.5f}")
        self._logger.info(f"Final model selected: {model_type}")

        # --- Log categorical feature usage ---
        if self._categorical_features:
            self._logger.info(
                f"🔖 Categorical features used: {len(self._categorical_features)}"
            )

    # Common function shared by both LightGBM, XGBoost and Catboost:
    def _validate_input_parameters(
        self,
        train_dataframe: Optional[SparkDataFrame],
        valid_dataframe: Optional[SparkDataFrame],
        target_column: str,
        num_batches: int,
    ) -> None:
        """
        Validate input parameters before starting training.

        Parameters
        ----------
        train_dataframe : SparkDataFrame or None
            Training dataset. Must be a Spark DataFrame containing the target column.
        valid_dataframe : SparkDataFrame or None
            Validation dataset. Must be a Spark DataFrame containing the target column.
        target_column : str
            Name of the target column to predict. Must exist in both
            `train_dataframe` and `valid_dataframe`.
        num_batches : int
            Number of batches to split the training data into.
            Must be greater than or equal to 1.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If:
            - `train_dataframe` or `valid_dataframe` is None
            - `num_batches` is less than 1
            - `target_column` is missing from either DataFrame
        """
        self._logger.info("Validating input parameters...")
        if train_dataframe is None or valid_dataframe is None:
            raise ValueError("train dataframe and validation dataframe cannot be None")

        if num_batches <= 0:
            raise ValueError("the number of batches must be >= 1")

        if target_column not in train_dataframe.columns:
            raise ValueError(f"train dataframe must contain '{target_column}' column")

        if target_column not in valid_dataframe.columns:
            raise ValueError(f"valid dataframe must contain '{target_column}' column")

        self._logger.info("Input parameters validation passed.")

    def _setup_training_state(
        self,
        config_training: Dict[str, Any],
        num_batches: int,
    ) -> None:
        """
        Initialize and update the training state inside the provided config_training dictionary.

        This method enriches the training configuration with runtime state information
        required during the batch-wise training process (e.g., early stopping, best model,
        patience counter, etc.). It modifies the dictionary in place and does not return
        any value.

        Parameters
        ----------
        config_training : dict
            Training configuration parameters that will also store the runtime training state.
            Keys such as 'max_patience', 'use_sample_weight', and 'verbose' are read if present,
            otherwise defaults are used.
        num_batches : int
            Total number of batches planned for the training session.
        """
        # Initialize the training state with default values and config overrides
        config_training.update(
            {
                "previous_model": None,  # Model trained on the previous batch (used for warm start)
                "best_model": None,  # Best model found so far (updated when validation improves)
                "should_stop": False,  # Early stopping flag (True if training should end)
                "patience_counter": 0,  # Number of consecutive batches without improvement
                "best_valid_loss": float("inf"),  # Best validation loss achieved so far
                "max_patience": config_training.get("max_patience", 5),
                # Maximum number of batches allowed without improvement before stopping
                "num_batches": num_batches,
                # Total number of batches planned for the training session
                "use_sample_weight": config_training.get("use_sample_weight", False),
                # Whether to apply sample weights during training
                "verbose": config_training.get("verbose", True),
                # Controls verbosity of LightGBM training logs
            }
        )