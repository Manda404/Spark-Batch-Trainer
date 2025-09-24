from copy import deepcopy
from lightgbm import LGBMClassifier
from typing import Any, Optional, List, Dict
from pandas import DataFrame as PandasDataFrame
from pyspark.sql import DataFrame as SparkDataFrame
from spark_batch_trainer.core.base_trainer import BatchTrainer
from spark_batch_trainer.core.memory_optimizer import MemoryOptimizer
from spark_batch_trainer.core.class_weight_optimizer import OptimizedWeightCalculator


class LightGBMTrainer(BatchTrainer):
    """
    Batch trainer for LightGBM classifiers.

    This class supports incremental training in mini-batches on Spark DataFrames,
    with validation support, early stopping, sample weighting, and learning rate scheduling.
    """

    def __init__(self):
        """
        Initialize the LightGBM batch trainer.

        Attributes
        ----------
        _global_train_loss : list of list of float
            History of training loss values for each batch.
        _global_valid_loss : list of list of float
            History of validation loss values for each batch.
        _global_iterations : list of int
            Indices of processed batches during training.
        _model : LGBMClassifier or None
            The trained LightGBM model instance (set after training).
        _lr_schedulers : list of float
            History of learning rates applied per batch.
        _categorical_features : list of str or None
            Names of categorical features used for training.
        _weight_calculator : OptimizedWeightCalculator
            Utility to compute optimized sample weights per batch.
        _optimizer : MemoryOptimizer
            Utility for reducing memory usage of validation DataFrames
            (optimizes size in MB before training).
        """
        super().__init__()
        self._global_train_loss: List[List[float]] = []
        self._global_valid_loss: List[List[float]] = []
        self._global_iterations: List[int] = []
        self._model: Optional[LGBMClassifier] = None
        self._lr_schedulers: List[float] = []
        self._categorical_features: Optional[List[str]] = None
        self._weight_calculator = OptimizedWeightCalculator()
        self._optimizer = MemoryOptimizer(
            use_logger=True
        )  # Memory optimization of the validation DataFrame (reduce size in MB before training)

    def fit(
        self,
        train_dataframe: Optional[SparkDataFrame],
        valid_dataframe: Optional[SparkDataFrame],
        target_column: str,
        **kwargs,
    ) -> None:
        """
        Train a LightGBM classifier incrementally on Spark DataFrames in batches.

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
                LightGBM hyperparameters. ``learning_rate`` may be updated batch-wise.
            - config_training : dict
                Training options such as ``num_batches`` (int), ``eval_metric`` (str),
                ``use_sample_weight`` (bool).
            - config_lr_scheduler : dict or callable or None
                Learning rate scheduler configuration.

        Returns
        -------
        None
            The trained model is stored internally in ``self._model``.

        Raises
        ------
        ValueError
            If input DataFrames or target column are invalid.
        RuntimeError
            If no model is successfully trained.

        Notes
        -----
        - Supports early stopping based on validation loss.
        - Learning rate can be scheduled across batches.
        """
        # Extract configurations (model, training, learning rate scheduler)
        config_model: Dict[str, Any] = deepcopy(kwargs.get("config_model", {}))
        config_training: Dict[str, Any] = deepcopy(kwargs.get("config_training", {}))
        num_batches: int = config_training.get("num_batches", 10)
        lr_scheduler_config: Optional[Dict[str, Any]] = kwargs.get(
            "config_lr_scheduler", None
        )

        # Validate input parameters before starting training
        self._validate_input_parameters(
            train_dataframe, valid_dataframe, target_column, num_batches
        )

        # Preprocess training data and split into batches (generator)
        dataframe_generator = self._apply_pandas_processing_to_generator(
            train_dataframe, target_column, num_batches
        )

        # Prepare validation dataset (converted to pandas + preprocessing) and return dict (features, target, sample weights)
        valid_data = self._prepare_validation_data(valid_dataframe, target_column)

        # Initialize training state (best model, patience, eval metric, etc.)
        self._logger.info("Setup training state for LightGBM")
        self._setup_training_state(config_training, num_batches)

        # Log the start of training
        self._logger.info(f"🚀 Starting LightGBM training with {num_batches} batches")

        # --- Main training loop ---
        try:
            for batch_id, processed_batch in enumerate(dataframe_generator):
                self._logger.info(
                    f"\n--- 📦 Processing batch {batch_id + 1}/{num_batches} ---"
                )

                # Prepare the batch data (features, target, sample weights)
                current_batch_data = self._prepare_batch_data(
                    processed_batch, target_column, batch_id + 1
                )

                # Retrieve the current learning rate from the scheduler
                current_lr = self._get_current_learning_rate(
                    lr_scheduler_config, batch_id + 1
                )

                # Train the model on the current batch (with validation set)
                current_model = self._train_batch(
                    batch_data=current_batch_data,
                    valid_data=valid_data,
                    config_training=config_training,
                    config_model=config_model,
                    batch_id=batch_id + 1,
                    current_lr=current_lr,
                )

                # Evaluate the trained model and update training state
                self._evaluate_trained_model(
                    current_model=current_model,
                    config_training=config_training,
                    batch_id=batch_id + 1,
                )

                # Check early stopping condition (based on patience and validation loss)
                if config_training["should_stop"]:
                    self._logger.info("Early stopping triggered - Training completed")
                    break

        except Exception as e:
            # Log and re-raise exceptions for debugging
            self._logger.error(f"Training failed at batch {batch_id + 1}: {str(e)}")
            raise

        # Finalize training: select best model, clear cache, and generate visualizations
        self._wrap_up_training(
            config_training=config_training,
            lr_scheduler_config=lr_scheduler_config,
        )

    def get_trained_model(self) -> Any:
        """
        Retrieve the final trained model instance.

        Returns
        -------
        Any
            Trained model object (e.g. ``LGBMClassifier``).
        """
        return self._model

    def _train_batch(
        self,
        batch_data: Dict[str, Any],
        valid_data: Dict[str, Any],
        config_training: Dict[str, Any],
        config_model: Dict[str, Any],
        batch_id: int,
        current_lr: float,
    ) -> LGBMClassifier:
        """
        Train a LightGBM classifier on a single batch of data with dynamic
        learning rate adjustment, optional warm start, and sample weights.

        Parameters
        ----------
        batch_data : dict
            Training batch with:
            - "features" : pandas.DataFrame
                Feature matrix for the batch.
            - "target" : pandas.Series
                Target values for the batch.
            - "sample_weight" : array-like or None, optional
                Sample weights for the training batch.
        valid_data : dict
            Validation set with:
            - "features" : pandas.DataFrame
            - "target" : pandas.Series
            - "sample_weight" : array-like or None, optional
                Sample weights for the validation set.
        config_training : dict
            Training configuration and runtime state. Keys used include:
            - "eval_metric" : str or callable, optional
                Metric to monitor during training (default: "logloss").
            - "use_sample_weight" : bool, optional
                Whether to apply sample weights (default: False).
            - "previous_model" : LGBMClassifier or None
                Model to use for warm start if available.
        config_model : dict
            Base LightGBM hyperparameters for model initialization.
        batch_id : int
            Index of the current batch (1-based).
        current_lr : float
            Learning rate to apply for this batch, overriding config_model.

        Returns
        -------
        LGBMClassifier
            The LightGBM model trained on the given batch.

        Notes
        -----
        - Warm start is enabled if ``config_training["previous_model"]`` is provided.
        - Sample weights are applied to both training and validation sets when
        ``use_sample_weight=True`` and corresponding weights are available.
        - The learning rate is dynamically updated per batch with ``current_lr``.
        """
        # Log start of batch training
        self._logger.info(f"🏋️ Training LGBM on batch {batch_id}")

        # Merge model configuration with current learning rate
        params_batch = {**config_model, "learning_rate": current_lr}
        self._logger.info(
            "LGBMClassifier configuration",
            extra={
                "batch": batch_id,
                "learning_rate": current_lr,
                "lgbm_config": params_batch,
            },
        )

        # Instantiate model
        model = LGBMClassifier(**params_batch)

        # Safe extraction of optional sample weights
        sample_weight_train = (
            batch_data.get("sample_weight")
            if config_training.get("use_sample_weight")
            else None
        )
        sample_weight_valid = (
            valid_data.get("sample_weight")
            if config_training.get("use_sample_weight")
            else None
        )

        # Default eval_metric if not provided
        eval_metric = config_training.get("eval_metric", "logloss")

        # Warm start from previous model if available
        prev_model = config_training.get("previous_model")

        # Train model
        model.fit(
            batch_data["features"],
            batch_data["target"],
            eval_set=[
                (batch_data["features"], batch_data["target"]),  # Training set
                (valid_data["features"], valid_data["target"]),  # Validation set
            ],
            eval_names=["train", "valid"],
            eval_metric=eval_metric,
            categorical_feature=(
                self._categorical_features if self._categorical_features else "auto"
            ),
            sample_weight=sample_weight_train,
            eval_sample_weight=[sample_weight_train, sample_weight_valid]
            if config_training.get("use_sample_weight")
            else None,
            init_model=prev_model if prev_model is not None else None,
        )

        # Log completion
        self._logger.info(f"🎯 Completed LGBM training on batch {batch_id}")

        return model

    def _evaluate_trained_model(
        self,
        current_model: LGBMClassifier,
        config_training: dict[str, Any],
        batch_id: int,
    ) -> None:
        """
        Evaluate the LightGBM model after training on a batch and update
        the training state (best model, patience, early stopping).

        Parameters
        ----------
        current_model : LGBMClassifier
            Model trained on the current batch.
        config_training : dict
            Training configuration and runtime state. Keys updated include:
            - "previous_model" : last trained model
            - "best_model" : best model so far
            - "best_valid_loss" : lowest validation loss achieved
            - "patience_counter" : consecutive non-improving batches
            - "should_stop" : bool, early stopping flag
        batch_id : int
            Index of the current batch (1-based).

        Returns
        -------
        None

        Notes
        -----
        - Updates ``best_model`` if validation loss improves.
        - Resets or increments the patience counter depending on improvement.
        - Sets ``should_stop=True`` if patience exceeds ``max_patience``.
        """
        self._logger.info(f"🔍 Evaluating LightGBM model on batch {batch_id}")

        # Save current model as "previous_model"
        config_training["previous_model"] = deepcopy(current_model)

        # Extract metrics (requires .fit() with eval_set and eval_metric configured)
        train_scores, valid_scores, eval_metric = self._get_training_metrics(
            current_model, "lightgbm"
        )

        if not train_scores or not valid_scores:
            self._logger.warning("⚠️ No training history found for evaluation.")
            return

        # Track metrics across batches
        self._global_train_loss.append(train_scores)
        self._global_valid_loss.append(valid_scores)
        self._global_iterations.append(batch_id)

        # Get most recent scores
        current_train_loss = train_scores[-1]
        current_valid_loss = valid_scores[-1]

        self._logger.info(
            f"Batch {batch_id} - {eval_metric} | "
            f"Train: {current_train_loss:.5f} | "
            f"Valid: {current_valid_loss:.5f}"
        )

        # --- Early stopping logic ---
        if current_valid_loss < config_training["best_valid_loss"]:
            # Improvement → update best model and reset patience
            improvement = config_training["best_valid_loss"] - current_valid_loss
            self._logger.info(
                f"🎉 New best model found - {eval_metric}: {current_valid_loss:.5f} "
                f"(improvement: {improvement:.5f})"
            )
            config_training["best_valid_loss"] = current_valid_loss
            config_training["patience_counter"] = 0
            config_training["best_model"] = deepcopy(current_model)
            config_training["should_stop"] = False
        else:
            # No improvement → increment patience
            config_training["patience_counter"] += 1
            self._logger.info(
                f"⏳ No improvement - Patience: "
                f"{config_training['patience_counter']}/{config_training.get('max_patience', 5)}"
            )
            config_training["should_stop"] = config_training[
                "patience_counter"
            ] >= config_training.get("max_patience", 5)

    def _wrap_up_training(
        self,
        config_training: dict[str, Any],
        lr_scheduler_config: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Finalize the LightGBM training process by selecting the final model,
        clearing caches, and generating optional visualizations.

        Parameters
        ----------
        config_training : dict
            Training configuration and runtime state. Expected keys include:
            - "best_model" : best model found (or None if no improvement)
            - "previous_model" : last trained model
            - "best_valid_loss" : lowest validation loss achieved
            - "show_learning_curve" : bool, whether to generate plots
            - "eval_metric" : str or callable, optional
                Metric used for evaluation (default: "logloss").
        lr_scheduler_config : dict or None, optional
            Learning rate scheduler configuration if one was used.

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If no model was successfully trained.
        """
        # Clear cached values from the weight calculator to free memory
        self._logger.info(f"Cleaning cache of {type(self._weight_calculator).__name__}")
        self._weight_calculator.clear_cache()

        # Select final model: best if available, otherwise last trained
        final_model = (
            config_training["best_model"]
            if config_training["best_model"] is not None
            else config_training["previous_model"]
        )

        if final_model is None:
            raise RuntimeError("No model was successfully trained")

        # Save as final model
        self._model = final_model

        # Generate plots if requested
        if config_training.get("show_learning_curve"):
            # Learning rate schedule
            if lr_scheduler_config is not None and self._lr_schedulers:
                self._plot_lr_schedule("Exponential Decay", self._lr_schedulers)

            # Training/validation loss curves
            self._plot_learning_curve(
                self._global_train_loss,
                self._global_valid_loss,
                self._global_iterations,
                "LGBM",
                config_training.get("eval_metric", "logloss"),
            )

        # Log training summary
        total_batches = len(self._global_iterations)
        final_valid_loss = config_training["best_valid_loss"]
        model_type = "best" if config_training["best_model"] is not None else "last"

        self._logger.info("LGBM training completed successfully")
        self._logger.info(f"Total batches processed: {total_batches}")
        self._logger.info(f"Best validation loss: {final_valid_loss:.5f}")
        self._logger.info(f"Final model selected: {model_type}")

        # Log categorical features if any
        if self._categorical_features:
            self._logger.info(
                f"Categorical features used: {len(self._categorical_features)}"
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
                "eval_metric": config_training.get(
                    "eval_metric", "binary_logloss"
                ),  # Evaluation metric for monitoring training (default: binary_logloss)
            }
        )

    # Common function shared by both LightGBM and XGBoost:
    def _get_current_learning_rate(
        self, lr_scheduler_config: Optional[dict[str, Any]], batch_id: int
    ) -> float:
        """
        Compute the learning rate for the current batch using an exponential decay schedule.

        If no scheduler configuration is provided, a default learning rate of 0.1 is used.

        Parameters
        ----------
        lr_scheduler_config : dict or None
            Configuration for the learning rate scheduler. Expected keys:
            - "initial_lr" : float, default=0.1
                Initial learning rate before decay.
            - "decay_rate" : float, default=0.95
                Multiplicative decay factor applied per batch.
            - "min_lr" : float, default=1e-4
                Minimum learning rate allowed to avoid vanishing updates.
        batch_id : int
            Index of the current batch (starting at 1).

        Returns
        -------
        float
            The learning rate computed for the given batch.
        """

        # If no scheduler configuration is provided, fall back to a constant learning rate
        if lr_scheduler_config is None:
            return 0.1

        # Log the use of exponential decay for the learning rate
        self._logger.info("Computing the current learning rate with exponential decay")

        # Compute the current learning rate using exponential decay:
        # LR = max(initial_lr * (decay_rate ** batch_id), min_lr)
        current_lr = self._exponential_lr_schedule(
            initial_lr=lr_scheduler_config.get("initial_lr", 0.1),  # Default initial LR
            decay_rate=lr_scheduler_config.get(
                "decay_rate", 0.95
            ),  # Default decay factor
            min_lr=lr_scheduler_config.get("min_lr", 1e-4),  # Lower bound for LR
            batch_id=batch_id,  # Current batch index
        )

        # Log the computed learning rate for monitoring
        self._logger.info(f"Learning rate for batch {batch_id}: {current_lr:.6f}")

        # Keep track of learning rates used across batches for later analysis/visualization
        self._lr_schedulers.append(current_lr)

        # Return the final learning rate for the current batch
        return current_lr

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
        valid_data_processed, cat_is_present = self._convert_object_to_category_dtype(
            valid_data_processed, target_column
        )

        # Detect categorical features in the validation set
        if cat_is_present:
            self._logger.info("Categorical features detected in validation set")
            self._categorical_features = (
                valid_data_processed.drop(columns=[target_column])
                .select_dtypes(include=["category"])
                .columns.tolist()
            )
            self._logger.info(
                f"Number of categorical features identified: {len(self._categorical_features)}"
            )

        # Compute sample weights based on the target distribution
        sample_weights_valid = self._weight_calculator.calculate_sample_weights(
            valid_data_processed[target_column]
        )

        # Log that sample weights have been successfully computed
        self._logger.info("Sample weights for validation set computed successfully")

        # Return the processed validation dataset
        return {
            "features": self._optimizer.optimize(
                valid_data_processed.drop(columns=[target_column])
            ),  # Feature matrix optimized
            "target": valid_data_processed[target_column],  # Target vector
            "sample_weight": sample_weights_valid,  # Class weights
        }
