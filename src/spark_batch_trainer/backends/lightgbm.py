from copy import deepcopy
from lightgbm import LGBMClassifier
from typing import Any, Optional, List, Dict
from pandas import DataFrame as PandasDataFrame
from pyspark.sql import DataFrame as SparkDataFrame
from spark_batch_trainer.training.base import BatchTrainer
from spark_batch_trainer.data import PandasMemoryOptimizer
from spark_batch_trainer.data import BalancedSampleWeightCalculator
from spark_batch_trainer.training.config import TrainingConfig
from spark_batch_trainer.training.early_stopping import GlobalEarlyStopping


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
        _weight_calculator : BalancedSampleWeightCalculator
            Utility to compute optimized sample weights per batch.
        _optimizer : PandasMemoryOptimizer
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
        self._sample_weight_calculator = BalancedSampleWeightCalculator()
        self._memory_optimizer = PandasMemoryOptimizer(enable_logging=True)

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

            - model_config : dict
                LightGBM hyperparameters. ``learning_rate`` may be updated batch-wise.
            - runtime_state : dict
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
        model_config: Dict[str, Any] = deepcopy(
            kwargs.get("model_config", kwargs.get("config_model", {}))
        )
        runtime_state: Dict[str, Any] = deepcopy(
            kwargs.get("training_config", kwargs.get("config_training", {}))
        )
        num_batches = TrainingConfig.from_mapping(runtime_state).num_batches
        learning_rate_config: Optional[Dict[str, Any]] = kwargs.get(
            "learning_rate_config", kwargs.get("config_lr_scheduler")
        )
        self._reset_run_history()

        # Validate input parameters before starting training
        self._validate_inputs(
            train_dataframe, valid_dataframe, target_column, num_batches
        )

        # Preprocess training data and split into batches (generator)
        dataframe_generator = self._iter_training_batches(
            train_dataframe, target_column, num_batches
        )

        # Prepare validation dataset (converted to pandas + preprocessing) and return dict (features, target, sample weights)
        validation_data = self._prepare_validation(valid_dataframe, target_column)

        # Initialize training state (best model, patience, eval metric, etc.)
        self._logger.info("Setup training state for LightGBM")
        self._initialize_runtime_state(runtime_state, num_batches)

        # Log the start of training
        self._logger.info(f"🚀 Starting LightGBM training with {num_batches} batches")

        # --- Main training loop ---
        try:
            for batch_id, pandas_batch in enumerate(dataframe_generator):
                self._logger.info(
                    f"\n--- 📦 Processing batch {batch_id + 1}/{num_batches} ---"
                )

                # Prepare the batch data (features, target, sample weights)
                current_batch_data = self._prepare_batch(
                    pandas_batch, target_column, batch_id + 1
                )

                # Retrieve the current learning rate from the scheduler
                learning_rate = self._resolve_learning_rate(
                    learning_rate_config,
                    batch_id + 1,
                    default_lr=float(model_config.get("learning_rate", 0.1)),
                )

                # Train the model on the current batch (with validation set)
                current_model = self._fit_batch(
                    batch_data=current_batch_data,
                    validation_data=validation_data,
                    runtime_state=runtime_state,
                    model_config=model_config,
                    batch_id=batch_id + 1,
                    learning_rate=learning_rate,
                )

                # Evaluate the trained model and update training state
                self._evaluate_batch(
                    current_model=current_model,
                    runtime_state=runtime_state,
                    batch_id=batch_id + 1,
                )

                # Check early stopping condition (based on patience and validation loss)
                if runtime_state["should_stop"]:
                    self._logger.info("Early stopping triggered - Training completed")
                    break

        except Exception as e:
            # Log and re-raise exceptions for debugging
            self._logger.error(f"Training failed at batch {batch_id + 1}: {str(e)}")
            raise

        # Finalize training: select best model, clear cache, and generate visualizations
        self._finalize_training(
            runtime_state=runtime_state,
            learning_rate_config=learning_rate_config,
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

    def _fit_batch(
        self,
        batch_data: Dict[str, Any],
        validation_data: Dict[str, Any],
        runtime_state: Dict[str, Any],
        model_config: Dict[str, Any],
        batch_id: int,
        learning_rate: float,
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
        validation_data : dict
            Validation set with:
            - "features" : pandas.DataFrame
            - "target" : pandas.Series
            - "sample_weight" : array-like or None, optional
                Sample weights for the validation set.
        runtime_state : dict
            Training configuration and runtime state. Keys used include:
            - "eval_metric" : str or callable, optional
                Metric to monitor during training (default: "logloss").
            - "use_sample_weight" : bool, optional
                Whether to apply sample weights (default: False).
            - "previous_model" : LGBMClassifier or None
                Model to use for warm start if available.
        model_config : dict
            Base LightGBM hyperparameters for model initialization.
        batch_id : int
            Index of the current batch (1-based).
        learning_rate : float
            Learning rate to apply for this batch, overriding model_config.

        Returns
        -------
        LGBMClassifier
            The LightGBM model trained on the given batch.

        Notes
        -----
        - Warm start is enabled if ``runtime_state["previous_model"]`` is provided.
        - Sample weights are applied to both training and validation sets when
        ``use_sample_weight=True`` and corresponding weights are available.
        - The learning rate is dynamically updated per batch with ``learning_rate``.
        """
        # Log start of batch training
        self._logger.info(f"🏋️ Training LGBM on batch {batch_id}")

        # Merge model configuration with current learning rate
        params_batch = {**model_config, "learning_rate": learning_rate}
        self._logger.info(
            "LGBMClassifier configuration",
            extra={
                "batch": batch_id,
                "learning_rate": learning_rate,
                "lgbm_config": params_batch,
            },
        )

        # Instantiate model
        model = LGBMClassifier(**params_batch)

        # Safe extraction of optional sample weights
        sample_weight_train = (
            batch_data.get("sample_weight")
            if runtime_state.get("use_sample_weight")
            else None
        )
        sample_weight_valid = (
            validation_data.get("sample_weight")
            if runtime_state.get("use_sample_weight")
            else None
        )

        # Default eval_metric if not provided
        eval_metric = runtime_state.get("eval_metric", "logloss")

        # Warm start from previous model if available
        prev_model = runtime_state.get("previous_model")

        # Train model
        model.fit(
            batch_data["features"],
            batch_data["target"],
            eval_set=[
                (batch_data["features"], batch_data["target"]),  # Training set
                (validation_data["features"], validation_data["target"]),  # Validation set
            ],
            eval_names=["train", "valid"],
            eval_metric=eval_metric,
            categorical_feature=(
                self._categorical_features if self._categorical_features else "auto"
            ),
            sample_weight=sample_weight_train,
            eval_sample_weight=[sample_weight_train, sample_weight_valid]
            if runtime_state.get("use_sample_weight")
            else None,
            init_model=prev_model if prev_model is not None else None,
        )

        # Log completion
        self._logger.info(f"🎯 Completed LGBM training on batch {batch_id}")

        return model

    def _evaluate_batch(
        self,
        current_model: LGBMClassifier,
        runtime_state: dict[str, Any],
        batch_id: int,
    ) -> None:
        """
        Evaluate the LightGBM model after training on a batch and update
        the training state (best model, patience, early stopping).

        Parameters
        ----------
        current_model : LGBMClassifier
            Model trained on the current batch.
        runtime_state : dict
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
        runtime_state["previous_model"] = deepcopy(current_model)

        # Extract metrics (requires .fit() with eval_set and eval_metric configured)
        train_scores, valid_scores, eval_metric = self._extract_metric_history(
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
        decision = GlobalEarlyStopping.observe(
            current_score=current_valid_loss,
            best_score=runtime_state.get("best_valid_score"),
            metric_name=eval_metric,
            patience_counter=runtime_state["patience_counter"],
            max_patience=runtime_state["max_patience"],
            mode=runtime_state["metric_mode"],
            min_delta=runtime_state["min_delta"],
        )
        runtime_state["patience_counter"] = decision.patience_counter
        if decision.improved:
            # Improvement → update best model and reset patience
            improvement = runtime_state["best_valid_loss"] - current_valid_loss
            self._logger.info(
                f"🎉 New best model found - {eval_metric}: {current_valid_loss:.5f} "
                f"(improvement: {improvement:.5f})"
            )
            runtime_state["best_valid_loss"] = current_valid_loss
            runtime_state["best_valid_score"] = decision.best_score
            runtime_state["best_model"] = deepcopy(current_model)
            runtime_state["should_stop"] = False
        else:
            # No improvement → increment patience
            self._logger.info(
                f"⏳ No improvement - Patience: "
                f"{runtime_state['patience_counter']}/{runtime_state.get('max_patience', 5)}"
            )
            runtime_state["should_stop"] = decision.should_stop

    def _finalize_training(
        self,
        runtime_state: dict[str, Any],
        learning_rate_config: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Finalize the LightGBM training process by selecting the final model,
        clearing caches, and generating optional visualizations.

        Parameters
        ----------
        runtime_state : dict
            Training configuration and runtime state. Expected keys include:
            - "best_model" : best model found (or None if no improvement)
            - "previous_model" : last trained model
            - "best_valid_loss" : lowest validation loss achieved
            - "show_learning_curve" : bool, whether to generate plots
            - "eval_metric" : str or callable, optional
                Metric used for evaluation (default: "logloss").
        learning_rate_config : dict or None, optional
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
        self._logger.info(f"Cleaning cache of {type(self._sample_weight_calculator).__name__}")
        self._sample_weight_calculator.clear_cache()

        # Select final model: best if available, otherwise last trained
        final_model = (
            runtime_state["best_model"]
            if runtime_state["best_model"] is not None
            else runtime_state["previous_model"]
        )

        if final_model is None:
            raise RuntimeError("No model was successfully trained")

        # Save as final model
        self._model = final_model

        # Generate plots if requested
        if runtime_state.get("show_learning_curve"):
            # Learning rate schedule
            if learning_rate_config is not None and self._lr_schedulers:
                self._plot_learning_rate_history("Exponential Decay", self._lr_schedulers)

            # Training/validation loss curves
            self._plot_metric_history(
                self._global_train_loss,
                self._global_valid_loss,
                self._global_iterations,
                "LGBM",
                runtime_state.get("eval_metric", "logloss"),
            )

        # Log training summary
        total_batches = len(self._global_iterations)
        final_valid_loss = runtime_state["best_valid_loss"]
        model_type = "best" if runtime_state["best_model"] is not None else "last"

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
    def _validate_inputs(
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

    def _initialize_runtime_state(
        self,
        runtime_state: Dict[str, Any],
        num_batches: int,
    ) -> None:
        """
        Initialize and update the training state inside the provided runtime_state dictionary.

        This method enriches the training configuration with runtime state information
        required during the batch-wise training process (e.g., early stopping, best model,
        patience counter, etc.). It modifies the dictionary in place and does not return
        any value.

        Parameters
        ----------
        runtime_state : dict
            Training configuration parameters that will also store the runtime training state.
            Keys such as 'max_patience', 'use_sample_weight', and 'verbose' are read if present,
            otherwise defaults are used.
        num_batches : int
            Total number of batches planned for the training session.
        """
        training_config = TrainingConfig.from_mapping(
            {**runtime_state, "num_batches": num_batches}
        )
        runtime_state.update(
            {
                "previous_model": None,  # Model trained on the previous batch (used for warm start)
                "best_model": None,  # Best model found so far (updated when validation improves)
                "should_stop": False,  # Early stopping flag (True if training should end)
                "patience_counter": 0,  # Number of consecutive batches without improvement
                "best_valid_loss": float("inf"),  # Best validation loss achieved so far
                "best_valid_score": None,
                "eval_metric": runtime_state.get(
                    "eval_metric", "binary_logloss"
                ),  # Evaluation metric for monitoring training (default: binary_logloss)
            }
        )
        training_config.apply_to(runtime_state)

    # Common function shared by both LightGBM and XGBoost:
    def _resolve_learning_rate(
        self,
        learning_rate_config: Optional[dict[str, Any]],
        batch_id: int,
        default_lr: float = 0.1,
    ) -> float:
        """
        Compute the learning rate for the current batch using an exponential decay schedule.

        If no scheduler configuration is provided, a default learning rate of 0.1 is used.

        Parameters
        ----------
        learning_rate_config : dict or None
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
        if learning_rate_config is None:
            return default_lr

        # Log the use of exponential decay for the learning rate
        self._logger.info("Computing the current learning rate with exponential decay")

        # Compute the current learning rate using exponential decay:
        # LR = max(initial_lr * (decay_rate ** batch_id), min_lr)
        learning_rate = self._calculate_scheduled_learning_rate(
            initial_lr=learning_rate_config.get("initial_lr", 0.1),  # Default initial LR
            decay_rate=learning_rate_config.get(
                "decay_rate", 0.95
            ),  # Default decay factor
            min_lr=learning_rate_config.get("min_lr", 1e-4),  # Lower bound for LR
            batch_id=batch_id,  # Current batch index
        )

        # Log the computed learning rate for monitoring
        self._logger.info(f"Learning rate for batch {batch_id}: {learning_rate:.6f}")

        # Keep track of learning rates used across batches for later analysis/visualization
        self._lr_schedulers.append(learning_rate)

        # Return the final learning rate for the current batch
        return learning_rate

    def _prepare_batch(
        self,
        pandas_batch: PandasDataFrame,
        target_column: str,
        batch_number: int,
    ) -> Dict[str, Any]:
        """
        Prepare a training batch by extracting features, targets, and sample weights.

        This method:
        - Computes sample weights for the batch based on the target distribution.
        - Converts object-type features to categorical dtype when applicable.
        - Optimizes memory usage of the feature matrix using the class-level optimizer.

        Parameters
        ----------
        pandas_batch : PandasDataFrame
            Batch of training data after preprocessing.
            Must include both input features and the target column.
        target_column : str
            Name of the target column in `pandas_batch`.
        batch_number : int
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
        self._logger.info(f"Calculating sample weights for batch {batch_number}")

        # Compute sample weights based on the distribution of the target values
        # (important for handling class imbalance)
        sample_weight = self._sample_weight_calculator.calculate_sample_weights(
            pandas_batch[target_column]
        )

        # Log the categorical conversion process
        self._logger.info("Converting all features of type 'object' to 'categorical'")

        # Convert columns with dtype "object" into pandas "category"
        # This is useful for models like XGBoost/CatBoost/lightgbm that can leverage categorical features
        pandas_batch, _ = self._convert_categorical_features(
            pandas_batch, target_column
        )

        # Return the batch data in a standardized dictionary format:
        # - Features (optimized using the instance-level PandasMemoryOptimizer)
        # - Target vector
        # - Sample weights
        return {
            "features": self._memory_optimizer.optimize(
                pandas_batch.drop(
                    columns=[target_column]
                )  # Optimized feature matrix
            ),
            "target": pandas_batch[target_column],  # Target vector
            "sample_weight": sample_weight,  # Sample weights for balancing
        }

    def _prepare_validation(
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
        valid_data_processed: PandasDataFrame = self._collect_validation_data(
            valid_dataframe
        )

        # Convert object-type columns to categorical dtype if applicable
        valid_data_processed, cat_is_present = self._convert_categorical_features(
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
        sample_weights_valid = self._sample_weight_calculator.calculate_sample_weights(
            valid_data_processed[target_column]
        )

        # Log that sample weights have been successfully computed
        self._logger.info("Sample weights for validation set computed successfully")

        # Return the processed validation dataset
        return {
            "features": self._memory_optimizer.optimize(
                valid_data_processed.drop(columns=[target_column])
            ),  # Feature matrix optimized
            "target": valid_data_processed[target_column],  # Target vector
            "sample_weight": sample_weights_valid,  # Class weights
        }
