from copy import deepcopy
from xgboost import XGBClassifier
from typing import Optional, Any, List, Dict
from pandas import DataFrame as PandasDataFrame
from pyspark.sql import DataFrame as SparkDataFrame
from spark_batch_trainer.training.base import BatchTrainer
from spark_batch_trainer.data import PandasMemoryOptimizer
from spark_batch_trainer.data import BalancedSampleWeightCalculator
from spark_batch_trainer.training.config import TrainingConfig
from spark_batch_trainer.training.early_stopping import GlobalEarlyStopping


class XGBoostTrainer(BatchTrainer):
    """
    Batch trainer for XGBoost classifiers.

    This class supports incremental training in mini-batches on Spark DataFrames,
    with validation support, early stopping, sample weighting, and learning rate scheduling.
    """

    def __init__(self):
        """
        Initialize the XGBoost batch trainer.

        Attributes
        ----------
        _global_train_loss : list of list of float
            History of training loss values for each batch.
        _global_valid_loss : list of list of float
            History of validation loss values for each batch.
        _global_iterations : list of int
            Indices of processed batches during training.
        _model : XGBClassifier or None
            The trained XGBoost model instance (set after training).
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
        self._model: Optional[XGBClassifier] = None
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
        Train an XGBoost classifier incrementally on Spark DataFrames in batches.

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
                XGBoost hyperparameters. ``learning_rate`` may be updated batch-wise.
            - runtime_state : dict
                Training options such as ``num_batches`` (int), ``eval_metric`` (str),
                ``use_sample_weight`` (bool), ``max_patience`` (int).
            - config_lr_scheduler : dict or None
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
        - Supports warm-restart training across batches.
        - Learning rate can be scheduled dynamically per batch.
        """
        # Extract model, training, and learning rate scheduler configurations
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
        self._logger.info("Validating input parameters")
        self._validate_inputs(
            train_dataframe, valid_dataframe, target_column, num_batches
        )

        # Preprocess training data and split into batches (generator)
        self._logger.info("Preparing training data")
        dataframe_generator = self._iter_training_batches(
            train_dataframe, target_column, num_batches
        )

        # Prepare validation dataset (converted to pandas + preprocessing) and return dict (features, target, sample weights)
        self._logger.info("Preparing validation data")
        validation_data = self._prepare_validation(
            validation_data=valid_dataframe,
            target_column=target_column,
            model_config=model_config,
        )

        # Initialize training state (best model, patience, eval metric, etc.)
        self._logger.info("Setup training state for XGBoost")
        self._initialize_runtime_state(
            runtime_state=runtime_state,
            num_batches=num_batches,
        )

        # Log the start of training
        self._logger.info(f"🚀 Starting XGBoost training with {num_batches} batches")

        # --- Main training loop ---
        try:
            for batch_id, pandas_batch in enumerate(dataframe_generator):
                self._logger.info(
                    f"\n--- 📦 Processing batch {batch_id + 1}/{num_batches} ---"
                )

                # Prepare current batch (features, target, sample weights)
                current_batch_data = self._prepare_batch(
                    pandas_batch,
                    target_column,
                    batch_id + 1,
                )

                # Retrieve the current learning rate from the scheduler
                self._logger.info("Updating XGBoost learning rate for current batch.")
                learning_rate = self._resolve_learning_rate(
                    learning_rate_config,
                    batch_id + 1,
                    default_lr=float(model_config.get("learning_rate", 0.1)),
                )

                # Create the first estimator or update its batch-specific parameters.
                runtime_state["model"] = self._create_or_update_model(
                    runtime_state["model"], model_config, learning_rate, batch_id + 1
                )

                # Train the model on the current batch with validation set
                self._fit_batch(
                    model=runtime_state["model"],
                    previous_model=runtime_state["previous_model"],
                    batch_data=current_batch_data,
                    validation_data=validation_data,
                    runtime_state=runtime_state,
                    batch_id=batch_id + 1,
                )

                # Evaluate the trained model and update best model if improved
                should_stop = self._evaluate_batch(
                    model=runtime_state["model"],
                    runtime_state=runtime_state,
                    batch_id=batch_id + 1,
                )

                # Check early stopping condition
                if should_stop:
                    self._logger.info("Early stopping triggered - Training completed")
                    break

        except Exception as e:
            # Log and propagate any exception during training
            self._logger.error(f"Training failed at batch {batch_id + 1}: {str(e)}")
            raise

        # Finalize training: select best model, log results, and generate diagnostics
        self._finalize_training(
            runtime_state=runtime_state,
            model_config=model_config,
            learning_rate_config=learning_rate_config,
        )

    def get_trained_model(self) -> Any:
        """
        Retrieve the final trained model instance.

        Returns
        -------
        Any
            Trained model object (e.g. ``XGBClassifier``).
        """
        return self._model

    # Helper function required for the proper execution of the fit method
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
                "model": None,  # Current XGBoost model being trained
                "should_stop": False,  # Early stopping flag (True if training should end)
                "patience_counter": 0,  # Number of consecutive batches without improvement
                "best_valid_loss": float("inf"),  # Best validation loss achieved so far
                "best_valid_score": None,
            }
        )
        training_config.apply_to(runtime_state)

    def _fit_batch(
        self,
        model: XGBClassifier,
        previous_model: Optional[XGBClassifier],
        batch_data: Dict[str, Any],
        validation_data: Dict[str, Any],
        runtime_state: Dict[str, Any],
        batch_id: int,
    ) -> None:
        """
        Train an XGBoost model on the current batch, with optional warm restart
        and support for sample weights.

        Parameters
        ----------
        model : XGBClassifier
            Booster instance to be trained on the current batch.
        previous_model : XGBClassifier or None
            Model produced by the preceding batch and used for continuation.
        batch_data : dict
            Dictionary containing training data:
            - "features" : pandas.DataFrame
            - "target" : pandas.Series
            - "sample_weight" : array-like (optional, required if `use_sample_weight=True`)
        validation_data : dict
            Dictionary containing validation data:
            - "features" : pandas.DataFrame
            - "target" : pandas.Series
            - "sample_weight" : array-like (optional, required if `use_sample_weight=True`)
        runtime_state : dict
            Training configuration and runtime state, including:
            - "use_sample_weight" : bool
            - "verbose" : bool
        batch_id : int
            Index of the current batch (1-based).

        Returns
        -------
        None

        Notes
        -----
        - If ``previous_model`` is provided and ``batch_id > 1``, training
          continues from the model produced by the preceding batch.
        - When ``sample_weight_eval_set`` is passed:
            - The first array corresponds to the training set (``validation_0``),
            - The second array corresponds to the validation set (``validation_1``).
        This ensures evaluation metrics (e.g., logloss, AUC) are computed while
        accounting for class imbalance.
        """
        self._logger.info(f"🏋️ Training XGBoost on batch {batch_id}")

        # Warm restart: continue training from the best model if available
        xgb_model_param = None
        if batch_id > 1 and previous_model is not None:
            xgb_model_param = previous_model.get_booster()
            self._logger.info("Continuing training from the previous batch model")

        # Train model on current batch
        model.fit(
            batch_data["features"],
            batch_data["target"],
            eval_set=[
                (
                    batch_data["features"],
                    batch_data["target"],
                ),  # Train set for monitoring
                (
                    validation_data["features"],
                    validation_data["target"],
                ),  # Validation set for monitoring
            ],
            xgb_model=xgb_model_param,  # Continue training from previous best model if set
            sample_weight=batch_data["sample_weight"]
            if runtime_state.get("use_sample_weight")
            else None,  # Apply sample weights to training set if enabled
            verbose=runtime_state.get("verbose"),  # XGBoost verbosity
            sample_weight_eval_set=[
                batch_data["sample_weight"],
                validation_data["sample_weight"],
            ],  # Apply sample weights to both train and validation sets
        )

        self._logger.info(f"🎯 Completed XGBoost training on batch {batch_id}")

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

    def _create_or_update_model(
        self,
        current_model: Optional[XGBClassifier],
        model_config: Dict[str, Any],
        learning_rate: float,
        batch_id: int,
    ) -> XGBClassifier:
        """
        Initialize or update the XGBoost model.

        Parameters
        ----------
        current_model : XGBClassifier or None
            Existing model, or ``None`` for the first batch.
        model_config : dict
            XGBoost hyperparameters.
        learning_rate : float
            Learning rate for the current batch.
        batch_id : int
            One-based batch number.

        Returns
        -------
        XGBClassifier
            Updated or newly initialized model.
        """
        if current_model is None:
            # First initialization of the XGBoost classifier with the given configuration
            model_config["learning_rate"] = learning_rate
            self._logger.info(
                f"[Batch {batch_id}] Initializing XGBoost model with configuration parameters."
            )
            model = XGBClassifier(**model_config)
            return model
        else:
            # Update parameters of the existing model
            # - Dynamically adjust learning rate
            # - Update n_estimators if provided in the config, otherwise keep current
            current_model.set_params(
                learning_rate=learning_rate,
                n_estimators=model_config.get(
                    "n_estimators", current_model.n_estimators
                ),
            )
            return current_model

    def _finalize_training(
        self,
        runtime_state: Dict[str, Any],
        model_config: Dict[str, Any],
        learning_rate_config: Optional[Dict[str, Any]],
    ) -> None:
        """
        Finalize the XGBoost training process by selecting the final model,
        clearing caches, and generating optional visualizations.

        Parameters
        ----------
        runtime_state : dict
            Training configuration and runtime state. Expected keys include:
            - "best_model" : best model found (or None if no improvement)
            - "previous_model" : last trained model
            - "best_valid_loss" : lowest validation loss achieved
            - "show_learning_curve" : bool, whether to plot curves
        model_config : dict
            Model hyperparameters, used in particular for labeling plots.
            Expected key:
            - "eval_metric" : str or callable, optional (default: "logloss")
        learning_rate_config : dict or None
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

        # Select the final model: prefer best model if available, otherwise last
        final_model = (
            runtime_state["best_model"]
            if runtime_state["best_model"] is not None
            else runtime_state["previous_model"]
        )

        if final_model is None:
            raise RuntimeError("No model was successfully trained")

        # Store the selected model internally
        self._model = final_model

        # Optionally plot learning curves and LR schedule
        if runtime_state.get("show_learning_curve"):
            # Plot LR schedule if configured
            if learning_rate_config is not None and self._lr_schedulers:
                self._plot_learning_rate_history("Exponential Decay", self._lr_schedulers)

            # Plot train/validation losses across batches
            self._plot_metric_history(
                self._global_train_loss,
                self._global_valid_loss,
                self._global_iterations,
                "XGBoost",
                model_config.get("eval_metric", "logloss"),
            )

        # Log summary of the training session
        total_batches = len(self._global_iterations)
        final_valid_loss = runtime_state["best_valid_loss"]
        model_type = "best" if runtime_state["best_model"] is not None else "last"

        self._logger.info("XGBoost training completed")
        self._logger.info(f"Total batches processed: {total_batches}")
        self._logger.info(f"Best validation loss: {final_valid_loss:.5f}")
        self._logger.info(f"Final model selected: {model_type}")

        # Log categorical features if any were used
        if self._categorical_features:
            self._logger.info(
                f"Categorical features used: {len(self._categorical_features)}"
            )

    def _evaluate_batch(
        self,
        model: XGBClassifier,
        runtime_state: Dict[str, Any],
        batch_id: int,
    ) -> bool:
        """
        Evaluate the trained model on the current batch and update the training state.

        Parameters
        ----------
        model : XGBClassifier
            The model trained on the current batch.
        runtime_state : dict
            Training configuration and runtime state. Updated in place with:
            - "previous_model" : last trained model
            - "best_model" : best model found so far
            - "best_valid_loss" : lowest validation loss observed
            - "patience_counter" : consecutive non-improving batches
        batch_id : int
            Index of the current batch (1-based).

        Returns
        -------
        bool
            True if training should stop early (patience exhausted), False otherwise.

        Notes
        -----
        - Tracks training and validation metrics over time for global monitoring.
        - Updates the best model when validation loss improves.
        - Implements early stopping by comparing validation loss across batches.
        """
        self._logger.info(f"🔍 Evaluating model on batch {batch_id}")

        # Retrieve evaluation metrics from model training history
        train_scores, valid_scores, eval_metric = self._extract_metric_history(
            model, "xgboost"
        )

        # Take the last available score (most recent iteration)
        current_train_loss = train_scores[-1]
        current_valid_loss = valid_scores[-1]

        # Store metrics for global tracking/visualization
        self._global_train_loss.append(train_scores)
        self._global_valid_loss.append(valid_scores)
        self._global_iterations.append(batch_id)

        # Log results for this batch
        self._logger.info(
            f"Batch {batch_id} - {eval_metric} | "
            f"Train: {current_train_loss:.5f} | "
            f"Valid: {current_valid_loss:.5f}"
        )

        # Keep the current model as "previous_model"
        runtime_state["previous_model"] = deepcopy(model)

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
                f"🎉 New best model - {eval_metric}: {current_valid_loss:.5f} "
                f"(improvement: {improvement:.5f})"
            )
            runtime_state["best_valid_loss"] = current_valid_loss
            runtime_state["best_valid_score"] = decision.best_score
            runtime_state["best_model"] = deepcopy(model)
            return False
        else:
            # No improvement → increment patience counter
            self._logger.info(
                f"⏳ No improvement - Patience: "
                f"{runtime_state['patience_counter']}/{runtime_state['max_patience']}"
            )
            # Stop if patience exhausted
            return decision.should_stop

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
        validation_data: Optional[SparkDataFrame],
        target_column: str,
        model_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Prepare and preprocess the validation dataset for model training.

        This method:
        - Converts the input Spark DataFrame into a pandas DataFrame.
        - Applies preprocessing steps (e.g., optimization).
        - Detects categorical features and converts them to the proper categorical dtype.
        - Updates the model configuration to enable categorical support if needed.
        - Computes sample/class weights based on the target distribution.

        Parameters
        ----------
        validation_data : SparkDataFrame or None
            Validation dataset in Spark format. If None, no validation data is prepared.
        target_column : str
            Name of the target column in the dataset.
        model_config : dict
            Dictionary containing model configuration parameters. This dictionary may
            be updated in-place to enable categorical support.

        Returns
        -------
        dict
            Dictionary containing the processed validation data:
            - "features" (pandas.DataFrame): Optimized feature matrix.
            - "target" (pandas.Series): Target vector.
            - "sample_weight" (numpy.ndarray): Computed class/sample weights.
        """
        # Convert Spark DataFrame to pandas and apply preprocessing if required
        valid_data_processed: PandasDataFrame = self._collect_validation_data(
            validation_data
        )

        # Convert object-type columns to categorical dtype (excluding target column)
        valid_data_processed, cat_is_present = self._convert_categorical_features(
            valid_data_processed, target_column
        )

        # If categorical features are detected, enable categorical support in XGBoost
        if cat_is_present:
            self._logger.info(
                "Categorical features detected - Enabling categorical support for XGBoost model"
            )
            model_config["enable_categorical"] = True

        # Compute sample weights based on the target distribution
        sample_weights_valid = self._sample_weight_calculator.calculate_sample_weights(
            valid_data_processed[target_column]
        )

        # Return the processed validation dataset
        return {
            "features": self._memory_optimizer.optimize(
                valid_data_processed.drop(columns=[target_column])
            ),  # Feature matrix optimized
            "target": valid_data_processed[target_column],  # Target vector
            "sample_weight": sample_weights_valid,  # Class weights
        }
