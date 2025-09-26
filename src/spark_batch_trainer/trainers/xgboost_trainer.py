from copy import deepcopy
from xgboost import XGBClassifier
from typing import Optional, Any, List, Dict
from pandas import DataFrame as PandasDataFrame
from pyspark.sql import DataFrame as SparkDataFrame
from spark_batch_trainer.core.base_trainer import BatchTrainer
from spark_batch_trainer.core.memory_optimizer import MemoryOptimizer
from spark_batch_trainer.core.class_weight_optimizer import OptimizedWeightCalculator


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
        self._model: Optional[XGBClassifier] = None
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

            - config_model : dict
                XGBoost hyperparameters. ``learning_rate`` may be updated batch-wise.
            - config_training : dict
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
        config_model: Dict[str, Any] = deepcopy(kwargs.get("config_model", {}))
        config_training: Dict[str, Any] = deepcopy(kwargs.get("config_training", {}))
        num_batches: int = config_training.get("num_batches", 10)
        lr_scheduler_config: Optional[Dict[str, Any]] = kwargs.get(
            "config_lr_scheduler", None
        )

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

        # Prepare validation dataset (converted to pandas + preprocessing) and return dict (features, target, sample weights)
        self._logger.info("Preparing validation data")
        valid_data = self._prepare_validation_data(
            valid_data=valid_dataframe,
            target_column=target_column,
            config_model=config_model,
        )

        # Initialize training state (best model, patience, eval metric, etc.)
        self._logger.info("Setup training state for XGBoost")
        self._setup_training_state(
            config_training=config_training,
            num_batches=num_batches,
        )

        # Log the start of training
        self._logger.info(f"🚀 Starting XGBoost training with {num_batches} batches")

        # --- Main training loop ---
        try:
            for batch_id, processed_batch in enumerate(dataframe_generator):
                self._logger.info(
                    f"\n--- 📦 Processing batch {batch_id + 1}/{num_batches} ---"
                )

                # Prepare current batch (features, target, sample weights)
                current_batch_data = self._prepare_batch_data(
                    processed_batch,
                    target_column,
                    batch_id + 1,
                )

                # Retrieve the current learning rate from the scheduler
                self._logger.info("Updating XGBoost learning rate for current batch.")
                current_lr = self._get_current_learning_rate(
                    lr_scheduler_config, batch_id + 1
                )

                # Update or initialize the model (booster)
                config_training["booster"] = self._update_model(
                    config_training["booster"], config_model, current_lr, batch_id + 1
                )

                # Train the booster on the current batch with validation set
                self._train_batch(
                    booster=config_training["booster"],
                    best_model=config_training["best_model"],
                    batch_data=current_batch_data,
                    valid_data=valid_data,
                    config_training=config_training,
                    batch_id=batch_id + 1,
                )

                # Evaluate the trained booster and update best model if improved
                should_stop = self._evaluate_trained_model(
                    booster=config_training["booster"],
                    config_training=config_training,
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
        self._wrap_up_training(
            config_training=config_training,
            config_model=config_model,
            lr_scheduler_config=lr_scheduler_config,
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
                "booster": None,  # Current XGBoost booster being trained
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
                # Controls verbosity of XGBoost training logs
                "show_learning_curve": config_training.get("show_learning_curve", True), # Afficher la courbe d’apprentissage
            }
        )

    def _train_batch(
        self,
        booster: XGBClassifier,
        best_model: XGBClassifier,
        batch_data: Dict[str, Any],
        valid_data: Dict[str, Any],
        config_training: Dict[str, Any],
        batch_id: int,
    ) -> None:
        """
        Train an XGBoost booster on the current batch, with optional warm restart
        and support for sample weights.

        Parameters
        ----------
        booster : XGBClassifier
            Booster instance to be trained on the current batch.
        best_model : XGBClassifier
            Best model so far. If provided (and not the first batch), its booster
            is used to perform a warm restart.
        batch_data : dict
            Dictionary containing training data:
            - "features" : pandas.DataFrame
            - "target" : pandas.Series
            - "sample_weight" : array-like (optional, required if `use_sample_weight=True`)
        valid_data : dict
            Dictionary containing validation data:
            - "features" : pandas.DataFrame
            - "target" : pandas.Series
            - "sample_weight" : array-like (optional, required if `use_sample_weight=True`)
        config_training : dict
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
        - If ``best_model`` is provided and ``batch_id > 1``, warm restart training
        is performed by continuing from the previous best booster.
        - When ``sample_weight_eval_set`` is passed:
            - The first array corresponds to the training set (``validation_0``),
            - The second array corresponds to the validation set (``validation_1``).
        This ensures evaluation metrics (e.g., logloss, AUC) are computed while
        accounting for class imbalance.
        """
        self._logger.info(f"🏋️ Training XGBoost on batch {batch_id}")

        # Warm restart: continue training from the best booster if available
        xgb_model_param = None
        if batch_id > 1 and best_model is not None:
            xgb_model_param = best_model.get_booster()
            self._logger.info("Using warm restart from best model")

        # Train booster on current batch
        booster.fit(
            batch_data["features"],
            batch_data["target"],
            eval_set=[
                (
                    batch_data["features"],
                    batch_data["target"],
                ),  # Train set for monitoring
                (
                    valid_data["features"],
                    valid_data["target"],
                ),  # Validation set for monitoring
            ],
            xgb_model=xgb_model_param,  # Continue training from previous best booster if set
            sample_weight=batch_data["sample_weight"]
            if config_training.get("use_sample_weight")
            else None,  # Apply sample weights to training set if enabled
            verbose=config_training.get("verbose"),  # XGBoost verbosity
            sample_weight_eval_set=[
                batch_data["sample_weight"],
                valid_data["sample_weight"],
            ],  # Apply sample weights to both train and validation sets
        )

        self._logger.info(f"🎯 Completed XGBoost training on batch {batch_id}")

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

    def _update_model(
        self,
        current_booster: Optional[XGBClassifier],
        config_model: Dict[str, Any],
        current_lr: float,
        batch_id: int,
    ) -> XGBClassifier:
        """
        Initialize or update the XGBoost booster.

        Parameters
        ----------
        current_booster : XGBClassifier or None
            Existing booster model or None if this is the first batch.
        config_model : dict
            XGBoost hyperparameters.
        current_lr : float
            Learning rate for the current batch.
        batch_id : int
            Batch index.

        Returns
        -------
        XGBClassifier
            Updated or newly initialized booster.
        """
        if current_booster is None:
            # First initialization of the XGBoost classifier with the given configuration
            config_model["learning_rate"] = current_lr
            self._logger.info(
                f"[Batch {batch_id + 1}] Initializing XGBoost model with configuration parameters."
            )
            booster = XGBClassifier(**config_model)
            return booster
        else:
            # Update parameters of the existing booster
            # - Dynamically adjust learning rate
            # - Update n_estimators if provided in the config, otherwise keep current
            current_booster.set_params(
                learning_rate=current_lr,
                n_estimators=config_model.get(
                    "n_estimators", current_booster.n_estimators
                ),
            )
            return current_booster

    def _wrap_up_training(
        self,
        config_training: Dict[str, Any],
        config_model: Dict[str, Any],
        lr_scheduler_config: Optional[Dict[str, Any]],
    ) -> None:
        """
        Finalize the XGBoost training process by selecting the final model,
        clearing caches, and generating optional visualizations.

        Parameters
        ----------
        config_training : dict
            Training configuration and runtime state. Expected keys include:
            - "best_model" : best booster found (or None if no improvement)
            - "previous_model" : last trained booster
            - "best_valid_loss" : lowest validation loss achieved
            - "show_learning_curve" : bool, whether to plot curves
        config_model : dict
            Model hyperparameters, used in particular for labeling plots.
            Expected key:
            - "eval_metric" : str or callable, optional (default: "logloss")
        lr_scheduler_config : dict or None
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

        # Select the final model: prefer best model if available, otherwise last
        final_model = (
            config_training["best_model"]
            if config_training["best_model"] is not None
            else config_training["previous_model"]
        )

        if final_model is None:
            raise RuntimeError("No model was successfully trained")

        # Store the selected model internally
        self._model = final_model

        # Optionally plot learning curves and LR schedule
        if config_training.get("show_learning_curve"):
            # Plot LR schedule if configured
            if lr_scheduler_config is not None and self._lr_schedulers:
                self._plot_lr_schedule("Exponential Decay", self._lr_schedulers)

            # Plot train/validation losses across batches
            self._plot_learning_curve(
                self._global_train_loss,
                self._global_valid_loss,
                self._global_iterations,
                "XGBoost",
                config_model.get("eval_metric", "logloss"),
            )

        # Log summary of the training session
        total_batches = len(self._global_iterations)
        final_valid_loss = config_training["best_valid_loss"]
        model_type = "best" if config_training["best_model"] is not None else "last"

        self._logger.info("XGBoost training completed")
        self._logger.info(f"Total batches processed: {total_batches}")
        self._logger.info(f"Best validation loss: {final_valid_loss:.5f}")
        self._logger.info(f"Final model selected: {model_type}")

        # Log categorical features if any were used
        if self._categorical_features:
            self._logger.info(
                f"Categorical features used: {len(self._categorical_features)}"
            )

    def _evaluate_trained_model(
        self,
        booster: XGBClassifier,
        config_training: Dict[str, Any],
        batch_id: int,
    ) -> bool:
        """
        Evaluate the trained booster on the current batch and update the training state.

        Parameters
        ----------
        booster : XGBClassifier
            The model trained on the current batch.
        config_training : dict
            Training configuration and runtime state. Updated in place with:
            - "previous_model" : last trained booster
            - "best_model" : best booster found so far
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

        # Retrieve evaluation metrics from booster training history
        train_scores, valid_scores, eval_metric = self._get_training_metrics(
            booster, "xgboost"
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

        # Keep the current booster as "previous_model"
        config_training["previous_model"] = deepcopy(booster)

        # --- Early stopping logic ---
        if current_valid_loss < config_training["best_valid_loss"]:
            # Improvement → update best model and reset patience
            improvement = config_training["best_valid_loss"] - current_valid_loss
            self._logger.info(
                f"🎉 New best model - {eval_metric}: {current_valid_loss:.5f} "
                f"(improvement: {improvement:.5f})"
            )
            config_training["best_valid_loss"] = current_valid_loss
            config_training["patience_counter"] = 0
            config_training["best_model"] = deepcopy(booster)
            return False
        else:
            # No improvement → increment patience counter
            config_training["patience_counter"] += 1
            self._logger.info(
                f"⏳ No improvement - Patience: "
                f"{config_training['patience_counter']}/{config_training['max_patience']}"
            )
            # Stop if patience exhausted
            return (
                config_training["patience_counter"] >= config_training["max_patience"]
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
        valid_data: Optional[SparkDataFrame],
        target_column: str,
        config_model: Dict[str, Any],
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
        valid_data : SparkDataFrame or None
            Validation dataset in Spark format. If None, no validation data is prepared.
        target_column : str
            Name of the target column in the dataset.
        config_model : dict
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
        valid_data_processed: PandasDataFrame = self._apply_pandas_processing(
            valid_data
        )

        # Convert object-type columns to categorical dtype (excluding target column)
        valid_data_processed, cat_is_present = self._convert_object_to_category_dtype(
            valid_data_processed, target_column
        )

        # If categorical features are detected, enable categorical support in XGBoost
        if cat_is_present:
            self._logger.info(
                "Categorical features detected - Enabling categorical support for XGBoost model"
            )
            config_model["enable_categorical"] = True

        # Compute sample weights based on the target distribution
        sample_weights_valid = self._weight_calculator.calculate_sample_weights(
            valid_data_processed[target_column]
        )

        # Return the processed validation dataset
        return {
            "features": self._optimizer.optimize(
                valid_data_processed.drop(columns=[target_column])
            ),  # Feature matrix optimized
            "target": valid_data_processed[target_column],  # Target vector
            "sample_weight": sample_weights_valid,  # Class weights
        }
