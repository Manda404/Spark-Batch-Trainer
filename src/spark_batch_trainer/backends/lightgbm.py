"""LightGBM batch-trainer backend."""

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
    """Batch trainer for LightGBM classifiers.

    This class supports incremental training in mini-batches on Spark DataFrames,
    with validation support, early stopping, sample weighting, and learning rate scheduling.
    """

    def __init__(self):
        """Initialize per-run state: loss history, LR schedule, trained model."""
        super().__init__()
        self._global_train_loss: List[List[float]] = []
        self._global_valid_loss: List[List[float]] = []
        self._global_iterations: List[int] = []
        self._model: Optional[LGBMClassifier] = None
        self._lr_schedulers: List[float] = []
        self._categorical_features: Optional[List[str]] = None
        self._known_labels: Optional[List[Any]] = None
        self._sample_weight_calculator = BalancedSampleWeightCalculator()
        self._memory_optimizer = PandasMemoryOptimizer(enable_logging=True)

    def fit(
        self,
        train_dataframe: Optional[SparkDataFrame],
        valid_dataframe: Optional[SparkDataFrame],
        target_column: str,
        **kwargs,
    ) -> None:
        """Train a LightGBM classifier incrementally on Spark DataFrames in batches.

        Args:
            train_dataframe: Training dataset.
            valid_dataframe: Validation dataset. Required for metric
                tracking and early stopping.
            target_column: Target column name, present in both DataFrames.
            **kwargs: ``model_config`` (LightGBM hyperparameters),
                ``training_config`` (options such as ``num_batches``,
                ``eval_metric``, ``use_sample_weight``), and an optional
                ``learning_rate_config`` scheduler configuration.

        Returns:
            None: The trained model is stored internally in ``self._model``.

        Raises:
            ValueError: If input DataFrames or target column are invalid.
            RuntimeError: If no model is successfully trained.
        """
        model_config: Dict[str, Any] = deepcopy(
            kwargs.get("model_config", {})
        )
        runtime_state: Dict[str, Any] = deepcopy(
            kwargs.get("training_config", {})
        )
        num_batches = TrainingConfig.from_mapping(runtime_state).num_batches
        learning_rate_config: Optional[Dict[str, Any]] = kwargs.get(
            "learning_rate_config"
        )
        self._reset_run_history()

        self._validate_inputs(
            train_dataframe, valid_dataframe, target_column, num_batches
        )
        self._fit_category_schema(train_dataframe, valid_dataframe, target_column)
        self._fit_known_labels(train_dataframe, valid_dataframe, target_column)

        dataframe_generator = self._iter_training_batches(
            train_dataframe, target_column, num_batches
        )

        validation_data = self._prepare_validation(valid_dataframe, target_column)

        self._logger.info("Setup training state for LightGBM")
        self._initialize_runtime_state(runtime_state, num_batches)

        self._logger.info(f"🚀 Starting LightGBM training with {num_batches} batches")

        batch_id = 0
        try:
            for batch_id, pandas_batch in enumerate(dataframe_generator):
                self._logger.info(
                    f"\n--- 📦 Processing batch {batch_id + 1}/{num_batches} ---"
                )

                current_batch_data = self._prepare_batch(
                    pandas_batch, target_column, batch_id + 1
                )

                learning_rate = self._resolve_learning_rate(
                    learning_rate_config,
                    batch_id + 1,
                    default_lr=float(model_config.get("learning_rate", 0.1)),
                )

                current_model = self._fit_batch(
                    batch_data=current_batch_data,
                    validation_data=validation_data,
                    runtime_state=runtime_state,
                    model_config=model_config,
                    batch_id=batch_id + 1,
                    learning_rate=learning_rate,
                )

                self._evaluate_batch(
                    current_model=current_model,
                    runtime_state=runtime_state,
                    batch_id=batch_id + 1,
                )

                if runtime_state["should_stop"]:
                    self._logger.info("Early stopping triggered - Training completed")
                    break

        except Exception as e:
            self._logger.error(f"Training failed at batch {batch_id + 1}: {str(e)}")
            raise

        self._finalize_training(
            runtime_state=runtime_state,
            learning_rate_config=learning_rate_config,
        )

    def get_trained_model(self) -> Any:
        """Retrieve the final trained model instance.

        Returns:
            Any: Trained model object (e.g. ``LGBMClassifier``).
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
        """Fit LightGBM on one batch, warm-started from the previous batch's model."""
        self._logger.info(f"🏋️ Training LGBM on batch {batch_id}")

        params_batch = {**model_config, "learning_rate": learning_rate}
        self._logger.info(
            "LGBMClassifier configuration",
            extra={
                "batch": batch_id,
                "learning_rate": learning_rate,
                "lgbm_config": params_batch,
            },
        )

        model = LGBMClassifier(**params_batch)

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

        eval_metric = runtime_state.get("eval_metric", "logloss")
        prev_model = runtime_state.get("previous_model")

        model.fit(
            batch_data["features"],
            batch_data["target"],
            eval_set=[
                (batch_data["features"], batch_data["target"]),
                (validation_data["features"], validation_data["target"]),
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

        self._logger.info(f"🎯 Completed LGBM training on batch {batch_id}")
        return model

    def _evaluate_batch(
        self,
        current_model: LGBMClassifier,
        runtime_state: dict[str, Any],
        batch_id: int,
    ) -> None:
        """Score the batch, update best-model/patience state in ``runtime_state``."""
        self._logger.info(f"🔍 Evaluating LightGBM model on batch {batch_id}")

        runtime_state["previous_model"] = deepcopy(current_model)

        train_scores, valid_scores, eval_metric = self._extract_metric_history(
            current_model, "lightgbm"
        )

        if not train_scores or not valid_scores:
            self._logger.warning("⚠️ No training history found for evaluation.")
            return

        self._global_train_loss.append(train_scores)
        self._global_valid_loss.append(valid_scores)
        self._global_iterations.append(batch_id)

        current_train_loss = train_scores[-1]
        current_valid_loss = valid_scores[-1]

        self._logger.info(
            f"Batch {batch_id} - {eval_metric} | "
            f"Train: {current_train_loss:.5f} | "
            f"Valid: {current_valid_loss:.5f}"
        )

        decision = GlobalEarlyStopping.observe(
            current_score=current_valid_loss,
            best_score=runtime_state.get("best_valid_score"),
            metric_name=eval_metric,
            patience_counter=runtime_state["patience_counter"],
            max_patience=runtime_state["max_patience"],
            mode=runtime_state["metric_mode"],
            min_delta=runtime_state["min_delta"],
            logger=self._logger,
        )
        runtime_state["patience_counter"] = decision.patience_counter
        if decision.improved:
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
        """Select the final model, clear caches, and plot diagnostics if requested.

        Raises:
            RuntimeError: If no model was successfully trained.
        """
        self._logger.info(f"Cleaning cache of {type(self._sample_weight_calculator).__name__}")
        self._sample_weight_calculator.clear_cache()

        final_model = (
            runtime_state["best_model"]
            if runtime_state["best_model"] is not None
            else runtime_state["previous_model"]
        )

        if final_model is None:
            raise RuntimeError("No model was successfully trained")

        self._model = final_model

        if runtime_state.get("show_learning_curve"):
            if learning_rate_config is not None and self._lr_schedulers:
                self._plot_learning_rate_history("Exponential Decay", self._lr_schedulers)

            self._plot_metric_history(
                self._global_train_loss,
                self._global_valid_loss,
                self._global_iterations,
                "LGBM",
                runtime_state.get("eval_metric", "logloss"),
            )

        total_batches = len(self._global_iterations)
        final_valid_loss = runtime_state["best_valid_loss"]
        model_type = "best" if runtime_state["best_model"] is not None else "last"

        self._logger.info("LGBM training completed successfully")
        self._logger.info(f"Total batches processed: {total_batches}")
        self._logger.info(f"Best validation loss: {final_valid_loss:.5f}")
        self._logger.info(f"Final model selected: {model_type}")

        if self._categorical_features:
            self._logger.info(
                f"Categorical features used: {len(self._categorical_features)}"
            )

    # Duplicated identically across all three backends; kept per-backend
    # rather than shared to avoid a cross-backend dependency for this check.
    def _validate_inputs(
        self,
        train_dataframe: Optional[SparkDataFrame],
        valid_dataframe: Optional[SparkDataFrame],
        target_column: str,
        num_batches: int,
    ) -> None:
        """Validate input parameters before starting training.

        Raises:
            ValueError: If either DataFrame is ``None``, ``num_batches`` < 1,
                or ``target_column`` is missing from either DataFrame.
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
        """Populate ``runtime_state`` in place with validated config and initial run state."""
        training_config = TrainingConfig.from_mapping(
            {**runtime_state, "num_batches": num_batches}
        )
        runtime_state.update(
            {
                "previous_model": None,
                "best_model": None,
                "should_stop": False,
                "patience_counter": 0,
                "best_valid_loss": float("inf"),
                "best_valid_score": None,
                "eval_metric": runtime_state.get("eval_metric", "binary_logloss"),
            }
        )
        training_config.apply_to(runtime_state)

    # Duplicated identically in XGBoostTrainer; kept per-backend rather than
    # shared to avoid a cross-backend dependency for this small helper.
    def _resolve_learning_rate(
        self,
        learning_rate_config: Optional[dict[str, Any]],
        batch_id: int,
        default_lr: float = 0.1,
    ) -> float:
        """Resolve the batch learning rate from ``learning_rate_config``, or ``default_lr``."""
        if learning_rate_config is None:
            return default_lr

        self._logger.info("Computing the current learning rate with exponential decay")

        learning_rate = self._calculate_scheduled_learning_rate(
            initial_lr=learning_rate_config.get("initial_lr", 0.1),
            decay_rate=learning_rate_config.get("decay_rate", 0.95),
            min_lr=learning_rate_config.get("min_lr", 1e-4),
            batch_id=batch_id,
        )

        self._logger.info(f"Learning rate for batch {batch_id}: {learning_rate:.6f}")
        self._lr_schedulers.append(learning_rate)
        return learning_rate

    def _prepare_batch(
        self,
        pandas_batch: PandasDataFrame,
        target_column: str,
        batch_number: int,
    ) -> Dict[str, Any]:
        """Split a batch into weighted, memory-optimized features/target/weights."""
        self._logger.info(f"Calculating sample weights for batch {batch_number}")
        sample_weight = self._sample_weight_calculator.calculate_sample_weights(
            pandas_batch[target_column], known_labels=self._known_labels
        )

        self._logger.info("Converting all features of type 'object' to 'categorical'")
        pandas_batch, _ = self._convert_categorical_features(
            pandas_batch, target_column
        )

        return {
            "features": self._memory_optimizer.optimize(
                pandas_batch.drop(columns=[target_column])
            ),
            "target": pandas_batch[target_column],
            "sample_weight": sample_weight,
        }

    def _prepare_validation(
        self,
        valid_dataframe: Optional[SparkDataFrame],
        target_column: str,
    ) -> Dict[str, Any]:
        """Collect, categorize, and weight the validation set once before training."""
        self._logger.info("Preparing validation data...")
        valid_data_processed: PandasDataFrame = self._collect_validation_data(
            valid_dataframe
        )

        valid_data_processed, cat_is_present = self._convert_categorical_features(
            valid_data_processed, target_column
        )

        if cat_is_present:
            self._logger.info(
                f"Categorical features detected in validation set: "
                f"{len(self._categorical_features or [])}"
            )

        sample_weights_valid = self._sample_weight_calculator.calculate_sample_weights(
            valid_data_processed[target_column], known_labels=self._known_labels
        )

        return {
            "features": self._memory_optimizer.optimize(
                valid_data_processed.drop(columns=[target_column])
            ),
            "target": valid_data_processed[target_column],
            "sample_weight": sample_weights_valid,
        }
