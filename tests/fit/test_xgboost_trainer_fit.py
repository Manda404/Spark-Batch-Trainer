import pytest
from pandas import DataFrame
from tests.processing.label_encoder import SparkLabelEncoder
from spark_batch_trainer import XGBoostTrainer
from tests.data._data_mocks import (
    build_mock_obesity_df,
    build_mock_diabetes_df,
    stratified_split_sparkdf,
)


def convert_object_to_category(df: DataFrame) -> DataFrame:
    """
    Convert all object-type columns in a DataFrame to categorical dtype.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with object columns converted to category.
    """
    object_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if object_cols:
        print(f"Converting object columns to category: {object_cols}")
        for col in object_cols:
            df[col] = df[col].astype("category")
    return df


def predict_on_sample(model, df, target_col: str, n: int = 5):
    """
    Take a small sample, drop the target column, and return predictions.
    """
    sample = convert_object_to_category(df.limit(n).toPandas())
    preds = model.predict(sample.drop(columns=[target_col]))
    assert len(preds) == len(sample), "Predictions should match sample size"
    return preds


# ---- Fixtures: datasets ----
@pytest.fixture
def mock_data_obesity(spark):
    """Build a mock dataset for obesity (multi-class classification)."""
    df = build_mock_obesity_df(spark, n=200, seed=42)
    train_df, valid_df = stratified_split_sparkdf(
        df, target_col="NObeyesdad", valid_size=0.2, seed=42
    )
    return train_df, valid_df, "NObeyesdad"


@pytest.fixture
def mock_data_diabetes(spark):
    """Build a mock dataset for diabetes (binary classification)."""
    df = build_mock_diabetes_df(spark, n=200, seed=42)
    train_df, valid_df = stratified_split_sparkdf(
        df, target_col="diabetes", valid_size=0.2, seed=42
    )
    return train_df, valid_df, "diabetes"


# ---- Fixtures: configurations ----
@pytest.fixture
def config_training():
    """Provide default training configuration."""
    return {"num_batches": 2, "max_patience": 1, "show_learning_curve": False}


@pytest.fixture
def config_model_binary():
    """Provide XGBoost configuration for binary classification."""
    return {"objective": "binary:logistic", "n_estimators": 10, "max_depth": 3}


@pytest.fixture
def config_model_multiclass():
    """Provide base XGBoost configuration for multi-class classification."""
    # num_class must be set dynamically inside the test (depends on dataset)
    return {"objective": "multi:softmax", "n_estimators": 10, "max_depth": 3}


# ---- Tests ----
def test_xgboost_fit_runs_obesity(mock_data_obesity, config_training, config_model_multiclass):
    """Test that XGBoostTrainer.fit works correctly on the Obesity dataset."""
    train_df, valid_df, target_col = mock_data_obesity

    # Encode the target (replace directly)
    encoder = SparkLabelEncoder().fit(train_df, target_col)
    train_df_encoded = encoder.transform(train_df, target_col)
    valid_df_encoded = encoder.transform(valid_df, target_col)

    trainer = XGBoostTrainer()

    # Update num_class dynamically
    config_model_multiclass["num_class"] = len(encoder.get_classes())

    trainer.fit(
        train_dataframe=train_df_encoded,
        valid_dataframe=valid_df_encoded,
        target_column=target_col,
        training_config=config_training,
        model_config=config_model_multiclass,
    )

    model = trainer.get_trained_model()
    assert model is not None, "Trainer should return a trained XGBoost model"
    assert hasattr(model, "predict"), "Model should have a predict method"

    # Predict on a small sample (valid set already encoded)
    predict_on_sample(model, valid_df_encoded, target_col)


def test_xgboost_fit_runs_diabetes(mock_data_diabetes, config_training, config_model_binary):
    """Test that XGBoostTrainer.fit works correctly on the Diabetes dataset."""
    train_df, valid_df, target_col = mock_data_diabetes

    trainer = XGBoostTrainer()
    trainer.fit(
        train_dataframe=train_df,
        valid_dataframe=valid_df,
        target_column=target_col,
        training_config=config_training,
        model_config=config_model_binary,
    )

    model = trainer.get_trained_model()
    assert model is not None, "Trainer should return a trained XGBoost model"
    assert hasattr(model, "predict"), "Model should have a predict method"

    # Predict on a small sample (use diabetes valid_df, not obesity!)
    predict_on_sample(model, valid_df, target_col)
