import subprocess
import sys


def test_training_import_does_not_load_model_backends() -> None:
    code = (
        "import sys; import spark_batch_trainer.training; "
        "assert not {'xgboost', 'catboost', 'lightgbm'} & set(sys.modules)"
    )

    subprocess.run([sys.executable, "-c", code], check=True)
