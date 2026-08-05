"""Dataset loading helpers used by the example notebooks."""

from pathlib import Path
from typing import Union

from pandas import DataFrame as PandasDataFrame
from pandas import read_csv

PathLike = Union[str, Path]


def _find_project_root(start: PathLike = ".") -> Path:
    """Return the nearest parent directory containing ``pyproject.toml``.

    Parameters
    ----------
    start:
        File or directory from which to begin the upward search.

    Raises
    ------
    FileNotFoundError
        If no project configuration can be found.
    """
    resolved = Path(start).expanduser().resolve()
    current = resolved.parent if resolved.is_file() else resolved
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError(f"Could not locate pyproject.toml from {resolved}.")


def get_dataset_path(data_dir: PathLike = "data") -> Path:
    """Return the only CSV file contained in a project-relative directory."""
    directory = Path(data_dir)
    if not directory.is_absolute():
        directory = _find_project_root() / directory

    csv_files = sorted(directory.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found in {directory}.")
    if len(csv_files) > 1:
        raise ValueError(
            f"Expected one CSV file in {directory}, found {len(csv_files)}."
        )
    return csv_files[0]


def load_dataset(dataset_path: PathLike) -> PandasDataFrame:
    """Load a CSV dataset from an absolute or project-relative path."""
    path = Path(dataset_path).expanduser()
    if not path.is_absolute():
        path = _find_project_root() / path
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return read_csv(path)
