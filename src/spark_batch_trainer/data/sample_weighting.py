from threading import Lock
from typing import Optional, Sequence, Tuple, Dict

from numpy import (
    ndarray,
    float64,
    isnan,
    floating,
    bincount,
    array,
    asarray,
    unique,
    fromiter,
    issubdtype,
)

KeyType = Tuple[bytes, bytes, float, bool]


class BalancedSampleWeightCalculator:
    """
    Compute class-balanced sample weights with caching.

    This class implements an optimized version of balanced weights:

    .. math::

        sample_weights(c) = \\frac{sample_count}{class_count \\cdot n_c}

    where:
        - ``sample_count`` is the total number of samples,
        - ``class_count`` is the number of classes,
        - ``n_c`` is the number of samples in class ``c``.

    Features
    --------
    - Supports additive smoothing to avoid zero-count divisions.
    - Normalizes weights so that the mean equals 1 (optional).
    - Uses a cache for previously computed class weights to speed up repeated calls.
    - Ensures consistency across batches when ``known_labels`` is provided.

    Attributes
    ----------
    _cache : dict
        Internal cache mapping keys to precomputed class weights.
    _lock : threading.Lock
        Thread lock to ensure thread-safe access to the cache.
    """

    __slots__ = ("_cache", "_lock")

    def __init__(self) -> None:
        """
        Initialize the weight calculator.

        Initializes an empty cache and a thread lock to ensure thread-safe
        operations when computing or retrieving cached class weights.
        """
        self._cache: Dict[KeyType, ndarray] = {}
        self._lock = Lock()

    @staticmethod
    def _as_1d(y) -> ndarray:
        """
        Convert input array-like to a 1D numpy array.

        Parameters
        ----------
        y : array-like
            Input labels.

        Returns
        -------
        ndarray
            Flattened 1D numpy array of labels.
        """
        y = asarray(y)
        if y.ndim != 1:
            y = y.ravel()
        return y

    @staticmethod
    def _validate(y: ndarray) -> None:
        """
        Validate the input labels array.

        Parameters
        ----------
        y : ndarray
            Input labels array.

        Raises
        ------
        ValueError
            If ``y`` contains NaN values.
        """
        if y.size == 0:
            return
        if issubdtype(y.dtype, floating) and isnan(y).any():
            raise ValueError("target values must not contain NaN")

    @staticmethod
    def _key_from_arrays(
        classes: ndarray,
        counts: ndarray,
        smoothing: float,
        normalize: bool,
    ) -> KeyType:
        """
        Build a compact and stable cache key from classes and counts.

        Parameters
        ----------
        classes : ndarray
            Unique class labels.
        counts : ndarray
            Class counts (after smoothing).
        smoothing : float
            Additive smoothing applied to counts.
        normalize : bool
            Whether normalization is applied.

        Returns
        -------
        tuple of (bytes, bytes, float, bool)
            A tuple uniquely identifying the weight configuration.
        """
        kind = classes.dtype.kind  # 'i','u','f','U','S','O',...
        if kind in ("i", "u"):
            classes_bytes = asarray(classes, dtype="int64").tobytes()
        elif kind == "f":
            classes_bytes = asarray(classes, dtype=float64).tobytes()
        else:  # Text and object labels.
            classes_bytes = asarray(classes, dtype="U").tobytes()

        counts_bytes = counts.astype(float64, copy=False).tobytes()
        return (classes_bytes, counts_bytes, float(smoothing), bool(normalize))

    def calculate_sample_weights(
        self,
        target_values,
        known_labels: Optional[Sequence] = None,
        smoothing: float = 0.0,
        normalize: bool = True,
    ) -> ndarray:
        """
        Compute balanced sample weights for a training batch.

        Parameters
        ----------
        target_values : array-like
            Labels of the current batch.
        known_labels : sequence of shape (n_classes,), optional
            Complete set of possible labels. Ensures batch-to-batch consistency
            and allows normalization across all classes.
        smoothing : float, default=0.0
            Additive smoothing applied to class counts.
        normalize : bool, default=True
            Whether to normalize weights so that their mean equals 1.

        Returns
        -------
        ndarray of shape (n_samples,)
            Array of sample weights for the batch.

        Raises
        ------
        ValueError
            If unknown labels are found (when ``known_labels`` is provided).
        """
        target_array = self._as_1d(target_values)
        self._validate(target_array)

        if target_array.size == 0:
            return array([], dtype=float64)

        # Build the label space and one class index per sample.
        if known_labels is None:
            classes, class_indices = unique(target_array, return_inverse=True)
        else:
            classes = asarray(known_labels)
            label_to_index = {label: index for index, label in enumerate(classes)}
            try:
                class_indices = fromiter(
                    (label_to_index[label] for label in target_array),
                    dtype=int,
                    count=target_array.size,
                )
            except KeyError as error:
                raise ValueError(
                    f"unknown label {error.args[0]!r}; expected one of {classes}"
                ) from None

        class_count = classes.shape[0]
        sample_count = target_array.shape[0]

        # Count the samples assigned to every known class.
        counts = bincount(class_indices, minlength=class_count).astype(
            float64, copy=False
        )
        if smoothing > 0.0:
            counts = counts + float(smoothing)
        counts[counts == 0.0] = 1.0  # avoid division by zero

        cache_key = self._key_from_arrays(classes, counts, smoothing, normalize)
        with self._lock:
            class_weights = self._cache.get(cache_key)
            if class_weights is None:
                class_weights = (sample_count / (class_count * counts)).astype(
                    float64, copy=False
                )
                self._cache[cache_key] = class_weights

        # Map each class weight back to its corresponding sample.
        sample_weights = class_weights[class_indices].astype(float64, copy=False)

        # Normalize only when the complete label space is known.
        if normalize and known_labels is not None and sample_weights.size > 0:
            sample_weights *= sample_weights.size / sample_weights.sum()

        return sample_weights

    def clear_cache(self) -> None:
        """
        Clear the internal cache of computed class weights.

        Returns
        -------
        None
        """
        with self._lock:
            self._cache.clear()

    def get_cache_size(self) -> int:
        """
        Get the current size of the cache.

        Returns
        -------
        int
            Number of cached entries.
        """
        with self._lock:
            return len(self._cache)
