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


class OptimizedWeightCalculator:
    """
    Compute class-balanced sample weights with caching.

    This class implements an optimized version of balanced weights:

    .. math::

        w(c) = \\frac{N}{K \\cdot n_c}

    where:
        - ``N`` is the total number of samples,
        - ``K`` is the number of classes,
        - ``n_c`` is the number of samples in class ``c``.

    Features
    --------
    - Supports additive smoothing to avoid zero-count divisions.
    - Normalizes weights so that the mean equals 1 (optional).
    - Uses a cache for previously computed class weights to speed up repeated calls.
    - Ensures consistency across batches when ``labels_all`` is provided.

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
            raise ValueError("y_train_batch contient des NaN.")

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
        else:  # textes/objets
            classes_bytes = asarray(classes, dtype="U").tobytes()

        counts_bytes = counts.astype(float64, copy=False).tobytes()
        return (classes_bytes, counts_bytes, float(smoothing), bool(normalize))

    def calculate_sample_weights(
        self,
        y_train_batch,
        labels_all: Optional[Sequence] = None,
        smoothing: float = 0.0,
        normalize: bool = True,
    ) -> ndarray:
        """
        Compute balanced sample weights for a training batch.

        Parameters
        ----------
        y_train_batch : array-like
            Labels of the current batch.
        labels_all : sequence of shape (n_classes,), optional
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
            If unknown labels are found (when ``labels_all`` is provided).
        """
        y = self._as_1d(y_train_batch)
        self._validate(y)

        if y.size == 0:
            return array([], dtype=float64)

        # Espace des classes + indices inverses
        if labels_all is None:
            classes, inv = unique(y, return_inverse=True)
        else:
            classes = asarray(labels_all)
            idx_map = {c: i for i, c in enumerate(classes)}
            try:
                inv = fromiter((idx_map[c] for c in y), dtype=int, count=y.size)
            except KeyError as e:
                raise ValueError(
                    f"Label inconnu {e.args[0]} par rapport à labels_all={classes}."
                ) from None

        K = classes.shape[0]
        N = y.shape[0]

        # Effectifs par classe
        counts = bincount(inv, minlength=K).astype(float64, copy=False)
        if smoothing > 0.0:
            counts = counts + float(smoothing)
        counts[counts == 0.0] = 1.0  # avoid division by zero

        # --- CACHE ---
        cache_key = self._key_from_arrays(classes, counts, smoothing, normalize)
        with self._lock:
            class_weights = self._cache.get(cache_key)
            if class_weights is None:
                class_weights = (N / (K * counts)).astype(float64, copy=False)
                self._cache[cache_key] = class_weights

        # Projection par échantillon
        w = class_weights[inv].astype(float64, copy=False)

        # Normalisation (uniquement si labels_all fourni)
        if normalize and labels_all is not None and w.size > 0:
            w *= w.size / w.sum()

        return w

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