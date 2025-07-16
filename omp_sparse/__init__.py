"""
OpenMP Sparse Matrix Multiplication Package

High-performance dense @ sparse matrix multiplication using OpenMP-accelerated
Fortran algorithms optimized for scientific computing applications.

This package currently implements the v4 algorithm, which provides excellent
performance across a wide range of matrix sizes and sparsity patterns.
"""

__version__ = "1.0.0"
__author__ = "Zekun Lou"
__email__ = "null@null.null"

import warnings
from typing import Literal, Optional, Union

import numpy as np
import scipy.sparse

try:
    # full path: omp_sparse.lib.omp_sparse.omp_sparse_mod.dense_dot_sparse_v4
    from omp_sparse.lib.omp_sparse import omp_sparse_mod as _omp_sparse_mod

    _MODULE_AVAILABLE = True
except ImportError as e:
    warnings.warn(
        f"Failed to import compiled omp_sparse module: {e}. "
        "Please ensure the package is properly built with 'make' or 'pip install'.",
        ImportWarning,
        stacklevel=2,
    )
    _MODULE_AVAILABLE = False
    _omp_sparse_mod = None

# Algorithm type definitions (extensible for future versions)
AlgorithmType = Literal["v4"]
_AVAILABLE_ALGORITHMS = ["v4"]


class OMPSparseMultiplier:
    """
    OpenMP-accelerated dense @ sparse matrix multiplication.

    This class provides a high-performance interface for multiplying dense matrices
    with sparse matrices using OpenMP-parallelized Fortran algorithms.

    Currently supports:
    - v4: Optimized column-wise algorithm with dynamic scheduling (recommended)

    Future versions may include additional algorithms (v1-v3, v5-v10).

    Example:
        >>> import numpy as np
        >>> import scipy.sparse as sp
        >>> from omp_sparse import OMPSparseMultiplier
        >>>
        >>> # Create test matrices
        >>> dense = np.random.random((1000, 2000))
        >>> sparse = sp.random(2000, 3000, density=0.02, format='csc')
        >>>
        >>> # Initialize multiplier
        >>> multiplier = OMPSparseMultiplier(algorithm="v4")
        >>>
        >>> # Perform multiplication
        >>> result = multiplier.multiply(dense, sparse)
    """

    def __init__(self, algorithm: AlgorithmType = "v4"):
        """
        Initialize the sparse matrix multiplier.

        Args:
            algorithm: Algorithm variant to use. Currently only "v4" is supported.

        Raises:
            ValueError: If algorithm is not supported
            ImportError: If compiled module is not available
        """
        if not _MODULE_AVAILABLE:
            raise ImportError(
                "Compiled module not available. Please build the package with 'make' "
                "or install it properly with 'pip install'."
            )

        self.algorithm = algorithm
        self._validate_algorithm()

    def _validate_algorithm(self):
        """Validate the selected algorithm."""
        if self.algorithm not in _AVAILABLE_ALGORITHMS:
            raise ValueError(
                f"Algorithm '{self.algorithm}' is not supported. Available algorithms: {_AVAILABLE_ALGORITHMS}"
            )

    def _validate_inputs(
        self, dense_matrix: np.ndarray, sparse_matrix: Union[scipy.sparse.coo_matrix, scipy.sparse.csc_matrix]
    ):
        """Validate input matrices for multiplication."""
        # Check dense matrix
        if not isinstance(dense_matrix, np.ndarray):
            raise TypeError("Dense matrix must be a numpy array")

        if dense_matrix.ndim != 2:
            raise ValueError("Dense matrix must be 2-dimensional")

        if dense_matrix.dtype != np.float64:
            raise ValueError("Dense matrix must be float64 (double precision)")

        # Check sparse matrix
        if not scipy.sparse.issparse(sparse_matrix):
            raise TypeError("Sparse matrix must be a scipy sparse matrix")

        if sparse_matrix.ndim != 2:
            raise ValueError("Sparse matrix must be 2-dimensional")

        # Check dimension compatibility
        if dense_matrix.shape[1] != sparse_matrix.shape[0]:
            raise ValueError(
                f"Matrix dimension mismatch: dense matrix has {dense_matrix.shape[1]} columns, "
                f"sparse matrix has {sparse_matrix.shape[0]} rows. Cannot multiply."
            )

        # Check for empty matrices
        if dense_matrix.size == 0:
            raise ValueError("Dense matrix is empty")

        if sparse_matrix.nnz == 0:
            raise ValueError("Sparse matrix has no non-zero elements")

    def _prepare_inputs(self, dense_matrix: np.ndarray, sparse_matrix):
        """Prepare inputs for the v4 algorithm (CSC format)."""
        # Convert to CSC format if needed
        if not isinstance(sparse_matrix, scipy.sparse.csc_matrix):
            sparse_matrix = sparse_matrix.tocsc()

        # Ensure contiguous arrays
        dense_matrix = np.ascontiguousarray(dense_matrix, dtype=np.float64)

        M, K = dense_matrix.shape
        N = sparse_matrix.shape[1]

        # Ensure sparse matrix data is contiguous and correct dtype
        indptr = np.ascontiguousarray(sparse_matrix.indptr, dtype=np.int32)
        indices = np.ascontiguousarray(sparse_matrix.indices, dtype=np.int32)
        data = np.ascontiguousarray(sparse_matrix.data, dtype=np.float64)

        return dense_matrix, indptr, indices, data, M, K, N

    def multiply(
        self, dense_matrix: np.ndarray, sparse_matrix: Union[scipy.sparse.coo_matrix, scipy.sparse.csc_matrix]
    ) -> np.ndarray:
        """
        Multiply dense matrix with sparse matrix using OpenMP acceleration.

        Args:
            dense_matrix: Dense matrix of shape (M, K) with dtype float64
            sparse_matrix: Sparse matrix of shape (K, N) in COO or CSC format

        Returns:
            Result matrix of shape (M, N) as numpy array

        Raises:
            TypeError: If input types are incorrect
            ValueError: If matrix dimensions don't match or other validation fails
        """
        # Validate inputs
        self._validate_inputs(dense_matrix, sparse_matrix)

        # Prepare inputs for v4 algorithm
        dense_matrix, indptr, indices, data, M, K, N = self._prepare_inputs(dense_matrix, sparse_matrix)

        # Call the v4 algorithm
        return _omp_sparse_mod.dense_dot_sparse_v4(dense_matrix, indptr, indices, data, M, K, N)

    def benchmark(
        self,
        dense_matrix: np.ndarray,
        sparse_matrix: Union[scipy.sparse.coo_matrix, scipy.sparse.csc_matrix],
        baseline_result: Optional[np.ndarray] = None,
        repeats: int = 3,
    ) -> dict:
        """
        Benchmark the multiplication and return performance metrics.

        Args:
            dense_matrix: Dense matrix for multiplication
            sparse_matrix: Sparse matrix for multiplication
            baseline_result: Optional baseline result for correctness validation
            repeats: Number of times to repeat the benchmark

        Returns:
            Dictionary containing timing and correctness information
        """
        import gc
        import time

        # Validate inputs
        self._validate_inputs(dense_matrix, sparse_matrix)

        # Compute baseline if not provided
        if baseline_result is None:
            baseline_result = np.dot(dense_matrix, sparse_matrix.todense())

        # Run multiple times and collect statistics
        times = []
        results = []

        for _ in range(repeats):
            gc.collect()  # Clean memory before each run

            start_time = time.time()
            result = self.multiply(dense_matrix, sparse_matrix)
            end_time = time.time()

            times.append(end_time - start_time)
            results.append(result)

            gc.collect()  # Clean memory after each run

        # Calculate statistics
        mean_time = np.mean(times)
        std_time = np.std(times) if len(times) > 1 else 0.0

        # Check correctness using first result
        correct = np.allclose(baseline_result, results[0], rtol=1e-10, atol=1e-10)

        # Calculate matrix properties
        M, K = dense_matrix.shape
        N = sparse_matrix.shape[1]
        density = sparse_matrix.nnz / (K * N)

        return {
            "algorithm": self.algorithm,
            "mean_time": mean_time,
            "std_time": std_time,
            "correct": correct,
            "matrix_shape": (M, K, N),
            "density": density,
            "nnz": sparse_matrix.nnz,
            "repeats": repeats,
        }


def benchmark_dense_dot_dense(
    dense_matrix: np.ndarray,
    sparse_matrix: Union[scipy.sparse.coo_matrix, scipy.sparse.csc_matrix],
    repeats: int = 3,
) -> dict:
    """
    Benchmark dense @ dense matrix multiplication for performance comparison.

    This function performs the same computation as the sparse multiplication
    but using dense matrices with numpy's standard dot product, providing
    a baseline for performance comparison.

    Args:
        dense_matrix: Dense matrix of shape (M, K) with dtype float64
        sparse_matrix: Sparse matrix of shape (K, N) - will be converted to dense
        repeats: Number of times to repeat the benchmark

    Returns:
        Dictionary containing timing and matrix information for comparison
    """
    import gc
    import time

    # Convert sparse to dense for comparison
    dense_sparse = sparse_matrix.todense()

    # Ensure proper types
    if dense_matrix.dtype != np.float64:
        dense_matrix = dense_matrix.astype(np.float64)
    if dense_sparse.dtype != np.float64:
        dense_sparse = dense_sparse.astype(np.float64)

    # Run multiple times and collect statistics
    times = []
    results = []

    for _ in range(repeats):
        gc.collect()  # Clean memory before each run

        start_time = time.time()
        result = np.dot(dense_matrix, dense_sparse)
        end_time = time.time()

        times.append(end_time - start_time)
        results.append(result)

        gc.collect()  # Clean memory after each run

    # Calculate statistics
    mean_time = np.mean(times)
    std_time = np.std(times) if len(times) > 1 else 0.0

    # Calculate matrix properties
    M, K = dense_matrix.shape
    N = sparse_matrix.shape[1]
    density = sparse_matrix.nnz / (K * N)

    return {
        "algorithm": "numpy_dense",
        "mean_time": mean_time,
        "std_time": std_time,
        "correct": True,  # This is the baseline
        "matrix_shape": (M, K, N),
        "density": density,
        "nnz": sparse_matrix.nnz,
        "repeats": repeats,
    }


def multiply_dense_sparse(
    dense_matrix: np.ndarray,
    sparse_matrix: Union[scipy.sparse.coo_matrix, scipy.sparse.csc_matrix],
    algorithm: AlgorithmType = "v4",
) -> np.ndarray:
    """
    Convenience function for dense @ sparse matrix multiplication.

    Args:
        dense_matrix: Dense matrix of shape (M, K) with dtype float64
        sparse_matrix: Sparse matrix of shape (K, N) in COO or CSC format
        algorithm: Algorithm variant to use (currently only "v4" is supported)

    Returns:
        Result matrix of shape (M, N) as numpy array
    """
    multiplier = OMPSparseMultiplier(algorithm)
    return multiplier.multiply(dense_matrix, sparse_matrix)


def get_available_algorithms() -> list[str]:
    """
    Get list of available algorithms.

    Returns:
        List of available algorithm names
    """
    return _AVAILABLE_ALGORITHMS.copy()


def is_module_available() -> bool:
    """
    Check if the compiled module is available.

    Returns:
        True if module is available, False otherwise
    """
    return _MODULE_AVAILABLE


# Export main interface
__all__ = [
    "OMPSparseMultiplier",
    "multiply_dense_sparse",
    "get_available_algorithms",
    "is_module_available",
    "AlgorithmType",
]
