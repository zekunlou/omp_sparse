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

from omp_sparse.utils import (
    analyze_sparsity_pattern, 
    convert_to_segmented_format, 
    benchmark_sparse_algorithm,
    benchmark_numpy_dense_multiplication
)

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
AlgorithmType = Literal["v4", "v11"]
_AVAILABLE_ALGORITHMS = ["v4", "v11"]


class OMPSparseMultiplier:
    """
    OpenMP-accelerated dense @ sparse matrix multiplication.

    This class provides a high-performance interface for multiplying dense matrices
    with sparse matrices using OpenMP-parallelized Fortran algorithms.

    Currently supports:
    - v4: Optimized column-wise algorithm with dynamic scheduling
    - v11: Segmented sparse algorithm for matrices with one contiguous segment per row

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
            algorithm: Algorithm variant to use. Supports "v4" (general CSC) and "v11" (segmented).

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

    def _prepare_inputs_v4(self, dense_matrix: np.ndarray, sparse_matrix):
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

    def _prepare_inputs_v11(self, dense_matrix: np.ndarray, sparse_matrix):
        """Prepare inputs for the v11 algorithm (segmented format)."""
        # Analyze sparsity pattern
        analysis = analyze_sparsity_pattern(sparse_matrix)
        
        if not analysis['is_segmented_compatible']:
            warnings.warn(
                f"Matrix is not compatible with v11 segmented format "
                f"(compatibility: {analysis['compatibility_ratio']:.3f}, "
                f"max segments per row: {analysis['max_segments']}). "
                f"Falling back to v4 algorithm.",
                UserWarning,
                stacklevel=3
            )
            # Fallback to v4
            return self._prepare_inputs_v4(dense_matrix, sparse_matrix) + ("v4",)
        
        # Convert to segmented format
        row_start_col, row_segment_len, seg_data = convert_to_segmented_format(sparse_matrix)
        
        # Ensure contiguous arrays
        dense_matrix = np.ascontiguousarray(dense_matrix, dtype=np.float64)
        row_start_col = np.ascontiguousarray(row_start_col, dtype=np.int32)
        row_segment_len = np.ascontiguousarray(row_segment_len, dtype=np.int32)
        seg_data = np.ascontiguousarray(seg_data, dtype=np.float64)

        M, K = dense_matrix.shape
        N = sparse_matrix.shape[1]

        return dense_matrix, row_start_col, row_segment_len, seg_data, M, K, N, "v11"

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

        if self.algorithm == "v4":
            # Prepare inputs for v4 algorithm
            dense_matrix, indptr, indices, data, M, K, N = self._prepare_inputs_v4(dense_matrix, sparse_matrix)
            # Call the v4 algorithm
            return _omp_sparse_mod.dense_dot_sparse_v4(dense_matrix, indptr, indices, data, M, K, N)
        
        elif self.algorithm == "v11":
            # Prepare inputs for v11 algorithm (with possible fallback to v4)
            prepared = self._prepare_inputs_v11(dense_matrix, sparse_matrix)
            actual_algorithm = prepared[-1]  # Last element indicates which algorithm to use
            
            if actual_algorithm == "v4":
                # Fallback case
                dense_matrix, indptr, indices, data, M, K, N, _ = prepared
                return _omp_sparse_mod.dense_dot_sparse_v4(dense_matrix, indptr, indices, data, M, K, N)
            else:
                # v11 segmented algorithm
                dense_matrix, row_start_col, row_segment_len, seg_data, M, K, N, _ = prepared
                return _omp_sparse_mod.dense_dot_sparse_v11(dense_matrix, row_start_col, row_segment_len, seg_data, M, K, N)
        
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")

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
        # Validate inputs
        self._validate_inputs(dense_matrix, sparse_matrix)

        # Use unified benchmarking function from utils
        return benchmark_sparse_algorithm(
            multiplier=self,
            dense_matrix=dense_matrix,
            sparse_matrix=sparse_matrix,
            baseline_result=baseline_result,
            repeats=repeats
        )


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
    # Use unified benchmarking function from utils
    result = benchmark_numpy_dense_multiplication(dense_matrix, sparse_matrix, repeats)
    
    # Add 'repeats' field for backwards compatibility
    result["repeats"] = repeats
    result["correct"] = True  # This is the baseline
    
    return result


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
