"""
Utility functions for omp_sparse package.

Contains sparsity pattern analysis, format conversion utilities, and benchmarking functions.
"""

import gc
import time
from typing import Tuple, Optional, Dict, Any, Callable, Union

import numpy as np
import scipy.sparse


def analyze_sparsity_pattern(sparse_matrix: scipy.sparse.spmatrix) -> Dict[str, Any]:
    """
    Analyze sparsity pattern of a sparse matrix to determine if it's compatible
    with segmented sparse format (one contiguous segment per row).
    
    Args:
        sparse_matrix: Input sparse matrix in any scipy sparse format
        
    Returns:
        Dictionary containing analysis results:
        - is_segmented_compatible: bool, True if matrix has one segment per row
        - segments_per_row: list of int, number of segments in each row
        - max_segments: int, maximum segments in any row
        - total_rows: int, total number of rows
        - compatible_rows: int, number of rows with exactly one segment
        - compatibility_ratio: float, fraction of rows that are compatible
    """
    # Convert to CSR for efficient row-wise operations
    if not isinstance(sparse_matrix, scipy.sparse.csr_matrix):
        csr_matrix = sparse_matrix.tocsr()
    else:
        csr_matrix = sparse_matrix
    
    rows, cols = csr_matrix.shape
    segments_per_row = []
    
    for i in range(rows):
        row_start = csr_matrix.indptr[i]
        row_end = csr_matrix.indptr[i + 1]
        
        if row_start == row_end:
            # Empty row - considered as 0 segments
            segments_per_row.append(0)
            continue
            
        # Get column indices for this row
        row_cols = csr_matrix.indices[row_start:row_end]
        
        if len(row_cols) == 0:
            segments_per_row.append(0)
            continue
            
        # Count contiguous segments
        segments = 1
        for j in range(1, len(row_cols)):
            if row_cols[j] != row_cols[j-1] + 1:
                segments += 1
                
        segments_per_row.append(segments)
    
    # Calculate statistics
    segments_array = np.array(segments_per_row)
    max_segments = int(np.max(segments_array)) if len(segments_array) > 0 else 0
    compatible_rows = int(np.sum((segments_array == 1) | (segments_array == 0)))
    compatibility_ratio = compatible_rows / rows if rows > 0 else 0.0
    is_segmented_compatible = compatibility_ratio == 1.0
    
    return {
        'is_segmented_compatible': is_segmented_compatible,
        'segments_per_row': segments_per_row,
        'max_segments': max_segments,
        'total_rows': rows,
        'compatible_rows': compatible_rows,
        'compatibility_ratio': compatibility_ratio,
        'nnz': csr_matrix.nnz,
        'density': csr_matrix.nnz / (rows * cols) if rows * cols > 0 else 0.0
    }


def convert_to_segmented_format(sparse_matrix: scipy.sparse.spmatrix) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a sparse matrix to segmented format for v11 algorithm.
    
    The matrix must have exactly one contiguous segment per row (or be empty).
    
    Args:
        sparse_matrix: Input sparse matrix with one segment per row
        
    Returns:
        Tuple of (row_start_col, row_segment_len, data):
        - row_start_col: array of starting column indices for each row's segment
        - row_segment_len: array of segment lengths for each row
        - data: flattened array of non-zero values in row-major order
        
    Raises:
        ValueError: If matrix is not compatible with segmented format
    """
    # First verify compatibility
    analysis = analyze_sparsity_pattern(sparse_matrix)
    if not analysis['is_segmented_compatible']:
        raise ValueError(
            f"Matrix is not compatible with segmented format. "
            f"Compatibility ratio: {analysis['compatibility_ratio']:.3f}, "
            f"Max segments per row: {analysis['max_segments']}"
        )
    
    # Convert to CSR for efficient row-wise processing
    if not isinstance(sparse_matrix, scipy.sparse.csr_matrix):
        csr_matrix = sparse_matrix.tocsr()
    else:
        csr_matrix = sparse_matrix
    
    rows = csr_matrix.shape[0]
    row_start_col = np.zeros(rows, dtype=np.int32)
    row_segment_len = np.zeros(rows, dtype=np.int32)
    data_list = []
    
    for i in range(rows):
        row_start = csr_matrix.indptr[i]
        row_end = csr_matrix.indptr[i + 1]
        
        if row_start == row_end:
            # Empty row
            row_start_col[i] = 0
            row_segment_len[i] = 0
        else:
            # Get column indices and data for this row
            row_cols = csr_matrix.indices[row_start:row_end]
            row_data = csr_matrix.data[row_start:row_end]
            
            # Since we verified compatibility, we know there's exactly one segment
            row_start_col[i] = row_cols[0]  # First column in segment
            row_segment_len[i] = len(row_cols)  # Length of segment
            
            # Append data to list
            data_list.extend(row_data)
    
    # Convert data list to numpy array
    data = np.array(data_list, dtype=np.float64)
    
    return row_start_col, row_segment_len, data


def validate_segmented_format(row_start_col: np.ndarray, row_segment_len: np.ndarray, 
                             data: np.ndarray, expected_shape: Tuple[int, int]) -> bool:
    """
    Validate that segmented format data is consistent.
    
    Args:
        row_start_col: Starting column for each row's segment
        row_segment_len: Length of each row's segment  
        data: Flattened non-zero values
        expected_shape: Expected (rows, cols) shape of original matrix
        
    Returns:
        True if format is valid, False otherwise
    """
    rows, cols = expected_shape
    
    # Check array lengths
    if len(row_start_col) != rows or len(row_segment_len) != rows:
        return False
    
    # Check data length matches total segments
    expected_data_len = np.sum(row_segment_len)
    if len(data) != expected_data_len:
        return False
    
    # Check column bounds
    for i in range(rows):
        start_col = row_start_col[i]
        seg_len = row_segment_len[i]
        
        if seg_len > 0:
            if start_col < 0 or start_col + seg_len > cols:
                return False
        elif seg_len == 0:
            # Empty row is OK
            continue
        else:
            # Negative segment length is invalid
            return False
    
    return True


# =============================================================================
# Benchmarking Utilities
# =============================================================================

def benchmark_function(func: Callable, *args, repeats: int = 3, name: str = "Function", 
                      validate_result: Optional[Callable] = None, 
                      baseline_result: Optional[np.ndarray] = None, **kwargs) -> Dict[str, Any]:
    """
    Generic function benchmarking utility with timing and validation.
    
    Args:
        func: Function to benchmark
        *args: Positional arguments for the function
        repeats: Number of times to repeat the benchmark
        name: Name of the function for logging
        validate_result: Optional function to validate results
        baseline_result: Optional baseline result for correctness checking
        **kwargs: Keyword arguments for the function
        
    Returns:
        Dictionary with benchmark results including timing and correctness info
    """
    times = []
    results = []
    errors = []
    
    for i in range(repeats):
        gc.collect()
        
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            runtime = end_time - start_time
            times.append(runtime)
            results.append(result)
            errors.append(None)
        except Exception as e:
            end_time = time.time()
            runtime = end_time - start_time
            times.append(runtime)
            results.append(None)
            errors.append(str(e))
        
        gc.collect()
    
    # Calculate statistics
    valid_times = [t for t, e in zip(times, errors) if e is None]
    valid_results = [r for r, e in zip(results, errors) if e is None]
    
    if valid_times:
        mean_time = np.mean(valid_times)
        std_time = np.std(valid_times) if len(valid_times) > 1 else 0.0
        final_result = valid_results[0] if valid_results else None
        final_error = None
    else:
        mean_time = np.mean(times)
        std_time = np.std(times) if len(times) > 1 else 0.0
        final_result = None
        final_error = errors[0] if errors else "Unknown error"
    
    # Validate result if baseline provided
    correct = None
    if final_result is not None and baseline_result is not None:
        correct = np.allclose(baseline_result, final_result, rtol=1e-10, atol=1e-10)
    elif validate_result is not None and final_result is not None:
        correct = validate_result(final_result)
    
    return {
        "name": name,
        "result": final_result,
        "mean_time": mean_time,
        "std_time": std_time,
        "min_time": np.min(valid_times) if valid_times else float('inf'),
        "max_time": np.max(valid_times) if valid_times else float('inf'),
        "error": final_error,
        "correct": correct,
        "valid_runs": len(valid_times),
        "total_runs": repeats,
        "times": times
    }


def benchmark_numpy_dense_multiplication(dense_matrix: np.ndarray, 
                                        sparse_matrix: Union[scipy.sparse.coo_matrix, scipy.sparse.csc_matrix],
                                        repeats: int = 3) -> Dict[str, Any]:
    """
    Benchmark NumPy dense @ dense matrix multiplication as baseline.
    
    Args:
        dense_matrix: Dense matrix of shape (M, K) with dtype float64
        sparse_matrix: Sparse matrix of shape (K, N) - will be converted to dense
        repeats: Number of times to repeat the benchmark
        
    Returns:
        Dictionary containing timing and matrix information for comparison
    """
    # Convert sparse to dense for comparison
    dense_sparse = sparse_matrix.todense()
    
    # Ensure proper types
    if dense_matrix.dtype != np.float64:
        dense_matrix = dense_matrix.astype(np.float64)
    if dense_sparse.dtype != np.float64:
        dense_sparse = dense_sparse.astype(np.float64)
    
    def numpy_multiply():
        return np.dot(dense_matrix, dense_sparse)
    
    result = benchmark_function(
        func=numpy_multiply,
        repeats=repeats,
        name="NumPy dense @ dense"
    )
    
    # Add matrix properties
    M, K = dense_matrix.shape
    N = sparse_matrix.shape[1]
    density = sparse_matrix.nnz / (K * N)
    
    result.update({
        "algorithm": "numpy_dense",
        "matrix_shape": (M, K, N),
        "density": density,
        "nnz": sparse_matrix.nnz
    })
    
    return result


def benchmark_sparse_algorithm(multiplier, dense_matrix: np.ndarray,
                              sparse_matrix: Union[scipy.sparse.coo_matrix, scipy.sparse.csc_matrix],
                              baseline_result: Optional[np.ndarray] = None,
                              repeats: int = 3) -> Dict[str, Any]:
    """
    Benchmark a sparse matrix multiplication algorithm.
    
    Args:
        multiplier: OMPSparseMultiplier instance
        dense_matrix: Dense matrix for multiplication
        sparse_matrix: Sparse matrix for multiplication  
        baseline_result: Optional baseline result for correctness validation
        repeats: Number of times to repeat the benchmark
        
    Returns:
        Dictionary containing timing and correctness information
    """
    # Compute baseline if not provided
    if baseline_result is None:
        baseline_result = np.dot(dense_matrix, sparse_matrix.todense())
    
    def multiply_func():
        return multiplier.multiply(dense_matrix, sparse_matrix)
    
    result = benchmark_function(
        func=multiply_func,
        repeats=repeats,
        name=f"OMP Sparse {multiplier.algorithm}",
        baseline_result=baseline_result
    )
    
    # Add algorithm-specific properties
    M, K = dense_matrix.shape
    N = sparse_matrix.shape[1]
    density = sparse_matrix.nnz / (K * N)
    
    result.update({
        "algorithm": multiplier.algorithm,
        "matrix_shape": (M, K, N),
        "density": density,
        "nnz": sparse_matrix.nnz
    })
    
    return result