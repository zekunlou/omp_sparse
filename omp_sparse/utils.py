"""
Utility functions for omp_sparse package.

Contains sparsity pattern analysis and format conversion utilities.
"""

import numpy as np
import scipy.sparse
from typing import Tuple, Optional, Dict, Any


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