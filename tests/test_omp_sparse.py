"""
Comprehensive test suite for omp_sparse package.

Tests correctness, performance, and edge cases for v4 and v11 algorithms.
"""

import gc
import os

import numpy as np
import pytest
import scipy.sparse

# Add parent directory to path to import omp_sparse
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import omp_sparse
from omp_sparse import OMPSparseMultiplier, multiply_dense_sparse

package_path = os.path.dirname(omp_sparse.__file__)
print(f"Package path: {package_path}")

class TestOMPSparseMultiplier:
    """Test the OMPSparseMultiplier class."""

    def test_module_availability(self):
        """Test that the compiled module is available."""
        assert omp_sparse.is_module_available(), "Compiled module should be available"

    def test_available_algorithms(self):
        """Test available algorithms."""
        algorithms = omp_sparse.get_available_algorithms()
        assert "v4" in algorithms, "v4 algorithm should be available"
        assert "v11" in algorithms, "v11 algorithm should be available"
        assert len(algorithms) >= 2, "Both v4 and v11 algorithms should be available"

    def test_initialization(self):
        """Test multiplier initialization."""
        # Valid initialization
        multiplier = OMPSparseMultiplier("v4")
        assert multiplier.algorithm == "v4"

        # Default initialization
        multiplier_default = OMPSparseMultiplier()
        assert multiplier_default.algorithm == "v4"

        # Invalid algorithm
        with pytest.raises(ValueError, match="not supported"):
            OMPSparseMultiplier("invalid")

    def test_input_validation(self):
        """Test input validation."""
        multiplier = OMPSparseMultiplier("v4")

        # Create valid matrices
        dense_mat = np.random.random((100, 200)).astype(np.float64)
        sparse_mat = scipy.sparse.random(200, 300, density=0.05, format="csc")

        # Valid inputs should work
        result = multiplier.multiply(dense_mat, sparse_mat)
        assert result.shape == (100, 300)

        # Test type validation
        with pytest.raises(TypeError, match="Dense matrix must be a numpy array"):
            multiplier.multiply([[1, 2], [3, 4]], sparse_mat)

        with pytest.raises(TypeError, match="Sparse matrix must be a scipy sparse matrix"):
            multiplier.multiply(dense_mat, [[1, 2], [3, 4]])

        # Test dimension validation
        with pytest.raises(ValueError, match="Dense matrix must be 2-dimensional"):
            multiplier.multiply(np.array([1, 2, 3]), sparse_mat)

        # Test dtype validation
        with pytest.raises(ValueError, match="Dense matrix must be float64"):
            multiplier.multiply(dense_mat.astype(np.float32), sparse_mat)

        # Test dimension compatibility
        wrong_sparse = scipy.sparse.random(300, 400, density=0.05, format="csc")
        with pytest.raises(ValueError, match="Matrix dimension mismatch"):
            multiplier.multiply(dense_mat, wrong_sparse)

        # Test empty matrices
        with pytest.raises(ValueError, match="Matrix dimension mismatch"):
            multiplier.multiply(np.array([]).reshape(0, 0), sparse_mat)

        empty_sparse = scipy.sparse.csc_matrix((200, 300))
        with pytest.raises(ValueError, match="Sparse matrix has no non-zero elements"):
            multiplier.multiply(dense_mat, empty_sparse)

    def test_v11_initialization(self):
        """Test v11 algorithm initialization."""
        # Valid v11 initialization
        multiplier = OMPSparseMultiplier("v11")
        assert multiplier.algorithm == "v11"


class TestV11Algorithm:
    """Test v11 segmented sparse algorithm."""

    @pytest.fixture
    def segmented_matrix(self):
        """Create a test matrix with segmented structure."""
        # 6x8 matrix with one segment per row
        rows, cols = 6, 8
        matrix = scipy.sparse.lil_matrix((rows, cols))
        
        # Row 0: segment from col 2-4
        matrix[0, 2:5] = [1.0, 2.0, 3.0]
        # Row 1: segment from col 0-1  
        matrix[1, 0:2] = [4.0, 5.0]
        # Row 2: empty row
        # Row 3: segment from col 5-7
        matrix[3, 5:8] = [6.0, 7.0, 8.0]
        # Row 4: single element (segment length 1)
        matrix[4, 3] = 9.0
        # Row 5: segment from col 1-3
        matrix[5, 1:4] = [10.0, 11.0, 12.0]
        
        return matrix.tocsr()

    @pytest.fixture
    def non_segmented_matrix(self):
        """Create a test matrix with non-segmented structure."""
        # 4x6 matrix with multiple segments per row
        rows, cols = 4, 6
        matrix = scipy.sparse.lil_matrix((rows, cols))
        
        # Row 0: two segments [0-1] and [4-5]
        matrix[0, 0:2] = [1.0, 2.0]
        matrix[0, 4:6] = [3.0, 4.0]
        # Row 1: three segments [1], [3], [5]
        matrix[1, 1] = 5.0
        matrix[1, 3] = 6.0
        matrix[1, 5] = 7.0
        
        return matrix.tocsr()

    def test_v11_correctness_segmented(self, segmented_matrix):
        """Test v11 correctness with segmented matrices."""
        # Create dense matrix
        M, K = 3, segmented_matrix.shape[0]
        dense_matrix = np.random.random((M, K)).astype(np.float64)
        
        # Test v4 and v11 produce same results
        multiplier_v4 = OMPSparseMultiplier("v4")
        multiplier_v11 = OMPSparseMultiplier("v11")
        
        result_v4 = multiplier_v4.multiply(dense_matrix, segmented_matrix)
        result_v11 = multiplier_v11.multiply(dense_matrix, segmented_matrix)
        
        assert np.allclose(result_v4, result_v11, rtol=1e-10, atol=1e-10), "v11 and v4 results should match"

    def test_v11_fallback_non_segmented(self, non_segmented_matrix):
        """Test v11 fallback to v4 for non-segmented matrices."""
        # Create dense matrix
        M, K = 3, non_segmented_matrix.shape[0]
        dense_matrix = np.random.random((M, K)).astype(np.float64)
        
        # Test that v11 falls back to v4 and generates warning
        multiplier_v11 = OMPSparseMultiplier("v11")
        multiplier_v4 = OMPSparseMultiplier("v4")
        
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result_v11 = multiplier_v11.multiply(dense_matrix, non_segmented_matrix)
            
            # Check that warning was generated
            assert len(w) > 0, "Warning should be generated for non-segmented matrix"
            assert "not compatible with v11" in str(w[0].message)
        
        result_v4 = multiplier_v4.multiply(dense_matrix, non_segmented_matrix)
        
        # Results should match since v11 fell back to v4
        assert np.allclose(result_v4, result_v11, rtol=1e-10, atol=1e-10), "Fallback results should match v4"

    def test_v11_sparsity_pattern_analysis(self, segmented_matrix, non_segmented_matrix):
        """Test sparsity pattern analysis."""
        from omp_sparse.utils import analyze_sparsity_pattern
        
        # Test segmented matrix
        analysis_seg = analyze_sparsity_pattern(segmented_matrix)
        assert analysis_seg['is_segmented_compatible'], "Segmented matrix should be compatible"
        assert analysis_seg['compatibility_ratio'] == 1.0, "Compatibility ratio should be 1.0"
        assert analysis_seg['max_segments'] <= 1, "Max segments should be 1 or 0"
        
        # Test non-segmented matrix
        analysis_non_seg = analyze_sparsity_pattern(non_segmented_matrix)
        assert not analysis_non_seg['is_segmented_compatible'], "Non-segmented matrix should not be compatible"
        assert analysis_non_seg['compatibility_ratio'] < 1.0, "Compatibility ratio should be < 1.0"
        assert analysis_non_seg['max_segments'] > 1, "Max segments should be > 1"


class TestMatrixMultiplication:
    """Test matrix multiplication correctness."""

    @pytest.fixture
    def small_matrices(self):
        """Create small test matrices."""
        np.random.seed(42)
        dense = np.random.random((50, 100)).astype(np.float64)
        sparse_coo = scipy.sparse.random(100, 150, density=0.1, format="coo", random_state=42)
        sparse_csc = sparse_coo.tocsc()
        return dense, sparse_coo, sparse_csc

    @pytest.fixture
    def medium_matrices(self):
        """Create medium test matrices."""
        np.random.seed(123)
        dense = np.random.random((200, 500)).astype(np.float64)
        sparse_coo = scipy.sparse.random(500, 800, density=0.02, format="coo", random_state=123)
        sparse_csc = sparse_coo.tocsc()
        return dense, sparse_coo, sparse_csc

    def test_correctness_small_coo(self, small_matrices):
        """Test correctness with small COO matrix."""
        dense, sparse_coo, _ = small_matrices

        # Compute reference result
        reference = np.dot(dense, sparse_coo.todense())

        # Compute using v4 algorithm
        multiplier = OMPSparseMultiplier("v4")
        result = multiplier.multiply(dense, sparse_coo)

        # Check correctness
        assert np.allclose(reference, result, rtol=1e-10, atol=1e-10)

    def test_correctness_small_csc(self, small_matrices):
        """Test correctness with small CSC matrix."""
        dense, _, sparse_csc = small_matrices

        # Compute reference result
        reference = np.dot(dense, sparse_csc.todense())

        # Compute using v4 algorithm
        multiplier = OMPSparseMultiplier("v4")
        result = multiplier.multiply(dense, sparse_csc)

        # Check correctness
        assert np.allclose(reference, result, rtol=1e-10, atol=1e-10)

    def test_correctness_medium(self, medium_matrices):
        """Test correctness with medium matrices."""
        dense, sparse_coo, sparse_csc = medium_matrices

        # Compute reference result
        reference = np.dot(dense, sparse_coo.todense())

        # Test both COO and CSC formats
        multiplier = OMPSparseMultiplier("v4")

        result_coo = multiplier.multiply(dense, sparse_coo)
        result_csc = multiplier.multiply(dense, sparse_csc)

        # Check correctness
        assert np.allclose(reference, result_coo, rtol=1e-10, atol=1e-10)
        assert np.allclose(reference, result_csc, rtol=1e-10, atol=1e-10)
        assert np.allclose(result_coo, result_csc, rtol=1e-10, atol=1e-10)

    def test_convenience_function(self, small_matrices):
        """Test the convenience function."""
        dense, sparse_coo, _ = small_matrices

        # Compute reference result
        reference = np.dot(dense, sparse_coo.todense())

        # Use convenience function
        result = multiply_dense_sparse(dense, sparse_coo, algorithm="v4")

        # Check correctness
        assert np.allclose(reference, result, rtol=1e-10, atol=1e-10)


class TestPerformance:
    """Test performance characteristics."""

    def test_benchmark_functionality(self):
        """Test the benchmark functionality."""
        # Create test matrices
        dense = np.random.random((100, 200)).astype(np.float64)
        sparse = scipy.sparse.random(200, 300, density=0.05, format="csc")

        # Run benchmark
        multiplier = OMPSparseMultiplier("v4")
        benchmark_result = multiplier.benchmark(dense, sparse, repeats=2)

        # Check benchmark result structure
        expected_keys = ["algorithm", "mean_time", "std_time", "correct", "matrix_shape", "density", "nnz", "repeats"]
        for key in expected_keys:
            assert key in benchmark_result, f"Missing key: {key}"

        # Check values
        assert benchmark_result["algorithm"] == "v4"
        assert benchmark_result["correct"] is True
        assert benchmark_result["mean_time"] > 0
        assert benchmark_result["std_time"] >= 0
        assert benchmark_result["matrix_shape"] == (100, 200, 300)
        assert benchmark_result["repeats"] == 2

    def test_memory_cleanup(self):
        """Test that memory is properly cleaned up."""
        # Create larger matrices
        dense = np.random.random((500, 1000)).astype(np.float64)
        sparse = scipy.sparse.random(1000, 1500, density=0.01, format="csc")

        multiplier = OMPSparseMultiplier("v4")

        # Perform multiple multiplications
        for _ in range(5):
            result = multiplier.multiply(dense, sparse)
            del result
            gc.collect()

        # Should not have memory issues


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_very_sparse_matrix(self):
        """Test with very sparse matrix."""
        dense = np.random.random((100, 500)).astype(np.float64)
        sparse = scipy.sparse.random(500, 300, density=0.001, format="csc")  # Very sparse

        multiplier = OMPSparseMultiplier("v4")
        result = multiplier.multiply(dense, sparse)

        # Check correctness
        reference = np.dot(dense, sparse.todense())
        assert np.allclose(reference, result, rtol=1e-10, atol=1e-10)

    def test_dense_sparse_matrix(self):
        """Test with relatively dense sparse matrix."""
        dense = np.random.random((50, 100)).astype(np.float64)
        sparse = scipy.sparse.random(100, 200, density=0.5, format="csc")  # Dense sparse

        multiplier = OMPSparseMultiplier("v4")
        result = multiplier.multiply(dense, sparse)

        # Check correctness
        reference = np.dot(dense, sparse.todense())
        assert np.allclose(reference, result, rtol=1e-10, atol=1e-10)

    def test_single_row_column(self):
        """Test with single row/column matrices."""
        # Single row dense matrix
        dense_row = np.random.random((1, 100)).astype(np.float64)
        sparse = scipy.sparse.random(100, 50, density=0.1, format="csc")

        multiplier = OMPSparseMultiplier("v4")
        result = multiplier.multiply(dense_row, sparse)

        reference = np.dot(dense_row, sparse.todense())
        assert np.allclose(reference, result, rtol=1e-10, atol=1e-10)

        # Single column sparse matrix
        dense = np.random.random((50, 100)).astype(np.float64)
        sparse_col = scipy.sparse.random(100, 1, density=0.2, format="csc")

        result = multiplier.multiply(dense, sparse_col)
        reference = np.dot(dense, sparse_col.todense())
        assert np.allclose(reference, result, rtol=1e-10, atol=1e-10)


@pytest.mark.skipif(
    not os.path.exists(f"{package_path}/../data"),
    reason="Scientific datasets not available",
)
class TestScientificDatasets:
    """Test with real scientific datasets if available."""

    @pytest.fixture
    def water_data(self):
        """Load water dataset."""
        try:
            dense = np.load(f"{package_path}/../data/data_dense_water.npy")
            sparse = scipy.sparse.load_npz(f"{package_path}/../data/data_sparse_water.npz")
            return dense, sparse
        except FileNotFoundError:
            pytest.skip("Water dataset not found")

    @pytest.fixture
    def graphene_data(self):
        """Load graphene dataset."""
        try:
            dense = np.load(f"{package_path}/../data/data_dense_graphene_3x3.npy")
            sparse = scipy.sparse.load_npz(f"{package_path}/../data/data_sparse_graphene_3x3.npz")
            return dense, sparse
        except FileNotFoundError:
            pytest.skip("Graphene dataset not found")

    def test_water_dataset(self, water_data):
        """Test with water dataset."""
        dense, sparse = water_data

        # Ensure correct dtype
        dense = dense.astype(np.float64)

        multiplier = OMPSparseMultiplier("v4")
        result = multiplier.multiply(dense, sparse)

        # Check correctness
        reference = np.dot(dense, sparse.todense())
        assert np.allclose(reference, result, rtol=1e-10, atol=1e-10)

    def test_graphene_dataset(self, graphene_data):
        """Test with graphene dataset."""
        dense, sparse = graphene_data

        # Ensure correct dtype
        dense = dense.astype(np.float64)

        multiplier = OMPSparseMultiplier("v4")
        result = multiplier.multiply(dense, sparse)

        # Check correctness (using slightly relaxed tolerance for larger matrices)
        reference = np.dot(dense, sparse.todense())
        assert np.allclose(reference, result, rtol=1e-9, atol=1e-9)


if __name__ == "__main__":
    # Print environment info
    print(f"OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'NOT SET')}")
    print(f"MKL_NUM_THREADS: {os.environ.get('MKL_NUM_THREADS', 'NOT SET')}")
    print(f"Module available: {omp_sparse.is_module_available()}")
    print(f"Available algorithms: {omp_sparse.get_available_algorithms()}")

    # Run tests
    pytest.main([__file__, "-v"])
