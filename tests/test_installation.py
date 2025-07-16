#!/usr/bin/env python3
"""
Comprehensive installation and functionality test for omp_sparse package.

This script tests the package in multiple ways:
1. Direct shared library import (fallback if package not installed)
2. Package import (if properly installed)
3. Correctness validation
4. Performance benchmarking
5. Environment checking
"""

import gc
import os
import sys
import time

import numpy as np
import scipy.sparse as sp


def test_package_import():
    """Test if omp_sparse package can be imported normally."""
    print("=" * 60)
    print("Testing Package Import")
    print("=" * 60)

    try:
        import omp_sparse

        print("✓ Package import successful")

        # Test package functions
        available = omp_sparse.is_module_available()
        algorithms = omp_sparse.get_available_algorithms()

        print(f"✓ Module available: {available}")
        print(f"✓ Available algorithms: {algorithms}")

        print(f"LOG: {algorithms=}")
        print(f"LOG: {available=}")
        if available and "v4" in algorithms:
            return omp_sparse, "package"
        else:
            print("⚠️  Package imported but module not available")
            return None, None

    except ImportError as e:
        print(f"⚠️  Package import failed: {e}")
        print("   Falling back to direct shared library import...")
        return None, None


def test_direct_import():
    """Test direct import of the compiled shared library."""
    print("\n" + "=" * 60)
    print("Testing Direct Shared Library Import")
    print("=" * 60)

    # Add lib directory to path
    lib_path = os.path.join(os.path.dirname(__file__), "omp_sparse", "lib")
    sys.path.insert(0, lib_path)

    print(f"Library path: {lib_path}")

    # Check if shared library exists
    so_files = [f for f in os.listdir(lib_path) if f.endswith(".so")]
    if not so_files:
        print("❌ No shared library (.so) files found")
        return None, None

    so_file = os.path.join(lib_path, so_files[0])
    print(f"✓ Shared library found: {so_files[0]}")
    print(f"  Size: {os.path.getsize(so_file)} bytes")

    try:
        import omp_sparse

        print("✓ Direct import successful")

        # Check module structure
        if hasattr(omp_sparse, "omp_sparse_mod"):
            mod = omp_sparse.omp_sparse_mod
            if hasattr(mod, "dense_dot_sparse_v4"):
                print("✓ dense_dot_sparse_v4 function found")
                return omp_sparse, "direct"
            else:
                print("❌ dense_dot_sparse_v4 function not found")
                return None, None
        else:
            print("❌ omp_sparse_mod not found")
            return None, None

    except ImportError as e:
        print(f"❌ Direct import failed: {e}")
        return None, None


def test_functionality(omp_sparse_module, import_type):
    """Test basic functionality of the imported module."""
    print(f"\n" + "=" * 60)
    print(f"Testing Functionality ({import_type} import)")
    print("=" * 60)

    # Create test matrices
    M, K, N = 500, 1000, 800
    density = 0.03

    print(f"Creating test matrices: {M}×{K} @ {K}×{N}, density={density:.3f}")

    np.random.seed(42)
    dense_matrix = np.random.random((M, K)).astype(np.float64)
    sparse_matrix = sp.random(K, N, density=density, format="csc", random_state=42)

    print(f"✓ Dense matrix shape: {dense_matrix.shape}")
    print(f"✓ Sparse matrix shape: {sparse_matrix.shape}")
    print(f"✓ Non-zero elements: {sparse_matrix.nnz}")

    try:
        if import_type == "package":
            # Use package interface
            multiplier = omp_sparse_module.OMPSparseMultiplier("v4")

            start_time = time.time()
            result = multiplier.multiply(dense_matrix, sparse_matrix)
            omp_time = time.time() - start_time

            print(f"✓ Package interface test successful")

        elif import_type == "direct":
            # Use direct interface
            dense_matrix = np.ascontiguousarray(dense_matrix, dtype=np.float64)
            indptr = np.ascontiguousarray(sparse_matrix.indptr, dtype=np.int32)
            indices = np.ascontiguousarray(sparse_matrix.indices, dtype=np.int32)
            data = np.ascontiguousarray(sparse_matrix.data, dtype=np.float64)

            start_time = time.time()
            result = omp_sparse_module.omp_sparse_mod.dense_dot_sparse_v4(dense_matrix, indptr, indices, data, M, K, N)
            omp_time = time.time() - start_time

            print(f"✓ Direct interface test successful")

        print(f"✓ Result shape: {result.shape}")
        print(f"✓ Result dtype: {result.dtype}")
        print(f"✓ Execution time: {omp_time:.6f} seconds")

        return result, omp_time

    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        import traceback

        traceback.print_exc()
        return None, None


def test_correctness(dense_matrix, sparse_matrix, omp_result):
    """Test correctness against NumPy baseline."""
    print(f"\n" + "=" * 60)
    print("Testing Correctness")
    print("=" * 60)

    try:
        # Force garbage collection
        gc.collect()

        print("Running NumPy baseline...")
        start_time = time.time()
        numpy_result = np.dot(dense_matrix, sparse_matrix.todense())
        numpy_time = time.time() - start_time

        print(f"✓ NumPy execution time: {numpy_time:.6f} seconds")

        # Check correctness
        correct = np.allclose(numpy_result, omp_result, rtol=1e-10, atol=1e-10)
        max_diff = np.max(np.abs(numpy_result - omp_result))

        print(f"✓ Correctness: {'PASS' if correct else 'FAIL'}")
        print(f"✓ Max absolute difference: {max_diff:.2e}")

        return correct, numpy_time

    except Exception as e:
        print(f"❌ Correctness test failed: {e}")
        return False, None


def test_performance(omp_time, numpy_time, matrix_info):
    """Analyze performance results."""
    print(f"\n" + "=" * 60)
    print("Performance Analysis")
    print("=" * 60)

    if numpy_time is None or omp_time is None:
        print("❌ Cannot analyze performance - missing timing data")
        return False

    speedup = numpy_time / omp_time if omp_time > 0 else 0

    print(f"Matrix size: {matrix_info['M']}×{matrix_info['K']} @ {matrix_info['K']}×{matrix_info['N']}")
    print(f"Density: {matrix_info['density']:.4f}")
    print(f"Non-zeros: {matrix_info['nnz']:,}")
    print(f"NumPy time: {numpy_time:.6f} seconds")
    print(f"OMP v4 time: {omp_time:.6f} seconds")
    print(f"Speedup: {speedup:.2f}x")

    # Performance evaluation
    if speedup > 3.0:
        print("✓ EXCELLENT performance (>3x speedup)")
        return True
    elif speedup > 2.0:
        print("✓ GOOD performance (>2x speedup)")
        return True
    elif speedup > 1.5:
        print("⚠️  MODERATE performance (>1.5x speedup)")
        print("   Consider optimizing OpenMP settings")
        return True
    else:
        print("❌ POOR performance (<1.5x speedup)")
        print("   Check OpenMP configuration")
        return False


def check_environment():
    """Check environment settings."""
    print(f"\n" + "=" * 60)
    print("Environment Check")
    print("=" * 60)

    omp_threads = os.environ.get("OMP_NUM_THREADS", "NOT SET")
    mkl_threads = os.environ.get("MKL_NUM_THREADS", "NOT SET")

    print(f"OMP_NUM_THREADS: {omp_threads}")
    print(f"MKL_NUM_THREADS: {mkl_threads}")

    recommendations = []

    if omp_threads == "NOT SET":
        recommendations.append("Set OMP_NUM_THREADS=8 for better performance")
    else:
        print(f"✓ OpenMP threads configured: {omp_threads}")

    if mkl_threads == "NOT SET":
        recommendations.append("Set MKL_NUM_THREADS=8 for consistent performance")
    else:
        print(f"✓ MKL threads configured: {mkl_threads}")

    # Check system info
    try:
        import multiprocessing

        cpu_count = multiprocessing.cpu_count()
        print(f"✓ System CPU cores: {cpu_count}")

        if omp_threads != "NOT SET":
            omp_int = int(omp_threads)
            if omp_int > cpu_count:
                recommendations.append(f"OMP_NUM_THREADS ({omp_int}) > CPU cores ({cpu_count})")
    except Exception as e:
        print(f"⚠️  Failed to check CPU cores: {e}")
        recommendations.append("Consider checking CPU core count manually")

    if recommendations:
        print("\n⚠️  Recommendations:")
        for rec in recommendations:
            print(f"   - {rec}")

    return len(recommendations) == 0


def main():
    """Run comprehensive test suite."""
    print("=" * 60)
    print("OMP Sparse Installation & Performance Test")
    print("=" * 60)

    success = True

    # Test 1: Try package import first
    omp_sparse_module, import_type = test_package_import()

    # Test 2: Fallback to direct import
    if omp_sparse_module is None:
        omp_sparse_module, import_type = test_direct_import()

    if omp_sparse_module is None:
        print(f"\n❌ CRITICAL FAILURE: Cannot import omp_sparse")
        print("   Solutions:")
        print("   1. Run 'make system' to build the shared library")
        print("   2. Or run 'make develop' to install the package")
        return False

    # Test 3: Functionality test
    print(f"\n✓ Using {import_type} import method")

    # Create test matrices for all subsequent tests
    M, K, N = 500, 1000, 800
    density = 0.03

    np.random.seed(42)
    dense_matrix = np.random.random((M, K)).astype(np.float64)
    sparse_matrix = sp.random(K, N, density=density, format="csc", random_state=42)

    matrix_info = {"M": M, "K": K, "N": N, "density": density, "nnz": sparse_matrix.nnz}

    omp_result, omp_time = test_functionality(omp_sparse_module, import_type)
    if omp_result is None:
        print(f"\n❌ CRITICAL FAILURE: Functionality test failed")
        return False

    # Test 4: Correctness
    correct, numpy_time = test_correctness(dense_matrix, sparse_matrix, omp_result)
    if not correct:
        print(f"\n❌ CRITICAL FAILURE: Correctness test failed")
        success = False

    # Test 5: Performance
    perf_good = test_performance(omp_time, numpy_time, matrix_info)
    if not perf_good:
        success = False

    # Test 6: Environment
    env_good = check_environment()
    if not env_good:
        success = False

    # Final summary
    print(f"\n" + "=" * 60)
    if success:
        print("🎉 INSTALLATION TEST: SUCCESS!")
        print("✓ OMP Sparse v4 is working correctly")
        print("✓ Performance is good")
        print("✓ Ready for production use")
    else:
        print("⚠️  INSTALLATION TEST: PARTIAL SUCCESS")
        print("✓ Basic functionality works")
        print("⚠️  Some issues detected (see above)")
        print("   Consider following the recommendations")

    print("=" * 60)

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
