"""
Benchmark module for omp_sparse package.

Provides command-line benchmarking functionality for performance testing.
"""

import argparse
import gc
import os
import sys

import numpy as np
import scipy.sparse
from scipy.sparse import coo_matrix

from . import OMPSparseMultiplier
from .utils import analyze_sparsity_pattern, benchmark_function

file_path = os.path.abspath(__file__)
data_path = os.path.abspath(f"{file_path}/../../data")
print(data_path)

def log(*args, **kwargs):
    """Logging utility with automatic flushing."""
    if "flush" not in kwargs:
        kwargs["flush"] = True
    print(*args, **kwargs)


def load_data(data_source: str, args) -> tuple:
    """
    Load test data from various sources.

    Args:
        data_source: Source of data ('random', 'water', 'graphene', 'TiS2', 'ZrS2')
        args: Command line arguments containing matrix dimensions

    Returns:
        Tuple of (dense_matrix, sparse_matrix, M, K, N, NNZ)
    """
    log(f"Loading data from source: {data_source}")

    if data_source == "random":
        # Matrix dimensions - use provided args or defaults
        M = getattr(args, "M", 1000)
        K = getattr(args, "K", 2000)
        N = getattr(args, "N", 3000)
        density = getattr(args, "density", 0.02)

        # Validate inputs
        if M <= 0 or K <= 0 or N <= 0:
            raise ValueError("Matrix dimensions must be positive")
        if not (0 < density <= 1):
            raise ValueError("Density must be between 0 and 1")
        if M * K * N > 1e9:  # Prevent memory issues
            log(f"Warning: Large matrix size ({M}x{K}) @ ({K}x{N}) may cause memory issues")

        # Seed for reproducibility
        rng = np.random.default_rng(42)

        # Generate random dense and sparse matrices
        dense_mat = rng.random((M, K)).astype(np.float64)
        nnz_ds = int(K * N * density)
        sparse_row_ds = rng.integers(0, K, size=nnz_ds)
        sparse_col_ds = rng.integers(0, N, size=nnz_ds)
        sparse_data_ds = rng.random(nnz_ds)
        coo_ds = coo_matrix((sparse_data_ds, (sparse_row_ds, sparse_col_ds)), shape=(K, N))

    elif data_source == "water":
        try:
            dense_mat = np.load(f"{data_path}/data_dense_water.npy").astype(np.float64)
            coo_ds = scipy.sparse.load_npz(f"{data_path}/data_sparse_water.npz")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Water dataset not found: {e}") from e

    elif data_source == "TiS2":
        try:
            dense_mat = np.load(f"{data_path}/data_dense_TiS2_3x3.npy").astype(np.float64)
            coo_ds = scipy.sparse.load_npz(f"{data_path}/data_sparse_TiS2_3x3.npz")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"TiS2 dataset not found: {e}") from e

    elif data_source == "graphene":
        try:
            dense_mat = np.load(f"{data_path}/data_dense_graphene_3x3.npy").astype(np.float64)
            coo_ds = scipy.sparse.load_npz(f"{data_path}/data_sparse_graphene_3x3.npz")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Graphene dataset not found: {e}") from e

    elif data_source == "ZrS2":
        try:
            dense_mat = np.load(f"{data_path}/data_dense_ZrS2_3x3.npy").astype(np.float64)
            coo_ds = scipy.sparse.load_npz(f"{data_path}/data_sparse_ZrS2_3x3.npz")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"ZrS2 dataset not found: {e}") from e

    else:
        raise ValueError(f"Unknown data source: {data_source}")

    M, K = dense_mat.shape
    N = coo_ds.shape[1]
    NNZ = coo_ds.nnz

    # Validate matrix dimensions for multiplication
    if coo_ds.shape[0] != K:
        raise ValueError(
            f"Matrix dimension mismatch: dense matrix has {K} columns, "
            f"sparse matrix has {coo_ds.shape[0]} rows. Cannot multiply."
        )

    density = coo_ds.nnz / (K * N)

    # Validate non-zero elements
    if NNZ == 0:
        raise ValueError("Sparse matrix has no non-zero elements")
    if NNZ > K * N:
        raise ValueError(f"Invalid sparse matrix: {NNZ} non-zero elements > {K * N} total elements")

    log(f"{dense_mat.shape=}, {coo_ds.shape=}, {coo_ds.nnz=}, density={100 * density:.2f}%")
    
    # Analyze sparsity pattern for v11 compatibility
    log("Analyzing sparsity pattern...")
    analysis = analyze_sparsity_pattern(coo_ds)
    log(f"Segmented compatible: {analysis['is_segmented_compatible']}")
    log(f"Compatibility ratio: {analysis['compatibility_ratio']:.3f}")
    log(f"Max segments per row: {analysis['max_segments']}")
    
    if not analysis['is_segmented_compatible']:
        log("⚠️  Dataset is not fully compatible with v11 algorithm (will fall back to v4)")

    return dense_mat, coo_ds, M, K, N, NNZ


def benchmark_algorithm(name: str, func, repeats: int = 3, *args, **kwargs) -> dict:
    """
    Benchmark a single algorithm multiple times and return results with statistics.
    
    This is a wrapper around the unified benchmark_function from utils.

    Args:
        name: Name of the algorithm
        func: Function to benchmark
        repeats: Number of times to repeat the benchmark
        *args, **kwargs: Arguments to pass to the function

    Returns:
        Dictionary with benchmark results
    """
    log(f"--- {name} ---")
    
    # Use unified benchmarking function from utils
    result = benchmark_function(func, *args, repeats=repeats, name=name, **kwargs)
    
    # Log results
    if result["error"] is None:
        log(f"{name} time: {result['mean_time']:.6f} ± {result['std_time']:.6f} seconds (n={result['valid_runs']})")
    else:
        log(f"{name} FAILED: {result['error']}")
    
    # Rename some fields for backwards compatibility
    return {
        "result": result["result"],
        "mean_time": result["mean_time"],
        "std_time": result["std_time"],
        "error": result["error"],
        "valid_runs": result["valid_runs"],
        "total_runs": result["total_runs"],
    }


def main():
    """Main benchmark function."""
    # Force garbage collection at the start for clean memory state
    gc.collect()

    parser = argparse.ArgumentParser(description="Benchmark OMP sparse matrix multiplication")
    parser.add_argument(
        "--data",
        choices=[
            "random",
            "water",
            "graphene",
            "TiS2",
            "ZrS2",
        ],
        default="random",
        help="Data source to use (default: random)",
    )
    parser.add_argument("--M", type=int, default=1000, help="Number of rows in dense matrix (for random data)")
    parser.add_argument("--K", type=int, default=2000, help="Number of columns in dense matrix (for random data)")
    parser.add_argument("--N", type=int, default=3000, help="Number of columns in sparse matrix (for random data)")
    parser.add_argument("--density", type=float, default=0.02, help="Density of sparse matrix (for random data)")
    parser.add_argument("--repeats", type=int, default=3, help="Number of times to repeat each test (default: 3)")
    parser.add_argument("--algorithm", nargs="+", default=["v4"], 
                        choices=["v4", "v11"], 
                        help="Algorithm(s) to benchmark (default: v4). Use multiple values to compare: --algorithm v4 v11")
    args = parser.parse_args()

    # Print environment variables for debugging
    log(f"OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'NOT SET')}")
    log(f"MKL_NUM_THREADS: {os.environ.get('MKL_NUM_THREADS', 'NOT SET')}")

    try:
        dense_mat, coo_ds, M, K, N, NNZ = load_data(args.data, args)
    except Exception as e:
        log(f"Error loading data: {e}")
        return 1

    # Benchmark results storage
    results = {}

    # Benchmark NumPy baseline
    baseline_result = benchmark_algorithm("NumPy baseline", lambda: np.dot(dense_mat, coo_ds.todense()), args.repeats)
    results["numpy"] = baseline_result

    if baseline_result["result"] is None:
        log("ERROR: NumPy baseline failed, cannot validate other algorithms")
        return 1

    # Benchmark specified algorithms
    for algorithm in args.algorithm:
        try:
            multiplier = OMPSparseMultiplier(algorithm)
            # Use a closure to properly capture the multiplier variable
            def create_benchmark_func(mult):
                return lambda: mult.multiply(dense_mat, coo_ds)
            
            alg_result = benchmark_algorithm(
                f"OMP Sparse {algorithm}",
                create_benchmark_func(multiplier),
                args.repeats,
            )
            results[f"omp_sparse_{algorithm}"] = alg_result

            if alg_result["result"] is not None:
                correct = np.allclose(baseline_result["result"], alg_result["result"], rtol=1e-10, atol=1e-10)
                results[f"omp_sparse_{algorithm}"]["correct"] = correct
                log(f"Result correct: {correct}")
            else:
                results[f"omp_sparse_{algorithm}"]["correct"] = False

            # Force garbage collection after algorithm test
            gc.collect()

        except Exception as e:
            log(f"Error benchmarking OMP Sparse {algorithm}: {e}")
            results[f"omp_sparse_{algorithm}"] = {"result": None, "mean_time": 0.0, "std_time": 0.0, "error": str(e), "correct": False}

    # Performance summary
    log("\n" + "=" * 60)
    log("PERFORMANCE SUMMARY")
    log("=" * 60)

    valid_results = [(k, v) for k, v in results.items() if v.get("correct", False) or k == "numpy"]
    omp_results = [(k, v) for k, v in valid_results if k.startswith("omp_sparse_")]
    
    if valid_results:
        log(f"{'Algorithm':<20} {'Time (s)':<20} {'vs NumPy':<12} {'Status'}")
        log("-" * 65)

        baseline_time = results["numpy"]["mean_time"]
        for name, data in valid_results:
            if name == "numpy":
                speedup_str = "1.00x"
                status = "✓ (baseline)"
            else:
                speedup = baseline_time / data["mean_time"] if data["mean_time"] > 0 else float("inf")
                speedup_str = f"{speedup:.2f}x"
                status = "✓" if data.get("correct", False) else "✗"

            time_str = f"{data['mean_time']:.6f} ± {data['std_time']:.6f}"
            display_name = name.replace("omp_sparse_", "OMP ") if name.startswith("omp_sparse_") else name
            log(f"{display_name:<20} {time_str:<20} {speedup_str:<12} {status}")

        # Inter-algorithm comparison if multiple OMP algorithms were tested
        if len(omp_results) > 1:
            log(f"\n{'Algorithm Comparison':<40}")
            log("-" * 40)
            
            # Sort by performance (fastest first)
            omp_sorted = sorted(omp_results, key=lambda x: x[1]["mean_time"])
            fastest_name, fastest_data = omp_sorted[0]
            fastest_time = fastest_data["mean_time"]
            
            for name, data in omp_sorted:
                if name == fastest_name:
                    comparison = "fastest"
                else:
                    ratio = data["mean_time"] / fastest_time
                    comparison = f"{ratio:.2f}x slower"
                
                display_name = name.replace("omp_sparse_", "")
                log(f"{display_name:<20} {comparison}")

        # Show failed algorithms
        failed_results = [(k, v) for k, v in results.items() if v.get("error") and k != "numpy"]
        if failed_results:
            log("\nFailed algorithms:")
            for name, data in failed_results:
                time_str = f"{data['mean_time']:.6f} ± {data['std_time']:.6f}"
                display_name = name.replace("omp_sparse_", "OMP ") if name.startswith("omp_sparse_") else name
                log(f"{display_name:<20} {time_str:<20} {'N/A':<12} ERROR")
    else:
        log("No valid results to compare")

    return 0


if __name__ == "__main__":
    sys.exit(main())
