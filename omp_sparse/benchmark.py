"""
Benchmark module for omp_sparse package.

Provides command-line benchmarking functionality for performance testing.
"""

import argparse
import gc
import os
import sys
import time
from typing import Optional

import numpy as np
import scipy.sparse
from scipy.sparse import coo_matrix

from . import OMPSparseMultiplier

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

    return dense_mat, coo_ds, M, K, N, NNZ


def benchmark_algorithm(name: str, func, repeats: int = 3, *args, **kwargs) -> dict:
    """
    Benchmark a single algorithm multiple times and return results with statistics.

    Args:
        name: Name of the algorithm
        func: Function to benchmark
        repeats: Number of times to repeat the benchmark
        *args, **kwargs: Arguments to pass to the function

    Returns:
        Dictionary with benchmark results
    """
    log(f"--- {name} ---")
    times = []
    results = []
    errors = []

    for i in range(repeats):
        # Force garbage collection before each run
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
            log(f"{name} FAILED on run {i + 1}: {e}")

        # Force garbage collection after each run
        gc.collect()

    # Calculate statistics
    valid_times = [t for t, e in zip(times, errors) if e is None]

    if valid_times:
        mean_time = np.mean(valid_times)
        std_time = np.std(valid_times) if len(valid_times) > 1 else 0.0
        # Use the first successful result for correctness checking
        final_result = next((r for r, e in zip(results, errors) if e is None), None)
        final_error = None
        log(f"{name} time: {mean_time:.6f} ± {std_time:.6f} seconds (n={len(valid_times)})")
    else:
        mean_time = np.mean(times)
        std_time = np.std(times) if len(times) > 1 else 0.0
        final_result = None
        final_error = errors[0] if errors else "Unknown error"
        log(f"{name} FAILED: {final_error}")

    return {
        "result": final_result,
        "mean_time": mean_time,
        "std_time": std_time,
        "error": final_error,
        "valid_runs": len(valid_times),
        "total_runs": repeats,
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
    parser.add_argument("--algorithm", default="v4", help="Algorithm to use (default: v4)")
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

    # Benchmark v4 algorithm
    try:
        multiplier = OMPSparseMultiplier(args.algorithm)
        v4_result = benchmark_algorithm(
            f"OMP Sparse {args.algorithm}",
            lambda: multiplier.multiply(dense_mat, coo_ds),
            args.repeats,
        )
        results["omp_sparse"] = v4_result

        if v4_result["result"] is not None:
            correct = np.allclose(baseline_result["result"], v4_result["result"])
            results["omp_sparse"]["correct"] = correct
            log(f"Result correct: {correct}")
        else:
            results["omp_sparse"]["correct"] = False

        # Force garbage collection after algorithm test
        gc.collect()

    except Exception as e:
        log(f"Error benchmarking OMP Sparse: {e}")
        results["omp_sparse"] = {"result": None, "mean_time": 0.0, "std_time": 0.0, "error": str(e), "correct": False}

    # Performance summary
    log("\n" + "=" * 60)
    log("PERFORMANCE SUMMARY")
    log("=" * 60)

    valid_results = [(k, v) for k, v in results.items() if v.get("correct", False) or k == "numpy"]
    if valid_results:
        log(f"{'Algorithm':<20} {'Time (s)':<20} {'Speedup':<10} {'Status'}")
        log("-" * 60)

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
            log(f"{name:<20} {time_str:<20} {speedup_str:<10} {status}")

        # Show failed algorithms
        failed_results = [(k, v) for k, v in results.items() if v.get("error") and k != "numpy"]
        if failed_results:
            log("\nFailed algorithms:")
            for name, data in failed_results:
                time_str = f"{data['mean_time']:.6f} ± {data['std_time']:.6f}"
                log(f"{name:<20} {time_str:<20} {'N/A':<10} ERROR")
    else:
        log("No valid results to compare")

    return 0


if __name__ == "__main__":
    sys.exit(main())
