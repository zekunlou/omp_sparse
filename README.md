# OMP Sparse

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![OpenMP](https://img.shields.io/badge/OpenMP-enabled-orange.svg)](https://www.openmp.org/)

A high-performance Python package for OpenMP-accelerated dense-sparse matrix multiplication, optimized for scientific computing and computational chemistry applications.

## Features

- **Dual Algorithm Support**: Both v4 (general CSC) and v11 (segmented) algorithms for different matrix types
- **High Performance**: OpenMP-parallelized Fortran implementation with up to 2.3x speedup over NumPy
- **Intelligent Algorithm Selection**: Automatic sparsity pattern analysis with smart fallback mechanisms
- **Scientific Computing Ready**: Validated on real computational chemistry datasets (water, graphene, TiS2)
- **Easy Integration**: Clean Python API with NumPy and SciPy compatibility
- **Memory Efficient**: Automatic memory management and garbage collection
- **Comprehensive Testing**: Full test suite with correctness validation and edge case handling

## Installation

### Prerequisites

- Python 3.8+
- NumPy >= 1.20.0
- SciPy >= 1.6.0
- GCC with OpenMP support
- F2Py (included with NumPy)

### From Source

```bash
git clone https://github.com/zekunlou/omp_sparse.git
cd omp_sparse
make  # Build the Fortran extension with system f2py (default)
pip install -e .  # Install in development mode
```

### Build Methods

The package supports two build backends via f2py:

```bash
# Using f2py with meson backend (often faster performance)
make meson
pip install -e .

# Using system f2py with numpy backend (doesn't work if using python>=3.12 or numpy>=1.23.0)
# this will be removed in the future
make system
pip install -e .
```

The meson backend requires meson to be installed.

## Quick Start

### Basic Usage

```python
import numpy as np
import scipy.sparse as sp
from omp_sparse import OMPSparseMultiplier, multiply_dense_sparse

# Create test matrices
dense_matrix = np.random.random((1000, 2000)).astype(np.float64)
sparse_matrix = sp.random(2000, 3000, density=0.02, format='csc')

# Method 1: Using v4 algorithm (general purpose)
multiplier_v4 = OMPSparseMultiplier(algorithm="v4")
result_v4 = multiplier_v4.multiply(dense_matrix, sparse_matrix)

# Method 2: Using v11 algorithm (segmented matrices with auto-fallback)
multiplier_v11 = OMPSparseMultiplier(algorithm="v11")
result_v11 = multiplier_v11.multiply(dense_matrix, sparse_matrix)

# Method 3: Using convenience function
result = multiply_dense_sparse(dense_matrix, sparse_matrix, algorithm="v4")

print(f"Result shape: {result_v4.shape}")
print(f"Results match: {np.allclose(result_v4, result_v11)}")
```

### Algorithm Selection Guide

```python
from omp_sparse.utils import analyze_sparsity_pattern

# Analyze your sparse matrix
analysis = analyze_sparsity_pattern(sparse_matrix)
print(f"Segmented compatible: {analysis['is_segmented_compatible']}")
print(f"Compatibility ratio: {analysis['compatibility_ratio']:.3f}")

# Choose algorithm based on sparsity pattern
if analysis['is_segmented_compatible']:
    algorithm = "v11"  # May have different performance characteristics
else:
    algorithm = "v4"   # Recommended for general use
```

### Benchmarking

Please set `OMP_NUM_THREADS` to the number of CPU cores you want to use before running benchmarks.

```python
import time
import numpy as np
import scipy.sparse as sp
from omp_sparse import OMPSparseMultiplier, benchmark_dense_dot_dense

# Create test matrices
dense = np.random.random((1000, 2000)).astype(np.float64)
sparse = sp.random(2000, 3000, density=0.02, format='csc')

# Benchmark against NumPy
multiplier = OMPSparseMultiplier("v4")
benchmark_result = multiplier.benchmark(dense, sparse, repeats=5)

print(f"Algorithm: {benchmark_result['algorithm']}")
print(f"Mean time: {benchmark_result['mean_time']:.6f} +- {benchmark_result['std_time']:.6f} seconds")
print(f"Correct: {benchmark_result['correct']}")
print(f"Matrix shape: {benchmark_result['matrix_shape']}")
print(f"Density: {benchmark_result['density']:.4f}")

# Compare with dense-dense multiplication baseline
dense_baseline = benchmark_dense_dot_dense(dense, sparse, repeats=5)

print(f"\nBaseline (NumPy dense @ dense):")
print(f"Mean time: {dense_baseline['mean_time']:.6f} +- {dense_baseline['std_time']:.6f} seconds")
print(f"Matrix shape: {dense_baseline['matrix_shape']}")
print(f"Density: {dense_baseline['density']:.4f}")

# Calculate speedup
speedup = dense_baseline['mean_time'] / benchmark_result['mean_time']
print(f"\nSpeedup over dense multiplication: {speedup:.2f}x")
```

## Command Line Interface

The package includes a command-line benchmarking tool with algorithm comparison support:

```bash
# Set number of threads for optimal performance
export OMP_NUM_THREADS=8 && export MKL_NUM_THREADS=8

# Benchmark single algorithm
python -m omp_sparse.benchmark --data water --algorithm v4

# Compare v4 vs v11 algorithms
python -m omp_sparse.benchmark --data water --algorithm v4 v11 --repeats 5

# Test with scientific datasets
python -m omp_sparse.benchmark --data graphene --algorithm v4 v11
python -m omp_sparse.benchmark --data TiS2 --algorithm v4 v11

# Benchmark with random data
python -m omp_sparse.benchmark --data random --M 1000 --K 2000 --N 3000 --density 0.02 --algorithm v4 v11
```

**New Features:**
- **Multi-algorithm comparison**: Compare v4 vs v11 performance in a single run
- **Automatic sparsity analysis**: Shows matrix compatibility with v11 algorithm
- **Enhanced output**: Performance comparison tables and statistical analysis

## Performance Tuning

### OpenMP Configuration

For optimal performance, configure OpenMP thread settings:

```bash
# Set the number of threads to match your CPU cores
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# For Intel MKL users
export MKL_NUM_THREADS=8
export MKL_DOMAIN_NUM_THREADS="MKL_DOMAIN_BLAS=8"
```

### Memory Considerations

- Use `float64` (double precision) for dense matrices
- CSC format is preferred for sparse matrices (automatic conversion if needed)
- The package handles memory management automatically with garbage collection

## Algorithm Details

### v4 Algorithm (Recommended)

The v4 algorithm provides excellent general-purpose performance:

- **CSC format optimization**: Processes compressed sparse column format efficiently
- **Column-wise parallelization**: Distributes sparse matrix columns across threads
- **Dynamic scheduling**: Automatically balances load across threads
- **O(nnz) complexity**: Only processes non-zero elements
- **Proven performance**: 1.5-2.3x speedup over NumPy on real datasets

### v11 Algorithm (Segmented Sparse)

The v11 algorithm is specialized for matrices with specific sparsity patterns:

- **Segmented format**: Optimized for matrices where each row has one contiguous segment
- **Column-wise parallelization**: Avoids race conditions by parallelizing over output columns
- **Automatic compatibility detection**: Falls back to v4 for incompatible matrices
- **O(N×K) complexity**: Trades computation for potential memory access improvements
- **Research foundation**: Enables future cache-optimization techniques

### Performance Characteristics

| Algorithm | Best For | Complexity | Typical Performance |
|-----------|----------|------------|-------------------|
| **v4** | General sparse matrices | O(nnz) | 1.5-2.3x vs NumPy |
| **v11** | Segmented sparse matrices | O(N×K) | Research/specialized use |

### Technical Specifications

- **Input**: Dense matrix (MxK, float64) and sparse matrix (KxN, COO/CSC)
- **Output**: Dense result matrix (MxN, float64)
- **Parallelization**: OpenMP with static/dynamic scheduling
- **Memory**: Automatic contiguous array conversion
- **Precision**: Double precision (float64) throughout
- **Compatibility**: Both algorithms support the same matrix formats

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
make test

# Run specific test categories
python -m pytest tests/test_omp_sparse.py::TestMatrixMultiplication -v
python -m pytest tests/test_omp_sparse.py::TestV11Algorithm -v
python -m pytest tests/test_omp_sparse.py::TestPerformance -v
python -m pytest tests/test_omp_sparse.py::TestEdgeCases -v

# Run with scientific datasets (if available)
python -m pytest tests/test_omp_sparse.py::TestScientificDatasets -v
```

## API Reference

### OMPSparseMultiplier

```python
class OMPSparseMultiplier:
    def __init__(self, algorithm: str = "v4")  # "v4" or "v11"
    def multiply(self, dense_matrix: np.ndarray, sparse_matrix: Union[coo_matrix, csc_matrix]) -> np.ndarray
    def benchmark(self, dense_matrix: np.ndarray, sparse_matrix: Union[coo_matrix, csc_matrix], 
                  baseline_result: Optional[np.ndarray] = None, repeats: int = 3) -> dict
```

### Convenience Functions

def multiply_dense_sparse(dense_matrix: np.ndarray, sparse_matrix: Union[coo_matrix, csc_matrix], 
                         algorithm: str = "v4") -> np.ndarray  # "v4" or "v11"

def get_available_algorithms() -> List[str]  # Returns ["v4", "v11"]
def is_module_available() -> bool

# Sparsity analysis utilities
from omp_sparse.utils import analyze_sparsity_pattern, convert_to_segmented_format
def analyze_sparsity_pattern(sparse_matrix) -> dict
def convert_to_segmented_format(sparse_matrix) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
```

## Development

### Building from Source

```bash
# Clone repository
git clone https://github.com/zekunlou/omp_sparse.git
cd omp_sparse

# Install development dependencies
pip install -e .[dev]

# Build extension
make develop

# Run tests
make test

# Run benchmarks
make benchmark
```

### Future Enhancements

The package is designed for easy extension:

- Additional algorithms (v1-v3, v5-v10) can be added
- Support for other sparse formats (COO, CSR)
- GPU acceleration support
- Additional optimization techniques

## Troubleshooting

### Common Issues

1. **Import Error**: Ensure the Fortran extension is compiled with `make`
2. **OpenMP Not Working**: Check that GCC has OpenMP support (`gcc -fopenmp`)
3. **Performance Issues**: Set `OMP_NUM_THREADS` to match your CPU cores
4. **Memory Errors**: Reduce matrix size or increase available memory

### Debug Information

```python
import omp_sparse

# Check module availability
print(f"Module available: {omp_sparse.is_module_available()}")
print(f"Available algorithms: {omp_sparse.get_available_algorithms()}")

# Check environment
import os
print(f"OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'NOT SET')}")
```

## Citation

If you use this software in academic research, please cite:

```bibtex
@software{omp_sparse,
  author = {Zekun Lou},
  title = {OMP Sparse: High-Performance Dense-Sparse Matrix Multiplication},
  year = {2025},
  url = {https://github.com/zekunlou/omp_sparse}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
