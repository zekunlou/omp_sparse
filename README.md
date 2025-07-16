# OMP Sparse

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![OpenMP](https://img.shields.io/badge/OpenMP-enabled-orange.svg)](https://www.openmp.org/)

A high-performance Python package for OpenMP-accelerated dense-sparse matrix multiplication, optimized for scientific computing and computational chemistry applications.

## Features

- **High Performance**: OpenMP-parallelized Fortran implementation with up to 8x speedup (of course it depends)
- **Optimized Algorithm**: v4 algorithm provides excellent performance across matrix sizes and sparsity patterns
- **Scientific Computing Ready**: Tested on real computational chemistry datasets
- **Easy Integration**: Clean Python API with NumPy and SciPy compatibility
- **Memory Efficient**: Automatic memory management and garbage collection

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
# Using system f2py with numpy backend (most compatible, default)
make system
pip install -e .

# Using f2py with meson backend (often faster performance)
make meson
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

# Method 1: Using the multiplier class
multiplier = OMPSparseMultiplier(algorithm="v4")
result = multiplier.multiply(dense_matrix, sparse_matrix)

# Method 2: Using the convenience function
result = multiply_dense_sparse(dense_matrix, sparse_matrix, algorithm="v4")

print(f"Result shape: {result.shape}")
print(f"Result type: {type(result)}")
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

The package includes a command-line benchmarking tool:

```bash
# set number of threads
# export MKL_NUM_THREADS=16 && export OMP_NUM_THREADS=16 && <other commands>

# Benchmark with random data
omp-sparse-benchmark --data random --M 1000 --K 2000 --N 3000 --density 0.02

# Benchmark with scientific datasets (if available)
omp-sparse-benchmark --data water
omp-sparse-benchmark --data graphene
omp-sparse-benchmark --data TiS2

# Custom parameters
omp-sparse-benchmark --data random --M 500 --K 1000 --N 1500 --repeats 5
```

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

### v4 Algorithm

The v4 algorithm is the current implementation, featuring:

- **Column-wise parallelization**: Distributes sparse matrix columns across threads
- **Dynamic scheduling**: Automatically balances load across threads
- **Vectorized operations**: Optimized inner loops for better performance
- **CSC format optimization**: Efficient access patterns for compressed sparse columns

### Technical Specifications

- **Input**: Dense matrix (MxK, float64) and sparse matrix (KxN, COO/CSC)
- **Output**: Dense result matrix (MxN, float64)
- **Parallelization**: OpenMP with dynamic scheduling
- **Memory**: Automatic contiguous array conversion
- **Precision**: Double precision (float64) throughout

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
make test

# Run specific test categories
python -m pytest tests/test_omp_sparse.py::TestMatrixMultiplication -v
python -m pytest tests/test_omp_sparse.py::TestPerformance -v
python -m pytest tests/test_omp_sparse.py::TestEdgeCases -v

# Run with scientific datasets (if available)
python -m pytest tests/test_omp_sparse.py::TestScientificDatasets -v
```

## API Reference

### OMPSparseMultiplier

```python
class OMPSparseMultiplier:
    def __init__(self, algorithm: str = "v4")
    def multiply(self, dense_matrix: np.ndarray, sparse_matrix: Union[coo_matrix, csc_matrix]) -> np.ndarray
    def benchmark(self, dense_matrix: np.ndarray, sparse_matrix: Union[coo_matrix, csc_matrix], 
                  baseline_result: Optional[np.ndarray] = None, repeats: int = 3) -> dict
```

### Convenience Functions

```python
def multiply_dense_sparse(dense_matrix: np.ndarray, sparse_matrix: Union[coo_matrix, csc_matrix], 
                         algorithm: str = "v4") -> np.ndarray

def get_available_algorithms() -> List[str]
def is_module_available() -> bool
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
