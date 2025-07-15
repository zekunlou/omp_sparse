# Makefile for omp_sparse package
# Provides both meson and system f2py build targets

# Fortran compiler
FC = gfortran

# Fortran compiler flags
FFLAGS = -fopenmp -fPIC -O2 -march=native -mtune=native

# Output directory
LIBDIR = omp_sparse/lib

# Target library name
TARGET = $(LIBDIR)/omp_sparse.cpython-311-x86_64-linux-gnu.so

# Source file
SRC = src/fortran/omp_sparse.f90

# Python executable (prefer conda if available)
PYTHON = python

# Default: use f2py with distutils backend (MOST COMPATIBLE)
F2PY_SYSTEM := /usr/bin/f2py
F2PY_FLAGS_SYSTEM = -c --f90flags="$(FFLAGS)" -lgomp -lm

# Meson: use f2py with meson backend
F2PY_MESON := $(shell which f2py || which f2py3)
F2PY_FLAGS_MESON = -c --backend meson --f90flags="$(FFLAGS)" --opt="-O2" -lgomp -lm

.PHONY: all clean install test system meson develop

# Default target - use system f2py for maximum compatibility
all: system

# Create lib directory if it doesn't exist
$(LIBDIR):
	mkdir -p $(LIBDIR)

# System f2py build (most compatible)
system: $(TARGET)

$(TARGET): $(SRC) | $(LIBDIR)
	cd $(LIBDIR) && $(F2PY_SYSTEM) $(F2PY_FLAGS_SYSTEM) -m omp_sparse ../../$(SRC)

# Meson backend build
meson: $(LIBDIR)
	cd $(LIBDIR) && $(F2PY_MESON) $(F2PY_FLAGS_MESON) -m omp_sparse ../../$(SRC)

# Install package in development mode
develop: meson
	$(PYTHON) -m pip install -e .

# Install package
install: 
	$(PYTHON) -m pip install .

# Build wheel
wheel:
	$(PYTHON) -m build --wheel

# Build source distribution
sdist:
	$(PYTHON) -m build --sdist

# Run tests
test: system
	$(PYTHON) -m pytest tests/ -v

# Run benchmarks
benchmark: system
	$(PYTHON) -m pytest tests/test_benchmark.py -v --benchmark-only

# Clean build artifacts
clean:
	rm -rf $(LIBDIR)
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf __pycache__/
	rm -rf omp_sparse/__pycache__/
	rm -rf tests/__pycache__/
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	find . -name "*.mod" -delete
	find . -name "*.o" -delete

# Help target
help:
	@echo "Available targets:"
	@echo "  make         - Build with system f2py (default, most compatible)"
	@echo "  make system  - Build with system f2py"
	@echo "  make meson   - Build with meson backend"
	@echo "  make develop - Install in development mode"
	@echo "  make install - Install package"
	@echo "  make wheel   - Build wheel distribution"
	@echo "  make sdist   - Build source distribution"
	@echo "  make test    - Run test suite"
	@echo "  make benchmark - Run benchmarks"
	@echo "  make clean   - Clean build artifacts"
	@echo "  make help    - Show this help message"
	@echo ""
	@echo "For development:"
	@echo "  1. make develop  # Install in development mode"
	@echo "  2. make test     # Run tests"
	@echo "  3. make benchmark # Run performance benchmarks"