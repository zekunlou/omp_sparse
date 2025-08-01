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

# Default: use f2py with distutils backend  (doesn't work if using python>=3.12 or numpy>=1.23.0)
F2PY_SYSTEM := $(shell which f2py || which f2py3)
F2PY_FLAGS_SYSTEM = -c --f90flags="$(FFLAGS)" -lgomp -lm

# Meson: use f2py with meson backend
F2PY_MESON := $(shell which f2py || which f2py3)
F2PY_FLAGS_MESON = -c --backend meson --f90flags="$(FFLAGS)" --opt="-O2" -lgomp -lm

.PHONY: all clean install test system meson develop

# Default target - use meson backend for modern Python/NumPy compatibility
all: meson

# Create lib directory if it doesn't exist
$(LIBDIR):
	mkdir -p $(LIBDIR)

# System f2py build
system: $(TARGET)

$(TARGET): $(SRC) | $(LIBDIR)
	cd $(LIBDIR) && $(F2PY_SYSTEM) $(F2PY_FLAGS_SYSTEM) -m omp_sparse ../../$(SRC)

# Meson backend build
meson: $(LIBDIR)
	cd $(LIBDIR) && env FFLAGS="$(FFLAGS)" $(F2PY_MESON) $(F2PY_FLAGS_MESON) -m omp_sparse ../../$(SRC)
# NOTE: it seems `env FFLAGS="$(FFLAGS)"` is critical to meson's using openmp, but IDK why

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
test: meson
	$(PYTHON) -m pytest tests/test_omp_sparse.py -v

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
	@echo "  make         - Build with meson backend (default, modern Python/NumPy)"
	@echo "  make system  - Build with system f2py (legacy, Python<3.12/NumPy<1.23)"
	@echo "  make meson   - Build with meson backend"
	@echo "  make develop - Install in development mode"
	@echo "  make install - Install package"
	@echo "  make wheel   - Build wheel distribution"
	@echo "  make sdist   - Build source distribution"
	@echo "  make test    - Run test suite"
	@echo "  make clean   - Clean build artifacts"
	@echo "  make help    - Show this help message"
	@echo ""
	@echo "For development:"
	@echo "  1. make develop  # Install in development mode"
	@echo "  2. make test     # Run tests"
