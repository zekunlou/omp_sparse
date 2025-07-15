project('omp_sparse', 
        ['c', 'fortran'],
        version: '1.0.0',
        default_options: [
            'warning_level=1',
            'buildtype=release',
        ])

# Python and NumPy dependencies
py = import('python').find_installation(pure: false)
py_dep = py.dependency()

# Find NumPy
run_command(py, ['-c', 'import numpy; print(numpy.get_include())'], check: true)
numpy_include = run_command(py, ['-c', 'import numpy; print(numpy.get_include())'], check: true).stdout().strip()

# Fortran compiler
fc = meson.get_compiler('fortran')

# OpenMP dependency
openmp_dep = dependency('openmp', language: 'fortran', required: true)

# Fortran compiler flags
fortran_args = ['-fopenmp', '-fPIC', '-O3']

# Add architecture-specific optimizations if available
if fc.get_id() == 'gcc'
    fortran_args += ['-march=native', '-mtune=native']
endif

# F2PY module generation
fortran_sources = files('src/fortran/omp_sparse.f90')

# Custom target for F2PY compilation
f2py_module = custom_target(
    'omp_sparse_f2py',
    input: fortran_sources,
    output: 'omp_sparse.cpython-311-x86_64-linux-gnu.so',
    command: [
        'f2py',
        '-c',
        '--f90flags=' + ' '.join(fortran_args),
        '--opt=-O3',
        '-lgomp',
        '-lm',
        '-m', 'omp_sparse',
        '@INPUT@'
    ],
    install: true,
    install_dir: py.get_install_dir() / 'omp_sparse' / 'lib'
)

# Install Python package files
py.install_sources([
    'omp_sparse/__init__.py',
], subdir: 'omp_sparse')

# Install data files if they exist
if fs.exists('tests/data')
    install_subdir('tests/data', install_dir: py.get_install_dir() / 'omp_sparse' / 'data')
endif