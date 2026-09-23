"""kilix-ml. Training dependencies are never imported by the runtime."""
import os

# Tiny matrix products are faster without a machine-wide BLAS thread pool.
# Explicit caller settings win; set these before importing NumPy elsewhere too.
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_key, '1')
__version__ = "0.1.0"
