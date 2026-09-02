"""Dynamic-vocabulary learned sparse retrieval - pilot study package."""
import os

# This box has 256 cores; OpenBLAS is built for at most 128 threads and aborts if
# every core tries to enter BLAS at once.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "32")
