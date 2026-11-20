"""Cap CPU thread pools. Call ``limit_threads()`` before numpy or torch are imported."""

import os

_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
         "VECLIB_MAXIMUM_THREADS")


def limit_threads(default_cap: int = 8) -> int:
    """Use $CCLM_NUM_THREADS, else $SLURM_CPUS_PER_TASK, else min(8, cpu count). Existing settings win."""
    n = os.environ.get("CCLM_NUM_THREADS") or os.environ.get("SLURM_CPUS_PER_TASK")
    n = int(n) if n else min(default_cap, os.cpu_count() or default_cap)
    for var in _VARS:
        os.environ.setdefault(var, str(n))
    return n
