import numpy as np

# Ensure NumPy 2.x compatibility with Numba CUDA arrayobj
if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack
