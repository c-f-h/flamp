import numpy as np
import gmpy2
import numbers

def zeros(shape):
    """Create array of zeros of the given shape with MP floating point numbers."""
    zero = gmpy2.mpfr(0)
    return np.full(shape, zero)

def ones(shape):
    """Create array of ones of the given shape with MP floating point numbers."""
    one = gmpy2.mpfr(1)
    return np.full(shape, one)

def empty(shape):
    """Create array off the given shape suitable for holding MP floating point numbers."""
    return np.empty(shape, dtype=object)

def eye(n):
    """Create identity matrix of size `n` with MP floating point numbers."""
    I = zeros((n, n))
    one = gmpy2.mpfr(1)
    np.fill_diagonal(I, one)
    return I

def swap_rows(A, i, j):
    """Swap rows i and j of 2D numpy array."""
    A[[i, j]] = A[[j, i]]

def contains_complex(A):
    """Return True if the array contains any (non-real) complex numbers."""
    return any(isinstance(x, numbers.Complex) and not isinstance(x, numbers.Real)
            for x in A.flat)

def to_mp(A):
    """Ensures an array contains mpf or mpc numbers. Always copies the input."""
    if contains_complex(A):
        return np.vectorize(gmpy2.mpc)(A)
    else:
        return np.vectorize(gmpy2.mpfr)(A)

def vector_norm(x):
    """Compute Euclidean norm of vector `x`."""
    return gmpy2.sqrt(gmpy2.fsum((abs(x)**2).flat))
