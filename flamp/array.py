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

def linspace(start, stop, num, endpoint=True):
    """Return an array of evenly spaced multiprecision numbers over a specified interval.

    This behaves like the numpy version.
    """
    if endpoint:
        if num == 1:
            x = zeros(1)
        x = np.arange(num) / gmpy2.mpf(num - 1)
    else:
        x = np.arange(num) / gmpy2.mpf(num)
    start, stop = gmpy2.mpfr(start), gmpy2.mpfr(stop)
    return (stop - start) * x + start

def swap_rows(A, i, j):
    """Swap rows i and j of 2D numpy array."""
    A[[i, j]] = A[[j, i]]

def contains_complex(A):
    """Return True if the array contains any (non-real) complex numbers."""
    return any(isinstance(x, numbers.Complex) and not isinstance(x, numbers.Real)
            for x in A.flat)

def to_mp(A):
    """Ensures an array contains mpf or mpc numbers. Always copies the input."""
    A = np.asanyarray(A)
    if issubclass(A.dtype.type, numbers.Integral):
        # mpfr cannot deal with numpy fixed-width integer types - convert to Python int
        return np.vectorize(lambda x: gmpy2.mpfr(int(x)))(A)
    if contains_complex(A):
        return np.vectorize(gmpy2.mpc)(A)
    else:
        return np.vectorize(gmpy2.mpfr)(A)

def vector_norm(x):
    """Compute Euclidean norm of vector `x`."""
    return gmpy2.sqrt(gmpy2.fsum((abs(x)**2).flat))
