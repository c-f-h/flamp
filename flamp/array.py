import numpy as np
import gmpy2

def zeros(shape):
    zero = gmpy2.mpfr(0)
    return np.full(shape, zero)

def ones(shape):
    one = gmpy2.mpfr(1)
    return np.full(shape, one)

def eye(n):
    I = zeros((n, n))
    one = gmpy2.mpfr(1)
    np.fill_diagonal(I, one)
    return I
