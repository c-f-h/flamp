__version__ = '1.0.1'

import gmpy2
import numpy as np

def ldexp(x, n):
    # work around a bug in gmpy2:
    # mul_2exp does not accept negative exponents
    if n >= 0:
        return gmpy2.mul_2exp(x, n)
    else:
        return x / gmpy2.mul_2exp(1, -n)

# monkey-patch for compatibility with mpmath context
gmpy2.mpf = gmpy2.mpfr
gmpy2.isinf = gmpy2.is_infinite
gmpy2.isnan = gmpy2.is_nan
gmpy2.ldexp = ldexp

# precision in binary digits
gmpy2.precision = lambda: gmpy2.get_context().precision
# unit roundoff
gmpy2.epsilon   = lambda: ldexp(gmpy2.mpf(1), 1 - gmpy2.precision())

def prec_to_dps(n):
    """Return number of accurate decimals that can be represented
    with a precision of n bits."""
    return max(1, int(round(int(n)/3.3219280948873626)-1))

def dps_to_prec(n):
    """Return the number of bits required to represent n decimals
    accurately."""
    return max(1, int(round((int(n)+1)*3.3219280948873626)))

def get_precision():
    """Return the current precision in binary digits."""
    return gmpy2.get_context().precision

def set_precision(prec):
    """Set the working precision in binary digits."""
    gmpy2.get_context().precision = prec

def get_dps():
    """Return the current precision in decimal digits (approximate)."""
    return prec_to_dps(get_precision())

def set_dps(dps):
    """Set the working precision in decimal digits (approximate)."""
    set_precision(dps_to_prec(dps))

def extraprec(n):
    """Return a context manager (for use in a `with` statement) which
    temporarily increases the working precision by the given amount."""
    prec = get_precision()
    return gmpy2.local_context(precision=prec + n)


import functools
from . import linalg
from . import eigen
from . import eigen_symmetric

# import functions while fixing ctx argument to gmpy2
for orig in [
        linalg.lu_solve, linalg.qr_solve, linalg.cholesky_solve,
        linalg.L_solve, linalg.U_solve,
        linalg.inverse, linalg.det,
        linalg.lu, linalg.qr, linalg.cholesky,
        eigen.eig, eigen.hessenberg, eigen.schur,
        eigen_symmetric.eigh, eigen_symmetric.svd,
    ]:
    _func = functools.partial(orig, gmpy2)
    _func.__name__ = orig.__name__
    _func.__doc__ = orig.__doc__
    globals()[orig.__name__] = _func

from .array import zeros, ones, empty, eye, linspace, vector_norm, to_mp, _gmpy2_vectorize

# array-aware versions of some special functions

exp    = _gmpy2_vectorize(gmpy2.exp)
sqrt   = _gmpy2_vectorize(gmpy2.sqrt)
sin    = _gmpy2_vectorize(gmpy2.sin)
cos    = _gmpy2_vectorize(gmpy2.cos)
tan    = _gmpy2_vectorize(gmpy2.tan)
sinh   = _gmpy2_vectorize(gmpy2.sinh)
cosh   = _gmpy2_vectorize(gmpy2.cosh)
tanh   = _gmpy2_vectorize(gmpy2.tanh)
square = _gmpy2_vectorize(gmpy2.square)
log    = _gmpy2_vectorize(gmpy2.log)
log2   = _gmpy2_vectorize(gmpy2.log2)
log10  = _gmpy2_vectorize(gmpy2.log10)
