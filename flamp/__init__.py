__version__ = '1.0.0'

import gmpy2

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
gmpy2.ldexp = ldexp

# precision in binary digits
gmpy2.precision = lambda: gmpy2.get_context().precision
# unit roundoff
gmpy2.epsilon   = lambda: ldexp(gmpy2.mpf(1), 1 - gmpy2.precision())


import functools
from . import linalg
from . import eigen
from . import eigen_symmetric

# import functions while fixing ctx argument to gmpy2
for orig in [
        linalg.qr,
        eigen.eig, eigen.hessenberg, eigen.schur,
        eigen_symmetric.eigh, eigen_symmetric.svd,
    ]:
    _func = functools.partial(orig, gmpy2)
    _func.__name__ = orig.__name__
    _func.__doc__ = orig.__doc__
    globals()[orig.__name__] = _func

from .array import zeros, ones, eye
