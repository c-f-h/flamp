
# flamp - Faster linear algebra with multiple precision [![Build Status](https://github.com/c-f-h/flamp/actions/workflows/python-package.yml/badge.svg)](https://github.com/c-f-h/flamp/actions/workflows/python-package.yml) [![PyPI version](https://badge.fury.io/py/flamp.svg)](https://badge.fury.io/py/flamp)

`flamp` contains ports of many real and complex linear algebra routines from
the [`mpmath`](https://mpmath.org/) package, but using numpy object arrays
containing `gmpy2` multiprecision floating point numbers instead of the
`mpmath` floating point numbers. The resulting linear algebra routines are
typically by a factor of 10x-15x faster than those in `mpmath`.

`flamp` is based on `mpmath` by Fredrik Johansson and mpmath contributors;
in particular, the eigenvalue algorithms therein are by Timo Hartmann.
`flamp` is BSD-licensed.

## Installation

The package is written in pure Python and can simply be installed by

    pip install flamp

Its only dependencies are `numpy` and `gmpy2`, both of which have pre-built
packages readily available.

## List of functions

The following is a list of supported functions in the `flamp` module by category. All matrix and
vector arguments should be supplied as numpy arrays of `gmpy2` numbers,
although standard floating point numpy arrays will be automatically converted
in most cases.

Refer to the docstrings for further information.

### Linear algebra

These behave essentially like the corresponding functions in `mpmath`, with
some slight modifications. For instance, all functions for solving linear
systems accept either a single vector or an array of vectors for the right-hand
side.

- `lu_solve(A, b)` - solve a linear system using LU decomposition
- `qr_solve(A, b)` - solve a linear system using QR decomposition
- `cholesky_solve(A, b)` - solve a symmetric positive definite system using Cholesky decomposition
- `L_solve(L, b, unit_diag=False)` - solve a lower triangular system
- `U_solve(U, y)` - solve an upper triangular system
- `inverse(A)` - compute inverse of a square matrix
- `det(A)` - compute determinant of a square matrix
- `lu(A)` - compute LU decomposition of a square matrix
- `qr(A, mode='full')` - compute QR decomposition of a matrix; mode=`full`, `reduced`, `raw`
- `cholesky(A)` - compute Cholesky decomposition of a symmetric positive definite matrix
- `eig(A, left=False, right=True)` - compute eigenvalues and (optionally) left and right eigenvectors of a matrix
- `eigh(A, eigvals_only=False)` - compute eigenvalues and (optionally) the orthonormal eigenvectors of a real symmetric or complex Hermitian square matrix
- `hessenberg(A)` - compute Hessenberg decomposition `(Q, R)` of a square matrix
- `schur(A)` - compute Schur decomposition of a square matrix
- `svd(A, full_matrices=False, compute_uv=True)` - compute singular value decomposition (singular values and optionally the left and right singular vectors) of a matrix

### Array functions

Most of these behave essentially like their numpy counterparts, but work with
`gmpy2` extended precision numbers.

- `zeros(shape)`
- `ones(shape)`
- `empty(shape)`
- `eye(n)`
- `linspace(start, stop, num, endpoint=True)`
- `vector_norm(x)` - computes Euclidean norm of a vector
- `to_mp(A)` - converts an arbitrary numpy array (or list/tuple) into an array of `gmpy2` numbers, copying the input

### Utility functions

These functions are used to manipulate the working precision of the `gmpy2` library.

- `prec_to_dps(n)` - number of accurate decimals that can be represented with a precision of `n` bits
- `dps_to_prec(n)` - number of bits required to represent `n` decimals accurately
- `get_precision()` - get the current precision in binary digits
- `set_precision(prec)` - set the working precision in binary digits
- `get_dps()` get the current precision in decimal digits (approximate)
- `set_dps(dps)` - set the working precision in decimal digits (approximate)
- `extraprec(n)` - returns a context manager (for use in a `with` statement) which temporarily increases the working precision by the given amount

### Array-aware special functions

These functions work much like the corresponding functions in numpy in that they
automatically distribute over numpy arrays while computing in extended
precision.

- `exp(x)`
- `sqrt(x)`
- `sin(x)`
- `cos(x)`
- `tan(x)`
- `sinh(x)`
- `cosh(x)`
- `tanh(x)`
- `square(x)`
