import numpy as np
import numbers

from .array import (zeros, empty, eye, swap_rows,
        contains_complex, to_mp, vector_norm)

# Code based on the linalg module from mpmath.

def LU_decomp(ctx, A, overwrite=False):
    """
    LU-factorization of a n*n matrix using the Gauss algorithm.
    Returns L and U in one matrix and the pivot indices.

    Use overwrite to specify whether A will be overwritten with L and U.
    """
    if not A.shape[0] == A.shape[1]:
        raise ValueError('need n*n matrix')
    if not overwrite:
        A = to_mp(A)
    tol = abs(np.linalg.norm(A, ord=1) * ctx.epsilon()) # each pivot element has to be bigger
    n = A.shape[0]
    p = [None]*(n - 1)
    for j in range(n - 1):
        # pivoting, choose max(abs(reciprocal row sum)*abs(pivot element))
        biggest = 0
        for k in range(j, n):
            s = ctx.fsum([abs(A[k,l]) for l in range(j, n)])
            if abs(s) <= tol:
                raise ZeroDivisionError('matrix is numerically singular')
            elif ctx.isnan(s):
                raise ValueError('matrix contains nans')
            current = 1/s * abs(A[k,j])
            if current > biggest: # TODO: what if equal?
                biggest = current
                p[j] = k
        # swap rows according to p
        swap_rows(A, j, p[j])
        if abs(A[j,j]) <= tol:
            raise ZeroDivisionError('matrix is numerically singular')
        # calculate elimination factors and add rows
        for i in range(j + 1, n):
            A[i,j] /= A[j,j]
            for k in range(j + 1, n):
                A[i,k] -= A[i,j]*A[j,k]
    if abs(A[n - 1, n - 1]) <= tol:
        raise ZeroDivisionError('matrix is numerically singular')
    return A, p

def L_solve(ctx, L, b, p=None, unit_diag=False):
    """
    Solve the lower part of a LU factorized matrix for y.
    If `unit_diag` is True, the diagonal of L is assumed to be 1.

    b may be a vector or matrix.
    """
    if L.shape[0] != L.shape[1]:
        raise RuntimeError("need n*n matrix")
    n = L.shape[0]
    if b.shape[0] != n:
        raise ValueError("vector b has incorrect shape")
    b = to_mp(b)
    if p: # swap b according to p
        for k in range(len(p)):
            swap_rows(b, k, p[k])
    # solve
    for i in range(n):
        for j in range(i):
            b[i] -= L[i,j] * b[j]
        if not unit_diag:
            b[i] /= L[i,i]
    return b

def U_solve(ctx, U, y):
    """
    Solve the upper part of a LU factorized matrix for x.

    y may be a vector or matrix.
    """
    if U.shape[0] != U.shape[1]:
        raise RuntimeError("need n*n matrix")
    n = U.shape[0]
    if y.shape[0] != n:
        raise ValueError("vector y has incorrect shape")
    x = to_mp(y)
    for i in range(n - 1, -1, -1):
        for j in range(i + 1, n):
            x[i] -= U[i,j] * x[j]
        x[i] /= U[i,i]
    return x

def lu_solve(ctx, A, b, real=False):
    """
    Ax = b => x

    b may be a vector or matrix.

    Solve a determined or overdetermined linear equations system.
    Fast LU decomposition is used, which is less accurate than QR decomposition
    (especially for overdetermined systems), but it's twice as efficient.
    Use qr_solve if you want more precision or have to solve a very ill-
    conditioned system.

    If you specify real=True, it does not check for overdeterminded complex
    systems.
    """
    # do not overwrite A nor b
    A, b = to_mp(A), to_mp(b)
    if A.shape[0] != b.shape[0]:
        raise ValueError('right-hand side has incorrect size')
    if A.shape[0] < A.shape[1]:
        raise ValueError('cannot solve underdetermined system')
    if A.shape[0] > A.shape[1]:
        # use least-squares method if overdetermined
        # (this increases errors)
        AH = A.T.conj()
        A = AH @ A
        b = AH @ b
        if real or not contains_complex(A):
            x = cholesky_solve(ctx, A, b)
        else:
            x = lu_solve(ctx, A, b)
    else:
        # LU factorization
        A, p = LU_decomp(ctx, A)
        b = L_solve(ctx, A, b, p, unit_diag=True)
        x = U_solve(ctx, A, b)
    return x

def lu(ctx, A):
    """
    A -> P, L, U

    LU factorisation of a square matrix A. L is the lower, U the upper part.
    P is the permutation matrix indicating the row swaps.

    P*A = L*U

    If you need efficiency, use the low-level method LU_decomp instead, it's
    much more memory efficient.
    """
    # get factorization
    A, p = LU_decomp(ctx, A)
    L = np.tril(A, -1)
    np.fill_diagonal(L, ctx.mpf(1))
    U = np.triu(A)
    # calculate permutation matrix
    P = eye(A.shape[0])
    for k in range(len(p)):
        swap_rows(P, k, p[k])
    return P, L, U

def unitvector(ctx, n, i):
    """
    Return the i-th n-dimensional unit vector.
    """
    assert 0 <= i < n, 'this unit vector does not exist'
    e = zeros(n)
    e[i] = ctx.mpf(1)
    return e

def inverse(ctx, A):
    """
    Calculate the inverse of a matrix.

    If you want to solve an equation system Ax = b, it's recommended to use
    solve(A, b) instead, it's about 3 times more efficient.
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError('can only compute inverse of square matrix')
    # do not overwrite A
    A = to_mp(A)
    # get LU factorisation
    A, p = LU_decomp(ctx, A)
    B = empty(A.shape)
    # calculate unit vectors and solve corresponding system to get columns
    n = A.shape[0]
    for i in range(n):
        e = unitvector(ctx, n, i)
        y = L_solve(ctx, A, e, p, unit_diag=True)
        B[:, i] = U_solve(ctx, A, y)
    return B

def householder(ctx, A, num_dofs):
    """
    (A|b) -> H, p, x, res

    (A|b) is the coefficient matrix with left hand side of an optionally
    overdetermined linear equation system.
    H and p contain all information about the transformation matrices.
    x is the solution, res the residual.
    """
    m, n = A.shape
    if m < num_dofs:
        raise RuntimeError("system is underdetermined")
    # calculate Householder matrix
    p = empty(num_dofs)
    eps = ctx.epsilon()
    for j in range(num_dofs):
        #s = ctx.fsum(abs(A[i,j])**2 for i in range(j, m))
        s = ctx.fsum(abs(A[j:m,j])**2)
        if not abs(s) > eps:
            raise ValueError('matrix is numerically singular')
        p[j] = -ctx.sign(A[j,j].real) * ctx.sqrt(s)
        kappa = ctx.mpf(1) / (s - p[j] * A[j,j])
        A[j,j] -= p[j]
        for k in range(j+1, n):
            y = ctx.fsum(A[i,j].conjugate() * A[i,k] for i in range(j, m)) * kappa
            for i in range(j, m):
                A[i,k] -= A[i,j] * y
    # solve Rx = c1
    x = A[:num_dofs, num_dofs:].copy()      # collect all right-hand sides
    for k in range(A.shape[1] - num_dofs):
        for i in range(num_dofs - 1, -1, -1):
            x[i, k] -= ctx.fsum(A[i,j] * x[j, k] for j in range(i + 1, num_dofs))
            x[i, k] /= p[i]
    # calculate residual
    if m > num_dofs:        # overdetermined system
        # TODO: use slicing
        r = np.array([A[m-1-i, num_dofs:] for i in range(m - num_dofs)])
    else:
        # determined system, residual should be 0
        r = zeros(m)
    return A, p, x, r

def qr_solve(ctx, A, b, norm=None, res=False):
    """
    Ax = b => x, ||Ax - b||

    Solve a determined or overdetermined linear equations system and
    calculate the norm of the residual (error).
    QR decomposition using Householder factorization is applied, which gives very
    accurate results even for ill-conditioned matrices. qr_solve is twice as
    efficient.
    """
    # NB: unclear what `res` does for overdetermined systems!
    if norm is None:
        norm = vector_norm
    # do not overwrite A nor b
    A, b = to_mp(A), to_mp(b)
    if A.shape[0] < A.shape[1]:
        raise ValueError('cannot solve underdetermined system')
    H, p, x, r = householder(ctx, np.column_stack((A, b)), A.shape[1])
    if b.ndim == 1:
        x = x.ravel()       # only one solution vector; x has shape (n,1)
    return (x, r) if res else x

def cholesky(ctx, A, tol=None):
    r"""
    Cholesky decomposition of a symmetric positive-definite matrix `A`.
    Returns a lower triangular matrix `L` such that `A = L \times L^T`.
    More generally, for a complex Hermitian positive-definite matrix,
    a Cholesky decomposition satisfying `A = L \times L^H` is returned.

    The Cholesky decomposition can be used to solve linear equation
    systems twice as efficiently as LU decomposition, or to
    test whether `A` is positive-definite.

    The optional parameter ``tol`` determines the tolerance for
    verifying positive-definiteness.

    **Examples**

    Cholesky decomposition of a positive-definite symmetric matrix::

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> A = eye(3) + hilbert(3)
        >>> nprint(A)
        [     2.0      0.5  0.333333]
        [     0.5  1.33333      0.25]
        [0.333333     0.25       1.2]
        >>> L = cholesky(A)
        >>> nprint(L)
        [ 1.41421      0.0      0.0]
        [0.353553  1.09924      0.0]
        [0.235702  0.15162  1.05899]
        >>> chop(A - L*L.T)
        [0.0  0.0  0.0]
        [0.0  0.0  0.0]
        [0.0  0.0  0.0]

    Cholesky decomposition of a Hermitian matrix::

        >>> A = eye(3) + matrix([[0,0.25j,-0.5j],[-0.25j,0,0],[0.5j,0,0]])
        >>> L = cholesky(A)
        >>> nprint(L)
        [          1.0                0.0                0.0]
        [(0.0 - 0.25j)  (0.968246 + 0.0j)                0.0]
        [ (0.0 + 0.5j)  (0.129099 + 0.0j)  (0.856349 + 0.0j)]
        >>> chop(A - L*L.H)
        [0.0  0.0  0.0]
        [0.0  0.0  0.0]
        [0.0  0.0  0.0]

    Attempted Cholesky decomposition of a matrix that is not positive
    definite::

        >>> A = -eye(3) + hilbert(3)
        >>> L = cholesky(A)
        Traceback (most recent call last):
          ...
        ValueError: matrix is not positive-definite

    **References**

    1. [Wikipedia]_ http://en.wikipedia.org/wiki/Cholesky_decomposition

    """
    if not A.shape[0] == A.shape[1]:
        raise ValueError('need n*n matrix')
    if tol is None:
        tol = ctx.epsilon()
    n = A.shape[0]
    L = zeros((n, n))
    for j in range(n):
        c = A[j,j].real
        if abs(c-A[j,j]) > tol:
            raise ValueError('matrix is not Hermitian')
        s = c - ctx.fsum((abs(L[j,k])**2 for k in range(j)))
        if s < tol:
            raise ValueError('matrix is not positive-definite')
        L[j,j] = ctx.sqrt(s)
        for i in range(j, n):
            #it1 = (L[i,k] for k in range(j))
            #it2 = (L[j,k] for k in range(j))
            #t = ctx.fdot(it1, it2, conjugate=True)
            t = np.sum(L[i,:j] * L[j,:j].conj())
            L[i,j] = (A[i,j] - t) / L[j,j]
    return L

def cholesky_solve(ctx, A, b):
    """
    Ax = b => x

    Solve a symmetric positive-definite linear equation system.
    This is twice as efficient as lu_solve.

    Typical use cases:
    * A.T*A
    * Hessian matrix
    * differential equations
    """
    # do not overwrite A nor b
    A, b = to_mp(A), to_mp(b)
    if A.shape[0] != A.shape[1]:
        raise ValueError('can only solve determined system')
    # Cholesky factorization
    L = cholesky(ctx, A)
    # solve
    b = L_solve(ctx, L, b, unit_diag=False)
    return U_solve(ctx, L.T, b)

def det(ctx, A):
    """
    Calculate the determinant of a matrix.
    """
    # do not overwrite A
    A = to_mp(A)
    # use LU factorization to calculate determinant
    try:
        R, p = LU_decomp(ctx, A)
    except ZeroDivisionError:
        return 0
    z = np.prod(np.diag(R))
    for i, e in enumerate(p):
        if i != e:
            z *= -1
    return z

def cond(ctx, A, norm=None):
    """
    Calculate the spectral condition number of a matrix using a specified matrix norm.

    The condition number estimates the sensitivity of a matrix to errors.
    Example: small input errors for ill-conditioned coefficient matrices
    alter the solution of the system dramatically.

    For ill-conditioned matrices it's recommended to use qr_solve() instead
    of lu_solve(). This does not help with input errors however, it just avoids
    to add additional errors.

    Definition:    cond(A) = ||A|| * ||A**-1||
    """
    if norm is None:
        norm = lambda x: np.linalg.norm(x, ord=1)
    return norm(A) * norm(inverse(ctx, A))   # TODO: use svd?


def qr(ctx, A, mode='full'):
    """
    Compute a QR factorization $A = QR$ where
    A is an m x n matrix of real or complex numbers where m >= n

    mode has following meanings:
    (1) mode = 'raw' returns two matrixes (A, tau) in the
        internal format used by LAPACK
    (2) mode = 'reduced' returns the leading n columns of Q
        and n rows of R
    (3) Any other value returns the leading m columns of Q
        and m rows of R
    """
    m, n = A.shape
    assert n >= 0
    assert m >= n

    # check for complex data type
    cmplx = contains_complex(A)

    tau = empty(n)
    A = to_mp(A)

    # ---------------
    # FACTOR MATRIX A
    # ---------------
    if cmplx:
        one = ctx.mpc(1)
        zero = ctx.mpc(0)
        rzero = ctx.mpf(0)

        # main loop to factor A (complex)
        for j in range(n):
            alpha = A[j,j]
            alphr, alphi = alpha.real, alpha.imag

            if (m-j) >= 2:
                #xnorm = ctx.fsum(A[i,j] * A[i,j].conjugate() for i in range(j+1, m))
                xnorm = sum(A[i,j] * A[i,j].conjugate() for i in range(j+1, m))
                xnorm = ctx.sqrt(xnorm).real
            else:
                xnorm = rzero

            if (xnorm == rzero) and (alphi == rzero):
                tau[j] = zero
                continue

            if alphr < rzero:
                beta = ctx.sqrt(alphr**2 + alphi**2 + xnorm**2)
            else:
                beta = -ctx.sqrt(alphr**2 + alphi**2 + xnorm**2)

            tau[j] = ctx.mpc((beta - alphr) / beta, -alphi / beta)
            t = -tau[j].conjugate()
            za = one / (alpha - beta)

            for i in range(j+1, m):
                A[i,j] *= za

            A[j,j] = one
            for k in range(j+1, n):
                #y = ctx.fsum(A[i,j] * A[i,k].conjugate() for i in range(j, m))
                y = sum(A[i,j] * A[i,k].conjugate() for i in range(j, m))
                temp = t * y.conjugate()
                for i in range(j, m):
                    A[i,k] += A[i,j] * temp

            A[j,j] = ctx.mpc(beta, 0)
    else:
        one = ctx.mpf(1)
        zero = ctx.mpf(0)

        # main loop to factor A (real)
        for j in range(n):
            alpha = A[j,j]

            if m - j > 2:
                xnorm = ctx.fsum(A[i,j]**2 for i in range(j+1, m))
                xnorm = ctx.sqrt(xnorm)
            elif m - j == 2:
                xnorm = abs(A[m-1,j])
            else:
                xnorm = zero

            if xnorm == zero:
                tau[j] = zero
                continue

            if alpha < zero:
                beta = ctx.hypot(alpha, xnorm)
            else:
                beta = -ctx.hypot(alpha, xnorm)

            tau[j] = (beta - alpha) / beta
            t = -tau[j]
            da = one / (alpha - beta)

            for i in range(j+1, m):
                A[i,j] *= da

            A[j,j] = one
            for k in range(j+1, n):
                y = ctx.fsum(A[i,j] * A[i,k] for i in range(j, m))
                temp = t * y
                for i in range(j,m):
                    A[i,k] += A[i,j] * temp

            A[j,j] = beta

    # return factorization in same internal format as LAPACK
    if mode == 'raw':
        return A, tau

    # ----------------------------------
    # FORM Q USING BACKWARD ACCUMULATION
    # ----------------------------------

    # form R before the values are overwritten
    R = A.copy()
    for j in range(n):
        for i in range(j+1, m):
            R[i,j] = zero

    # set the value of p (number of columns of Q to return)
    p = m
    if mode == 'reduced' or mode == 'r':
        p = n

    # add columns to A if needed and initialize
    A = np.hstack((A, zeros((m, p - n))))
    for j in range(p):
        A[j,j] = one
        for i in range(j):
            A[i,j] = zero

    # main loop to form Q
    for j in range(n-1, -1, -1):
        t = -tau[j]
        A[j,j] += t

        for k in range(j+1, p):
            if cmplx:
                #y = ctx.fsum(A[i,j] * A[i,k].conjugate() for i in range(j+1, m))
                y = sum(A[i,j] * A[i,k].conjugate() for i in range(j+1, m))
                temp = t * y.conjugate()
            else:
                y = ctx.fsum(A[i,j] * A[i,k] for i in range(j+1, m))
                temp = t * y
            A[j,k] = temp
            for i in range(j+1, m):
                A[i,k] += A[i,j] * temp

        for i in range(j+1, m):
            A[i, j] *= t

    return A, R[0:p, 0:n]
