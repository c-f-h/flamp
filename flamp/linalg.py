import numpy as np
import numbers

# Code based on the linalg module from mpmath.

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
    cmplx = any(isinstance(x, numbers.Complex) and not isinstance(x, numbers.Real)
                for x in A.flat)

    tau = np.empty(n, dtype=A.dtype)
    A = A.copy()

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
    #A.cols += (p-n)
    A = np.hstack((A, np.empty((m, p - n), dtype=A.dtype)))
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
