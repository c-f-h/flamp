import numpy as np
import gmpy2
from gmpy2 import mpfr, mpc

import flamp

def to_fp(A):
    return np.array(A, float)
def to_cpx(A):
    return np.array(A, complex)

### linalg

def test_qr_real():
    n = 5
    A = np.random.rand(n, n)
    AA = mpfr(1) * A
    Q, R = flamp.qr(AA)
    assert Q.shape == (n, n) and R.shape == (n, n)
    assert np.allclose(to_fp(Q.T @ Q), np.eye(n))
    assert np.allclose(to_fp(Q @ R), A)
    assert np.all(np.tril(R, -1) == 0)
    ## special case: size 0 matrix
    AA = flamp.zeros((4, 0))
    Q, R = flamp.qr(AA)
    assert np.allclose(to_fp(Q), np.eye(4))
    assert R.shape == (4, 0)

def test_qr_complex():
    n = 5
    A = np.random.rand(n, n) + 1j * np.random.rand(n, n)
    AA = mpfr(1) * A
    Q, R = flamp.qr(AA)
    assert Q.shape == (n, n) and R.shape == (n, n)
    assert np.allclose(to_cpx(Q.T.conj() @ Q), np.eye(n))
    assert np.allclose(to_cpx(Q @ R), A)
    assert np.all(np.tril(R, -1) == 0)

def test_inverse_real():
    n = 5
    A = np.random.rand(n, n)
    AA = mpfr(1) * A
    Ainv = flamp.inverse(AA)
    assert A.shape == (n, n)
    assert np.allclose(to_fp(Ainv @ A), np.eye(n))

def test_inverse_complex():
    n = 5
    A = np.random.rand(n, n) + 1j * np.random.rand(n, n)
    AA = mpfr(1) * A
    Ainv = flamp.inverse(AA)
    assert A.shape == (n, n)
    assert np.allclose(to_cpx(Ainv @ A), np.eye(n))

def test_lu_solve_real():
    n = 5
    A, b = np.random.rand(n, n), np.random.rand(n)
    AA = mpfr(1) * A
    x = flamp.lu_solve(AA, b)
    assert x.shape == (n,)
    assert np.allclose(to_fp(A @ x), b)

def test_lu_solve_real_block():
    n = 5
    A, b = np.random.rand(n, n), np.random.rand(n, 3)
    AA = mpfr(1) * A
    x = flamp.lu_solve(AA, b)
    assert x.shape == (n, 3)
    assert np.allclose(to_fp(A @ x), b)

def test_lu_solve_complex():
    n = 5
    A, b = np.random.rand(n, n) + 1j * np.random.rand(n, n), np.random.rand(n)
    AA = mpfr(1) * A
    x = flamp.lu_solve(AA, b)
    assert x.shape == (n,)
    assert np.allclose(to_cpx(A @ x), b)

def test_lu():
    n = 5
    A = np.random.rand(n, n) + 1j * np.random.rand(n, n)
    AA = mpfr(1) * A
    P, L, U = flamp.lu(AA)
    assert np.allclose(to_cpx(P @ AA), to_cpx(L @ U))

def test_cholesky_solve_real():
    n = 5
    A, b = np.random.rand(n, n), np.random.rand(n)
    A = A.T @ A
    AA = mpfr(1) * A
    x = flamp.cholesky_solve(AA, b)
    assert x.shape == (n,)
    assert np.allclose(to_fp(A @ x), b)

def test_cholesky_solve_real_block():
    n = 5
    A, b = np.random.rand(n, n), np.random.rand(n, 3)
    A = A.T @ A
    AA = mpfr(1) * A
    x = flamp.cholesky_solve(AA, b)
    assert x.shape == (n, 3)
    assert np.allclose(to_fp(A @ x), b)

def test_qr_solve_real():
    n = 5
    A, b = np.random.rand(n, n), np.random.rand(n)
    AA = mpfr(1) * A
    x = flamp.qr_solve(AA, b)
    assert x.shape == (n,)
    assert np.allclose(to_fp(A @ x), b)

def test_qr_solve_real_block():
    n = 5
    A, b = np.random.rand(n, n), np.random.rand(n, 3)
    AA = mpfr(1) * A
    x = flamp.qr_solve(AA, b)
    assert x.shape == (n, 3)
    assert np.allclose(to_fp(A @ x), b)

def test_solve_real_overdet():
    n = 5
    A, b = np.random.rand(n + 2, n), np.random.rand(n + 2, 3)
    AA = mpfr(1) * A
    x = flamp.qr_solve(AA, b)
    x2 = flamp.lu_solve(AA, b)
    assert x.shape == (n, 3)
    assert x2.shape == (n, 3)
    assert np.allclose(to_fp(x), to_fp(x2))

def test_det():
    n = 5
    E = np.random.rand(n)   # random eigenvalues
    U = mpfr(1) * np.random.rand(n, n)
    Uinv = flamp.inverse(U)
    A = U @ np.diag(E) @ Uinv
    det = flamp.det(A)
    assert np.allclose(to_fp(det), np.prod(E))

### eigen

def test_eig_real():
    A = mpfr(1) * np.arange(9).reshape((3, 3))
    E, UL, UR = flamp.eig(A, left=True, right=True)
    assert np.allclose(to_cpx(A @ UR), to_cpx(E[None, :] * UR))
    assert np.allclose(to_cpx(UL @ A), to_cpx(E[:, None] * UL))
    # compute only eigenvalues
    E2 = flamp.eig(A, left=False, right=False)
    assert np.all(E == E2)

def test_eig_complex():
    A = mpfr(1) * (np.random.rand(5, 5) + 1j * np.random.rand(5, 5))
    E, UL, UR = flamp.eig(A, left=True, right=True)
    assert np.allclose(to_cpx(A @ UR), to_cpx(E[None, :] * UR))
    assert np.allclose(to_cpx(UL @ A), to_cpx(E[:, None] * UL))
    # compute only eigenvalues
    E2 = flamp.eig(A, left=False, right=False)
    assert np.all(E == E2)

def test_hessenberg_real():
    n = 5
    A = np.random.rand(n, n)
    AA = mpfr(1) * A
    Q, H = flamp.hessenberg(AA)
    assert Q.shape == (n, n) and H.shape == (n, n)
    assert np.allclose(to_fp(Q.T @ Q), np.eye(n))
    assert np.allclose(to_fp(Q @ H @ Q.T), A)
    assert np.all(np.tril(H, -2) == 0)

def test_hessenberg_complex():
    n = 5
    A = np.random.rand(n, n) + 1j * np.random.rand(n, n)
    AA = mpfr(1) * A
    Q, H = flamp.hessenberg(AA)
    assert Q.shape == (n, n) and H.shape == (n, n)
    assert np.allclose(to_cpx(Q.T.conj() @ Q), np.eye(n))
    assert np.allclose(to_cpx(Q @ H @ Q.T.conj()), A)
    assert np.all(np.tril(H, -2) == 0)

def test_schur():
    n = 5
    A = np.random.rand(n, n) + 1j * np.random.rand(n, n)
    AA = mpfr(1) * A
    Q, R = flamp.schur(AA)
    assert Q.shape == (n, n) and R.shape == (n, n)
    assert np.allclose(to_cpx(Q.T.conj() @ Q), np.eye(n))
    assert np.allclose(to_cpx(Q @ R @ Q.T.conj()), A)
    assert np.all(np.tril(R, -1) == 0)

### eigen_symmetric

def test_eigh_real():
    n = 5
    A = np.random.rand(n, n)
    A = A + A.T
    AA = mpfr(1) * A
    E, Q = flamp.eigh(AA)
    assert np.allclose(to_fp(Q.T @ Q), np.eye(n))
    assert E.shape == (n,)
    assert np.allclose(to_fp(Q @ np.diag(E) @ Q.T), A)
    # compute only eigenvalues
    E2 = flamp.eigh(AA, eigvals_only=True)
    assert np.all(E == E2)

def test_eigh_complex():
    n = 5
    A = np.random.rand(n, n) + 1j * np.random.rand(n, n)
    A = A + A.T.conj()
    AA = mpfr(1) * A
    E, Q = flamp.eigh(AA)
    assert np.allclose(to_cpx(Q.T.conj() @ Q), np.eye(n))
    assert E.shape == (n,)
    assert np.allclose(to_cpx(Q @ np.diag(E) @ Q.T.conj()), A)
    # compute only eigenvalues
    E2 = flamp.eigh(AA, eigvals_only=True)
    assert np.all(E == E2)

def test_svd_real():
    n = 5
    A = np.random.rand(n, n)
    AA = mpfr(1) * A
    U, S, V = flamp.svd(AA)
    assert np.allclose(to_fp(U.T @ U), np.eye(n))
    assert np.allclose(to_fp(V.T @ V), np.eye(n))
    assert S.shape == (n,)
    assert np.allclose(to_fp((U * S[None, :]) @ V), A)
    # compute only singular values
    S2 = flamp.svd(AA, compute_uv=False)
    assert np.all(S == S2)

def test_svd_complex():
    n = 5
    A = np.random.rand(n, n) + 1j * np.random.rand(n, n)
    AA = mpfr(1) * A
    U, S, V = flamp.svd(AA)
    assert np.allclose(to_cpx(U.T.conj() @ U), np.eye(n))
    assert np.allclose(to_cpx(V.T.conj() @ V), np.eye(n))
    assert S.shape == (n,)
    assert np.allclose(to_cpx((U * S[None, :]) @ V), A)
    # compute only singular values
    S2 = flamp.svd(AA, compute_uv=False)
    assert np.all(S == S2)


### eigen internals

import flamp.eigen

def test_hessenberg_qr():
    A = np.triu(np.ones((3, 3)), -1)
    AA = gmpy2.mpfr(1) * A
    Q = flamp.eye(3)
    flamp.eigen.hessenberg_qr(gmpy2, AA, Q)
    assert np.allclose(to_fp(Q.T @ Q), np.eye(3))
    assert np.allclose(to_fp(Q @ AA @ Q.T), A)

def test_eig_tr_r():
    R = np.triu(np.ones((3, 3)))
    U = flamp.eigen.eig_tr_r(gmpy2, gmpy2.mpfr(1) * R)
    assert np.allclose(to_fp(U), [[1, -1, 1], [0, 0, 0], [0, 0, 0]])


### utility

def test_prec():
    assert flamp.get_precision() == 53
    assert flamp.get_dps() == 15
    with flamp.extraprec(336 - flamp.get_precision()):
        assert flamp.get_precision() == 336
        assert flamp.get_dps() == 100
    flamp.set_precision(88)
    assert flamp.get_precision() == 88
    flamp.set_dps(54)
    assert flamp.get_dps() == 54

def test_to_mp():
    x = flamp.to_mp([3.4, 5.6])
    assert x.shape == (2,) and x.dtype == 'O' and np.allclose(to_fp(x), [3.4, 5.6])
    x = flamp.to_mp(np.arange(10))
    assert x.shape == (10,) and x.dtype == 'O' and x[4] == 4

def test_linspace():
    x = flamp.linspace(3, 5, 17)
    assert np.allclose(to_fp(x), np.linspace(3, 5, 17))
    x = flamp.linspace(4.5, -3.8, 7, endpoint=False)
    assert np.allclose(to_fp(x), np.linspace(4.5, -3.8, 7, endpoint=False))
