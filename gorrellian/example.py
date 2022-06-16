"""
Incremental PCA, SVD using Hebbian updates (from data only).

  PCA problem:
  Given many data vectors x, find the eigenvectors of the correlation matrix R,
  where E(xx') = R.

  SVD problem:
  Given many data vector pairs (a,b), where a=Mb, find the singular vectors of M.

  References:

  Zhang, “Complex-Valued Neural Networks“, 2003.
    https://www.mbari.org/wp-content/uploads/2016/01/Zhang_bookchapter_2003.pdf

  G. Gorrell, “Generalized Hebbian algorithm for incremental singular value
  decomposition in natural language processing,” EACL 2006, 11st Conference of
  the European Chapter of the Association for Computational Linguistics,
  Proceedings of the Conference, April 3-7, 2006, Trento, Italy (D. McCarthy
  and S. Wintner, eds.), The Association for Computer Linguistics, 2006.

"""

import numpy

from numpy import array, diag, outer, conj, eye, tril, triu, exp
from numpy import vdot as inner  # complex vector inner product
from numpy.linalg import svd, eig, inv, norm
from numpy.random import default_rng


N = 50
M = 100
L = 2

numpy.set_printoptions(precision=2, suppress=True)

rng = default_rng(1)
A = rng.standard_normal([N, M]) + 1j * rng.standard_normal([N, M])


def outer(a, b):
    """
    Complex outer product of a and b: a x b^T.
    """
    return numpy.outer(a, conj(b))


def gen_pair():
    """
    Generate a random complex data pair, a = Mb.
      You can generate either a and find b, or the other way round.
      The Hebbian solver doesn't know.
    """
    b = rng.standard_normal(M) + 1j * rng.standard_normal(M)
    return {"b": b, "a": A @ b}


def pca_update(W, x, eta=1e-3):
    """
    Incremental update to approximate eigenvector matrix W of correlation matrix R
      with one data vector x. Returns updated W.
    See Zhang's book and the Wikipedia page on GHA.
    """
    y = conj(W.T) @ x
    return W + eta * ((outer(x, x) @ W) - (W @ triu(outer(y, y))))


def pca_solve(W, iterations=100000, eta=1e-4, d=1e-4, verbose=False):
    """
    Solve iteratively for the eigenvectors of M.
    Exponential decay d on learning rate eta.
    """
    for n in range(iterations):
        data_pair = gen_pair()
        eta_decayed = eta * exp(-d * n)
        if verbose:
            print(n, W[..., 1], eta_decayed)
        W = pca_update(W, data_pair['a'], eta=eta_decayed)
    return W


def pca_test(**kwargs):
    R = A @ conj(A.T)
    L_ref, W_ref = eig(R)
    W = pca_solve(rng.standard_normal([N, N]), **kwargs)
    L = diag(inv(W) @ R @ W)
    print("A")
    print(A)
    print()
    print("R")
    print(R)
    print()
    print("W (ref)")
    print(W_ref)
    print()
    print("W (GHA)")
    print(W)
    print()
    print("R W (ref)")
    print(R @ W_ref)
    print()
    print("R W (GHA)")
    print(R @ W)
    print()
    print("L (ref)")
    print(L_ref)
    print()
    print("L (GHA)")
    print(L)
    print()


def svd_update(U, V, data_pair, eta=1e-3):
    """
    Incremental update to approximate SVD matrices U, V of operator A
      with one data vector pair a, b. Returns updated U, V.
    See Gorrell's paper.
    """
    a = data_pair["a"]
    b = data_pair["b"]
    ya = conj(U.T) @ a
    yb = conj(V.T) @ b
    return {
        "U": (U + (eta * (outer(a, yb) - (U @ triu(outer(ya, ya)))))),
        "V": (V + (eta * (outer(b, ya) - (V @ triu(outer(yb, yb)))))),
    }


def svd_solve(U, V, iterations=100000, eta=1e-4, d=1e-4, verbose=False):
    """
    Solve iteratively for the eigenvectors of M.
    Exponential decay d on learning rate eta.
    """
    for n in range(iterations):
        data_pair = gen_pair()
        eta_decayed = eta * exp(-d * n)
        UV = svd_update(U, V, data_pair, eta=eta_decayed)
        if verbose:
            print(n, U[..., 1], eta_decayed)
        U = UV["U"]
        V = UV["V"]
    return {"U": U, "V": V}


def svd_test(**kwargs):
    U_ref, S_ref, V_ref = svd(A)
    U0 = eye(N, L)
    V0 = eye(M, L)
    UV = svd_solve(U0, V0, **kwargs)
    U = UV["U"]
    V = UV["V"]
    S = conj(U.T) @ A @ V
    print("A")
    print(A)
    print()
    print("U (ref)")
    print(U_ref)
    print()
    print("U (GHA)")
    print(U)
    print()
    print("V (ref)")
    print(V_ref)
    print()
    print("V (GHA)")
    print(V)
    print()
    print("S (ref)")
    print(diag(S_ref))
    print()
    print("S (GHA)")
    print(S)
    print()
