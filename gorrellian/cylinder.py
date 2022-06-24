"""
Incremental SVD using Hebbian updates (from data only).

  Applied to S. Timme's cylinder test case.

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


import os
import jax

# sol has a GPU, which is 32 bit
if os.uname()[1] != 'sol':
    jax.config.update("jax_enable_x64", True)

import scipy
import scipy.sparse.linalg as sla
from scipy.io import loadmat

import numpy
from numpy.random import default_rng

from hebbian_svd import svd_update, svd_test


try:
    print("Loading A...")
    A = numpy.load('../data/A.npy')
except:
    omega = 1
    print("A not found, loading matrices...")
    # these are loaded as scipy csc_matrix sparse arrays
    J = loadmat("../data/Jac.mat")['J']
    Q = loadmat("../data/Vol.mat")['Q']

    N = Q.shape[0]
    M = N

    # doing each bit separately to avoid core dump
    P = Q.sqrt()        # since Q real and diagonal
    del Q
    print("Forming A_inv...")
    A_inv = 1j * omega * scipy.sparse.eye(N) - P @ J @ sla.inv(P)
    del P, J
    print("passing dense matrix to Jax/GPU...")
    A_inv = A_inv.todense()
    A_inv = jax.device_put(A_inv)       # too big for my GPU :(
    print("Forming A = inv(A_inv)...")
    A = jax.numpy.linalg.inv(A_inv)
    with open('../data/A.npy', 'wb') as f:
        jax.numpy.save(f, A)
    del A_inv


N = A.shape[0]
M = A.shape[1]

L = 4

rng = default_rng(0)


def gen_pair(key):
    """
    Generate a random complex data pair, a = Ab.
      You can generate either a and find b, or the other way round.
      The Hebbian solver doesn't know.

    NB: in the scheme we use, a and b are stored as just 1D arrays,
    not Mx1 vector (or Nx1).
    """
    b = rng.standard_normal(M) + 1j * rng.standard_normal(M)
    return A @ b, b, key


def run(L=L, eta=0.0001, verbose=True, **kwargs):
    svd_test(A, gen_pair, L=L, eta=eta, verbose=verbose, **kwargs)
