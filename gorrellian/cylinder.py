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

if os.uname()[1] == 'sol':
    print('On sol, limiting number of threads.')
    os.environ["MKL_NUM_THREADS"] = "40" 
    os.environ["OPENBLAS_NUM_THREADS"] = "40"
    os.environ["NUMEXPR_NUM_THREADS"] = "40" 
    os.environ["OMP_NUM_THREADS"] = "40" 

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import scipy
import scipy.sparse.linalg as sla
from scipy.sparse import bsr_array
from scipy.io import loadmat

import numpy
from numpy.random import default_rng

from time import gmtime, strftime

from hebbian_svd import svd_test


try:
    print("Loading A...")
    print()
    A = numpy.load('../data/A.npy')
    A = jax.numpy.array(A)
except:
    omega = 0.7 * numpy.sqrt(1.4) * 0.2
    print(f"omega = {omega}")
    print("A not found, loading matrices...")
    print()
    # these are loaded as scipy csc_matrix sparse arrays
    J = loadmat("../data/Jac.mat")['J']
    Q = loadmat("../data/Vol.mat")['Q']

    N = Q.shape[0]
    M = N

    # doing each bit separately to avoid core dump
    P = Q.sqrt()        # since Q real and diagonal
    del Q
    print("Forming sparse A_inv...")
    A_inv = 1j * omega * scipy.sparse.eye(N) - P @ J @ sla.inv(P)
    del P, J
    print("Forming A = inv(A_inv)...")
    A = sla.inv(A_inv)
    A = jax.numpy.array(A)
    with open('../data/A.npy', 'wb') as f:
        print("Saving A...")
        numpy.save(f, A)
    del A_inv


N = A.shape[0]
M = A.shape[1]

rng = default_rng(0)
rkey = jax.random.PRNGKey(1)


def gen_pair(key):
    """
    Generate a random complex data pair, a = Ab.
      You can generate either a and find b, or the other way round.
      The Hebbian solver doesn't know.

    NB: in the scheme we use, a and b are stored as just 1D arrays,
    not Mx1 vector/matrix (or Nx1).
    """
    #b = rng.standard_normal(M) + 1j * rng.standard_normal(M)
    key, subkey = jax.random.split(key)
    b = jax.random.normal(key, shape=(M,)) + 1j * jax.random.normal(key, shape=(M,))
    return A @ b, b, key


def run(L=4, eta=1e-3, verbose=True, **kwargs):
    print(strftime("%H:%M:%S", gmtime()))
    UVS = svd_test(gen_pair, L=L, eta=eta, verbose=verbose, **kwargs)
    print(strftime("%H:%M:%S", gmtime()))
    print("found singular values (from snapshots):")
    print(UVS['S'])
    return UVS
