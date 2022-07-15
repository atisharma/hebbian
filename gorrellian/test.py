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
from jax import jit

jax.config.update("jax_enable_x64", True)

from time import gmtime, strftime, time

from hebbian_svd import svd_test


# example case of size similar to cylinder
N = 40000
M = 50000
L = 10

rkey = jax.random.PRNGKey(1)
# generate reasonable low-rank matrix A=USV'
# random orthogonal matrices U, V
U, _ = jax.numpy.linalg.qr(
        jax.random.normal(rkey, shape=[N, L]) + \
        1j * jax.random.normal(rkey, shape=[N, L]) )
V, _ = jax.numpy.linalg.qr(
        jax.random.normal(rkey, shape=[M, L]) + \
        1j * jax.random.normal(rkey, shape=[M, L]) )
# random singular values
S_sqrt = 100 * jax.random.normal(rkey, shape=[L])
S = jax.numpy.diag((S_sqrt * S_sqrt).sort())
print("true singular values:")
print(jax.numpy.flip(S.diagonal()))


@jit
def gen_pair(key):
    """
    Generate a random complex data pair, a = Ab.

    NB: in the scheme we use, a and b are stored as just 1D arrays,
    not Mx1 vector (or Nx1).

    Here, we use
        a = U S V' b = (U S) (V' b)
    """
    key, subkey = jax.random.split(key)
    b = jax.random.normal(key, shape=(M,)) + 1j * jax.random.normal(key, shape=(M,))
    return (U @ S) @ (V.T.conjugate() @ b), b, key


def run(L=5, eta=1e-3, **kwargs):
    print(strftime("%H:%M:%S", gmtime()))
    UVS = svd_test(gen_pair, L=L, eta=eta, **kwargs)
    print(strftime("%H:%M:%S", gmtime()))
    print("true singular values:")
    print(jax.numpy.flip(S.diagonal()))

    print("found singular values (from snapshots):")
    print(UVS['S'])
    print("found singular values (using orthogonal projection):")
    print( ( (UVS["U"].T.conjugate() @ U) @ S @ (V.T.conjugate() @ UVS["V"]) ).diagonal().real )
    return UVS
