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

if os.uname()[1] != 'sol':
    jax.config.update("jax_enable_x64", True)

from jax import jit

from hebbian_svd import svd_update, svd_test


N = 100
M = 200
rkey = jax.random.PRNGKey(1)
A = jax.random.normal(rkey, shape=[N, M]) + \
        1j * jax.random.normal(rkey, shape=[N, M])
#A_inv = jax.numpy.linalg.inv(A)

L = 4


@jit
def gen_pair(key):
    """
    Generate a random complex data pair, a = Ab.
      You can generate either a and find b, or the other way round.
      The Hebbian solver doesn't know.

    NB: in the scheme we use, a and b are stored as just 1D arrays,
    not Mx1 vector (or Nx1).
    """
    key, subkey = jax.random.split(key)
    b = jax.random.normal(key, shape=(M,)) + 1j * jax.random.normal(key, shape=(M,))
    return A @ b, b, key


def run(L=L, eta=0.0001, **kwargs):
    svd_test(A, gen_pair, L=L, eta=eta, **kwargs)
