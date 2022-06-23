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


import jax
jax.config.update("jax_enable_x64", True)

from jax import jit

from hebbian_svd import svd_update, svd_test




test = True;
if test:
    N = 10
    M = 50
    rkey = jax.random.PRNGKey(1)
    A = jax.random.normal(rkey, shape=[N, M]) + \
            1j * jax.random.normal(rkey, shape=[N, M])

else:
    import scipy.sparse
    from scipy.io import loadmat
    from scipy.sparse.linalg import inv as sparse_inv

    omega = 1
    print("Loading matrices...")
    # these are loaded as scipy csc_matrix sparse arrays
    J = loadmat("../data/Jac.mat")['J']
    Q = loadmat("../data/Vol.mat")['Q']

    N = Q.shape[0]
    M = N

    print("Forming A_inv...")
    # TODO: check sense of Q
    A_inv = 1j * omega * scipy.sparse.eye(N) - Q @ J @ sparse_inv(Q)
    print("Forming A = inv(A)...")
    A = inv(A_inv.todense())


N = A.shape[0]
M = A.shape[1]
L = 5


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
    b = jax.random.normal(key, shape=(M,)) + 1j + jax.random.normal(key, shape=(M,))
    #b = b / jax.numpy.linalg.norm(b)
    return A @ b, b, key


def run(L=L, **kwargs):
    svd_test(A, gen_pair, L=L, **kwargs)
