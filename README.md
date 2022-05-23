# Low-memory Resolvent modes from snapshot data

Incremental PCA, SVD using Hebbian updates (from data only) as per Gorrell's paper.

### PCA problem

Given many data vectors $\{x\}$, find the eigenvectors of the correlation matrix $R$,
where $E(xx') = R$.

### SVD problem

Given many data vector pairs $\{(a,b)\}$, where $a=Mb$, find the singular vectors of $M$.

### Convergence

Although there is a proof of convergence, in practice you need a small enough $\eta$ (update weight) and $d$ (eta decay).
If it fails to converge, try halving both and doubling $N$.
As a guide, to get the leading two singular vectors of a 200x500 matrix, I used $\eta=0.0001$ and $d=0.00005$.

### Running the code

I wrote the code in Hy, which is a lisp-like 'dialect' of python.
You will need to pip install hy first, then you can import from within regular python by doing
```
import hy
import example as ex
```
then (for example)
```
ex.svd_test(iterations=100000, eta=1e-4, d=5e-5)
```
as normal.

### Performance

No effort has been made in this regard in this code. Numpy is slow.

### Dynamic Mode Decomposition

You could probably get DMD modes too using $u(t)$ and $u(t+1)$ data pairs.

### References

Zhang, “Complex-Valued Neural Networks“, 2003.
https://www.mbari.org/wp-content/uploads/2016/01/Zhang_bookchapter_2003.pdf

G. Gorrell, “Generalized Hebbian algorithm for incremental singular value
decomposition in natural language processing,” EACL 2006, 11st Conference of
the European Chapter of the Association for Computational Linguistics,
Proceedings of the Conference, April 3-7, 2006, Trento, Italy (D. McCarthy
and S. Wintner, eds.), The Association for Computer Linguistics, 2006.

