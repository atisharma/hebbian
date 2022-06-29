# Low-memory Resolvent modes from snapshot data

Incremental PCA or SVD using Hebbian updates (from data snapshots only) as per Gorrell's paper.

### PCA problem

Given many data vectors {$x$}, find the eigenvectors of the correlation matrix $R$,
where $E(xx') = R$. This code is not finished.

### SVD problem

Given many data vector pairs {$(a,b)$}, where $a=Mb$, find the singular vectors of $M$.

### Convergence

Although there is a proof of convergence, in practice you need a small enough $\eta$ (interpreted as update weight or update step size).
The current code implements some simple adaptive step size logic. The default $\eta$ should be OK for most cases.

### Running the code

The current version of the code uses [jax](https://jax.readthedocs.io/en/latest/), a performant numpy replacement that has GPU capabilities.

To run the test example,
```
import test
test.run()
```

### Performance

No effort has been made in this regard in this code. It is probably slow. It is written with clarity in mind.
There exists a well optimised implementation in Common Lisp, and a reference C++ version provided by Gorrell.

### Dynamic Mode Decomposition and other modal decompositions

You could probably get DMD modes too using $u(t)$ and $u(t+1)$ data pairs.

### References

Zhang, “Complex-Valued Neural Networks“, 2003.
https://www.mbari.org/wp-content/uploads/2016/01/Zhang_bookchapter_2003.pdf

G. Gorrell, “Generalized Hebbian algorithm for incremental singular value
decomposition in natural language processing,” EACL 2006, 11st Conference of
the European Chapter of the Association for Computational Linguistics,
Proceedings of the Conference, April 3-7, 2006, Trento, Italy (D. McCarthy
and S. Wintner, eds.), The Association for Computer Linguistics, 2006.

