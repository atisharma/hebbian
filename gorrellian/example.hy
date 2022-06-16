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


(import numpy)
(import numpy [complex array diag outer conj eye tril triu exp])
(import numpy [vdot :as inner])             ; complex vector inner product
(import numpy.linalg [svd eig inv norm])
(import numpy.random [default-rng])


(setv N 50
      M 100
      L 2)
(numpy.set_printoptions :precision 2 :suppress True)

(setv rng (default-rng 1))
(setv A (+ (rng.standard-normal [N M])
           (* 1j (rng.standard-normal [N M]))))


(defn outer [a b]
  """
  Complex outer product of a and b: a x b^T.
  """
  (numpy.outer a (conj b)))


(defn gen-pair []
  """
  Generate a random complex data pair, a = Mb.
    You can generate either a and find b, or the other way round.
    The Hebbian solver doesn't know.
  """
  (let [b (+ (rng.standard-normal M)
             (* 1j (rng.standard-normal M)))]
    {"b" b
     "a" (@ A b)}))


(defn pca-update [W x [eta 1e-3]]
  """
  Incremental update to approximate eigenvector matrix W of correlation matrix R
    with one data vector x. Returns updated W.
  See Zhang's book and the Wikipedia page on GHA.
  """
  (let [y (@ (conj W.T) x)]
       (+ W
          (* eta (- (@ (outer x x) W)
                    (@ W (triu (outer y y))))))))


(defn pca-solve [W [iterations 100000] [eta 1e-4] [d 1e-4] [verbose False]]
  """
  Solve iteratively for the eigenvectors of M.
  Exponential decay d on learning rate eta.
  """
  (for [n (range iterations)]
    (let [data-pair (gen-pair)
          eta-decayed (* eta (exp (* -1 d n)))]
      (when verbose
        (print n (get W Ellipsis 1) eta-decayed))
      (setv W (pca-update W (:a data-pair) :eta eta-decayed))))
  W)


(defn pca-test [#** kwargs]
  (let [R (@ A (conj A.T))
        [L-ref W-ref] (eig R)
        W (pca-solve (rng.standard-normal [N N]) #** kwargs)
        L (diag (@ (inv W) R W))]
      (print "A")
      (print A)
      (print)
      (print "R")
      (print R)
      (print)
      (print "W (ref)")
      (print W-ref)
      (print)
      (print "W (GHA)")
      (print W)
      (print)
      (print "R W (ref)")
      (print (@ R W-ref))
      (print)
      (print "R W (GHA)")
      (print (@ R W))
      (print)
      (print "L (ref)")
      (print L-ref)
      (print)
      (print "L (GHA)")
      (print L)
      (print)))


(defn svd-update [U V data-pair [eta 1e-3]]
  """
  Incremental update to approximate SVD matrices U, V of operator A
    with one data vector pair a, b. Returns updated U, V.
  See Gorrell's paper.
  """
  (let [a (:a data-pair)
        b (:b data-pair)
        ya (@ (conj U.T) a)
        yb (@ (conj V.T) b)]
    {"U" (+ U
            (* eta (- (outer a yb)
                      (@ U (triu (outer ya ya))))))
     "V" (+ V
            (* eta (- (outer b ya)
                      (@ V (triu (outer yb yb))))))}))


(defn svd-solve [U V [iterations 100000] [eta 1e-4] [d 1e-4] [verbose False]]
  """
  Solve iteratively for the eigenvectors of M.
  Exponential decay d on learning rate eta.
  """
  (for [n (range iterations)]
    (let [data-pair (gen-pair)
          eta-decayed (* eta (exp (* -1 d n)))
          UV (svd-update U V data-pair :eta eta-decayed)]
      (when verbose
        (print n (get U Ellipsis 1) eta-decayed))
      (setv [U V] [(:U UV) (:V UV)])))
  {"U" U
   "V" V})


(defn svd-test [#** kwargs]
  (let [[U-ref S-ref V-ref] (svd A)
        U0 (eye N L)
        V0 (eye M L)
        UV (svd-solve U0 V0 #** kwargs)
        U (:U UV)
        V (:V UV)
        S (@ (conj U.T) A V)]
      (print "A")
      (print A)
      (print)
      (print "U (ref)")
      (print U-ref)
      (print)
      (print "U (GHA)")
      (print U)
      (print)
      (print "V (ref)")
      (print V-ref)
      (print)
      (print "V (GHA)")
      (print V)
      (print)
      (print "S (ref)")
      (print (diag S-ref))
      (print)
      (print "S (GHA)")
      (print S)
      (print)))
