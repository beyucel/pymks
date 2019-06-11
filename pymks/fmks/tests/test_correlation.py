"""MKS Correlation Module
For computing auto and cross corelations under assumption
of  periodic or  non-periodic boundary conditions using discreete fourier
transform.


Note that input microstrucure should be 4 dimensional array.
where X=[n_sample,x,y.n_basis]
"""

def test():
    """
    Below is an example of using MKSStructureAnalysis using FastICA.

    >>> from pymks.datasets import make_microstructure
    >>> from pymks.bases import PrimitiveBasis
    >>> from sklearn.decomposition import FastICA

    >>> leg_basis = PrimitiveBasis(n_states=2, domain=[0, 1])
    >>> reducer = FastICA(n_components=3)
    >>> analyzer = MKSStructureAnalysis(basis=leg_basis, mean_center=False,
    ...                                 dimension_reducer=reducer)

    >>> X = make_microstructure(n_samples=4, size=(13, 13), grain_size=(3, 3))
    >>> print(analyzer.fit_transform(X)) # doctest: +ELLIPSIS
    [[ 0.5 -0.5 -0.5]
     [ 0.5  0.5  0.5]
     [-0.5 -0.5  0.5]
     [-0.5  0.5 -0.5]]
     """
