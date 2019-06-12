
"""
The MKS homogenization module test cases

"""
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from pymks.fmks.correlations import FlattenTransformer, TwoPointcorrelation
from pymks.datasets import make_cahn_hilliard
from pymks.fmks.bases.legendre import LegendreTransformer


def test_classification():
    """
This test basically creates Legendre microstructures in both times: 0 and t.
Then builds homogenization classification linkages to classify if  newly
generated microstructures are at time 0 or t
    """
    reducer = PCA(n_components=3)
    linker = LogisticRegression()
    homogenization_pipeline = Pipeline(
        steps=[
            ("discretize", LegendreTransformer(n_state=3, min_=-1.0, max_=1.0)),
            (
                "Correlations",
                TwoPointcorrelation(
                    boundary="periodic", cutoff=10, correlations=[1, 1]
                ),
            ),
            ("flatten", FlattenTransformer()),
            ("reducer", reducer),
            ("connector", linker),
        ]
    )
    x0_phase, x1_phase = make_cahn_hilliard(n_samples=50)
    y0_class = np.zeros(x0_phase.shape[0])
    y1_class = np.ones(x1_phase.shape[0])
    x_combined = np.concatenate((x0_phase, x1_phase))
    y_combined = np.concatenate((y0_class, y1_class))
    homogenization_pipeline.fit(x_combined, y_combined)
    x0_test, x1_test = make_cahn_hilliard(n_samples=3)
    y1_test = homogenization_pipeline.predict(x1_test)
    y0_test = homogenization_pipeline.predict(x0_test)
    assert np.allclose(y0_test, [0, 0, 0])
    assert np.allclose(y1_test, [1, 1, 1])
