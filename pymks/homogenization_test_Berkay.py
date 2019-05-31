from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from dask.multiprocessing import get
import dask.array as da
import numpy as np

"""
Test cases for fmks homogenization framewok. This framework utilizes the
scikitlearn pipeline framewok so it is modular and it can be uesd to
build robust structure property correlations

"""
from pymks.bases import LegendreBasis
from pymks.fmks.data.cahn_hilliard import generate, solve
import numpy as np
n_states=3
domain=[-1,1]
leg_basis = LegendreBasis(n_states=n_states, domain=domain)
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
reducer = PCA(n_components=3)
linker = LogisticRegression()
