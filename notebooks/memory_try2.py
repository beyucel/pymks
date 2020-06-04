
import dask.array as da
import warnings

import dask.array as da
from sklearn.pipeline import Pipeline
from dask_ml.model_selection import train_test_split
from dask_ml.model_selection import GridSearchCV
from dask_ml.decomposition import PCA
from dask_ml.preprocessing import PolynomialFeatures
from dask_ml.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import pandas
from toolz.curried import groupby, valmap, pipe, pluck, merge_with, merge
from toolz.curried import map as fmap

from pymks.fmks.data.elastic_fe import solve
from pymks.fmks.data.multiphase import generate

from pymks.fmks.plot import plot_microstructures
from pymks.fmks.bases.primitive import PrimitiveTransformer
from pymks.fmks.correlations import TwoPointCorrelation, FlattenTransformer
from dask.distributed import Client, progress
import time
from mpl_toolkits import mplot3d


da.random.seed(10)
np.random.seed(10)

tmp = [
    generate(shape=(100, 101, 101,101), grain_size=x, volume_fraction=(0.5, 0.2,0.3), chunks=25, percent_variance=0.15)
    for x in [(15, 2,2), (2,2, 15), (7,7, 7), (9, 9,3), (9,9, 9), (2, 2,2)]
]
x_data_gen = da.concatenate(tmp)


@profile
def HomogenizationPipeline(x):
    a1=PrimitiveTransformer(n_state=2, min_=0.0, max_=1.0).transform(x).compute()
    a2=TwoPointCorrelation(periodic_boundary=True, cutoff=15,correlations=[(1,1)]).transform(a1).compute()
    a3=FlattenTransformer().transform(a2).compute()
    a4=PCA(n_components=3).fit_transform(a3).compute()
    return a4

HomogenizationPipeline(x_data_gen)
