# Originally written by Olivier Grisel as part of the scikit-learn documentation and released under the following license:
# # Author: Olivier Grisel <olivier.grisel@ensta.org>
# # License: BSD 3 clause

import sys
import numpy as np

from sklearn.utils import shuffle
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
from meticulous import Experiment

# Number of run (with randomly generated dataset) for each strategy so as
# to be able to compute an estimate of the standard deviation
n_runs = 5

# k-means models can do several random inits so as to be able to trade
# CPU time for convergence robustness
n_init_range = [1, 5, 10, 15, 20]

# Datasets generation parameters
n_samples_per_center = 100
grid_size = 3
scale = 0.1
n_clusters = grid_size ** 2


def make_data(random_state, n_samples_per_center, grid_size, scale):
    centers = np.array([[i, j]
                        for i in range(grid_size)
                        for j in range(grid_size)])
    n_clusters_true, n_features = centers.shape

    noise = np.random.normal(
        scale=scale, size=(n_samples_per_center, centers.shape[1]))

    X = np.concatenate([c + noise for c in centers])
    y = np.concatenate([[i] * n_samples_per_center
                        for i in range(n_clusters_true)])
    return shuffle(X, y)

cases = [
    (KMeans, 'k-means++', {}),
    (KMeans, 'random', {}),
    (MiniBatchKMeans, 'k-means++', {'max_no_improvement': 3}),
    (MiniBatchKMeans, 'random', {'max_no_improvement': 3, 'init_size': 500}),
]

for factory, init, params in cases:
    for run_id in range(n_runs):
        X, y = make_data(run_id, n_samples_per_center, grid_size, scale)
        for n_init in n_init_range:
            with Experiment({"estimator": factory.__name__, "init": init, "n_init": int(n_init), **params}) as exp:
                km = factory(n_clusters=n_clusters, init=init,
                            n_init=n_init, **params).fit(X)
                print(factory.__name__, init, km.inertia_)
                if np.random.choice([True, False], p=[0.05, 0.95]):
                    asfa
                exp.summary(dict(inertia=km.inertia_))

print("Experiments done. Checkout the results with \n" + \
      "> meticulous experiments/ --groupby 'args_init, args_estimator, args_n_init' --sort summary_inertia_mean --sort_reverse")