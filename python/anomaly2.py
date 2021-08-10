import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
X, y_true = make_blobs(n_samples=500, centers=1,
                       cluster_std=0.60, random_state=5)
X_append, y_true_append = make_blobs(
    n_samples=20, centers=1, cluster_std=5, random_state=5)
X = np.vstack([X, X_append])
y_true = np.hstack([y_true, [1 for _ in y_true_append]])
X = X[:, ::-1]
plt.scatter(X[:, 0], X[:, 1], marker="x")
