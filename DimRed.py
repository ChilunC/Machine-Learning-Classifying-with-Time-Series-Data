import numpy as np
from scipy.stats import norm
from pyts.transformation import StandardScaler
from pyts.visualization import plot_standardscaler
from pyts.transformation import PAA
from pyts.visualization import plot_paa

n_samples = 10
n_features = 48
n_classes = 2

rng = np.random.RandomState(41)

delta = 0.5
dt = 1

X = (norm.rvs(scale=delta ** 2 * dt, size=n_samples * n_features, random_state=rng)
     .reshape((n_samples, n_features)))
X[:, 0] = 0
X = np.cumsum(X, axis=1)

y = rng.randint(n_classes, size=n_samples)

standardscaler = StandardScaler(epsilon=1e-2)
X_standardized = standardscaler.transform(X)

plot_standardscaler(X[0])

paa = PAA(window_size=None, output_size=8, overlapping=True)
X_paa = paa.transform(X_standardized)
plot_paa(X_standardized[0], window_size=None, output_size=8, overlapping=True, marker='o')