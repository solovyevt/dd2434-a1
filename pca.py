import numpy as np
import pandas as pd
from scipy import linalg

from util import normalize, plot

k = 2
data = pd.read_csv('data/zoo.data', header=None).to_numpy()
names = data[:, 0]
features = data[:, 1:16].astype(np.float)
labels = data[:, -1]

X_n = normalize(features)

u, s, vh = linalg.svd(X_n)
V = vh.T
Y = np.dot(X_n, V)[:, 0:k]

plot(names, labels, Y, s, title="PCA")
