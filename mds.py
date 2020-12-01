import pandas as pd
import numpy as np
from scipy import linalg

from util import normalize, plot, distance_matrix, dist2similarity

k = 2
data = pd.read_csv('data/zoo.data', header=None).to_numpy()
names = data[:, 0]
features = data[:, 1:16].astype(np.float)
labels = data[:, -1]

X_n = normalize(features)

D = distance_matrix(X_n)
S = dist2similarity(D)

L, U = linalg.eig(S)
L = np.real(L)
U = np.real(U)
sigma_arr = np.sqrt(L)
sigma = np.diag(sigma_arr)

Y = np.dot(U, sigma)[:, 0:k]

plot(names, labels, Y, sigma_arr[:2], "MDS")
