import matplotlib.pyplot as plt
import numpy as np


def normalize(X):
    return (X - X.mean(axis=0)) / X.std(axis=0)


def plot(names, labels, Y, s, title):
    plt.scatter(Y[:, 0], Y[:, 1], c=labels)
    plt.title(title)
    for i, txt in enumerate(labels):
        plt.annotate(str(int(txt)), Y[i])
    plt.xlabel("PC_1, $\sigma_1$ = " + str(round(s[0] / s.mean(), 2)))
    plt.ylabel("PC_2, $\sigma_2$ = " + str(round(s[1] / s.mean(), 2)))
    plt.savefig('img/{}.png'.format(title))
    plt.show()


def distance_matrix(X):
    n = X.shape[0]
    D = np.zeros((n, n))
    for x_i in range(n):
        for x_j in range(n):
            D[x_i][x_j] = np.linalg.norm(X[x_i] - X[x_j])
    return D


def dist2similarity(D):
    return -1 / 2 * (D - np.matmul(D, np.ones(D.shape)) / D.shape[0]
                     - np.matmul(np.ones(D.shape), D) / D.shape[1] + np.mean(D))
