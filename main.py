import math

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def test_data(count=100):
    x1 = np.random.multivariate_normal([5, 5], [[1, -0.2], [-0.2, 1]], count)
    x2 = np.random.multivariate_normal([2, 2], [[1, -0.2], [-0.2, 1]], count)
    return np.vstack((x1, x2)), np.hstack((np.ones(count), np.zeros(count)))


def gradient_step(X, Y, t):
    h = sigmoid(X @ t)
    gr = X.T @ (h - Y)
    return gr


def fit(X, Y, learning_rate=0.1, epochs=100):
    X_new = np.ones((X.shape[0], 1))
    X_new = np.hstack((X_new, X))
    weights = np.zeros((X_new.shape[1], 1))
    Y_new = np.reshape(Y, (Y.shape[0], 1))
    for i in range(epochs):
        weights -= learning_rate * gradient_step(X_new, Y_new, weights)
    return weights


def mask(data, i, j, a):
    return data[data[:, j] == a][:, i]


def show_results(min=-2, max=8):
    x, y = test_data()
    data = np.copy(x)
    data = np.column_stack((data, y))
    plt.scatter(mask(data, 0, 2, 1.), mask(data, 1, 2, 1.))
    plt.scatter(mask(data, 0, 2, 0.), mask(data, 1, 2, 0.))
    w = fit(x, y)
    print(f'b = {w[0]}\n w1 = {w[1]}\n w2 = {w[2]}')
    plt.plot([min, max], [-w[0] / w[2] - w[1] / w[2] * min, -w[0] / w[2] - w[1] / w[2] * max])
    plt.xlim(min, max)
    plt.ylim(min, max)
    plt.show()


def test_poly_data(count1=100, count2=200):
    x1 = np.random.multivariate_normal([1, 1], [[0.5, -0.6], [-0.6, 0.5]], count1)
    x2 = np.random.multivariate_normal([2, 2], [[3, 0], [0, 3]], count2)
    return np.vstack((x1, x2)), np.hstack((np.ones(count1), np.zeros(count2)))


def fit_poly(X, Y, learning_rate=0.001, iterations=200):
    X_new = np.column_stack((
        np.ones((X.shape[0], 1)),
        X,
        np.multiply(X[:, 0], X[:, 0]),
        np.multiply(X[:, 0], X[:, 1]),
        np.multiply(X[:, 1], X[:, 1]),
    ))
    Y_new = np.reshape(Y, (Y.shape[0], 1))

    weights = np.zeros((X_new.shape[1], 1))

    for i in range(iterations):
        weights -= learning_rate * gradient_step(X_new, Y_new, weights)
    return weights


def show_results_poly(min=-2, max=8):
    x, y = test_poly_data()
    w = fit_poly(x, y)
    print(f'b = {w[0]}\n w1 = {w[1]}\n w2 = {w[2]}\n w3 = {w[3]}\n w4 = {w[4]}\n w5 = {w[5]}\n')
    data = np.copy(x)
    data = np.column_stack((data, y))
    plt.scatter(mask(data, 0, 2, 1.), mask(data, 1, 2, 1.), color='b')
    plt.scatter(mask(data, 0, 2, 0.), mask(data, 1, 2, 0.), color='r')
    plt.xlim(min, max)
    plt.ylim(min, max)
    x1_list = np.linspace(min, max, 500)
    x2_list = np.linspace(min, max, 500)
    X1, X2 = np.meshgrid(x1_list, x2_list)
    Z = np.sqrt(w[0] + w[1] * X1 + w[2] * X2 + w[3] * X1 ** 2 + w[4] * X1 * X2 + w[5] * X2 ** 2)
    plt.contour(X2, X1, Z, levels=[0.2])
    plt.show()


if __name__ == '__main__':
    # show_results()
    show_results_poly()
    # test_data()
