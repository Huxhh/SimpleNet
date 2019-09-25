# coding=utf-8

import numpy as np


def _numercial_gradient_no_batch(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp = x[idx]
        x[idx] = float(tmp) - h
        fx1 = f(x)

        x[idx] = float(tmp) + h
        fx2 = f(x)

        grad[idx] = (fx2 - fx1) / 2 * h
        x[idx] = tmp

    return grad


def numerical_gradient_2d(f, X):
    if X.ndim == 1:
        return _numercial_gradient_no_batch(f, X)
    else:
        grad = np.zeros_like(X)

        for idx, x in enumerate(X):
            grad[idx] = _numercial_gradient_no_batch(f, x)

        return grad


def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp = x[idx]
        x[idx] = float(tmp) + h
        fx1 = f(x)

        x[idx] = tmp - h
        fx2 = f(x)
        grad[idx] = (fx1 - fx2) / 2 * h

        x[idx] = tmp
        it.iternext()

    return grad


def gradient_descent(f, init_x, lr, step_num=100):
    x = init_x

    for _ in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x
