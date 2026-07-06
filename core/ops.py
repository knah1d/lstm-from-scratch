import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def dsigmoid(x):
    s = sigmoid(x)
    return s * (1.0 - s)


def tanh(x):
    return np.tanh(x)


def dtanh(x):
    t = np.tanh(x)
    return 1.0 - t ** 2


def xavier_init(shape, rng=None):
    rng = rng or np.random
    fan_in, fan_out = shape[0], shape[1]
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=shape)


def orthogonal_init(shape, rng=None):
    rng = rng or np.random
    a = rng.normal(0.0, 1.0, size=shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == shape else v
    return q
