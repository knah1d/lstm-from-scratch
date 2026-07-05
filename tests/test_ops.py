import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.ops import sigmoid, dsigmoid, tanh, dtanh, xavier_init, orthogonal_init

EPS = 1e-6


def numerical_grad(f, x):
    return (f(x + EPS) - f(x - EPS)) / (2 * EPS)


def test_sigmoid_known_points():
    assert np.isclose(sigmoid(0.0), 0.5)
    assert sigmoid(100.0) > 0.999
    assert sigmoid(-100.0) < 0.001


def test_tanh_known_points():
    assert np.isclose(tanh(0.0), 0.0)
    assert tanh(100.0) > 0.999
    assert tanh(-100.0) < -0.999


def test_dsigmoid_matches_numerical():
    x = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    analytical = dsigmoid(x)
    numerical = numerical_grad(sigmoid, x)
    assert np.allclose(analytical, numerical, atol=1e-5)


def test_dtanh_matches_numerical():
    x = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    analytical = dtanh(x)
    numerical = numerical_grad(tanh, x)
    assert np.allclose(analytical, numerical, atol=1e-5)


def test_xavier_init_shape_and_range():
    w = xavier_init((4, 8))
    assert w.shape == (4, 8)
    limit = np.sqrt(6.0 / (4 + 8))
    assert np.all(np.abs(w) <= limit)


def test_orthogonal_init_is_orthogonal():
    w = orthogonal_init((6, 6))
    assert w.shape == (6, 6)
    assert np.allclose(w @ w.T, np.eye(6), atol=1e-6)


if __name__ == "__main__":
    test_sigmoid_known_points()
    test_tanh_known_points()
    test_dsigmoid_matches_numerical()
    test_dtanh_matches_numerical()
    test_xavier_init_shape_and_range()
    test_orthogonal_init_is_orthogonal()
    print("All ops tests passed.")
