import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.cell import LSTMCell
from core.layer import LSTMLayer

INPUT_SIZE = 3
HIDDEN_SIZE = 5
BATCH = 4
SEQ_LEN = 6


def make_cell(seed=0):
    rng = np.random.RandomState(seed)
    return LSTMCell(INPUT_SIZE, HIDDEN_SIZE, rng)


def test_forward_output_shapes():
    cell = make_cell()
    x_t = np.random.randn(BATCH, INPUT_SIZE)
    h_prev = np.zeros((BATCH, HIDDEN_SIZE))
    C_prev = np.zeros((BATCH, HIDDEN_SIZE))

    h_t, C_t, cache = cell.forward(x_t, h_prev, C_prev)

    assert h_t.shape == (BATCH, HIDDEN_SIZE)
    assert C_t.shape == (BATCH, HIDDEN_SIZE)


def test_forward_output_ranges():
    cell = make_cell()
    x_t = np.random.randn(BATCH, INPUT_SIZE) * 5  # large inputs stress-test saturation
    h_prev = np.random.randn(BATCH, HIDDEN_SIZE)
    C_prev = np.random.randn(BATCH, HIDDEN_SIZE) * 3

    h_t, C_t, cache = cell.forward(x_t, h_prev, C_prev)

    # h_t = o_t * tanh(C_t), o_t in [0,1], tanh in [-1,1] -> h_t in [-1,1]
    assert np.all(h_t >= -1.0) and np.all(h_t <= 1.0)
    # gate activations must be valid probabilities
    for gate in ("f_t", "i_t", "o_t"):
        assert np.all(cache[gate] >= 0.0) and np.all(cache[gate] <= 1.0)
    assert np.all(cache["g_t"] >= -1.0) and np.all(cache["g_t"] <= 1.0)


def test_forward_deterministic_given_same_weights():
    cell = make_cell(seed=42)
    x_t = np.random.RandomState(1).randn(BATCH, INPUT_SIZE)
    h_prev = np.zeros((BATCH, HIDDEN_SIZE))
    C_prev = np.zeros((BATCH, HIDDEN_SIZE))

    h_t_1, C_t_1, _ = cell.forward(x_t, h_prev, C_prev)
    h_t_2, C_t_2, _ = cell.forward(x_t, h_prev, C_prev)

    assert np.allclose(h_t_1, h_t_2)
    assert np.allclose(C_t_1, C_t_2)
    