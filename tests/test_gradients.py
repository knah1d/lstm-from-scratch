import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.cell import LSTMCell
from core.layer import LSTMLayer

INPUT_SIZE = 3
HIDDEN_SIZE = 4
BATCH = 2
SEQ_LEN = 4
EPS = 1e-5
REL_TOL = 1e-5


def make_problem(seed=0):
    rng = np.random.RandomState(seed)
    cell = LSTMCell(INPUT_SIZE, HIDDEN_SIZE, rng)
    x_t = rng.randn(BATCH, INPUT_SIZE)
    h_prev = rng.randn(BATCH, HIDDEN_SIZE)
    C_prev = rng.randn(BATCH, HIDDEN_SIZE)
    return cell, x_t, h_prev, C_prev


def loss(h_t):
    # Simple scalar loss with a clean gradient: dL/dh_t = h_t.
    return 0.5 * np.sum(h_t ** 2)


def compute_loss(cell, x_t, h_prev, C_prev):
    h_t, _, _ = cell.forward(x_t, h_prev, C_prev)
    return loss(h_t)


def rel_error(analytic, numeric):
    return np.abs(analytic - numeric) / max(1e-8, np.abs(analytic) + np.abs(numeric))


def analytic_grads(cell, x_t, h_prev, C_prev):
    h_t, C_t, cache = cell.forward(x_t, h_prev, C_prev)
    dh_ext = h_t.copy()  # dL/dh_t for L = 0.5 * sum(h_t**2)
    dh_next = np.zeros_like(h_t)
    dC_next = np.zeros_like(C_t)
    dx_t, dh_prev, dC_prev, grads = cell.backward(dh_ext, dh_next, dC_next, cache)
    return dx_t, dh_prev, dC_prev, grads


def check_array(name, analytic, numeric_fn, shape, max_checks=6):
    rng = np.random.RandomState(123)
    idxs = [tuple(rng.randint(0, s) for s in shape) for _ in range(min(max_checks, np.prod(shape)))]
    max_err = 0.0
    for idx in idxs:
        numeric = numeric_fn(idx)
        err = rel_error(analytic[idx], numeric)
        max_err = max(max_err, err)
        assert err < REL_TOL, f"{name}{idx}: analytic={analytic[idx]:.8f} numeric={numeric:.8f} rel_err={err:.2e}"
    return max_err


def test_gradcheck_weights_and_biases():
    cell, x_t, h_prev, C_prev = make_problem()
    _, _, _, grads = analytic_grads(cell, x_t, h_prev, C_prev)

    for gate in "fiog":
        def numeric_Wx(idx, gate=gate):
            i, j = idx
            orig = cell.Wx[gate][i, j]
            cell.Wx[gate][i, j] = orig + EPS
            Lp = compute_loss(cell, x_t, h_prev, C_prev)
            cell.Wx[gate][i, j] = orig - EPS
            Lm = compute_loss(cell, x_t, h_prev, C_prev)
            cell.Wx[gate][i, j] = orig
            return (Lp - Lm) / (2 * EPS)

        check_array(f"Wx[{gate}]", grads["Wx"][gate], numeric_Wx, cell.Wx[gate].shape)

        def numeric_Wh(idx, gate=gate):
            i, j = idx
            orig = cell.Wh[gate][i, j]
            cell.Wh[gate][i, j] = orig + EPS
            Lp = compute_loss(cell, x_t, h_prev, C_prev)
            cell.Wh[gate][i, j] = orig - EPS
            Lm = compute_loss(cell, x_t, h_prev, C_prev)
            cell.Wh[gate][i, j] = orig
            return (Lp - Lm) / (2 * EPS)

        check_array(f"Wh[{gate}]", grads["Wh"][gate], numeric_Wh, cell.Wh[gate].shape)

        def numeric_b(idx, gate=gate):
            (i,) = idx
            orig = cell.b[gate][i]
            cell.b[gate][i] = orig + EPS
            Lp = compute_loss(cell, x_t, h_prev, C_prev)
            cell.b[gate][i] = orig - EPS
            Lm = compute_loss(cell, x_t, h_prev, C_prev)
            cell.b[gate][i] = orig
            return (Lp - Lm) / (2 * EPS)

        check_array(f"b[{gate}]", grads["b"][gate], numeric_b, cell.b[gate].shape)


def test_gradcheck_inputs():
    cell, x_t, h_prev, C_prev = make_problem()
    dx_t, dh_prev, dC_prev, _ = analytic_grads(cell, x_t, h_prev, C_prev)

    def numeric_x(idx):
        i, j = idx
        orig = x_t[i, j]
        x_t[i, j] = orig + EPS
        Lp = compute_loss(cell, x_t, h_prev, C_prev)
        x_t[i, j] = orig - EPS
        Lm = compute_loss(cell, x_t, h_prev, C_prev)
        x_t[i, j] = orig
        return (Lp - Lm) / (2 * EPS)

    check_array("x_t", dx_t, numeric_x, x_t.shape)

    def numeric_h(idx):
        i, j = idx
        orig = h_prev[i, j]
        h_prev[i, j] = orig + EPS
        Lp = compute_loss(cell, x_t, h_prev, C_prev)
        h_prev[i, j] = orig - EPS
        Lm = compute_loss(cell, x_t, h_prev, C_prev)
        h_prev[i, j] = orig
        return (Lp - Lm) / (2 * EPS)

    check_array("h_prev", dh_prev, numeric_h, h_prev.shape)

    def numeric_C(idx):
        i, j = idx
        orig = C_prev[i, j]
        C_prev[i, j] = orig + EPS
        Lp = compute_loss(cell, x_t, h_prev, C_prev)
        C_prev[i, j] = orig - EPS
        Lm = compute_loss(cell, x_t, h_prev, C_prev)
        C_prev[i, j] = orig
        return (Lp - Lm) / (2 * EPS)

    check_array("C_prev", dC_prev, numeric_C, C_prev.shape)


def make_seq_problem(seed=0):
    rng = np.random.RandomState(seed)
    layer = LSTMLayer(INPUT_SIZE, HIDDEN_SIZE, rng)
    x_seq = rng.randn(BATCH, SEQ_LEN, INPUT_SIZE)
    h0 = rng.randn(BATCH, HIDDEN_SIZE)
    C0 = rng.randn(BATCH, HIDDEN_SIZE)
    return layer, x_seq, h0, C0


def compute_seq_loss(layer, x_seq, h0, C0):
    h_seq, _, _ = layer.forward(x_seq, h0, C0)
    return loss(h_seq)


def analytic_seq_grads(layer, x_seq, h0, C0):
    h_seq, _, caches = layer.forward(x_seq, h0, C0)
    dh_seq = h_seq.copy()  # dL/dh_t = h_t at every timestep, for L = 0.5*sum(h_seq**2)
    dx_seq, dh0, dC0, grads = layer.backward(dh_seq, caches)
    return dx_seq, dh0, dC0, grads


def test_gradcheck_bptt_weights_and_biases():
    layer, x_seq, h0, C0 = make_seq_problem()
    _, _, _, grads = analytic_seq_grads(layer, x_seq, h0, C0)
    cell = layer.cell

    for gate in "fiog":
        def numeric_Wx(idx, gate=gate):
            i, j = idx
            orig = cell.Wx[gate][i, j]
            cell.Wx[gate][i, j] = orig + EPS
            Lp = compute_seq_loss(layer, x_seq, h0, C0)
            cell.Wx[gate][i, j] = orig - EPS
            Lm = compute_seq_loss(layer, x_seq, h0, C0)
            cell.Wx[gate][i, j] = orig
            return (Lp - Lm) / (2 * EPS)

        check_array(f"seq Wx[{gate}]", grads["Wx"][gate], numeric_Wx, cell.Wx[gate].shape)

        def numeric_Wh(idx, gate=gate):
            i, j = idx
            orig = cell.Wh[gate][i, j]
            cell.Wh[gate][i, j] = orig + EPS
            Lp = compute_seq_loss(layer, x_seq, h0, C0)
            cell.Wh[gate][i, j] = orig - EPS
            Lm = compute_seq_loss(layer, x_seq, h0, C0)
            cell.Wh[gate][i, j] = orig
            return (Lp - Lm) / (2 * EPS)

        check_array(f"seq Wh[{gate}]", grads["Wh"][gate], numeric_Wh, cell.Wh[gate].shape)

        def numeric_b(idx, gate=gate):
            (i,) = idx
            orig = cell.b[gate][i]
            cell.b[gate][i] = orig + EPS
            Lp = compute_seq_loss(layer, x_seq, h0, C0)
            cell.b[gate][i] = orig - EPS
            Lm = compute_seq_loss(layer, x_seq, h0, C0)
            cell.b[gate][i] = orig
            return (Lp - Lm) / (2 * EPS)

        check_array(f"seq b[{gate}]", grads["b"][gate], numeric_b, cell.b[gate].shape)


def test_gradcheck_bptt_inputs_and_initial_state():
    layer, x_seq, h0, C0 = make_seq_problem()
    dx_seq, dh0, dC0, _ = analytic_seq_grads(layer, x_seq, h0, C0)

    def numeric_x(idx):
        b, t, j = idx
        orig = x_seq[b, t, j]
        x_seq[b, t, j] = orig + EPS
        Lp = compute_seq_loss(layer, x_seq, h0, C0)
        x_seq[b, t, j] = orig - EPS
        Lm = compute_seq_loss(layer, x_seq, h0, C0)
        x_seq[b, t, j] = orig
        return (Lp - Lm) / (2 * EPS)

    check_array("seq x_seq", dx_seq, numeric_x, x_seq.shape)

    def numeric_h0(idx):
        i, j = idx
        orig = h0[i, j]
        h0[i, j] = orig + EPS
        Lp = compute_seq_loss(layer, x_seq, h0, C0)
        h0[i, j] = orig - EPS
        Lm = compute_seq_loss(layer, x_seq, h0, C0)
        h0[i, j] = orig
        return (Lp - Lm) / (2 * EPS)

    check_array("seq h0", dh0, numeric_h0, h0.shape)

    def numeric_C0(idx):
        i, j = idx
        orig = C0[i, j]
        C0[i, j] = orig + EPS
        Lp = compute_seq_loss(layer, x_seq, h0, C0)
        C0[i, j] = orig - EPS
        Lm = compute_seq_loss(layer, x_seq, h0, C0)
        C0[i, j] = orig
        return (Lp - Lm) / (2 * EPS)

    check_array("seq C0", dC0, numeric_C0, C0.shape)


if __name__ == "__main__":
    test_gradcheck_weights_and_biases()
    test_gradcheck_inputs()
    test_gradcheck_bptt_weights_and_biases()
    test_gradcheck_bptt_inputs_and_initial_state()
    print("All gradient checks passed.")
