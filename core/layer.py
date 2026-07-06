import numpy as np

from core.cell import LSTMCell


class LSTMLayer:
    def __init__(self, input_size, hidden_size, rng=None):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell = LSTMCell(input_size, hidden_size, rng)

    def forward(self, x_seq, h0=None, C0=None):
        # x_seq: (batch, T, input_size)
        batch, T, _ = x_seq.shape
        h_prev = h0 if h0 is not None else np.zeros((batch, self.hidden_size))
        C_prev = C0 if C0 is not None else np.zeros((batch, self.hidden_size))

        h_seq = np.zeros((batch, T, self.hidden_size))
        caches = []
        for t in range(T):
            h_t, C_t, cache = self.cell.forward(x_seq[:, t, :], h_prev, C_prev)
            h_seq[:, t, :] = h_t
            caches.append(cache)
            h_prev, C_prev = h_t, C_t

        return h_seq, C_prev, caches

    def backward(self, dh_seq, caches, dC_final=None):
        # dh_seq: (batch, T, hidden) external gradient on h_t at every timestep.
        batch, T, _ = dh_seq.shape
        dh_next = np.zeros((batch, self.hidden_size))
        dC_next = dC_final if dC_final is not None else np.zeros((batch, self.hidden_size))

        grads = {
            "Wx": {k: np.zeros_like(self.cell.Wx[k]) for k in "fiog"},
            "Wh": {k: np.zeros_like(self.cell.Wh[k]) for k in "fiog"},
            "b": {k: np.zeros_like(self.cell.b[k]) for k in "fiog"},
        }
        dx_seq = np.zeros((batch, T, self.input_size))

        for t in reversed(range(T)):
            dx_t, dh_next, dC_next, grads_t = self.cell.backward(
                dh_seq[:, t, :], dh_next, dC_next, caches[t]
            )
            dx_seq[:, t, :] = dx_t
            for k in "fiog":
                grads["Wx"][k] += grads_t["Wx"][k]
                grads["Wh"][k] += grads_t["Wh"][k]
                grads["b"][k] += grads_t["b"][k]

        # after the loop, dh_next/dC_next hold the gradient w.r.t. h0/C0
        return dx_seq, dh_next, dC_next, grads
