import numpy as np

from core.ops import sigmoid, tanh, xavier_init, orthogonal_init


class LSTMCell:
    def __init__(self, input_size, hidden_size, rng=None):
        rng = rng or np.random
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Wx: input -> gate, Wh: hidden -> gate. Gates: f, i, o, g (candidate).
        self.Wx = {g: xavier_init((input_size, hidden_size), rng) for g in "fiog"}
        self.Wh = {g: orthogonal_init((hidden_size, hidden_size), rng) for g in "fiog"}
        self.b = {g: np.zeros(hidden_size) for g in "fiog"}
        # Forget gate bias starts at 1: keeps the forget gate open early in
        # training so gradients aren't killed before the model learns anything.
        self.b["f"][:] = 1.0

    def forward(self, x_t, h_prev, C_prev):
        f_pre = x_t @ self.Wx["f"] + h_prev @ self.Wh["f"] + self.b["f"]
        i_pre = x_t @ self.Wx["i"] + h_prev @ self.Wh["i"] + self.b["i"]
        o_pre = x_t @ self.Wx["o"] + h_prev @ self.Wh["o"] + self.b["o"]
        g_pre = x_t @ self.Wx["g"] + h_prev @ self.Wh["g"] + self.b["g"]

        f_t = sigmoid(f_pre)
        i_t = sigmoid(i_pre)
        o_t = sigmoid(o_pre)
        g_t = tanh(g_pre)

        C_t = f_t * C_prev + i_t * g_t
        h_t = o_t * tanh(C_t)

        cache = {
            "x_t": x_t, "h_prev": h_prev, "C_prev": C_prev,
            "f_pre": f_pre, "i_pre": i_pre, "o_pre": o_pre, "g_pre": g_pre,
            "f_t": f_t, "i_t": i_t, "o_t": o_t, "g_t": g_t,
            "C_t": C_t, "h_t": h_t,
        }
        return h_t, C_t, cache

    def backward(self, dh_ext, dh_next, dC_next, cache):
        # See notes/derivations.md for the full chain-rule derivation.
        x_t, h_prev, C_prev = cache["x_t"], cache["h_prev"], cache["C_prev"]
        f_t, i_t, o_t, g_t = cache["f_t"], cache["i_t"], cache["o_t"], cache["g_t"]
        C_t = cache["C_t"]

        dh_t = dh_ext + dh_next
        dC_t = dC_next + dh_t * o_t * (1.0 - tanh(C_t) ** 2)

        df_t = dC_t * C_prev
        di_t = dC_t * g_t
        dg_t = dC_t * i_t
        do_t = dh_t * tanh(C_t)

        d_pre = {
            "f": df_t * f_t * (1.0 - f_t),
            "i": di_t * i_t * (1.0 - i_t),
            "o": do_t * o_t * (1.0 - o_t),
            "g": dg_t * (1.0 - g_t ** 2),
        }

        grads = {"Wx": {}, "Wh": {}, "b": {}}
        dx_t = np.zeros_like(x_t)
        dh_prev = np.zeros_like(h_prev)
        for k, dp in d_pre.items():
            grads["Wx"][k] = x_t.T @ dp
            grads["Wh"][k] = h_prev.T @ dp
            grads["b"][k] = dp.sum(axis=0)
            dx_t += dp @ self.Wx[k].T
            dh_prev += dp @ self.Wh[k].T

        dC_prev = dC_t * f_t

        return dx_t, dh_prev, dC_prev, grads
