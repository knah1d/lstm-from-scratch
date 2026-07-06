import numpy as np


class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, params, grads):
        if isinstance(params, dict):
            for k in params:
                self.step(params[k], grads[k])
        else:
            params -= self.lr * grads


class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = {}
        self.v = {}

    def step(self, params, grads):
        self.t += 1
        self._update(params, grads)

    def _update(self, params, grads):
        if isinstance(params, dict):
            for k in params:
                self._update(params[k], grads[k])
            return

        pid = id(params)
        if pid not in self.m:
            self.m[pid] = np.zeros_like(params)
            self.v[pid] = np.zeros_like(params)

        self.m[pid] = self.beta1 * self.m[pid] + (1 - self.beta1) * grads
        self.v[pid] = self.beta2 * self.v[pid] + (1 - self.beta2) * (grads ** 2)
        m_hat = self.m[pid] / (1 - self.beta1 ** self.t)
        v_hat = self.v[pid] / (1 - self.beta2 ** self.t)
        params -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
