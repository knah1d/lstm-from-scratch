import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.layer import LSTMLayer
from core.losses import mse_loss
from core.optimizer import SGD, Adam

WINDOW = 20
HIDDEN_SIZE = 16


def generate_sine_data(n_points=600, window=WINDOW):
    t = np.linspace(0, 30 * np.pi, n_points)
    series = np.sin(t)
    X, y = [], []
    for i in range(len(series) - window):
        X.append(series[i:i + window])
        y.append(series[i + window])
    X = np.array(X)[..., None]  # (N, window, 1)
    y = np.array(y)[..., None]  # (N, 1)
    return X, y


def init_head(hidden_size, rng):
    Wy = rng.randn(hidden_size, 1) * 0.1
    by = np.zeros(1)
    return Wy, by


def forward_head(h_last, Wy, by):
    return h_last @ Wy + by


def train(optimizer_name, num_epochs, lr, batch_size=32, hidden_size=HIDDEN_SIZE, seed=0):
    rng = np.random.RandomState(seed)
    X, y = generate_sine_data()

    layer = LSTMLayer(input_size=1, hidden_size=hidden_size, rng=rng)
    Wy, by = init_head(hidden_size, rng)

    optimizer = SGD(lr=lr) if optimizer_name == "sgd" else Adam(lr=lr)

    n = X.shape[0]
    losses = []
    for epoch in range(num_epochs):
        perm = rng.permutation(n)
        epoch_loss, num_batches = 0.0, 0

        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            x_batch, y_batch = X[idx], y[idx]

            h_seq, C_final, caches = layer.forward(x_batch)
            h_last = h_seq[:, -1, :]
            pred = forward_head(h_last, Wy, by)

            loss, dpred = mse_loss(pred, y_batch)
            epoch_loss += loss
            num_batches += 1

            dWy = h_last.T @ dpred
            dby = dpred.sum(axis=0)
            dh_last = dpred @ Wy.T

            dh_seq = np.zeros_like(h_seq)
            dh_seq[:, -1, :] = dh_last

            _, _, _, grads = layer.backward(dh_seq, caches)

            optimizer.step(
                {"Wx": layer.cell.Wx, "Wh": layer.cell.Wh, "b": layer.cell.b},
                {"Wx": grads["Wx"], "Wh": grads["Wh"], "b": grads["b"]},
            )
            optimizer.step({"Wy": Wy, "by": by}, {"Wy": dWy, "by": dby})

        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        if epoch % 20 == 0 or epoch == num_epochs - 1:
            print(f"[{optimizer_name}] epoch {epoch:4d}  loss {avg_loss:.6f}")

    return layer, Wy, by, X, y, losses


def plot_results(sgd_losses, adam_losses, layer, Wy, by, X, y, out_path):
    h_seq, _, _ = layer.forward(X)
    preds = forward_head(h_seq[:, -1, :], Wy, by)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(sgd_losses, label="sgd")
    axes[0].plot(adam_losses, label="adam")
    axes[0].set_title("training loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("MSE")
    axes[0].legend()

    axes[1].plot(y.flatten(), label="true")
    axes[1].plot(preds.flatten(), label="pred (adam)", linestyle="--")
    axes[1].set_title("prediction vs actual")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path)
    print(f"saved plot to {out_path}")


if __name__ == "__main__":
    _, _, _, _, _, sgd_losses = train(optimizer_name="sgd", num_epochs=100, lr=0.3)
    print("final loss (sgd):", sgd_losses[-1])

    layer, Wy, by, X, y, adam_losses = train(optimizer_name="adam", num_epochs=100, lr=0.01)
    print("final loss (adam):", adam_losses[-1])

    out_dir = os.path.dirname(os.path.abspath(__file__))
    plot_results(sgd_losses, adam_losses, layer, Wy, by, X, y,
                 os.path.join(out_dir, "toy_task_result.png"))
