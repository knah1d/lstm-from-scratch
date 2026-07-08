"""
Train an embedding + LSTMLayer + dense-softmax head to predict the next
bash command given the previous SEQ_LEN commands and the directory each of
them (and the predicted command itself) ran in.

Usage:  python training/train.py [--epochs N] [--hidden-size H] [--seq-len L]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.layer import LSTMLayer
from core.losses import cross_entropy_loss
from core.optimizer import Adam
from preprocessing.dataset import load_dataset

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "weights.npz"
)


def init_params(vocab_size, dir_vocab_size, embed_dim, dir_embed_dim, hidden_size, rng):
    E = rng.randn(vocab_size, embed_dim) * 0.1
    D = rng.randn(dir_vocab_size, dir_embed_dim) * 0.1
    layer = LSTMLayer(input_size=embed_dim + dir_embed_dim, hidden_size=hidden_size, rng=rng)
    Wy = rng.randn(hidden_size + dir_embed_dim, vocab_size) * 0.1
    by = np.zeros(vocab_size)
    return E, D, layer, Wy, by


def forward(X_ids, X_dir_ids, target_dir_ids, E, D, layer, Wy, by):
    cmd_embed = E[X_ids]                # (batch, seq_len, embed_dim)
    dir_embed_hist = D[X_dir_ids]       # (batch, seq_len, dir_embed_dim)
    x_seq = np.concatenate([cmd_embed, dir_embed_hist], axis=-1)

    h_seq, _, caches = layer.forward(x_seq)
    h_last = h_seq[:, -1, :]

    target_dir_embed = D[target_dir_ids]           # (batch, dir_embed_dim)
    head_input = np.concatenate([h_last, target_dir_embed], axis=-1)
    logits = head_input @ Wy + by

    return logits, head_input, h_last, h_seq, caches


def backward(dlogits, head_input, h_seq, caches, X_ids, X_dir_ids, target_dir_ids,
             E, D, layer, Wy, embed_dim):
    hidden_size = h_seq.shape[-1]

    dWy = head_input.T @ dlogits
    dby = dlogits.sum(axis=0)
    dhead_input = dlogits @ Wy.T

    dh_last = dhead_input[:, :hidden_size]
    dtarget_dir_embed = dhead_input[:, hidden_size:]

    dD = np.zeros_like(D)
    np.add.at(dD, target_dir_ids, dtarget_dir_embed)

    dh_seq = np.zeros_like(h_seq)
    dh_seq[:, -1, :] = dh_last
    dx_seq, _, _, grads = layer.backward(dh_seq, caches)

    dcmd_embed = dx_seq[:, :, :embed_dim]
    ddir_embed_hist = dx_seq[:, :, embed_dim:]

    dE = np.zeros_like(E)
    np.add.at(dE, X_ids.reshape(-1), dcmd_embed.reshape(-1, embed_dim))
    np.add.at(dD, X_dir_ids.reshape(-1), ddir_embed_hist.reshape(-1, ddir_embed_hist.shape[-1]))

    return dE, dD, grads, dWy, dby


def run_training(epochs=200, seq_len=3, hidden_size=64, embed_dim=32, dir_embed_dim=16,
                  lr=0.01, batch_size=32, seed=0,
                  data_dir=None, out_path=DEFAULT_MODEL_PATH, verbose=True):
    rng = np.random.RandomState(seed)
    kwargs = {"seq_len": seq_len}
    if data_dir is not None:
        kwargs["data_dir"] = data_dir
    X, X_dir, target_dir, y, vocab, dir_vocab = load_dataset(**kwargs)
    vocab_size = len(vocab["command_to_id"])
    dir_vocab_size = len(dir_vocab["dir_to_id"])
    if len(X) == 0:
        raise ValueError(f"Not enough commands to build a single window of seq_len={seq_len}.")

    E, D, layer, Wy, by = init_params(vocab_size, dir_vocab_size, embed_dim, dir_embed_dim, hidden_size, rng)
    optimizer = Adam(lr=lr)

    n = X.shape[0]
    losses = []
    for epoch in range(epochs):
        perm = rng.permutation(n)
        epoch_loss, num_batches = 0.0, 0

        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            X_batch, X_dir_batch = X[idx], X_dir[idx]
            target_dir_batch, y_batch = target_dir[idx], y[idx]

            logits, head_input, h_last, h_seq, caches = forward(
                X_batch, X_dir_batch, target_dir_batch, E, D, layer, Wy, by
            )
            loss, dlogits = cross_entropy_loss(logits, y_batch)
            epoch_loss += loss
            num_batches += 1

            dE, dD, grads, dWy, dby = backward(
                dlogits, head_input, h_seq, caches, X_batch, X_dir_batch, target_dir_batch,
                E, D, layer, Wy, embed_dim,
            )

            optimizer.step({"E": E, "D": D}, {"E": dE, "D": dD})
            optimizer.step(
                {"Wx": layer.cell.Wx, "Wh": layer.cell.Wh, "b": layer.cell.b},
                {"Wx": grads["Wx"], "Wh": grads["Wh"], "b": grads["b"]},
            )
            optimizer.step({"Wy": Wy, "by": by}, {"Wy": dWy, "by": dby})

        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        if verbose and (epoch % max(1, epochs // 20) == 0 or epoch == epochs - 1):
            print(f"epoch {epoch:4d}  loss {avg_loss:.4f}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(
        out_path,
        E=E, D=D,
        **{f"Wx_{g}": layer.cell.Wx[g] for g in "fiog"},
        **{f"Wh_{g}": layer.cell.Wh[g] for g in "fiog"},
        **{f"b_{g}": layer.cell.b[g] for g in "fiog"},
        Wy=Wy, by=by,
        hidden_size=hidden_size, embed_dim=embed_dim, dir_embed_dim=dir_embed_dim, seq_len=seq_len,
    )
    if verbose:
        print(f"saved model -> {out_path}")

    return losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--dir-embed-dim", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    run_training(
        epochs=args.epochs, seq_len=args.seq_len, hidden_size=args.hidden_size,
        embed_dim=args.embed_dim, dir_embed_dim=args.dir_embed_dim,
        lr=args.lr, batch_size=args.batch_size,
    )
