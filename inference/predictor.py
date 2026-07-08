"""
Load models/weights.npz and predict the next bash command given recent
history and the directory the next command will run in.

Usage:  python -m inference.predictor --cwd /path/to/dir "git status" "git add ."
"""
import argparse
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.layer import LSTMLayer
from core.losses import softmax
from preprocessing.dataset import (
    DATA_DIR, load_vocab, load_dir_vocab, load_processed,
    encode, decode, encode_dir,
)

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "weights.npz"
)


class Predictor:
    def __init__(self, weights, vocab, dir_vocab, command_counts=None):
        self.vocab = vocab
        self.dir_vocab = dir_vocab
        self.command_counts = command_counts or Counter()
        self.seq_len = int(weights["seq_len"])
        self.hidden_size = int(weights["hidden_size"])
        self.embed_dim = int(weights["embed_dim"])
        self.dir_embed_dim = int(weights["dir_embed_dim"])

        self.E = weights["E"]
        self.D = weights["D"]
        self.layer = LSTMLayer(input_size=self.embed_dim + self.dir_embed_dim, hidden_size=self.hidden_size)
        for g in "fiog":
            self.layer.cell.Wx[g] = weights[f"Wx_{g}"]
            self.layer.cell.Wh[g] = weights[f"Wh_{g}"]
            self.layer.cell.b[g] = weights[f"b_{g}"]
        self.Wy = weights["Wy"]
        self.by = weights["by"]

    def _encode_history(self, history):
        window = [(cmd.strip(), d) for cmd, d in history if cmd.strip()][-self.seq_len:]
        window = [("<UNK>", "<UNK_DIR>")] * (self.seq_len - len(window)) + window
        cmd_ids = [encode(cmd, self.vocab) for cmd, _ in window]
        dir_ids = [encode_dir(d, self.dir_vocab) for _, d in window]
        return np.array([cmd_ids], dtype=np.int64), np.array([dir_ids], dtype=np.int64)

    def predict_proba(self, history, current_dir):
        cmd_ids, dir_ids = self._encode_history(history)
        cmd_embed = self.E[cmd_ids]
        dir_embed_hist = self.D[dir_ids]
        x_seq = np.concatenate([cmd_embed, dir_embed_hist], axis=-1)

        h_seq, _, _ = self.layer.forward(x_seq)
        h_last = h_seq[:, -1, :]

        target_dir_id = np.array([encode_dir(current_dir, self.dir_vocab)])
        target_dir_embed = self.D[target_dir_id]
        head_input = np.concatenate([h_last, target_dir_embed], axis=-1)
        logits = head_input @ self.Wy + self.by
        return softmax(logits)[0]

    def predict_top_k(self, history, current_dir, k=3):
        probs = self.predict_proba(history, current_dir)
        top_idx = np.argsort(probs)[::-1][:k]
        return [(decode(int(i), self.vocab), float(probs[i])) for i in top_idx]

    def prefix_matches(self, prefix, limit=5):
        """Frequency-ranked fallback for when no top-k LSTM prediction matches
        what the user is currently typing. Not directory-aware."""
        candidates = [
            c for c in self.vocab["command_to_id"]
            if c != "<UNK>" and c != prefix and c.startswith(prefix)
        ]
        candidates.sort(key=lambda c: self.command_counts.get(c, 0), reverse=True)
        return candidates[:limit]


def load_model(model_path=DEFAULT_MODEL_PATH, data_dir=DATA_DIR):
    weights = np.load(model_path)
    vocab = load_vocab(data_dir)
    dir_vocab = load_dir_vocab(data_dir)
    try:
        command_counts = Counter(load_processed(data_dir))
    except FileNotFoundError:
        command_counts = Counter()
    return Predictor(weights, vocab, dir_vocab, command_counts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cwd", default=os.getcwd())
    parser.add_argument("commands", nargs="+")
    args = parser.parse_args()

    predictor = load_model()
    history = [(cmd, args.cwd) for cmd in args.commands]
    for i, (cmd, prob) in enumerate(predictor.predict_top_k(history, args.cwd, k=3), start=1):
        print(f"{i}. {cmd}  (p={prob:.3f})")
