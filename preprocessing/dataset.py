"""
data/processed.txt + data/vocab.json + data/dirs.txt + data/dir_vocab.json
-> windowed (X, X_dir, target_dir, y) id arrays for training.
"""
import json
import os

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
UNK = "<UNK>"
UNK_DIR = "<UNK_DIR>"
DEFAULT_SEQ_LEN = 3


def load_vocab(data_dir=DATA_DIR):
    with open(os.path.join(data_dir, "vocab.json")) as f:
        vocab = json.load(f)
    vocab["id_to_command"] = {int(k): v for k, v in vocab["id_to_command"].items()}
    return vocab


def load_dir_vocab(data_dir=DATA_DIR):
    with open(os.path.join(data_dir, "dir_vocab.json")) as f:
        dir_vocab = json.load(f)
    dir_vocab["id_to_dir"] = {int(k): v for k, v in dir_vocab["id_to_dir"].items()}
    return dir_vocab


def load_processed(data_dir=DATA_DIR):
    with open(os.path.join(data_dir, "processed.txt")) as f:
        return [line.strip() for line in f if line.strip()]


def load_dirs(data_dir=DATA_DIR):
    with open(os.path.join(data_dir, "dirs.txt")) as f:
        return [line.strip() for line in f if line.strip()]


def encode(cmd, vocab):
    return vocab["command_to_id"].get(cmd, 0)


def decode(cmd_id, vocab):
    return vocab["id_to_command"].get(cmd_id, UNK)


def encode_dir(d, dir_vocab):
    return dir_vocab["dir_to_id"].get(d, 0)


def decode_dir(dir_id, dir_vocab):
    return dir_vocab["id_to_dir"].get(dir_id, UNK_DIR)


def make_windows(commands, dirs, vocab, dir_vocab, seq_len=DEFAULT_SEQ_LEN):
    """X[i] = ids of commands[i:i+seq_len], X_dir[i] = ids of the directory
    each of those commands ran in, target_dir[i] = id of the directory the
    predicted command (commands[i+seq_len]) ran in, y[i] = id of that
    predicted command."""
    cmd_ids = [encode(cmd, vocab) for cmd in commands]
    dir_ids = [encode_dir(d, dir_vocab) for d in dirs]

    X, X_dir, target_dir, y = [], [], [], []
    for i in range(len(cmd_ids) - seq_len):
        X.append(cmd_ids[i:i + seq_len])
        X_dir.append(dir_ids[i:i + seq_len])
        target_dir.append(dir_ids[i + seq_len])
        y.append(cmd_ids[i + seq_len])

    return (
        np.array(X, dtype=np.int64),
        np.array(X_dir, dtype=np.int64),
        np.array(target_dir, dtype=np.int64),
        np.array(y, dtype=np.int64),
    )


def load_dataset(seq_len=DEFAULT_SEQ_LEN, data_dir=DATA_DIR):
    vocab = load_vocab(data_dir)
    dir_vocab = load_dir_vocab(data_dir)
    commands = load_processed(data_dir)
    dirs = load_dirs(data_dir)
    X, X_dir, target_dir, y = make_windows(commands, dirs, vocab, dir_vocab, seq_len)
    return X, X_dir, target_dir, y, vocab, dir_vocab
