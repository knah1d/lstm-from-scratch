import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.dataset import (
    load_vocab, load_dir_vocab, load_processed, load_dirs,
    encode, decode, encode_dir, decode_dir, make_windows, load_dataset,
)

COMMANDS = ["git status", "git add .", "git commit", "git push", "git status", "git add ."]
DIRS = ["/home/x", "/home/x", "/home/x", "/home/x", "/home/y", "/home/y"]


def make_vocab():
    command_to_id = {"<UNK>": 0}
    for cmd in COMMANDS:
        if cmd not in command_to_id:
            command_to_id[cmd] = len(command_to_id)
    # id_to_command uses int keys here, matching what load_vocab() produces after
    # parsing vocab.json (JSON itself only allows string keys on disk).
    return {"command_to_id": command_to_id, "id_to_command": {v: k for k, v in command_to_id.items()}}


def make_dir_vocab():
    dir_to_id = {"<UNK_DIR>": 0}
    for d in DIRS:
        if d not in dir_to_id:
            dir_to_id[d] = len(dir_to_id)
    return {"dir_to_id": dir_to_id, "id_to_dir": {v: k for k, v in dir_to_id.items()}}


def write_fixture(tmp_path):
    (tmp_path / "processed.txt").write_text("\n".join(COMMANDS) + "\n")
    (tmp_path / "dirs.txt").write_text("\n".join(DIRS) + "\n")
    vocab = make_vocab()
    dir_vocab = make_dir_vocab()
    (tmp_path / "vocab.json").write_text(json.dumps(vocab))
    (tmp_path / "dir_vocab.json").write_text(json.dumps(dir_vocab))
    return vocab, dir_vocab


def test_encode_decode_roundtrip():
    vocab = make_vocab()
    assert decode(encode("git push", vocab), vocab) == "git push"
    assert encode("never seen", vocab) == 0
    assert decode(0, vocab) == "<UNK>"


def test_encode_decode_dir_roundtrip():
    dir_vocab = make_dir_vocab()
    assert decode_dir(encode_dir("/home/y", dir_vocab), dir_vocab) == "/home/y"
    assert encode_dir("/never/seen", dir_vocab) == 0
    assert decode_dir(0, dir_vocab) == "<UNK_DIR>"


def test_make_windows_shapes_and_content():
    vocab = make_vocab()
    dir_vocab = make_dir_vocab()
    X, X_dir, target_dir, y = make_windows(COMMANDS, DIRS, vocab, dir_vocab, seq_len=3)

    # len(COMMANDS) - seq_len windows
    assert X.shape == (3, 3)
    assert X_dir.shape == (3, 3)
    assert target_dir.shape == (3,)
    assert y.shape == (3,)

    # first window: [git status, git add ., git commit] -> target git push
    assert X[0].tolist() == [encode(c, vocab) for c in COMMANDS[0:3]]
    assert X_dir[0].tolist() == [encode_dir(d, dir_vocab) for d in DIRS[0:3]]
    assert y[0] == encode(COMMANDS[3], vocab)
    assert target_dir[0] == encode_dir(DIRS[3], dir_vocab)


def test_load_dataset_reads_from_data_dir(tmp_path):
    vocab, dir_vocab = write_fixture(tmp_path)

    X, X_dir, target_dir, y, loaded_vocab, loaded_dir_vocab = load_dataset(seq_len=3, data_dir=str(tmp_path))

    assert X.shape[1] == 3
    assert len(X) == len(X_dir) == len(target_dir) == len(y) == len(COMMANDS) - 3
    assert load_processed(str(tmp_path)) == COMMANDS
    assert load_dirs(str(tmp_path)) == DIRS
    assert loaded_vocab["command_to_id"] == vocab["command_to_id"]
    assert loaded_dir_vocab["dir_to_id"] == dir_vocab["dir_to_id"]
