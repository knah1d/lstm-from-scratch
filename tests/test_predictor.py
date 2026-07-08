import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.preprocess import build_vocab, build_dir_vocab
from training.train import run_training
from inference.predictor import load_model

CYCLE = ["git status", "git add .", "git commit", "git push"]
REPEATS = 20
SAME_DIR = "/home/project"


def write_fixture(tmp_path, commands, dirs):
    (tmp_path / "processed.txt").write_text("\n".join(commands) + "\n")
    (tmp_path / "dirs.txt").write_text("\n".join(dirs) + "\n")
    vocab = build_vocab(commands)
    dir_vocab = build_dir_vocab(dirs)
    (tmp_path / "vocab.json").write_text(json.dumps(vocab))
    (tmp_path / "dir_vocab.json").write_text(json.dumps(dir_vocab))
    return tmp_path


def test_predictor_learns_a_deterministic_command_cycle(tmp_path):
    commands = CYCLE * REPEATS
    dirs = [SAME_DIR] * len(commands)
    data_dir = write_fixture(tmp_path, commands, dirs)
    model_path = tmp_path / "weights.npz"

    run_training(
        epochs=150, seq_len=3, hidden_size=8, embed_dim=8, dir_embed_dim=8,
        lr=0.05, batch_size=8, data_dir=str(data_dir), out_path=str(model_path), verbose=False,
    )

    predictor = load_model(model_path=str(model_path), data_dir=str(data_dir))

    history = [("git status", SAME_DIR), ("git add .", SAME_DIR), ("git commit", SAME_DIR)]
    top1_cmd, top1_prob = predictor.predict_top_k(history, SAME_DIR, k=1)[0]

    assert top1_cmd == "git push"
    assert top1_prob > 0.5


def test_identical_history_predicts_differently_by_current_directory(tmp_path):
    """Regression test: the same two-command prefix ("build", "test") is
    followed by a different next command depending on which project
    directory it ran in. If the model ignores current_dir, it can't tell
    these apart and will predict the same thing both times."""
    dir_a, dir_b = "/home/project_a", "/home/project_b"
    cycle_a = ["build", "test", "deploy_prod"]
    cycle_b = ["build", "test", "deploy_staging"]

    commands = cycle_a * REPEATS + cycle_b * REPEATS
    dirs = [dir_a] * len(cycle_a) * REPEATS + [dir_b] * len(cycle_b) * REPEATS

    data_dir = write_fixture(tmp_path, commands, dirs)
    model_path = tmp_path / "weights.npz"

    run_training(
        epochs=150, seq_len=2, hidden_size=8, embed_dim=8, dir_embed_dim=8,
        lr=0.05, batch_size=8, data_dir=str(data_dir), out_path=str(model_path), verbose=False,
    )

    predictor = load_model(model_path=str(model_path), data_dir=str(data_dir))

    history_a = [("build", dir_a), ("test", dir_a)]
    history_b = [("build", dir_b), ("test", dir_b)]

    top1_a = predictor.predict_top_k(history_a, dir_a, k=1)[0][0]
    top1_b = predictor.predict_top_k(history_b, dir_b, k=1)[0][0]

    assert top1_a == "deploy_prod"
    assert top1_b == "deploy_staging"
