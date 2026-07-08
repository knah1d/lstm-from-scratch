"""
Interactive shell with LSTM ghost-text command suggestions, Tab to accept.

Usage:  python terminal/terminal.py
"""
import getpass
import html
import json
import os
import socket
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings

from inference.predictor import load_model
from preprocessing.dataset import DATA_DIR, load_processed, load_dirs
from preprocessing.preprocess import build_vocab, build_dir_vocab
from training.train import run_training
from terminal import executor
from terminal.suggest import GhostSuggest
from terminal.history_store import HistoryStore

SEED_HISTORY = 10
predictor_lock = threading.Lock()


def build_key_bindings():
    kb = KeyBindings()

    @kb.add("tab")
    def _accept_suggestion(event):
        buffer = event.current_buffer
        suggestion = buffer.suggestion
        if suggestion:
            buffer.insert_text(suggestion.text)
        else:
            buffer.insert_text("    ")

    return kb


def prompt_text():
    cwd = os.getcwd()
    home = os.path.expanduser("~")
    if cwd == home or cwd.startswith(home + os.sep):
        cwd = "~" + cwd[len(home):]
    user = html.escape(getpass.getuser())
    host = html.escape(socket.gethostname())
    cwd = html.escape(cwd)
    return HTML(f"<ansigreen>{user}@{host}</ansigreen>:<ansiblue>{cwd}</ansiblue>$ ")


def make_retrain_callback(predictor):
    def retrain(store):
        new_rows, up_to_id = store.commands_since_last_train()
        if not new_rows:
            store.mark_trained(up_to_id)
            return

        new_commands = [cmd for cmd, _ in new_rows]
        new_dirs = [cwd for _, cwd in new_rows]

        with open(os.path.join(DATA_DIR, "processed.txt"), "a") as f:
            f.write("\n".join(new_commands) + "\n")
        with open(os.path.join(DATA_DIR, "dirs.txt"), "a") as f:
            f.write("\n".join(new_dirs) + "\n")

        vocab = build_vocab(load_processed(DATA_DIR))
        with open(os.path.join(DATA_DIR, "vocab.json"), "w") as f:
            json.dump(vocab, f, indent=2)

        dir_vocab = build_dir_vocab(load_dirs(DATA_DIR))
        with open(os.path.join(DATA_DIR, "dir_vocab.json"), "w") as f:
            json.dump(dir_vocab, f, indent=2)

        run_training(verbose=False)
        store.mark_trained(up_to_id)

        new_predictor = load_model()
        with predictor_lock:
            predictor.__dict__.update(new_predictor.__dict__)

    return retrain


def main():
    predictor = load_model()
    store = HistoryStore(on_retrain=make_retrain_callback(predictor))

    try:
        recent = list(zip(load_processed(DATA_DIR), load_dirs(DATA_DIR)))[-SEED_HISTORY:]
    except FileNotFoundError:
        recent = []

    session = PromptSession(
        auto_suggest=GhostSuggest(predictor, history_provider=lambda: recent),
        key_bindings=build_key_bindings(),
    )

    print("terminal-ai — Tab accepts the gray suggestion, Ctrl+D to quit")
    while True:
        try:
            command = session.prompt(prompt_text())
        except KeyboardInterrupt:
            continue
        except EOFError:
            break

        command = command.strip()
        if not command:
            continue
        if command in ("exit", "quit"):
            break

        cwd_before = os.getcwd()
        executor.run(command)
        recent.append((command, cwd_before))
        store.append(command, cwd_before)


if __name__ == "__main__":
    main()
