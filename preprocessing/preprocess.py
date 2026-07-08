"""
~/.bash_history -> data/processed.txt + data/vocab.json
                 -> data/dirs.txt + data/dir_vocab.json (reconstructed, see below)

Usage:  python preprocessing/preprocess.py [--input PATH] [--outdir DIR]

Plain bash history records no cwd or timestamp per command, so the directory
each command ran in is *reconstructed* by replaying every `cd` seen in the
(chronologically-ordered) history, starting from an assumed $HOME. This is
approximate — it can't know real session boundaries — but it's the only
signal available, and it self-corrects going forward since the live
terminal tags every new command with its real os.getcwd().
"""
import argparse
import json
import os
import shlex
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.cwd import resolve_cd_target

UNK = "<UNK>"
UNK_DIR = "<UNK_DIR>"

DEFAULT_INPUT = os.path.expanduser("~/.bash_history")
DEFAULT_OUTDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def normalize(line):
    return line.strip()


def load_commands(path):
    with open(path, "r", errors="ignore") as f:
        lines = f.readlines()
    return [normalize(line) for line in lines if normalize(line)]


def build_vocab(commands):
    command_to_id = {UNK: 0}
    for cmd in commands:
        if cmd not in command_to_id:
            command_to_id[cmd] = len(command_to_id)
    id_to_command = {str(i): cmd for cmd, i in command_to_id.items()}
    return {"command_to_id": command_to_id, "id_to_command": id_to_command}


def build_dir_vocab(dirs):
    dir_to_id = {UNK_DIR: 0}
    for d in dirs:
        if d not in dir_to_id:
            dir_to_id[d] = len(dir_to_id)
    id_to_dir = {str(i): d for d, i in dir_to_id.items()}
    return {"dir_to_id": dir_to_id, "id_to_dir": id_to_dir}


def simulate_directories(commands, start_dir=None):
    """Best-effort reconstruction of the directory each command ran in, by
    replaying `cd` calls in order. A `cd` only takes effect if its resolved
    target actually exists on disk right now (mirrors a real shell, where a
    `cd` to a bad path fails and cwd is unchanged)."""
    current = start_dir or os.path.expanduser("~")
    oldpwd = current
    dirs = []
    for cmd in commands:
        dirs.append(current)
        stripped = cmd.strip()
        if stripped == "cd" or stripped.startswith("cd ") or stripped.startswith("cd\t"):
            parts = shlex.split(stripped)
            arg = parts[1] if len(parts) > 1 else None
            target = resolve_cd_target(current, oldpwd, arg)
            if os.path.isdir(target):
                oldpwd = current
                current = target
    return dirs


def run(input_path=DEFAULT_INPUT, outdir=DEFAULT_OUTDIR):
    os.makedirs(outdir, exist_ok=True)
    commands = load_commands(input_path)
    dirs = simulate_directories(commands)

    with open(os.path.join(outdir, "raw_history.txt"), "w") as f:
        f.write("\n".join(commands) + "\n")

    with open(os.path.join(outdir, "processed.txt"), "w") as f:
        f.write("\n".join(commands) + "\n")

    with open(os.path.join(outdir, "dirs.txt"), "w") as f:
        f.write("\n".join(dirs) + "\n")

    vocab = build_vocab(commands)
    with open(os.path.join(outdir, "vocab.json"), "w") as f:
        json.dump(vocab, f, indent=2)

    dir_vocab = build_dir_vocab(dirs)
    with open(os.path.join(outdir, "dir_vocab.json"), "w") as f:
        json.dump(dir_vocab, f, indent=2)

    print(f"{len(commands)} commands -> {outdir}/processed.txt")
    print(f"vocab size: {len(vocab['command_to_id'])} -> {outdir}/vocab.json")
    print(f"dir vocab size: {len(dir_vocab['dir_to_id'])} -> {outdir}/dir_vocab.json")
    return commands, vocab, dirs, dir_vocab


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    args = parser.parse_args()
    run(args.input, args.outdir)
