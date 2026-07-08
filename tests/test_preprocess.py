import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.preprocess import load_commands, build_vocab, build_dir_vocab, simulate_directories, run
from preprocessing.cwd import resolve_cd_target


def test_load_commands_strips_and_drops_empty_lines(tmp_path):
    history = tmp_path / "bash_history"
    history.write_text("git status\n\n  git add .  \n\ngit push\n")

    commands = load_commands(str(history))

    assert commands == ["git status", "git add .", "git push"]


def test_build_vocab_assigns_unk_id_zero_and_unique_ids():
    vocab = build_vocab(["git status", "git add .", "git status"])

    assert vocab["command_to_id"]["<UNK>"] == 0
    assert vocab["command_to_id"]["git status"] == 1
    assert vocab["command_to_id"]["git add ."] == 2
    assert len(vocab["command_to_id"]) == 3  # "git status" not duplicated
    assert vocab["id_to_command"]["1"] == "git status"


def test_build_dir_vocab_assigns_unk_dir_id_zero_and_unique_ids():
    dir_vocab = build_dir_vocab(["/home/a", "/home/b", "/home/a"])

    assert dir_vocab["dir_to_id"]["<UNK_DIR>"] == 0
    assert dir_vocab["dir_to_id"]["/home/a"] == 1
    assert dir_vocab["dir_to_id"]["/home/b"] == 2
    assert len(dir_vocab["dir_to_id"]) == 3
    assert dir_vocab["id_to_dir"]["1"] == "/home/a"


def test_resolve_cd_target_variants():
    assert resolve_cd_target("/home/alice/project", None, None) == os.path.expanduser("~")
    assert resolve_cd_target("/home/alice/project", None, "sub") == "/home/alice/project/sub"
    assert resolve_cd_target("/home/alice/project", None, "/etc") == "/etc"
    assert resolve_cd_target("/home/alice/project", "/home/alice", "-") == "/home/alice"
    assert resolve_cd_target("/a/b/c", None, "..") == "/a/b"


def test_simulate_directories_replays_cd_and_ignores_bad_targets(tmp_path):
    project = tmp_path / "project"
    sub = project / "sub"
    sub.mkdir(parents=True)

    commands = ["ls", "cd project", "ls", "cd sub", "pwd", "cd nonexistent", "ls"]
    dirs = simulate_directories(commands, start_dir=str(tmp_path))

    assert dirs[0] == str(tmp_path)          # ls, before any cd
    assert dirs[1] == str(tmp_path)          # cd project, run *from* tmp_path
    assert dirs[2] == str(project)           # ls, now inside project
    assert dirs[3] == str(project)           # cd sub, run *from* project
    assert dirs[4] == str(sub)               # pwd, now inside sub
    assert dirs[5] == str(sub)               # cd nonexistent, run from sub
    assert dirs[6] == str(sub)               # ls — failed cd left us in sub


def test_run_writes_processed_dirs_and_vocab_files(tmp_path):
    history = tmp_path / "bash_history"
    history.write_text("ls\ncd /tmp\nls\n")
    outdir = tmp_path / "data"

    commands, vocab, dirs, dir_vocab = run(str(history), str(outdir))

    assert commands == ["ls", "cd /tmp", "ls"]
    assert len(dirs) == 3
    assert dirs[2] == "/tmp"  # "cd /tmp" is absolute and exists on any unix machine

    assert (outdir / "processed.txt").read_text().splitlines() == ["ls", "cd /tmp", "ls"]
    assert (outdir / "dirs.txt").read_text().splitlines() == dirs

    saved_vocab = json.loads((outdir / "vocab.json").read_text())
    assert saved_vocab == vocab
    saved_dir_vocab = json.loads((outdir / "dir_vocab.json").read_text())
    assert saved_dir_vocab == dir_vocab
