"""
Run a command against the real shell, inheriting stdio.

`cd` is a shell builtin: it only changes the state of the process that runs
it. A subprocess's `cd` has no effect on this Python process's own cwd, so
it's special-cased here and applied in-process via os.chdir() instead —
every other command still goes through subprocess.run(), which inherits
whatever cwd we've set.
"""
import os
import shlex
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.cwd import resolve_cd_target


def run(command):
    stripped = command.strip()
    if stripped == "cd" or stripped.startswith("cd ") or stripped.startswith("cd\t"):
        return _run_cd(stripped)
    return subprocess.run(command, shell=True)


def _run_cd(command):
    parts = shlex.split(command)
    arg = parts[1] if len(parts) > 1 else None
    target = resolve_cd_target(os.getcwd(), os.environ.get("OLDPWD"), arg)

    previous = os.getcwd()
    try:
        os.chdir(target)
    except OSError as e:
        print(f"cd: {target}: {e.strerror}")
        return None

    os.environ["OLDPWD"] = previous
    os.environ["PWD"] = os.getcwd()
    return None
