"""
Pure path resolution for the `cd` builtin — shared by the real executor
(terminal/executor.py) and the historical directory replay (preprocess.py),
so both agree on exactly what `cd X` means.
"""
import os


def resolve_cd_target(current_dir, oldpwd, arg):
    if not arg:
        target = os.path.expanduser("~")
    elif arg == "-":
        target = oldpwd or current_dir
    else:
        target = os.path.expanduser(arg)
        if not os.path.isabs(target):
            target = os.path.join(current_dir, target)
    return os.path.normpath(target)
