"""
Ghost-text suggestion source for prompt_toolkit.

- Empty buffer (right after a command executes): suggest the LSTM's single
  best next-command prediction, conditioned on the real current directory.
- Non-empty buffer (user mid-typing, e.g. "git p"): prefer an LSTM top-k
  prediction that matches the typed prefix; fall back to a frequency-ranked
  scan of the full vocab if none of the top-k match.

The LSTM conditions on a *learned embedding* of the current directory — it
has no notion of whether a `cd` target actually exists on disk (that's just
a categorical feature to it, and it degrades to a generic guess for any
directory it never saw during training). So any candidate `cd ...` command
is additionally checked against the real filesystem here before ever being
shown, using the same path resolution the real executor uses.
"""
import os
import shlex

from prompt_toolkit.auto_suggest import AutoSuggest, Suggestion

from preprocessing.cwd import resolve_cd_target


def _is_valid_cd(cmd):
    stripped = cmd.strip()
    if stripped != "cd" and not stripped.startswith("cd ") and not stripped.startswith("cd\t"):
        return True  # not a cd command, nothing to validate

    parts = shlex.split(stripped)
    arg = parts[1] if len(parts) > 1 else None
    target = resolve_cd_target(os.getcwd(), os.environ.get("OLDPWD"), arg)
    return os.path.isdir(target)


def _real_dir_completions(arg_typed, limit=5):
    """Plain filesystem directory listing — the bottom-rung fallback for
    `cd` completion, used when nothing the model has ever seen verbatim is a
    valid target here. Unlike the model/vocab paths, this can't rank by
    usage, but it's guaranteed to point at a real directory."""
    head, partial = os.path.split(arg_typed)
    if head == "":
        base_dir = os.getcwd()
    else:
        expanded = os.path.expanduser(head)
        base_dir = expanded if os.path.isabs(expanded) else os.path.normpath(os.path.join(os.getcwd(), expanded))

    try:
        names = sorted(
            name for name in os.listdir(base_dir)
            if name.startswith(partial)
            and (partial.startswith(".") or not name.startswith("."))  # hide dotdirs unless asked for
            and os.path.isdir(os.path.join(base_dir, name))
        )
    except OSError:
        return [], partial

    return names[:limit], partial


class GhostSuggest(AutoSuggest):
    def __init__(self, predictor, history_provider, k=5):
        self.predictor = predictor
        self.history_provider = history_provider
        self.k = k

    def get_suggestion(self, buffer, document):
        history = self.history_provider()
        if not history:
            return None

        text = document.text
        current_dir = os.getcwd()
        try:
            predictions = self.predictor.predict_top_k(history, current_dir, k=self.k)
        except Exception:
            return None

        if not text:
            for cmd, _ in predictions:
                if cmd != "<UNK>" and _is_valid_cd(cmd):
                    return Suggestion(cmd)
            return None

        for cmd, _ in predictions:
            if cmd != "<UNK>" and cmd != text and cmd.startswith(text) and _is_valid_cd(cmd):
                return Suggestion(cmd[len(text):])

        for cmd in self.predictor.prefix_matches(text, limit=10):
            if _is_valid_cd(cmd):
                return Suggestion(cmd[len(text):])

        if text == "cd" or text.startswith("cd ") or text.startswith("cd\t"):
            arg_typed = text[3:] if len(text) > 2 else ""
            lead_in = "" if len(text) > 2 else " "
            names, partial = _real_dir_completions(arg_typed)
            if names:
                return Suggestion(lead_in + names[0][len(partial):] + "/")

        return None
