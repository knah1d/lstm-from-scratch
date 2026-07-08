"""
Append-only sqlite log of every command executed in the shell (with the
directory it ran in), plus a retrain-trigger counter for continuous
learning (Phase 8).
"""
import os
import sqlite3
import threading
import time

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DB_PATH = os.path.join(DATA_DIR, "history.db")
TRAIN_EVERY = int(os.environ.get("TRAIN_EVERY", "500"))


class HistoryStore:
    def __init__(self, db_path=DB_PATH, train_every=TRAIN_EVERY, on_retrain=None):
        self.train_every = train_every
        self.on_retrain = on_retrain
        self._retraining = False

        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS commands ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, command TEXT NOT NULL, "
            "cwd TEXT NOT NULL DEFAULT '', ts REAL NOT NULL)"
        )
        self.conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value INTEGER NOT NULL)")
        self.conn.execute("INSERT OR IGNORE INTO meta (key, value) VALUES ('last_trained_id', 0)")
        self._migrate_add_cwd_column()
        self.conn.commit()

    def _migrate_add_cwd_column(self):
        """Older versions of this db (before directory-aware suggestions)
        didn't log a cwd per command — add the column instead of breaking
        on their existing history.db."""
        columns = [row[1] for row in self.conn.execute("PRAGMA table_info(commands)").fetchall()]
        if "cwd" not in columns:
            self.conn.execute("ALTER TABLE commands ADD COLUMN cwd TEXT NOT NULL DEFAULT ''")

    def append(self, command, cwd):
        self.conn.execute(
            "INSERT INTO commands (command, cwd, ts) VALUES (?, ?, ?)", (command, cwd, time.time())
        )
        self.conn.commit()

        if self.on_retrain is not None and not self._retraining and self.pending_count() >= self.train_every:
            self._start_retrain()

    def _max_id(self):
        row = self.conn.execute("SELECT COALESCE(MAX(id), 0) FROM commands").fetchone()
        return row[0]

    def _last_trained_id(self):
        row = self.conn.execute("SELECT value FROM meta WHERE key = 'last_trained_id'").fetchone()
        return row[0] if row else 0

    def pending_count(self):
        return self._max_id() - self._last_trained_id()

    def commands_since_last_train(self):
        last_trained = self._last_trained_id()
        rows = self.conn.execute(
            "SELECT command, cwd FROM commands WHERE id > ? ORDER BY id", (last_trained,)
        ).fetchall()
        return [(r[0], r[1]) for r in rows], self._max_id()

    def mark_trained(self, up_to_id):
        self.conn.execute("UPDATE meta SET value = ? WHERE key = 'last_trained_id'", (up_to_id,))
        self.conn.commit()

    def all_commands(self):
        rows = self.conn.execute("SELECT command, cwd FROM commands ORDER BY id").fetchall()
        return [(r[0], r[1]) for r in rows]

    def _start_retrain(self):
        self._retraining = True

        def wrapper():
            try:
                self.on_retrain(self)
            finally:
                self._retraining = False

        threading.Thread(target=wrapper, daemon=True).start()
