"""
fractal_brain/storage.py
A thin, dependency-free SQLite persistence layer -- sqlite3 is in the Python standard
library, so this doesn't compromise the project's zero-dependency design.

Covers the "Storage schema" item from To-Do.md, using the schema recommended there:
vocab, samples, documents (for RAG persistence), checkpoints, and metrics, plus a
generic memory key-value table.

This complements checkpoint.py rather than replacing it: checkpoint.py knows how to
turn a FractalBrain into/from a JSON-serializable blob; Storage is where you might keep
many such blobs (plus vocab, samples, RAG documents, and metrics) queryable in one file
instead of scattered across loose files.
"""
import sqlite3
import json
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS vocab (
    token TEXT PRIMARY KEY,
    token_id INTEGER NOT NULL,
    freq INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT,
    token_ids_json TEXT NOT NULL,
    label_json TEXT,
    split TEXT DEFAULT 'train'
);
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT,
    text TEXT,
    embedding_blob BLOB,
    created_at TEXT
);
CREATE TABLE IF NOT EXISTS checkpoints (
    version TEXT PRIMARY KEY,
    model_blob BLOB NOT NULL,
    config_json TEXT,
    created_at TEXT
);
CREATE TABLE IF NOT EXISTS metrics (
    step INTEGER,
    loss REAL,
    acc REAL,
    timestamp TEXT
);
CREATE TABLE IF NOT EXISTS memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT,
    value TEXT,
    type TEXT,
    updated_at TEXT
);
"""


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class Storage:
    """
    A single-file SQLite store for everything in To-Do.md's suggested schema: tokenizer
    vocabulary, dataset samples, RAG documents (with embeddings), model checkpoints, and
    training metrics.

    Usage:
        with Storage("project.db") as db:
            db.save_vocab(tokenizer)
            db.save_samples(dataset, split="train")
            db.log_metric(step, loss)
            db.save_checkpoint_blob("epoch-5", json.dumps(serialize_brain(brain)).encode())
    """
    def __init__(self, path):
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()

    # ---- vocab ----
    def save_vocab(self, tokenizer):
        """
        Persist a tokenizer's token -> id mapping for quick lookup/inspection (matches
        the `vocab(token, token_id, freq)` schema suggested in To-Do.md). This does NOT
        store merge order, so it's not a full round-trip on its own -- use
        `tokenizer.BPETokenizer.save()`/`.load()` (JSON, includes merges) for that, and
        treat this table as a queryable index alongside it.
        """
        rows = [(tok, tid, 0) for tok, tid in tokenizer.token_to_id.items()]
        self.conn.executemany(
            "INSERT OR REPLACE INTO vocab (token, token_id, freq) VALUES (?, ?, ?)", rows)
        self.conn.commit()

    def load_vocab(self):
        """Return {token: token_id}."""
        cur = self.conn.execute("SELECT token, token_id FROM vocab")
        return dict(cur.fetchall())

    # ---- samples ----
    def save_sample(self, token_ids, label=None, text=None, split="train"):
        self.conn.execute(
            "INSERT INTO samples (text, token_ids_json, label_json, split) VALUES (?, ?, ?, ?)",
            (text, json.dumps(token_ids), json.dumps(label) if label is not None else None, split))
        self.conn.commit()

    def save_samples(self, examples, split="train"):
        """Bulk insert. `examples`: iterable of (token_ids, label) pairs, e.g. a
        dataset.TextDataset or dataset.DatasetView."""
        rows = [(None, json.dumps(list(ctx)), json.dumps(list(tgt)), split) for ctx, tgt in examples]
        self.conn.executemany(
            "INSERT INTO samples (text, token_ids_json, label_json, split) VALUES (?, ?, ?, ?)", rows)
        self.conn.commit()

    def iter_samples(self, split=None):
        """Yield (token_ids, label) pairs, optionally filtered by split."""
        if split is None:
            cur = self.conn.execute("SELECT token_ids_json, label_json FROM samples")
        else:
            cur = self.conn.execute(
                "SELECT token_ids_json, label_json FROM samples WHERE split = ?", (split,))
        for token_ids_json, label_json in cur:
            yield json.loads(token_ids_json), (json.loads(label_json) if label_json else None)

    def count_samples(self, split=None):
        if split is None:
            return self.conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
        return self.conn.execute(
            "SELECT COUNT(*) FROM samples WHERE split = ?", (split,)).fetchone()[0]

    # ---- documents (RAG persistence) ----
    def save_document(self, vector, source=None, text=None):
        """vector: a Vector, or any list of floats. Returns the new row's id."""
        data = vector.to_list() if hasattr(vector, "to_list") else list(vector)
        blob = json.dumps(data).encode("utf-8")
        cur = self.conn.execute(
            "INSERT INTO documents (source, text, embedding_blob, created_at) VALUES (?, ?, ?, ?)",
            (source, text, blob, _now()))
        self.conn.commit()
        return cur.lastrowid

    def load_documents(self):
        """Return a list of (id, source, text, embedding_as_list_of_floats)."""
        cur = self.conn.execute("SELECT id, source, text, embedding_blob FROM documents")
        return [(doc_id, source, text, json.loads(blob.decode("utf-8")))
                for doc_id, source, text, blob in cur]

    def load_into_vector_store(self, vector_store):
        """Populate a rag.VectorStore from every saved document (uses each row's id as
        the doc_id)."""
        for doc_id, _source, _text, embedding in self.load_documents():
            vector_store.add(embedding, doc_id)

    # ---- checkpoints ----
    def save_checkpoint_blob(self, version, blob_bytes, config=None):
        """Store a checkpoint blob (e.g. json.dumps(checkpoint.serialize_brain(brain))
        .encode('utf-8')) under a name/version string."""
        self.conn.execute(
            "INSERT OR REPLACE INTO checkpoints (version, model_blob, config_json, created_at) "
            "VALUES (?, ?, ?, ?)",
            (version, blob_bytes, json.dumps(config) if config is not None else None, _now()))
        self.conn.commit()

    def load_checkpoint_blob(self, version):
        """Returns (blob_bytes, config_or_None). Raises KeyError if not found."""
        row = self.conn.execute(
            "SELECT model_blob, config_json FROM checkpoints WHERE version = ?", (version,)).fetchone()
        if row is None:
            raise KeyError(f"no checkpoint named {version!r}")
        blob, config_json = row
        return blob, (json.loads(config_json) if config_json else None)

    def list_checkpoints(self):
        """Returns [(version, created_at), ...] ordered oldest to newest."""
        return self.conn.execute(
            "SELECT version, created_at FROM checkpoints ORDER BY created_at").fetchall()

    # ---- metrics ----
    def log_metric(self, step, loss, acc=None):
        self.conn.execute(
            "INSERT INTO metrics (step, loss, acc, timestamp) VALUES (?, ?, ?, ?)",
            (step, loss, acc, _now()))
        self.conn.commit()

    def load_metrics(self):
        """Returns [(step, loss, acc, timestamp), ...] ordered by step."""
        return self.conn.execute(
            "SELECT step, loss, acc, timestamp FROM metrics ORDER BY step").fetchall()

    # ---- memory (generic key/value, e.g. training config, experiment notes) ----
    def set_memory(self, key, value, value_type="json"):
        payload = json.dumps(value) if value_type == "json" else str(value)
        self.conn.execute(
            "INSERT INTO memory (key, value, type, updated_at) VALUES (?, ?, ?, ?)",
            (key, payload, value_type, _now()))
        self.conn.commit()

    def get_memory(self, key, default=None):
        """Returns the most recently set value for `key`, or `default` if never set."""
        row = self.conn.execute(
            "SELECT value, type FROM memory WHERE key = ? ORDER BY updated_at DESC, id DESC LIMIT 1",
            (key,)).fetchone()
        if row is None:
            return default
        value, value_type = row
        return json.loads(value) if value_type == "json" else value
