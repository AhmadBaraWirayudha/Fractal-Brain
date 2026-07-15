from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class MemoryDocument:
    doc_id: str
    text: str
    embedding: list[float]
    metadata: dict[str, Any]
    success_count: int = 0
    created_at: float = 0.0
    updated_at: float = 0.0
    score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


@dataclass
class InteractionRecord:
    interaction_id: str
    input_text: str
    retrieved_doc_ids: list[str]
    plan: dict[str, Any]
    final_output: str
    success: Optional[bool]
    metadata: dict[str, Any]


class VectorMemoryStore:
    def __init__(self, sqlite_path: str | Path, embedder: Any, backend: str = 'auto', max_documents: int = 10000, base_path: str | Path | None = None) -> None:
        self.base_path = Path(base_path) if base_path is not None else Path.cwd()
        self.sqlite_path = self._resolve_path(sqlite_path)
        self.embedder = embedder
        self.backend = backend
        self.max_documents = max_documents
        self.conn: sqlite3.Connection | None = None
        self.documents: list[MemoryDocument] = []
        self.interactions: dict[str, InteractionRecord] = {}
        self._faiss_index = None

    @classmethod
    def from_config(cls, config: dict[str, Any], embedder: Any, base_path: str | Path | None = None) -> 'VectorMemoryStore':
        return cls(
            config['paths']['sqlite_db'],
            embedder,
            config.get('retrieval', {}).get('backend', 'auto'),
            int(config.get('retrieval', {}).get('max_documents', 10000)),
            base_path=base_path,
        )

    def initialize(self) -> None:
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.sqlite_path)
        self.conn.execute(
            '''CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                embedding TEXT NOT NULL,
                metadata TEXT NOT NULL,
                success_count INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )'''
        )
        self.conn.execute(
            '''CREATE TABLE IF NOT EXISTS interactions (
                interaction_id TEXT PRIMARY KEY,
                input_text TEXT NOT NULL,
                retrieved_doc_ids TEXT NOT NULL,
                plan TEXT NOT NULL,
                final_output TEXT NOT NULL,
                success INTEGER,
                metadata TEXT NOT NULL,
                corrected_output TEXT,
                notes TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )'''
        )
        self.conn.commit()
        self.load_documents()
        self._rebuild_optional_index()

    def _resolve_path(self, path: str | Path) -> Path:
        p = Path(path)
        if p.is_absolute():
            return p
        return (self.base_path / p).resolve()

    def load_bootstrap_records(self, dataset_path: Path) -> list[dict[str, Any]]:
        dataset_path = self._resolve_path(dataset_path)
        return [
            json.loads(line)
            for line in dataset_path.read_text(encoding='utf-8').splitlines()
            if line.strip()
        ]

    def bootstrap(self, records: list[dict[str, Any]]) -> None:
        for r in records:
            text = r.get('solution') or r['query']
            meta = dict(r.get('metadata', {}))
            meta.update(
                {
                    'problem_type': r.get('problem_type'),
                    'query': r.get('query'),
                    'actions': r.get('actions', []),
                    'step_texts': r.get('step_texts', []),
                    'bootstrap': True,
                }
            )
            # Stable, content-derived id (rather than a fresh uuid) so that
            # bootstrapping the same dataset on every engine.initialize()
            # is idempotent instead of inserting a new duplicate row each
            # time. See CHANGELOG.
            doc_id = 'bootstrap:' + _stable_bootstrap_id(r.get('query', ''), text)
            self.add_document(text, self.embedder.embed_text(text), meta, doc_id=doc_id)

    def load_documents(self) -> list[MemoryDocument]:
        if self.conn is None:
            return []
        rows = self.conn.execute(
            'SELECT doc_id, text, embedding, metadata, success_count, created_at, updated_at FROM documents'
        ).fetchall()
        self.documents = [
            MemoryDocument(
                row[0],
                row[1],
                json.loads(row[2]),
                json.loads(row[3]),
                int(row[4]),
                float(row[5]),
                float(row[6]),
            )
            for row in rows
        ][: self.max_documents]
        return self.documents

    def add_document(self, text: str, embedding: list[float], metadata: dict[str, Any], doc_id: str | None = None) -> str:
        if self.conn is None:
            raise RuntimeError('Memory store not initialized')
        doc_id = doc_id or str(uuid.uuid4())
        now = time.time()
        emb = [float(x) for x in embedding]
        cur = self.conn.execute(
            'INSERT OR IGNORE INTO documents VALUES (?, ?, ?, ?, ?, ?, ?)',
            (doc_id, text, json.dumps(emb), json.dumps(metadata), 0, now, now),
        )
        self.conn.commit()
        if cur.rowcount == 0:
            # A document with this id was already present (e.g. bootstrap
            # running again on an already-loaded dataset). Skip adding it to
            # the in-memory index too, so we don't accumulate duplicates
            # there either. See CHANGELOG.
            return doc_id
        self.documents.append(MemoryDocument(doc_id, text, emb, metadata, 0, now, now))
        self.documents = self.documents[-self.max_documents :]
        self._rebuild_optional_index()
        return doc_id

    def retrieve(self, query_embedding: list[float], top_k: int = 3) -> list[MemoryDocument]:
        if not self.documents:
            return []
        scores = [cosine_similarity(d.embedding, query_embedding) for d in self.documents]
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        out: list[MemoryDocument] = []
        for idx in order:
            d = self.documents[int(idx)]
            out.append(
                MemoryDocument(
                    d.doc_id,
                    d.text,
                    d.embedding,
                    d.metadata,
                    d.success_count,
                    d.created_at,
                    d.updated_at,
                    float(scores[int(idx)]),
                )
            )
        return out

    def store_pending_interaction(self, record: InteractionRecord) -> None:
        if self.conn is None:
            raise RuntimeError('Memory store not initialized')
        now = time.time()
        self.conn.execute(
            'INSERT OR REPLACE INTO interactions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, COALESCE((SELECT created_at FROM interactions WHERE interaction_id=?), ?), ?)',
            (
                record.interaction_id,
                record.input_text,
                json.dumps(record.retrieved_doc_ids),
                json.dumps(record.plan),
                record.final_output,
                None if record.success is None else int(record.success),
                json.dumps(record.metadata),
                None,
                None,
                record.interaction_id,
                now,
                now,
            ),
        )
        self.conn.commit()
        self.interactions[record.interaction_id] = record

    def finalize_interaction(self, interaction_id: str, success: bool, corrected_output: str | None = None, notes: str | None = None) -> dict[str, Any]:
        if self.conn is None:
            raise RuntimeError('Memory store not initialized')
        now = time.time()
        self.conn.execute(
            'UPDATE interactions SET success=?, corrected_output=?, notes=?, updated_at=? WHERE interaction_id=?',
            (int(success), corrected_output, notes, now, interaction_id),
        )
        self.conn.commit()
        return {'interaction_id': interaction_id, 'success': success, 'corrected_output': corrected_output, 'notes': notes}

    def _rebuild_optional_index(self) -> None:
        self._faiss_index = None


def _stable_bootstrap_id(query: str, text: str) -> str:
    digest = hashlib.sha256(f'{query}\x1f{text}'.encode('utf-8')).hexdigest()
    return digest[:24]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    length = min(len(a), len(b))
    dot = sum(float(a[i]) * float(b[i]) for i in range(length))
    na = math.sqrt(sum(float(x) * float(x) for x in a[:length])) or 1.0
    nb = math.sqrt(sum(float(x) * float(x) for x in b[:length])) or 1.0
    return dot / (na * nb)
