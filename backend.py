"""Layered memory backend — SQLite + Qdrant + layer pipeline.

This is the engine. The MemoryProvider in __init__.py is a thin wrapper.

Public API:
    - retrieve(query, max_layer, scope, data_type, data_id, session_name, limit) → list of result dicts
    - peek(query, layer) → list of result dicts at specific layer
    - add(content, summary, topic, keywords, scope, data_type, data_id, session_name, sensitivity, ttl, priority) → uuid
    - update(uuid, **fields) → None
    - delete(uuid) → None
    - rebuild(since=None) → dict with stats
    - sync_check() → dict with counts
    - list(topic, scope, limit, sort) → list of dicts
    - count() → int
    - close() → None
"""

from __future__ import annotations

import json
import logging
import os
import pwd
import sqlite3
import time
import uuid as uuid_mod
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger("hermes-layered-memory")

# File-based logging — writes to profile logs dir
_log_initialized = False
def _setup_logger():
    global _log_initialized
    if _log_initialized:
        return
    # Also check if logger already has handlers (prevents duplicates from multiple imports)
    if logger.handlers:
        _log_initialized = True
        return
    _log_initialized = True

    # Resolve log path — env var overrides default
    log_path = os.environ.get("HERMES_LAYERED_LOG_FILE")
    if log_path:
        log_file = os.path.expanduser(log_path)
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    else:
        # Default: hermes home / logs
        try:
            from hermes_constants import get_hermes_home
            log_dir = get_hermes_home() / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = str(log_dir / "hermes-layered-memory.log")
        except Exception:
            log_file = "/tmp/hermes-layered-memory.log"

    # File handler — append mode, rotates naturally
    fh = logging.FileHandler(str(log_file))
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s",
                                       datefmt="%H:%M:%S"))

    # Console handler — also keep stderr for debugging
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("[h-l-m @ %(asctime)s] %(levelname)s %(message)s",
                                       datefmt="%H:%M:%S"))

    # Set level from env or default to INFO
    _env_level = os.environ.get("HERMES_LAYERED_LOG", "INFO").upper()
    level = getattr(logging, _env_level, logging.INFO)

    logger.setLevel(level)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.propagate = False

# Auto-setup on first use
_setup_logger()

# UUID format conversion (SQLite stores 32-char hex, Qdrant wants standard UUID)
def _to_qdrant_id(uuid_hex: str) -> str:
    """Convert 32-char hex UUID to standard format with dashes."""
    return f"{uuid_hex[:8]}-{uuid_hex[8:12]}-{uuid_hex[12:16]}-{uuid_hex[16:20]}-{uuid_hex[20:]}"

def _from_qdrant_id(qid) -> str:
    """Convert Qdrant ID (str or int) back to 32-char hex UUID."""
    if isinstance(qid, int):
        return format(qid, '032x')[:32]
    return str(qid).replace('-', '')

# Embedding model — lazy import
_embedding_fn = None

def _get_embedding_fn():
    """Get embedding function. Uses remote endpoint if configured, else local sentence-transformers."""
    global _embedding_fn
    if _embedding_fn is not None:
        return _embedding_fn
    import os
    embed_url = os.environ.get("HERMES_LAYERED_EMBED_URL")
    embed_model = os.environ.get("HERMES_LAYERED_EMBED_MODEL")
    if embed_url and embed_model:
        # Remote embedding endpoint (Ollama, llama.cpp, etc.)
        import json as _json
        import urllib.request
        def remote_embed(texts):
            payload = _json.dumps({"model": embed_model, "input": texts}).encode()
            req = urllib.request.Request(
                embed_url,
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = _json.loads(resp.read())
                return result.get("embeddings", [])
        _embedding_fn = remote_embed
        logger.info("Using remote embedding: %s with %s", embed_url, embed_model)
    else:
        # Local sentence-transformers
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        def _embedding_fn(texts):
            return model.encode(texts).tolist()
    return _embedding_fn


# ---------------------------------------------------------------------------
# LayeredBackend
# ---------------------------------------------------------------------------

class LayeredBackend:
    """SQLite store + Qdrant index + layer pipeline.

    Supports multiple Qdrant collections mapped by data_type via config:
    {
        "collections": {
            "SYSTEM": "memories",
            "USER-DATA": "memories",
            "ENV-DATA": "memories",
            "OBSIDIAN": "vault",
            "CODE": "code",
            "SESSION-DATA": "sessions"
        }
    }
    Default: all data_types map to the configured qdrant_collection.
    """

    def __init__(self, db_path: str, qdrant_url: str = "http://localhost:6333",
                 qdrant_collection: str = "memories", embedding_model: str = "all-MiniLM-L6-v2",
                 layer0_top_k: int = 30, config: dict = None, profile_name: str = None):
        # Expand ~ in db_path
        self._db_path = os.path.expandvars(os.path.expanduser(db_path))
        self._qdrant_url = qdrant_url
        self._default_collection = qdrant_collection
        self._embedding_model = embedding_model
        self._layer0_top_k = layer0_top_k
        self._profile_name = profile_name
        self._conn = None
        self._qdrant = None
        self._config = config or {}
        # Env var overrides (higher priority than JSON config)
        self._apply_env_overrides()
        # Collection mapping: data_type -> qdrant_collection
        self._collection_map = self._config.get("collections", {})
        if not self._collection_map:
            # Default: all data_types use the configured collection
            self._collection_map = {
                "SYSTEM": qdrant_collection,
                "USER-DATA": qdrant_collection,
                "ENV-DATA": qdrant_collection,
                "SESSION-DATA": qdrant_collection,
                "CUSTOM": qdrant_collection,
            }
        # Circuit breaker state (YantrikDB pattern)
        self._qdrant_failures = 0
        self._qdrant_broken_until = None
        self._init_db()
        self._init_qdrant()

    def _apply_env_overrides(self):
        """Apply env var overrides to config (higher priority than JSON)."""
        import os
        import pwd
        if os.environ.get("HERMES_LAYERED_DB_PATH"):
            # Hermes sets HOME to profile dir, breaking ~ expansion.
            # Use the actual user home directory from passwd.
            real_home = pwd.getpwuid(os.getuid()).pw_dir
            dbp = os.environ["HERMES_LAYERED_DB_PATH"]
            if dbp.startswith("~"):
                dbp = dbp.replace("~", real_home, 1)
            elif "$" in dbp:
                dbp = os.path.expandvars(dbp)
            self._db_path = dbp
        if os.environ.get("HERMES_LAYERED_QDRANT_URL"):
            self._qdrant_url = os.environ["HERMES_LAYERED_QDRANT_URL"]
        if os.environ.get("HERMES_LAYERED_MAX_LAYER"):
            self._config["max_layer"] = int(os.environ["HERMES_LAYERED_MAX_LAYER"])
        if os.environ.get("HERMES_LAYERED_LAYER3_MODE"):
            self._config["layer3_mode"] = os.environ["HERMES_LAYERED_LAYER3_MODE"]
        if os.environ.get("HERMES_LAYERED_LAYER3_MODEL"):
            self._config["layer3_model"] = os.environ["HERMES_LAYERED_LAYER3_MODEL"]
        if os.environ.get("HERMES_LAYERED_LAYER3_BASE_URL"):
            if "layer3_provider_config" not in self._config:
                self._config["layer3_provider_config"] = {}
            self._config["layer3_provider_config"]["base_url"] = os.environ["HERMES_LAYERED_LAYER3_BASE_URL"]
        if os.environ.get("HERMES_LAYERED_LAYER3_API_KEY"):
            if "layer3_provider_config" not in self._config:
                self._config["layer3_provider_config"] = {}
            self._config["layer3_provider_config"]["api_key"] = os.environ["HERMES_LAYERED_LAYER3_API_KEY"]
        if os.environ.get("HERMES_LAYERED_REASONING_EFFORT"):
            self._config["layer3_reasoning_effort"] = os.environ["HERMES_LAYERED_REASONING_EFFORT"]
        if os.environ.get("HERMES_LAYERED_ENRICH_LLM"):
            self._config["enrich_llm"] = os.environ["HERMES_LAYERED_ENRICH_LLM"].lower() in ("true", "1", "yes")
        if os.environ.get("HERMES_LAYERED_LOW_TRUST_ARCHIVE_DAYS"):
            self._config["low_trust_archive_days"] = int(os.environ["HERMES_LAYERED_LOW_TRUST_ARCHIVE_DAYS"])

        # Instance-level heuristic reason (not class-level — avoids shared mutable state)
        self._layer_suggest_reason = ""
        self._entity_patterns_cache = None

    def _load_entity_patterns(self) -> List[dict]:
        """Load entity extraction patterns from JSON files in entity-patterns/ folder.

        Loading order:
        1. entity_patterns_path config override (full replacement, single file)
        2. All .json files in entity-patterns/ folder (shipped or user-provided)

        Files are loaded alphabetically, merged by name (same name = override, new = add).
        default.json is a sample — not recommended to edit. Copy to custom*.json instead.

        If entity-patterns/ folder is missing or empty, auto-creates it with default.json.
        If user deleted default.json but has other files, works with user's files only.
        """
        if self._entity_patterns_cache is not None:
            return self._entity_patterns_cache

        import json as _json

        # Option 1: config override (full replacement, single file)
        patterns_path = self._config.get("entity_patterns_path")
        if patterns_path:
            patterns_path = os.path.expanduser(patterns_path)
            try:
                with open(patterns_path) as f:
                    data = _json.load(f)
                    self._entity_patterns_cache = data.get("patterns", [])
                    logger.debug("Loaded %d entity patterns from config override: %s",
                                len(self._entity_patterns_cache), patterns_path)
                    return self._entity_patterns_cache
            except Exception as e:
                logger.debug("Config entity patterns override failed (%s): %s",
                            patterns_path, e)

        # Option 2: all .json files in entity-patterns/ folder
        patterns_dir = os.path.join(os.path.dirname(__file__), "entity-patterns")
        patterns = self._load_all_patterns_dir(patterns_dir)

        # Auto-restore if missing/empty
        if not patterns:
            self._ensure_entity_patterns_dir(patterns_dir)
            patterns = self._load_all_patterns_dir(patterns_dir)

        self._entity_patterns_cache = patterns
        return self._entity_patterns_cache

    def _load_all_patterns_dir(self, dir_path: str) -> List[dict]:
        """Load all .json pattern files from directory, merge by name."""
        if not os.path.exists(dir_path):
            return []

        files = sorted(f for f in os.listdir(dir_path)
                      if f.endswith('.json') and not f.startswith('.'))
        if not files:
            return []

        all_patterns = []
        for fname in files:
            fpath = os.path.join(dir_path, fname)
            loaded = self._load_json_patterns(fpath, fname)
            self._merge_patterns(all_patterns, loaded)

        logger.debug("Loaded %d entity patterns from %d files in %s",
                     len(all_patterns), len(files), dir_path)
        return all_patterns

    def _ensure_entity_patterns_dir(self, dir_path: str):
        """Create entity-patterns directory with default.json if missing."""
        import json as _json
        default_content = {
            "description": "DEFAULT — do not edit. Copy to custom*.json for your patterns.",
            "usage": "Create custom*.json files in this folder. All .json files are loaded at startup, merged by pattern name. Same name = override, new name = add.",
            "patterns": [
                {"name": "gpu_cpu_models", "regex": r"\b(RTX|GTX|Quadro|Threadripper|Ryzen|Core|i[3579]|Xeon|EPYC|Apple\s+[A-ZM])\s*[\d\-\.A-Z]\w*\b", "flags": "IGNORECASE", "enabled": True},
                {"name": "software_versions", "regex": r"\b(\w+)\s+(\d+\.\d+(?:\.\d+)?)\b", "enabled": True, "filter": "exclude_stopwords", "stopwords": ["the", "and", "for", "with", "from", "in", "on"]},
                {"name": "llm_models", "regex": r"\b(qwen|llama|gpt|mistral|phi|vicuna|dolphin|llava|mixtral|falcon|bloom)\s*[-\.]?[\d\-b.]+\b", "flags": "IGNORECASE", "enabled": True},
                {"name": "service_hostnames", "regex": r"\b[a-z][\w.-]*\.\w+(?:\.\w+)?(?::\d+)?\b", "enabled": True, "filter": "min_length", "min_length": 5, "exclude_tlds": [".com", ".net", ".org", ".io", ".dev"]},
            ]
        }
        try:
            os.makedirs(dir_path, exist_ok=True)
            default_path = os.path.join(dir_path, "default.json")
            with open(default_path, 'w') as f:
                _json.dump(default_content, f, indent=4)
                f.write('\n')
            logger.debug("Restored entity-patterns/default.json (folder was missing/empty)")
        except Exception as e:
            logger.debug("Failed to restore entity-patterns: %s", e)

    def _load_json_patterns(self, path: str, label: str) -> List[dict]:
        """Load patterns from a JSON file."""
        import json as _json
        try:
            with open(path) as f:
                data = _json.load(f)
                result = data.get("patterns", [])
                logger.debug("Loaded %d entity patterns from %s (%s)",
                            len(result), path, label)
                return result
        except Exception as e:
            logger.debug("Failed to load entity patterns %s from %s: %s",
                        label, path, e)
            return []

    def _merge_patterns(self, base: List[dict], overlay: List[dict]):
        """Merge overlay patterns into base, overriding by name."""
        overlay_by_name = {p.get("name"): p for p in overlay}
        for i, p in enumerate(base):
            name = p.get("name")
            if name and name in overlay_by_name:
                base[i] = overlay_by_name[name]
        # Add new patterns from overlay
        existing_names = {p.get("name") for p in base}
        for p in overlay:
            if p.get("name") not in existing_names:
                base.append(p)

    def _get_collection(self, data_type: str) -> str:
        """Get Qdrant collection for a data_type."""
        return self._collection_map.get(data_type, self._default_collection)

    def _record_qdrant_failure(self, e):
        """Record a Qdrant failure with circuit breaker after 5 consecutive."""
        self._qdrant_failures += 1
        if self._qdrant_failures >= 5:
            self._qdrant_broken_until = time.time() + 120
            logger.warning("Qdrant circuit OPEN after %d failures, 120s cooldown",
                           self._qdrant_failures)
        else:
            logger.debug("Qdrant upsert failed (%d/5): %s",
                         self._qdrant_failures, e)

    # ---- Database init --------------------------------------------------

    def _init_db(self):
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                uuid              TEXT PRIMARY KEY,
                content           TEXT NOT NULL,
                summary           TEXT,
                keywords          TEXT,
                topic             TEXT,
                scope             TEXT DEFAULT 'personal',
                data_type         TEXT DEFAULT 'CUSTOM',
                data_id           TEXT,
                session_name      TEXT,
                sensitivity       INTEGER DEFAULT 0,
                source            TEXT DEFAULT 'agent',
                source_url        TEXT,
                created_at        TEXT NOT NULL,
                updated_at        TEXT NOT NULL,
                ttl               TEXT,
                status            TEXT DEFAULT 'active',
                trust_score       REAL DEFAULT 0.5,
                reference_count   INTEGER DEFAULT 0,
                backlinks         TEXT DEFAULT '[]',
                layer3_flags      TEXT DEFAULT '{}',
                metadata          TEXT DEFAULT '{}',
                priority          INTEGER DEFAULT 0,
                sequence          INTEGER,
                embedding         BLOB,
                llm_review_status TEXT,          -- NULL=unreviewed, 'keep', 'delete', 'pending'
                llm_reviewed_at   TEXT,          -- ISO timestamp of last LLM review
                compacted_into    TEXT,          -- UUID of merged record (on originals)
                first_observed_at TEXT           -- oldest created_at from merged batch
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                content, summary, keywords, topic, data_id,
                content='memories', content_rowid='rowid'
            );
            CREATE INDEX IF NOT EXISTS idx_mem_scope ON memories(scope);
            CREATE INDEX IF NOT EXISTS idx_mem_data_type ON memories(data_type);
            CREATE INDEX IF NOT EXISTS idx_mem_data_id ON memories(data_id);
            CREATE INDEX IF NOT EXISTS idx_mem_session_name ON memories(session_name);
            CREATE INDEX IF NOT EXISTS idx_mem_sensitivity ON memories(sensitivity);
            CREATE INDEX IF NOT EXISTS idx_mem_ttl ON memories(ttl);
            CREATE INDEX IF NOT EXISTS idx_mem_status ON memories(status);
            CREATE INDEX IF NOT EXISTS idx_mem_created ON memories(created_at);
            CREATE INDEX IF NOT EXISTS idx_mem_topic ON memories(topic);
            CREATE INDEX IF NOT EXISTS idx_mem_priority ON memories(priority);

            -- FTS5 sync triggers (automatic, replaces manual sync)
            CREATE TRIGGER IF NOT EXISTS fts5_ai AFTER INSERT ON memories BEGIN
                INSERT INTO memories_fts(rowid, content, summary, keywords, topic, data_id)
                VALUES (new.rowid, new.content, new.summary, new.keywords, new.topic, new.data_id);
            END;
            CREATE TRIGGER IF NOT EXISTS fts5_au AFTER UPDATE ON memories BEGIN
                UPDATE memories_fts SET
                    content = new.content,
                    summary = new.summary,
                    keywords = new.keywords,
                    topic = new.topic,
                    data_id = new.data_id
                WHERE rowid = new.rowid;
            END;
            CREATE TRIGGER IF NOT EXISTS fts5_ad AFTER DELETE ON memories BEGIN
                DELETE FROM memories_fts WHERE rowid = old.rowid;
            END;
        """)
        # Migration: add llm_review columns if missing (v7)
        try:
            self._conn.execute("ALTER TABLE memories ADD COLUMN llm_review_status TEXT")
            self._conn.execute("ALTER TABLE memories ADD COLUMN llm_reviewed_at TEXT")
            self._conn.commit()
            logger.info("Migration: added llm_review_status and llm_reviewed_at columns")
        except Exception:
            pass  # Columns already exist
        # Migration: add compaction columns if missing (v8)
        try:
            self._conn.execute("ALTER TABLE memories ADD COLUMN compacted_into TEXT")
            self._conn.execute("ALTER TABLE memories ADD COLUMN first_observed_at TEXT")
            self._conn.commit()
            logger.info("Migration: added compacted_into and first_observed_at columns")
        except Exception:
            pass  # Columns already exist
        self._conn.commit()

    def _init_qdrant(self):
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
            self._qdrant = QdrantClient(url=self._qdrant_url)
            # Create all configured collections
            collections = set(self._collection_map.values())
            # Detect embedding dimension from configured model
            fn = _get_embedding_fn()
            sample = fn(["dimension detection"])[0]
            dim = len(sample)
            logger.info("Embedding dimension: %d", dim)
            for coll in collections:
                try:
                    self._qdrant.get_collection(coll)
                except Exception:
                    self._qdrant.create_collection(
                        collection_name=coll,
                        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                    )
                    # Create keyword indexes for efficient filtering
                    from qdrant_client.models import PayloadSchemaType
                    for field in ("data_type", "data_id", "session_name", "profile_name"):
                        self._qdrant.create_payload_index(coll, field_name=field,
                            field_schema=PayloadSchemaType.KEYWORD)
                    logger.info("Created Qdrant collection: %s (dim=%d, indexes: data_type, data_id, session_name, profile_name)", coll, dim)
        except ImportError:
            logger.warning("qdrant_client not installed — vector search disabled")
        except Exception as e:
            logger.warning("Qdrant connection failed: %s — vector search disabled", e)

    # ---- CRUD ------------------------------------------------------------

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _check_duplicate(self, embedding: List[float], data_type: str, threshold: float):
        """Check for semantically similar existing memories.

        Returns None if no duplicate found, or dict with uuid/similarity/note if duplicate detected.
        """
        if not self._qdrant:
            return None

        try:
            coll = self._get_collection(data_type or "CUSTOM")
            # Search for similar vectors in Qdrant
            from qdrant_client.models import Filter, FieldCondition, MatchValue as MatchValueQ
            scroll_filter = Filter(must=[])
            if self._profile_name:
                scroll_filter.must.append(
                    FieldCondition(key="profile_name", match=MatchValueQ(value=self._profile_name))
                )
            # Note: status filter removed — status is not stored in Qdrant payload

            resp = self._qdrant.query_points(
                collection_name=coll,
                query=embedding,
                query_filter=scroll_filter,
                limit=3,  # Check top 3 most similar
            )

            for hit in resp.points:
                # query_points returns score directly (cosine similarity: 1.0=identical, 0=orthogonal)
                similarity = hit.score if hasattr(hit, 'score') else 1.0 - hit.distance
                if similarity >= threshold:
                    uuid = _from_qdrant_id(hit.id)
                    # Get content from SQLite for the note
                    row = self._conn.execute(
                        "SELECT content, topic FROM memories WHERE uuid = ?", (uuid,)
                    ).fetchone()
                    note = f"Similar to: {row[1] if row and row[1] else row[0][:50] if row else 'unknown'}" if row else None
                    return {"uuid": uuid, "similarity": similarity, "note": note}

            return None
        except Exception as e:
            logger.debug("Dedup check failed: %s", e)
            return None

    def add(self, content: str, summary: str = None, topic: str = None,
            keywords: List[str] = None, scope: str = "personal",
            data_type: str = "CUSTOM", data_id: str = None,
            session_name: str = None,
            sensitivity: int = 0, ttl: str = None, priority: int = 0,
            source: str = "agent", source_url: str = None,
            metadata: dict = None) -> Union[str, dict]:
        # Embed FIRST (cheap), then dedup, then enrich (LLM call — expensive)
        # This saves tokens when adding near-duplicates
        fn = _get_embedding_fn()
        embedding = fn([content])[0]
        # Guard against None embedding (endpoint failure, empty content)
        if embedding is None:
            logger.warning("add: embedding returned None for content=%r", content[:60])
            embedding_blob = "null"
        else:
            embedding_blob = json.dumps(embedding)

        # Semantic dedup: check for existing similar memories (before enrichment)
        # Skip dedup if embedding is None (can't compare without vector)
        if embedding is not None:
            threshold = self._config.get("dedup_threshold", 0.97)
            if threshold > 0:
                existing = self._check_duplicate(embedding, data_type, threshold)
                if existing:
                    logger.debug("Dedup: duplicate detected (similarity=%.3f), returning existing %s",
                                 existing["similarity"], existing["uuid"])
                    return {
                        "uuid": existing["uuid"],
                        "status": "duplicate",
                        "similarity": existing["similarity"],
                        "note": "Existing memory with high semantic similarity. Use layered_update to merge content."
                    }

        # Hybrid enrichment: fill missing metadata fields (only after dedup passes)
        enriched = self._enrich_metadata(content, summary, data_type, data_id, topic, keywords)
        data_type = enriched.get("data_type", data_type) or "CUSTOM"
        data_id = enriched.get("data_id", data_id)
        topic = enriched.get("topic", topic)
        keywords = enriched.get("keywords", keywords) or []

        now = self._now()
        uuid = uuid_mod.uuid4().hex
        kw_json = json.dumps(keywords)
        metadata_json = json.dumps(metadata) if metadata else "null"

        self._conn.execute("""
            INSERT INTO memories (uuid, content, summary, keywords, topic, scope,
                data_type, data_id, session_name, sensitivity, source, source_url,
                created_at, updated_at, ttl, status, trust_score, reference_count,
                backlinks, layer3_flags, metadata, priority, sequence, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, 0, '[]',
                    '{}', ?, ?, NULL, ?)
        """, (uuid, content, summary or content[:100], kw_json, topic, scope,
              data_type or "CUSTOM", data_id, session_name, sensitivity, source, source_url,
              now, now, ttl, 0.5, metadata_json, priority, embedding_blob))

        # FTS5 sync handled by SQLite trigger (fts5_ai) — removed manual sync

        # Qdrant upsert (best effort) — route to correct collection by data_type
        try:
            from qdrant_client.models import PointStruct
            payload = {"data_type": data_type or "CUSTOM"}
            if data_id:
                payload["data_id"] = data_id
            if session_name:
                payload["session_name"] = session_name
            if self._profile_name:
                payload["profile_name"] = self._profile_name
            if embedding is not None:
                self._qdrant.upsert(
                    collection_name=self._get_collection(data_type or "CUSTOM"),
                    points=[PointStruct(id=_to_qdrant_id(uuid), vector=embedding, payload=payload)],
                )
            self._qdrant_failures = 0
        except Exception as e:
            self._record_qdrant_failure(e)
            logger.debug("Qdrant upsert failed: %s", e)

        self._conn.commit()
        logger.info("add: uuid=%s data_type=%s data_id=%s summary=%r",
                     uuid[:8], data_type, data_id, (summary or content[:60]))
        return uuid

    def update(self, uuid: str, **fields):
        if not fields:
            return
        # Ensure WAL checkpoint before writes to avoid corruption
        try:
            self._conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
        except Exception:
            pass
        allowed = {"content", "summary", "priority", "status", "trust_score",
                    "ttl", "sensitivity", "topic", "data_type", "data_id", "session_name", "keywords"}
        clean = {k: v for k, v in fields.items() if k in allowed and v is not None}
        if not clean:
            return

        sets = ", ".join(f"{k} = ?" for k in clean)
        values = []
        for k, v in clean.items():
            if isinstance(v, (list, dict)):
                values.append(json.dumps(v))
            else:
                values.append(v)
        values += [self._now(), uuid]
        self._conn.execute(f"UPDATE memories SET {sets}, updated_at = ? WHERE uuid = ?", values)

        # Re-embed if content changed
        if "content" in clean:
            fn = _get_embedding_fn()
            new_emb = fn([clean["content"]])[0]
            self._conn.execute("UPDATE memories SET embedding = ? WHERE uuid = ?",
                               (json.dumps(new_emb) if new_emb else "null", uuid))
            if new_emb is not None:
                try:
                    from qdrant_client.models import PointStruct
                    # Fetch data_type + data_id for payload
                    dt_row = self._conn.execute("SELECT data_type, data_id, session_name FROM memories WHERE uuid = ?", (uuid,)).fetchone()
                    dt = (dt_row[0] or "CUSTOM") if dt_row else "CUSTOM"
                    di = dt_row[1] if dt_row else None
                    sn = dt_row[2] if dt_row else None
                    payload = {"data_type": dt}
                    if di:
                        payload["data_id"] = di
                    if sn:
                        payload["session_name"] = sn
                    if self._profile_name:
                        payload["profile_name"] = self._profile_name
                    self._qdrant.upsert(
                        collection_name=self._get_collection(dt),
                        points=[PointStruct(id=_to_qdrant_id(uuid), vector=new_emb, payload=payload)],
                    )
                except Exception:
                    pass

        # FTS5 sync handled by SQLite triggers — removed manual sync (was redundant and buggy)
        self._conn.commit()

    def delete(self, uuid: str):
        # Get data_type to find the right collection
        row = self._conn.execute("SELECT data_type FROM memories WHERE uuid = ?", (uuid,)).fetchone()
        dt = (row[0] or "CUSTOM") if row else "CUSTOM"
        self._conn.execute("UPDATE memories SET status = 'deleted', updated_at = ? WHERE uuid = ?",
                           (self._now(), uuid))
        try:
            self._qdrant.delete(
                collection_name=self._get_collection(dt),
                points_selector=[_to_qdrant_id(uuid)],
            )
        except Exception:
            pass
        self._conn.commit()

    def _get_record(self, uuid: str) -> Optional[Dict[str, Any]]:
        cursor = self._conn.execute(
            "SELECT * FROM memories WHERE uuid = ? AND status = 'active'", (uuid,)
        )
        row = cursor.fetchone()
        if not row:
            return None
        cols = [d[0] for d in cursor.description]
        record = dict(zip(cols, row))
        # Parse JSON fields
        for jf in ("keywords", "backlinks", "layer3_flags", "metadata"):
            try:
                record[jf] = json.loads(record.get(jf, "null")) or {}
            except (json.JSONDecodeError, TypeError):
                record[jf] = {}
        # Strip embedding — lives in Qdrant, not in tool responses
        record.pop("embedding", None)
        return record

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM memories WHERE status = 'active'").fetchone()[0]

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    def backup(self, dest_dir: Optional[str] = None) -> dict:
        """Backup memories database to a copy file.

        Uses SQLite's atomic backup API (WAL-safe).

        Args:
            dest_dir: Directory for the backup file. Default: beside the original DB.

        Returns:
            dict with backup_path, timestamp, size_bytes, record_count.
        """
        from datetime import datetime, timezone

        if dest_dir is None:
            dest_dir = os.path.join(os.path.dirname(self._db_path), "backup")
        os.makedirs(dest_dir, exist_ok=True)

        now = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        base = os.path.basename(self._db_path)
        name, ext = os.path.splitext(base)
        backup_name = f"{name}.backup.{now}{ext}"
        backup_path = os.path.join(dest_dir, backup_name)

        # Atomic backup via SQLite API (WAL-safe)
        backup_conn = sqlite3.connect(backup_path)
        self._conn.backup(backup_conn)
        backup_conn.close()

        size = os.path.getsize(backup_path)
        record_count = self._conn.execute("SELECT count(*) FROM memories").fetchone()[0]
        logger.info("backup: %s (%d bytes, %d records)", backup_path, size, record_count)
        return {
            "backup_path": backup_path,
            "timestamp": now,
            "size_bytes": size,
            "record_count": record_count,
        }

    # ---- Layer Pipeline --------------------------------------------------

    def retrieve(self, query: str, max_layer: int = 2,
                 scope: str = None, data_type: str = None,
                 data_id: str = None, session_name: str = None,
                 profile_name: str = None, cross_profile: bool = False,
                 limit: int = 5, rerank: bool = False) -> List[Dict[str, Any]]:
        # Explicit rerank flag: override max_layer to at least 3
        if rerank and max_layer < 3:
            max_layer = 3
        logger.info("retrieve: query=%r max_layer=%d scope=%s data_type=%s data_id=%s session_name=%s profile=%s cross=%s limit=%d rerank=%s",
                     query[:50], max_layer, scope, data_type, data_id, session_name, profile_name, cross_profile, limit, rerank)
        # Determine effective profile filter
        effective_profile = profile_name if profile_name else (None if cross_profile else self._profile_name)
        result = self._run_pipeline(query, max_layer, scope, data_type, data_id,
                                    session_name, effective_profile, cross_profile, limit, rerank)
        logger.info("retrieve: returned %d results", len(result))
        # Track retrieval usage: increment reference_count for returned records
        if result:
            try:
                uuids = [r["uuid"] for r in result]
                placeholders = ",".join(["?"] * len(uuids))
                self._conn.execute(
                    f"UPDATE memories SET reference_count = reference_count + 1 "
                    f"WHERE uuid IN ({placeholders})",
                    uuids,
                )
                self._conn.commit()
            except Exception:
                pass
        # Checkpoint WAL to ensure clean state for subsequent writes
        try:
            self._conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
        except Exception:
            pass
        return result

    def peek(self, query: str, layer: int) -> List[Dict[str, Any]]:
        return self._run_pipeline(query, layer, limit=30)

    def _run_pipeline(self, query: str, max_layer: int,
                      scope: str = None, data_type: str = None,
                      data_id: str = None, session_name: str = None,
                      profile_name: str = None, cross_profile: bool = False,
                      limit: int = 5, rerank: bool = False) -> List[Dict[str, Any]]:
        # Heuristic gate: suggest max_layer based on query analysis
        # Skip when rerank=True (user explicitly requested LLM reranking)
        # Also skip when max_layer=4 — gap detection is an explicit request, not heuristic
        if max_layer > 2 and max_layer < 4 and not rerank:
            suggested = self._suggest_max_layer(query)
            if suggested < max_layer:
                logger.debug("heuristic gate: max_layer=%d → %d (%s)", max_layer, suggested, self._layer_suggest_reason)
                max_layer = suggested

        # Layer 0: Qdrant vector search → [uuid, distance]
        candidates = self._layer0(query, data_type=data_type, data_id=data_id, profile_name=profile_name)
        logger.debug("layer0: %d candidates", len(candidates))
        # Detect and clean orphaned Qdrant records
        candidates = self._detect_orphans(candidates)
        logger.debug("layer0: %d candidates after orphan check", len(candidates))

        # If Qdrant returns too few candidates, supplement with FTS5 results
        if len(candidates) < limit * 2:
            fts_uuids = self._fts5_fallback(query, scope, data_type, data_id, session_name, limit * 3)
            existing = {c[0] for c in candidates}
            for uid in fts_uuids:
                if uid not in existing:
                    candidates.append((uid, 0.5))  # mid-range distance for FTS5 matches
                    existing.add(uid)
            logger.debug("layer0: supplemented to %d candidates with FTS5", len(candidates))

        if not candidates:
            return []

        # Layer 1: SQLite join + structured filters
        records = self._layer1(candidates, query, scope=scope, data_id=data_id,
                               session_name=session_name, profile_name=profile_name,
                               cross_profile=cross_profile)
        logger.debug("layer1: %d records after filters", len(records))
        if not records or max_layer == 1:
            return records[:limit]

        # Layer 2: Fusion scoring
        scored = self._layer2(records, query)
        logger.debug("layer2: %d scored", len(scored))
        if max_layer == 2:
            return scored[:limit]

        # Layer 3: LLM reranker (skip if only gap detection needed)
        if max_layer == 4:
            # Layer 4 only needs summaries — skip expensive reranking
            result = self._layer4(scored, query)[:limit]
            logger.debug("layer4: %d final (skipped rerank)", len(result))
            return result

        reranked = self._layer3(scored, query, limit=limit)
        logger.debug("layer3: %d reranked", len(reranked))
        if max_layer == 3:
            return reranked

        # Layer 4: Gap detection (after reranking)
        result = self._layer4(reranked, query)[:limit]
        logger.debug("layer4: %d final", len(result))
        return result

    def _suggest_max_layer(self, query: str) -> int:
        """Heuristic gate: suggest max_layer based on query analysis."""
        import re
        q = query.lower()

        # Explicit gap detection request → layer 4 (check first)
        if re.search(r'\b(missing|gap|what else|what are we missing|not covered)\b', q):
            self._layer_suggest_reason = "explicit gap request"
            return 4

        # Stop words that aren't concepts
        stop = {"the", "a", "an", "is", "it", "to", "for", "of", "in", "on",
                "and", "or", "not", "with", "are", "was", "be", "that", "at",
                "this", "by", "from", "has", "had", "but", "what", "how", "do",
                "did", "can", "will", "should", "may", "could", "would"}

        # Count distinct meaningful words (not stop words)
        words = re.findall(r'\b[a-z]{3,}\b', q)
        concepts = [w for w in words if w not in stop]

        # Single-concept query → layer 2 is enough
        if len(concepts) <= 2:
            self._layer_suggest_reason = "single-concept query"
            return 2

        # Temporal expressions → layer 2 (temporal filter handles it)
        if re.search(r'\b(last|this|recent|next)\s+(week|month|year|quarter|spring|summer|fall|winter)\b', q):
            self._layer_suggest_reason = "temporal query"
            return 2

        # Multi-concept query → layer 3
        if len(concepts) > 2 or re.search(r'\b(vs|versus|compare)\b', q):
            self._layer_suggest_reason = "multi-concept query"
            return 3

        # Default: layer 2
        self._layer_suggest_reason = "default"
        return 2

    def _parse_frontmatter(self, file_path: str) -> tuple:
        """Parse YAML frontmatter and return (content, frontmatter_dict)."""
        with open(file_path, 'r') as f:
            text = f.read()

        if text.startswith('---'):
            parts = text.split('---', 2)
            if len(parts) >= 3:
                try:
                    import yaml
                    frontmatter = yaml.safe_load(parts[1]) or {}
                    content = parts[2].strip()
                    return content, frontmatter
                except Exception as e:
                    logger.debug("Failed to parse frontmatter %s: %s", file_path, e)

        return text, {}

    def _extract_wikilinks(self, content: str) -> List[str]:
        """Extract [[wikilinks]] from content."""
        import re
        return re.findall(r'\[\[([^\]]+)\]\]', content)

    def ingest_obsidian(self, vault_path: str, exclude: List[str] = None) -> dict:
        """Ingest Obsidian vault notes into HLM.

        Parses frontmatter, extracts tags and wikilinks, embeds all notes.
        """
        exclude = exclude or [".obsidian", "_resources", "Templates", "Excalidraw"]
        vault_path = os.path.abspath(vault_path)
        ingested = 0
        skipped = 0

        for root, dirs, files in os.walk(vault_path):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in exclude]

            for f in files:
                if not f.endswith('.md'):
                    continue

                file_path = os.path.join(root, f)
                relative = os.path.relpath(file_path, vault_path)

                try:
                    content, frontmatter = self._parse_frontmatter(file_path)
                    if not content.strip():
                        skipped += 1
                        continue

                    folder = os.path.dirname(relative).split('/')[0] if '/' in relative else ""
                    filename = os.path.splitext(f)[0]  # name without .md
                    tags = frontmatter.get('tags', [])
                    if isinstance(tags, str):
                        tags = [tags]
                    wikilinks = self._extract_wikilinks(content)

                    # data_type: CUSTOM (OBSIDIAN not in schema enum)
                    # data_id: folder name (partition: Concepts, Entities, etc.)
                    # topic: filename sans extension (specific subject)
                    self.add(
                        content=content,
                        summary=frontmatter.get('title', filename),
                        data_type="CUSTOM",
                        data_id=folder,
                        keywords=tags,
                        topic=filename,
                        source="obsidian",
                        source_url=relative,
                        metadata={"file_path": file_path, "tags": tags,
                                  "wikilinks": wikilinks, "frontmatter": frontmatter}
                    )
                    ingested += 1
                    logger.debug("Ingested: %s (data_id=%s, topic=%s)", relative, folder, filename)
                except Exception as e:
                    logger.debug("Failed to ingest %s: %s", relative, e)
                    skipped += 1

        logger.info("Obsidian ingestion complete: %d ingested, %d skipped", ingested, skipped)
        return {"ingested": ingested, "skipped": skipped}

    def _layer0(self, query: str, data_type: str = None,
                data_id: str = None, profile_name: str = None) -> List[tuple]:
        """Return [(uuid, distance), ...] from Qdrant ANN. Circuit breaker protected.

        Searches across all relevant collections and merges results.
        """
        # Circuit breaker: YantrikDB pattern (5 failures → 120s cooldown)
        if self._qdrant_broken_until and time.time() < self._qdrant_broken_until:
            remaining = int(self._qdrant_broken_until - time.time())
            logger.debug("Qdrant circuit open, %ds remaining — SQLite fallback", remaining)
            return self._fallback_uuids(data_type, data_id)
        if self._qdrant is None:
            return self._fallback_uuids(data_type, data_id)

        fn = _get_embedding_fn()
        query_vec = fn([query])[0]

        # Determine which collections to search
        if data_type:
            collections = [self._get_collection(data_type)]
        else:
            # Search all configured collections
            collections = list(set(self._collection_map.values()))

        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            all_results = []

            for coll in collections:
                conditions = []
                if data_type:
                    conditions.append(FieldCondition(key="data_type", match=MatchValue(value=data_type)))
                if profile_name:
                    conditions.append(FieldCondition(key="profile_name", match=MatchValue(value=profile_name)))
                search_filter = Filter(must=conditions) if conditions else None

                try:
                    results = self._qdrant.query_points(
                        collection_name=coll,
                        query=query_vec,
                        limit=self._layer0_top_k,
                        query_filter=search_filter,
                    )
                    # Convert Qdrant IDs back to SQLite hex UUIDs
                    for hit in results.points:
                        uuid_hex = _from_qdrant_id(hit.id)
                        all_results.append((uuid_hex, 1.0 - hit.score))
                except Exception as e:
                    logger.debug("Qdrant search failed for collection %s: %s", coll, e)
                    continue

            self._qdrant_failures = 0  # reset on success
            return all_results
        except Exception as e:
            self._qdrant_failures += 1
            if self._qdrant_failures >= 5:
                self._qdrant_broken_until = time.time() + 120
                logger.warning("Qdrant circuit OPEN after %d failures, 120s cooldown",
                               self._qdrant_failures)
            else:
                logger.debug("Qdrant search failed (%d/5): %s — SQLite fallback",
                             self._qdrant_failures, e)
            return self._fallback_uuids(data_type, data_id)

    def _fallback_uuids(self, data_type: str = None,
                        data_id: str = None) -> List[tuple]:
        """SQLite fallback when Qdrant is unavailable."""
        where = "SELECT uuid FROM memories WHERE status='active'"
        params = []
        if data_type:
            where += " AND data_type = ?"
            params.append(data_type)
        if data_id:
            where += " AND data_id = ?"
            params.append(data_id)
        where += " LIMIT ?"
        params.append(self._layer0_top_k)
        rows = self._conn.execute(where, params).fetchall()
        return [(r[0], 1.0) for r in rows]

    def _detect_orphans(self, candidates: List[tuple]) -> List[tuple]:
        """Detect and clean orphaned Qdrant records not in SQLite.

        Orphans: UUIDs in Qdrant but not in SQLite (status != 'active').
        Returns filtered candidates, removes orphans from Qdrant, logs warning.
        """
        if not candidates or not self._qdrant:
            return candidates

        # Get all UUIDs from candidates
        uuids = [c[0] for c in candidates]
        placeholders = ','.join(['?' for _ in uuids])

        # Check which UUIDs exist in SQLite as active
        rows = self._conn.execute(
            f"SELECT uuid FROM memories WHERE uuid IN ({placeholders}) AND status='active'",
            uuids
        ).fetchall()
        valid_uuids = {r[0] for r in rows}

        # Find orphans
        orphans = [c for c in candidates if c[0] not in valid_uuids]
        if orphans:
            logger.warning("orphan detection: %d orphaned records in Qdrant, cleaning", len(orphans))
            # Remove orphans from Qdrant
            try:
                from collections import defaultdict
                by_collection = defaultdict(list)
                for o in orphans:
                    # Try to determine collection (default to memories)
                    by_collection["memories"].append(_to_qdrant_id(o[0]))

                for coll, points in by_collection.items():
                    if self._qdrant.get_collection(coll):
                        self._qdrant.delete(collection_name=coll, points_selector=points)
                        logger.info("orphan cleanup: removed %d points from %s", len(points), coll)
            except Exception as e:
                logger.debug("orphan cleanup failed: %s", e)

            return [c for c in candidates if c[0] in valid_uuids]

        return candidates

    def _fts5_fallback(self, query: str, scope: str = None,
                       data_type: str = None, data_id: str = None,
                       session_name: str = None, limit: int = 30) -> List[str]:
        """Get UUIDs from FTS5 search for queries that vector search misses."""
        where = "status='active'"
        params = []
        if data_type:
            where += " AND data_type = ?"
            params.append(data_type)
        if data_id:
            where += " AND data_id = ?"
            params.append(data_id)
        if session_name:
            where += " AND session_name = ?"
            params.append(session_name)
        if scope:
            where += " AND scope = ?"
            params.append(scope)

        try:
            # Join FTS5 results with memories table
            # Use OR semantics for multi-word queries (FTS5 defaults to AND)
            import re
            tokens = re.findall(r'\w+', query.lower())
            if len(tokens) > 1:
                fts_query = " OR ".join(tokens)
            else:
                fts_query = query
            rows = self._conn.execute(f"""
                SELECT m.uuid FROM memories_fts f
                JOIN memories m ON m.rowid = f.rowid
                WHERE m.{where} AND memories_fts MATCH ?
                ORDER BY rank LIMIT ?
            """, params + [fts_query, limit]).fetchall()
            return [r[0] for r in rows]
        except Exception:
            return []

    # ---- Layer 1: SQLite join + filters ----------------------------------

    def _parse_temporal(self, query: str) -> Optional[tuple]:
        """Parse temporal expressions from query (Hindsight pattern).

        Returns (start_date, end_date) ISO strings or None.
        Handles: 'last week', 'this month', 'recent', 'last spring', etc.
        """
        import re
        q = query.lower()
        now = datetime.now(timezone.utc)

        # Simple patterns — good enough for Layer 1
        patterns = [
            (r'\blast\s+week\b', (now - timedelta(days=7)).isoformat(), now.isoformat()),
            (r'\bthis\s+week\b', (now - timedelta(days=now.weekday())).isoformat(),
             (now + timedelta(days=6 - now.weekday())).isoformat()),
            (r'\blast\s+month\b',
             (now.replace(day=1) - timedelta(days=1)).replace(day=1).isoformat(),
             now.replace(day=1).isoformat()),
            (r'\bthis\s+month\b', now.replace(day=1).isoformat(), now.isoformat()),
            (r'\brecent\b', (now - timedelta(days=30)).isoformat(), now.isoformat()),
            (r'\blast\s+year\b', (now - timedelta(days=365)).isoformat(), now.isoformat()),
            (r'\bthis\s+year\b', now.replace(month=1, day=1).isoformat(), now.isoformat()),
            # Seasons (Northern hemisphere, approximate)
            (r'\blast\s+spring\b',
             (now.year - 1 if now.month < 6 else now.year).__str__() + "-03-01T00:00:00+00:00",
             (now.year - 1 if now.month < 6 else now.year).__str__() + "-05-31T23:59:59+00:00"),
            (r'\blast\s+summer\b',
             (now.year - 1 if now.month < 9 else now.year).__str__() + "-06-01T00:00:00+00:00",
             (now.year - 1 if now.month < 9 else now.year).__str__() + "-08-31T23:59:59+00:00"),
            (r'\blast\s+fall\b',
             (now.year - 1 if now.month < 12 else now.year).__str__() + "-09-01T00:00:00+00:00",
             (now.year - 1 if now.month < 12 else now.year).__str__() + "-11-30T23:59:59+00:00"),
            (r'\blast\s+winter\b',
             (now.year - 1 if now.month < 3 else now.year).__str__() + "-12-01T00:00:00+00:00",
             (now.year - 1 if now.month < 3 else now.year).__str__() + "-02-28T23:59:59+00:00"),
        ]
        for pat, start, end in patterns:
            if re.search(pat, q):
                return (start, end)
        return None

    def _discover_profile_dbs(self) -> Dict[str, str]:
        """Find all profile SQLite DBs in the shared data directory.

        Derives the shared directory from the configured db_path:
        ~/.hermes/hermes-layered-memory-dbs/<profile>.db → uses that dir
        ~/.hermes/profiles/<profile>/hermes-layered-memory.db → checks both dirs
        """
        dbs = {}
        # 1. Directory containing self._db_path (profile-local DBs)
        local_dir = os.path.dirname(os.path.abspath(self._db_path))
        if os.path.isdir(local_dir):
            for f in os.listdir(local_dir):
                if f.endswith('.db'):
                    # If this is our own DB, use the configured profile_name
                    if os.path.join(local_dir, f) == os.path.abspath(self._db_path):
                        dbs[self._profile_name or f[:-3]] = os.path.join(local_dir, f)
                    elif f == "hermes-layered-memory.db":
                        # Profile-specific DB — use parent dir as profile name
                        dbs[os.path.basename(local_dir)] = os.path.join(local_dir, f)
                    else:
                        # Shared DB — filename is profile name
                        profile = f[:-3]  # strip .db
                        dbs[profile] = os.path.join(local_dir, f)

        # 2. If db_path already points to shared dir, we're done
        # 3. Otherwise, check ~/.hermes/hermes-layered-memory-dbs/ as fallback
        # Use actual home dir (~/.hermes), not HERMES_HOME (which may be profile dir)
        actual_hermes_home = os.path.expanduser("~/.hermes")
        candidate_shared = os.path.join(actual_hermes_home, "hermes-layered-memory-dbs")
        if os.path.isdir(candidate_shared) and candidate_shared != local_dir:
            for f in os.listdir(candidate_shared):
                if f.endswith('.db'):
                    profile = f[:-3]
                    dbs[profile] = os.path.join(candidate_shared, f)

        return dbs

    def _layer1(self, candidates: List[tuple], query: str,
                scope: str = None, data_id: str = None,
                session_name: str = None, profile_name: str = None,
                cross_profile: bool = False) -> List[Dict[str, Any]]:
        """Join UUIDs against SQLite, apply filters, add BM25 scores.
        When cross_profile=True, queries all discovered profile DBs."""
        if not candidates:
            return []

        uuids = [c[0] for c in candidates]
        uuid_map = {c[0]: c[1] for c in candidates}

        placeholders = ",".join("?" for _ in uuids)
        where = f"uuid IN ({placeholders}) AND status = 'active'"

        # TTL filter: exclude expired
        where += " AND (ttl IS NULL OR ttl > ?)"
        now = self._now()
        base_params = uuids + [now]

        # data_id filter
        if data_id:
            where += " AND data_id = ?"
            base_params.append(data_id)

        # session_name filter
        if session_name:
            where += " AND session_name = ?"
            base_params.append(session_name)

        # Temporal filter
        temporal = self._parse_temporal(query)
        if temporal:
            where += " AND created_at >= ? AND created_at <= ?"
            base_params.extend(temporal)

        # Scope filter
        if scope:
            where += " AND scope = ?"
            base_params.append(scope)

        # Determine which DBs to query
        if cross_profile or (profile_name and profile_name != self._profile_name):
            # Cross-profile: query all discovered DBs
            all_dbs = self._discover_profile_dbs()
            # If specific profile requested, filter to that one
            if profile_name:
                all_dbs = {profile_name: all_dbs.get(profile_name, "")}
        else:
            # Current profile only
            all_dbs = {self._profile_name or "default": self._db_path}

        # Query all relevant DBs
        all_records = []
        for prof, db_path in all_dbs.items():
            if not db_path or not os.path.exists(db_path):
                continue
            try:
                import sqlite3 as _sqlite3
                conn = _sqlite3.connect(db_path, check_same_thread=False)
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA busy_timeout=5000")
                cursor = conn.execute(f"SELECT rowid, * FROM memories WHERE {where}", base_params)
                rows = cursor.fetchall()
                cols = [d[0] for d in cursor.description]
                batch_records = []
                for row in rows:
                    record = dict(zip(cols, row))
                    record["distance"] = uuid_map.get(record["uuid"], 1.0)
                    record["profile_name"] = prof
                    # Parse JSON fields
                    for jf in ("keywords", "backlinks", "layer3_flags", "metadata"):
                        try:
                            record[jf] = json.loads(record.get(jf, "null")) or {}
                        except (json.JSONDecodeError, TypeError):
                            record[jf] = {}
                    # Strip embedding
                    record.pop("embedding", None)
                    batch_records.append(record)
                # BM25 for this profile's records
                self._add_bm25_conn(batch_records, query, conn)
                all_records.extend(batch_records)
                conn.close()
            except Exception as e:
                logger.debug("Failed to query profile DB %s: %s", prof, e)

        # Score threshold: drop all results if max BM25 below threshold
        min_bm25 = self._config.get("min_bm25_threshold", 0.0)
        if min_bm25 > 0 and all_records:
            max_bm25 = max(r.get("bm25_score", 0.0) for r in all_records)
            if max_bm25 < min_bm25:
                logger.debug("Score threshold: max BM25 %.3f < %.2f, dropping all results",
                             max_bm25, min_bm25)
                return []

        # Sort: pinned first, then by distance
        all_records.sort(key=lambda r: (-r.get("priority", 0), r.get("distance", 1.0)))
        return all_records

    def _add_bm25(self, records: List[Dict], query: str):
        """Add BM25 score from FTS5 for each record, normalized to [0,1]."""
        self._add_bm25_conn(records, query, self._conn)

    def _add_bm25_conn(self, records: List[Dict], query: str, conn):
        """Add BM25 score using a specific SQLite connection."""
        if not records:
            return
        rowids = [r.get("rowid") for r in records if r.get("rowid")]
        if not rowids:
            for r in records:
                r["bm25_score"] = 0.0
            return

        try:
            import re
            tokens = re.findall(r'\w+', query.lower())
            fts_query = " OR ".join(tokens) if len(tokens) > 1 else query

            # Query expansion: add keywords from candidate records
            expand = self._config.get("query_expand", False)
            if expand and tokens:
                expanded = set(tokens)
                for r in records[:5]:  # top-5 vector matches
                    for kw in (r.get("keywords") or []):
                        expanded.add(kw.lower())
                    for kw in (r.get("topic") or "").lower().split():
                        expanded.add(kw)
                if len(expanded) > len(tokens):
                    fts_query = " OR ".join(expanded)
                    logger.debug("Query expanded: %d → %d terms", len(tokens), len(expanded))

            rank_rows = conn.execute(
                "SELECT rowid, rank FROM memories_fts WHERE memories_fts MATCH ? ORDER BY rank",
                (fts_query,)
            ).fetchall()
            rank_map = {rowid: rank for rowid, rank in rank_rows}

            if rank_rows:
                best_rank = min(r[1] for r in rank_rows)
                worst_rank = max(r[1] for r in rank_rows)
                rank_range = worst_rank - best_rank if worst_rank != best_rank else 1.0

                for r in records:
                    rank = rank_map.get(r.get("rowid", 0), 0.0)
                    if rank != 0.0:
                        r["bm25_score"] = max(0.0, min(1.0, (worst_rank - rank) / rank_range))
                    else:
                        r["bm25_score"] = 0.0
            else:
                for r in records:
                    r["bm25_score"] = 0.0

        except Exception:
            for r in records:
                r["bm25_score"] = 0.0

    # ---- Layer 2: Fusion scoring ----------------------------------------

    # Default scoring weights — overridden by config if present
    _DEFAULT_WEIGHTS = {
        "bm25_weight": 0.7,
        "importance_weight": 0.15,
        "recency_weight": 0.15,
        "topic_boost": 0.3,
        "keyword_boost": 0.15,
        "pinned_boost": 1.5,
        "recency_half_life_days": 21,
        "rrf_constant": 60,
    }

    def _get_weight(self, key):
        return self._config.get("scoring", {}).get(key, self._DEFAULT_WEIGHTS[key])

    # ---- Metadata enrichment (heuristic + LLM fallback) ----------------

    # Heuristic keyword -> (data_type, data_id) mapping
    _HEURISTIC_MAP = {
        "ENV-DATA": {
            "hw": r"\b(gpu|cuda|ram|cpu|nvidia|rtx|tierra|blackwell|vram|wsl|ubuntu|linux|kernel|memory\s+size|cores?|clock|throttle|thermal)\b",
            "sw": r"\b(python|pip|docker|node|npm|vscode|git|ssh|proxy|package|install|brew|apt|pipx|venv|conda|wsl|driver|sdk|toolchain)\b",
            "net": r"\b(endpoint|api\s+key|url|hostname|port|dns|firewall|proxy|localhost|internal69|vllm|ollama|qdrant|whisper|http|https|tcp|grpc|socket|reverse\s+proxy|nginx)\b",
            "workflow": r"\b(kanban|delegate|worker|session|profile|orchestrat|pipeline|cron|schedule|automation|recurring|workflow|ci/cd|deployment)\b",
            "conventions": r"\b(naming|convention|format|standard|pattern|prefix|suffix|camelcase|snake.?case|kebab.?case|file\s+structure|directory\s+layout)\b",
            "troubleshooting": r"\b(error|fix|bug|workaround|issue|debug|crash|timeout|fail|resolve|resolve|patch|hotfix|segfault|oom|out\s+of\s+memory)\b",
        },
        "USER-DATA": {
            "preferences": r"\b(prefer|like|don't\s+like|always|never|want|wish|avoid|dislike|enjoy|favor|choice)\b",
            "habits": r"\b(always|usually|tend|habit|routine|convention|coding\s+style|comment|docstring|naming|indent)\b",
            "conventions": r"\b(naming|convention|format|standard|pattern|prefix|suffix|camelcase|snake.?case|kebab.?case|file\s+structure|directory\s+layout)\b",
            "style": r"\b(tone|concise|verbose|detailed|brief|formal|casual|language|respond|communication)\b",
        },
        "SYSTEM": {
            "identity": r"\b(agent|persona|identity|you\s+(are|should)|system|role|name\s+is|called\s|hermes|assistant|bot)\b",
            "rules": r"\b(must|must\s+not|never|always\s+do|forbidden|restricted|rule|constraint|requirement|mandatory|do\s+not|don't\s+(ever|do))\b",
        },
        "SESSION-DATA": {
            "session": r"\b(session|this\s+(run|task|work)|current|context|working\s+on|investigat|exploring)\b",
        },
    }

    def _heuristic_classify(self, content: str) -> tuple:
        """Classify content into (data_type, data_id) using keyword heuristics.

        Returns (data_type, data_id, confidence) where confidence is 0.0-1.0.
        Confidence 0 means no match; >= 0.6 means high confidence (skip LLM).
        """
        import re
        text = content.lower()
        best_type, best_id, best_count = None, None, 0

        for data_type, categories in self._HEURISTIC_MAP.items():
            for data_id, pattern in categories.items():
                matches = len(re.findall(pattern, text))
                if matches > best_count:
                    best_count = matches
                    best_type = data_type
                    best_id = data_id

        if best_count == 0:
            return ("CUSTOM", None, 0.0)

        # Confidence scales with match count, capped at 1.0
        confidence = min(best_count / 3.0, 1.0)
        return (best_type, best_id, confidence)

    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities from text using configurable patterns.

        Patterns loaded from entity-patterns/default.json or config override.
        Returns list of entity keywords. Enables cross-topic retrieval via
        entity-based keyword matching.
        """
        import re

        patterns = self._load_entity_patterns()
        entities = []

        for p in patterns:
            if not p.get("enabled", True):
                continue
            name = p.get("name", "custom")
            regex = p.get("regex")
            if not regex:
                continue

            try:
                flags = 0
                if p.get("flags") == "IGNORECASE":
                    flags = re.IGNORECASE

                for m in re.finditer(regex, text, flags):
                    raw = m.group().strip()
                    # Apply filters
                    if p.get("filter") == "exclude_stopwords":
                        if raw.lower().split()[0] in p.get("stopwords", []):
                            continue
                    elif p.get("filter") == "min_length":
                        if len(raw) < p.get("min_length", 0):
                            continue
                        exclude = p.get("exclude_tlds", [])
                        if any(raw.endswith(tld) for tld in exclude):
                            continue
                    entities.append(raw)
            except re.error:
                logger.debug("Entity pattern %s: invalid regex %s", name, regex)

        # Deduplicate and limit
        return list(dict.fromkeys(entities))[:15]

    def _llm_classify(self, content: str, summary: str = None) -> dict:
        """Classify content via LLM when heuristics are uncertain.

        Returns dict with data_type, data_id, topic, keywords.
        """
        text = summary or content[:500]
        prompt = f"""Classify this content into structured metadata. Return ONLY valid JSON:
{{
  "data_type": "USER-DATA" | "ENV-DATA" | "SESSION-DATA" | "SYSTEM" | "CUSTOM",
  "data_id": "hw" | "sw" | "net" | "workflow" | "conventions" | "troubleshooting" | "preferences" | "habits" | "style" | "identity" | "rules" | null,
  "topic": "short topic phrase",
  "keywords": ["keyword1", "keyword2", ...]
}}

Data type guide:
- ENV-DATA: hardware, software, tools, network, endpoints, configs, workflows, conventions, troubleshooting
- USER-DATA: user preferences, habits, communication style, coding conventions
- SESSION-DATA: per-session context, temporary facts
- SYSTEM: agent identity, behavioral rules, constraints
- CUSTOM: free-form notes, project-specific, imports

Content:
{text}"""

        response = self._call_llm(prompt)
        if response:
            try:
                import re as _re
                match = _re.search(r'\{.*?\}', response, _re.DOTALL)
                if match:
                    result = json.loads(match.group())
                    logger.debug("LLM classification: type=%s id=%s topic=%s",
                                 result.get("data_type"), result.get("data_id"), result.get("topic"))
                    return result
            except Exception as e:
                logger.debug("LLM classification parse failed: %s", e)
        return {"data_type": "CUSTOM", "data_id": None, "topic": None, "keywords": []}

    def _enrich_metadata(self, content: str, summary: str = None,
                         data_type: str = None, data_id: str = None,
                         topic: str = None, keywords: List[str] = None) -> dict:
        """Hybrid enrichment: heuristics first, LLM fallback for low confidence.

        Returns dict with (data_type, data_id, topic, keywords).
        Respects enrich_on_add config: "true", "false", "low_confidence" (default).
        """
        mode = self._config.get("enrich_on_add", "low_confidence")

        # If all fields are provided, skip enrichment
        if data_type and data_id and topic and keywords:
            return {"data_type": data_type, "data_id": data_id,
                    "topic": topic, "keywords": keywords}

        # Heuristic pass
        h_type, h_id, confidence = self._heuristic_classify(content)
        logger.debug("Heuristic classify: type=%s id=%s confidence=%.2f",
                      h_type, h_id, confidence)

        # If all fields are explicitly provided (not default CUSTOM), skip enrichment
        if data_type and data_type != "CUSTOM" and data_id and topic and keywords:
            return {"data_type": data_type, "data_id": data_id,
                    "topic": topic, "keywords": keywords}

        # If heuristics found a non-CUSTOM type with high confidence, use them
        if h_type != "CUSTOM" and confidence >= 0.6:
            result = {
                "data_type": data_type if (data_type and data_type != "CUSTOM") else h_type,
                "data_id": data_id or h_id,
                "topic": topic,
                "keywords": keywords or [],
            }
            # If enrich_llm is enabled and topic/keywords are empty, run LLM to fill gaps
            if self._config.get("enrich_llm") and (not topic or not keywords):
                llm_result = self._llm_classify(content, summary)
                return {
                    "data_type": result["data_type"],
                    "data_id": result["data_id"],
                    "topic": topic or llm_result.get("topic"),
                    "keywords": keywords or llm_result.get("keywords", []),
                }
            return result

        # LLM fallback: low_confidence mode or always enrich
        if mode == "true" or (mode == "low_confidence" and confidence < 0.6):
            llm_result = self._llm_classify(content, summary)
            result = {
                "data_type": data_type or llm_result.get("data_type", "CUSTOM"),
                "data_id": data_id or llm_result.get("data_id"),
                "topic": topic or llm_result.get("topic"),
                "keywords": keywords or llm_result.get("keywords", []),
            }
        else:
            # mode == "false" or high-confidence heuristic — return defaults
            result = {
                "data_type": data_type or "CUSTOM",
                "data_id": data_id,
                "topic": topic,
                "keywords": keywords or [],
            }

        # Entity extraction: merge entities into keywords
        keywords = result.get("keywords") or []
        entities = self._extract_entities(content)
        for e in entities:
            if e.lower() not in [k.lower() for k in keywords]:
                keywords.append(e)
        result["keywords"] = keywords
        return result

    def enrich_existing(self, since: str = None, max_items: int = 100) -> dict:
        """Batch-enrich existing memories that have empty or incorrect metadata.

        Pass 1: Records with empty topic/keywords (any data_type).
        Pass 2: data_type='CUSTOM' records (low-confidence heuristic leftover).
        """
        enriched = 0

        # Pass 1: empty fields
        where = "(topic IS NULL OR topic = '') AND (keywords IS NULL OR keywords = '[]' OR keywords = 'null')"\
                " AND status = 'active'"
        params = []
        if since:
            where += " AND updated_at > ?"
            params.append(since)
        where += " LIMIT ?"
        params.append(max_items)

        rows = self._conn.execute(
            f"SELECT uuid, content, summary FROM memories WHERE {where}", params
        ).fetchall()

        for uuid, content, summary in rows:
            try:
                result = self._enrich_metadata(content, summary)
                # Entity extraction: merge entities into keywords
                keywords = result.get("keywords") or []
                entities = self._extract_entities(content)
                for e in entities:
                    if e.lower() not in [k.lower() for k in keywords]:
                        keywords.append(e)
                kw_json = json.dumps(keywords)
                now = self._now()
                self._conn.execute(
                    "UPDATE memories SET data_type = ?, data_id = ?, topic = ?, keywords = ?, updated_at = ? WHERE uuid = ?",
                    (result.get("data_type"), result.get("data_id"), result.get("topic"), kw_json, now, uuid)
                )
                enriched += 1
                logger.debug("Enriched (pass1): %s -> type=%s id=%s topic=%s",
                             uuid[:8], result.get("data_type"), result.get("data_id"), result.get("topic"))
            except Exception as e:
                logger.debug("Enrich failed for %s: %s", uuid[:8], e)

        # Pass 2: CUSTOM type (incomplete heuristic classification)
        where2 = "data_type = 'CUSTOM' AND status = 'active'"
        params2 = []
        if since:
            where2 += " AND updated_at > ?"
            params2.append(since)
        where2 += " LIMIT ?"
        params2.append(max_items)

        rows2 = self._conn.execute(
            f"SELECT uuid, content, summary FROM memories WHERE {where2}", params2
        ).fetchall()

        for uuid, content, summary in rows2:
            try:
                result = self._enrich_metadata(content, summary)
                # Entity extraction: merge entities into keywords
                keywords = result.get("keywords") or []
                entities = self._extract_entities(content)
                for e in entities:
                    if e.lower() not in [k.lower() for k in keywords]:
                        keywords.append(e)
                kw_json = json.dumps(keywords)
                now = self._now()
                self._conn.execute(
                    "UPDATE memories SET data_type = ?, data_id = ?, topic = ?, keywords = ?, updated_at = ? WHERE uuid = ?",
                    (result.get("data_type"), result.get("data_id"), result.get("topic"), kw_json, now, uuid)
                )
                enriched += 1
                logger.debug("Enriched (pass2): %s -> type=%s id=%s topic=%s",
                             uuid[:8], result.get("data_type"), result.get("data_id"), result.get("topic"))
            except Exception as e:
                logger.debug("Enrich failed for %s: %s", uuid[:8], e)

        self._conn.commit()
        return {"enriched": enriched, "since": since, "max_items": max_items}

    def _layer2(self, records: List[Dict], query: str) -> List[Dict]:
        k = self._get_weight("rrf_constant")
        w_bm25 = self._get_weight("bm25_weight")
        w_imp = self._get_weight("importance_weight")
        w_rec = self._get_weight("recency_weight")
        w_topic = self._get_weight("topic_boost")
        w_kw = self._get_weight("keyword_boost")
        w_pinned = self._get_weight("pinned_boost")
        half_life = self._get_weight("recency_half_life_days")
        import math
        decay = 1.0 / (half_life * math.log(2))
        now_ts = datetime.now(timezone.utc)

        for rank, r in enumerate(records):
            # Base: RRF rank score
            r["fusion_score"] = 1.0 / (k + rank + 1)

            # BM25 contribution (normalized to [0,1] range)
            bm25 = r.get("bm25_score", 0.0)
            r["fusion_score"] += w_bm25 * bm25

            # Topic/keyword exact match boost
            query_words = set(query.lower().split())
            topic = (r.get("topic") or "").lower()
            keywords = [kw.lower() for kw in (r.get("keywords") or [])]
            if query_words and query_words & set(topic.split()):
                r["fusion_score"] += w_topic
            if query_words and query_words & set(keywords):
                r["fusion_score"] += w_kw

            # Importance (trust_score mapped to importance)
            trust = r.get("trust_score", 0.5)
            r["fusion_score"] += w_imp * trust

            # Recency: decay model
            prio = r.get("priority", 0)
            if prio < 3:  # pinned items don't decay
                try:
                    created = datetime.fromisoformat(r.get("created_at", self._now()))
                    days = max((now_ts - created).days, 0)
                    recency = 1.0 / (1.0 + days * decay)
                except (ValueError, TypeError):
                    recency = 0.5
            else:
                recency = 1.0
            r["fusion_score"] += w_rec * recency

            # Priority boost (overrides decay for pinned)
            if prio >= 3:
                r["fusion_score"] *= w_pinned

        records.sort(key=lambda r: r.get("fusion_score", 0), reverse=True)
        return records

    # ---- Layer 3: LLM reranker ------------------------------------------

    def _call_llm(self, prompt: str) -> str:
        """Call LLM for reranking/gap detection. Config: layer3_model."""
        model = self._config.get("layer3_model")
        if not model:
            return ""
        provider_cfg = self._config.get("layer3_provider_config", {})
        base_url = provider_cfg.get("base_url", "http://vllm.internal69.net:8000/v1")
        api_key = provider_cfg.get("api_key", "sk-")
        try:
            import urllib.request
            payload_obj = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a precise JSON formatter. Return ONLY valid JSON arrays or objects as instructed."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 1024,
                "temperature": 0.0,
                "stream": False,
            }
            # Add reasoning_effort if configured (e.g., "none" to disable thinking for Qwen3)
            reasoning_effort = self._config.get("layer3_reasoning_effort")
            if reasoning_effort:
                payload_obj["reasoning_effort"] = reasoning_effort
            payload = json.dumps(payload_obj).encode()
            req = urllib.request.Request(
                f"{base_url}/chat/completions",
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                    "Accept": "application/json",
                },
            )
            # Retry on transient failures (3 attempts, exponential backoff)
            last_err = None
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(req, timeout=120) as resp:
                        result = json.loads(resp.read())
                        content = result.get("choices", [{}])[0].get("message", {}).get("content")
                        if content is None:
                            logger.debug("LLM returned null content")
                            return ""
                        return content
                except Exception as e:
                    last_err = e
                    if attempt < 2:
                        wait = 2 ** attempt
                        logger.debug("LLM call failed (attempt %d/%d), retrying in %ds: %s",
                                     attempt + 1, 3, wait, e)
                        time.sleep(wait)
                    continue
            logger.debug("LLM call failed after 3 attempts: %s", last_err)
            return ""
        except Exception as e:
            logger.debug("LLM setup failed: %s", e)
            return ""

    def _layer3(self, records: List[Dict], query: str, limit: int = 5) -> List[Dict]:
        """LLM reranker — ask the model to re-rank by relevance to query.

        Requests more candidates than needed, lets LLM decide if reranking
        is necessary, then returns top `limit` results.
        """
        mode = self._config.get("layer3_mode", "inline")
        if mode != "inline":
            for i, r in enumerate(records):
                r["rerank_score"] = r.get("fusion_score", 0)
                r["rerank_position"] = i
                if not isinstance(r.get("layer3_flags"), dict):
                    r["layer3_flags"] = {}
                r["layer3_flags"].update({"reranked": False, "verified": False})
            return records

        # Build prompt for reranking — include more candidates than needed
        candidates = records[:min(limit * 3, len(records))]
        items = []
        for i, r in enumerate(candidates):
            items.append(f"<item id={i}>topic={r.get('topic')}, summary={r.get('summary','')[:100]}</item>")
        prompt = f"""You are a relevance scorer. Rank these items by how well they answer the query.

Query: "{query}"

Items:
{chr(10).join(items)}

If the top items already answer the query well, respond with ONLY:
{{"rerank_needed": false}}

If there is ambiguity or better ordering is possible, respond with ONLY:
{{"rerank_needed": true, "ranking": [[id, score], ...]}}
Score is 0.0-1.0 (1.0 = perfect match). Sort by score descending.
Example: {{"rerank_needed": true, "ranking": [[0, 0.95], [2, 0.8], [1, 0.6]]}}"""

        response = self._call_llm(prompt)
        if response:
            try:
                import re
                match = re.search(r'\{.*?\}', response, re.DOTALL)
                if match:
                    result = json.loads(match.group())
                    if result.get("rerank_needed") and "ranking" in result:
                        # Apply reranking (use dict for O(1) lookup instead of O(n) scan)
                        ranking = result["ranking"]
                        score_map = {idx: score for idx, score in ranking}
                        idx_map = {id(rec): i for i, rec in enumerate(candidates)}
                        for r in records:
                            idx = idx_map.get(id(r), -1)
                            if idx in score_map:
                                r["rerank_score"] = score_map[idx]
                                r["fusion_score"] = score_map[idx]
                            else:
                                r["rerank_score"] = r.get("fusion_score", 0)
                            r["rerank_position"] = idx
                            if not isinstance(r.get("layer3_flags"), dict):
                                r["layer3_flags"] = {}
                            r["layer3_flags"].update({"reranked": True, "verified": False})
                        records.sort(key=lambda r: r.get("rerank_score", 0), reverse=True)
                        logger.debug("layer3: reranked %d items", len(ranking))
                        return records[:limit]
                    else:
                        # BM25 already good — skip reranking
                        logger.debug("layer3: BM25 sufficient, skipping rerank")
                        for i, r in enumerate(records):
                            r["rerank_score"] = r.get("fusion_score", 0)
                            r["rerank_position"] = i
                            if not isinstance(r.get("layer3_flags"), dict):
                                r["layer3_flags"] = {}
                            r["layer3_flags"].update({
                                "reranked": False,
                                "verified": False,
                                "skip_reason": result.get("reason", "BM25 sufficient")
                            })
                        return records[:limit]
            except Exception as e:
                logger.debug("Layer 3 parse failed: %s", e)

        # Fallback: pass-through
        for i, r in enumerate(records):
            r["rerank_score"] = r.get("fusion_score", 0)
            r["rerank_position"] = i
            if not isinstance(r.get("layer3_flags"), dict):
                r["layer3_flags"] = {}
            r["layer3_flags"].update({"reranked": False, "verified": False})
        return records[:limit]

    # ---- Layer 4: Gap detection ------------------------------------------

    def _layer4(self, records: List[Dict], query: str) -> List[Dict]:
        """Gap detection — check if retrieved results answer the query."""
        if not records:
            return records

        # Build prompt for gap detection
        items = []
        for i, r in enumerate(records[:5]):  # Top 5 for gap analysis
            items.append(f"- {r.get('topic')}: {r.get('summary','')[:100]}")
        prompt = f"""Does the following retrieved information fully answer the query: "{query}"?

Retrieved:
{chr(10).join(items)}

Return JSON: {{"answers": true/false, "gaps": ["missing info 1", "missing info 2"], "confidence": 0-1}}"""

        response = self._call_llm(prompt)
        gaps = []
        confidence = 1.0
        if response:
            try:
                import re
                match = re.search(r'\{.*?\}', response, re.DOTALL)
                if match:
                    result = json.loads(match.group())
                    gaps = result.get("gaps", [])
                    confidence = result.get("confidence", 1.0)
                    logger.debug("layer4: gaps=%s confidence=%.2f", gaps, confidence)
            except Exception as e:
                logger.debug("Layer 4 parse failed: %s", e)

        # Annotate records with gap info (only on first record to save tokens)
        if records:
            if not isinstance(records[0].get("layer3_flags"), dict):
                records[0]["layer3_flags"] = {}
            records[0]["layer3_flags"]["gap_checked"] = True
            records[0]["layer3_flags"]["gaps"] = gaps
            records[0]["layer3_flags"]["confidence"] = confidence

        return records

    def list(self, topic: str = None, scope: str = None,
             limit: int = 20, sort: str = "created_at") -> List[Dict[str, Any]]:
        where = "status = 'active'"
        params = []
        if topic:
            where += " AND topic = ?"
            params.append(topic)
        if scope:
            where += " AND scope = ?"
            params.append(scope)
        order = "created_at DESC"
        if sort == "priority":
            order = "priority DESC, created_at DESC"
        elif sort == "trust_score":
            order = "trust_score DESC, created_at DESC"
        cursor = self._conn.execute(
            f"SELECT * FROM memories WHERE {where} ORDER BY {order} LIMIT ?",
            params + [limit]
        )
        rows = cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        results = []
        for row in rows:
            r = dict(zip(cols, row))
            for jf in ("keywords", "backlinks", "layer3_flags", "metadata"):
                try:
                    r[jf] = json.loads(r.get(jf, "null")) or {}
                except (json.JSONDecodeError, TypeError):
                    r[jf] = {}
            # Strip embedding (large, not needed in list output)
            r.pop("embedding", None)
            results.append(r)
        return results

    def rebuild(self, since: str = None) -> dict:
        if not self._qdrant:
            return {"status": "skipped", "reason": "qdrant not available"}
        fn = _get_embedding_fn()
        where = "status = 'active'"
        params = []
        if since:
            where += " AND updated_at > ?"
            params.append(since)
        rows = self._conn.execute(f"SELECT uuid, embedding, data_type, data_id, session_name FROM memories WHERE {where}", params).fetchall()
        count = 0
        # Group by collection
        batches = {}
        # Cache collection dimensions (avoid repeated HTTP calls)
        dim_cache = {}

        for uuid, emb_json, dt, di, sn in rows:
            try:
                emb = json.loads(emb_json) if emb_json else None
                # Check if re-embedding is needed (dimension mismatch)
                from qdrant_client.models import PointStruct
                coll = self._get_collection(dt or "CUSTOM")
                if coll not in dim_cache:
                    coll_info = self._qdrant.get_collection(coll)
                    dim_cache[coll] = coll_info.config.params.vectors.size
                expected_dim = dim_cache[coll]
                if emb and len(emb) != expected_dim:
                    # Re-embed with new model (dimension mismatch)
                    content = self._conn.execute("SELECT content FROM memories WHERE uuid=?", (uuid,)).fetchone()[0]
                    emb = fn([content])[0]
                    self._conn.execute("UPDATE memories SET embedding=? WHERE uuid=?", (json.dumps(emb) if emb else "null", uuid))
                    self._conn.commit()
                if emb is None:
                    # Re-embed records with NULL embedding (failed during original add)
                    content = self._conn.execute("SELECT content FROM memories WHERE uuid=?", (uuid,)).fetchone()[0]
                    emb = fn([content])[0]
                    self._conn.execute("UPDATE memories SET embedding=? WHERE uuid=?", (json.dumps(emb) if emb else "null", uuid))
                    self._conn.commit()
                    if emb is None:
                        logger.debug("rebuild: re-embed returned None for %s", uuid[:8])
                        continue
                payload = {"data_type": dt or "CUSTOM"}
                if di:
                    payload["data_id"] = di
                if sn:
                    payload["session_name"] = sn
                if self._profile_name:
                    payload["profile_name"] = self._profile_name
                batches.setdefault(coll, []).append(
                    PointStruct(id=_to_qdrant_id(uuid), vector=emb, payload=payload)
                )
                if len(batches[coll]) >= 100:
                    self._qdrant.upsert(collection_name=coll, points=batches[coll])
                    count += len(batches[coll])
                    batches[coll] = []
            except Exception as e:
                logger.debug("rebuild failed for %s: %s", uuid[:8], e)
                pass
        for coll, batch in batches.items():
            if batch:
                try:
                    self._qdrant.upsert(collection_name=coll, points=batch)
                    count += len(batch)
                except Exception as e:
                    logger.warning("rebuild: final batch upsert failed for collection %s (%d points): %s", coll, len(batch), e)

        # Cleanup: delete orphaned Qdrant points (archived/deleted records)
        from qdrant_client.models import PointIdsList, Filter, FieldCondition, MatchValue as MatchValueModel
        active_uuids = set(r[0] for r in rows)
        for coll in self._config.get("collections", {}).values():
            try:
                # Skip cleanup if no profile name (unprofiled instances)
                if not self._profile_name:
                    continue
                # Scroll all points in this collection for this profile
                scroll_uuids = []
                offset = None
                while True:
                    points, offset = self._qdrant.scroll(
                        collection_name=coll,
                        scroll_filter=Filter(must=[
                            FieldCondition(key="profile_name", match=MatchValueModel(value=self._profile_name))
                        ]),
                        limit=100,
                        offset=offset,
                        with_payload=False,
                        with_vectors=False,
                    )
                    if not points:
                        break
                    for p in points:
                        scroll_uuids.append(_from_qdrant_id(p.id))
                    if not offset:
                        break

                # Delete points not in active UUIDs
                orphan_ids = [_to_qdrant_id(u) for u in scroll_uuids if u not in active_uuids]
                if orphan_ids:
                    self._qdrant.delete(collection_name=coll, points_selector=PointIdsList(points=orphan_ids))
                    logger.debug("Rebuild cleanup: deleted %d orphaned points from %s", len(orphan_ids), coll)
            except Exception as e:
                logger.debug("Rebuild cleanup failed for %s: %s", coll, e)

        return {"status": "rebuilt", "count": count, "since": since}

    def sync_check(self) -> dict:
        sqlite_count = self.count()
        qdrant_count = 0
        profile_count = 0
        if self._qdrant:
            try:
                from qdrant_client.models import FieldCondition, MatchValue, Filter
                # Count points per profile in each collection
                for coll in set(self._collection_map.values()):
                    try:
                        info = self._qdrant.get_collection(coll)
                        qdrant_count += info.points_count
                        # Filter by current profile
                        if self._profile_name:
                            filter_ = Filter(must=[FieldCondition(key="profile_name",
                               match=MatchValue(value=self._profile_name))])
                            count_result = self._qdrant.count(coll, count_filter=filter_)
                            profile_count += count_result.count
                    except Exception:
                        pass
            except Exception:
                pass
        # Use profile-specific count for sync check
        effective_count = profile_count if self._profile_name else qdrant_count
        return {"sqlite_active": sqlite_count, "qdrant_total": qdrant_count,
                "qdrant_profile": effective_count,
                "in_sync": sqlite_count == effective_count}

    # ---- Trust score feedback --------------------------------------------

    def feedback(self, uuid: str, helpful: bool) -> dict:
        """Provide feedback on a memory's usefulness.

        Increments/decrements trust_score by 0.1, capped at 0.0-1.0.
        Low-trust records (<0.3) are auto-archived by layered_sleep.
        """
        row = self._conn.execute(
            "SELECT trust_score, status FROM memories WHERE uuid = ?", (uuid,)
        ).fetchone()
        if not row:
            return {"status": "not_found", "uuid": uuid}
        if row[1] != "active":
            return {"status": "not_active", "uuid": uuid, "current_status": row[1]}

        current = row[0] or 0.5
        delta = 0.1 if helpful else -0.1
        new_score = max(0.0, min(1.0, current + delta))

        self._conn.execute(
            "UPDATE memories SET trust_score = ? WHERE uuid = ?", (new_score, uuid)
        )
        self._conn.commit()

        logger.debug("Feedback: %s -> trust_score=%.2f (helpful=%s)", uuid, new_score, helpful)
        return {
            "uuid": uuid,
            "old_score": round(current, 2),
            "new_score": round(new_score, 2),
            "delta": delta
        }

    # ---- Sleep / Consolidation (Mnemosyne BEAM pattern) -----------------

    def sleep(self, max_items: int = 100, min_age_hours: float = 24.0) -> dict:
        """Consolidation: merge stale/low-trust memories, archive old ones.

        Mnemosyne BEAM pattern: working → episodic via summarization.
        Here: archive low-trust duplicates within the same data_type:data_id group,
        archive very old low-trust records.
        """
        now_ts = datetime.now(timezone.utc)
        cutoff = (now_ts - timedelta(hours=min_age_hours)).isoformat()
        merged = 0
        archived = 0
        ttl_expired = 0

        # 0. TTL auto-expiry: archive records past their TTL
        self._conn.execute("""
            UPDATE memories SET status = 'archived', updated_at = ?
            WHERE status = 'active' AND ttl IS NOT NULL AND ttl != '' AND ttl < ?
        """, (self._now(), now_ts.isoformat()))
        ttl_expired = self._conn.total_changes  # rough count

        # 1. Archive low-trust duplicates within the same data_type:data_id group
        # Group by data_type + data_id, keep the highest trust_score record, archive rest
        groups = self._conn.execute("""
            SELECT data_type, data_id FROM memories
            WHERE status = 'active' AND data_type IS NOT NULL AND data_id IS NOT NULL
            GROUP BY data_type, data_id HAVING COUNT(*) > 1
            LIMIT ?
        """, (max_items,)).fetchall()

        for (dt, di) in groups:
            members = self._conn.execute("""
                SELECT uuid, trust_score, priority
                FROM memories WHERE data_type = ? AND data_id = ? AND status = 'active'
                ORDER BY trust_score DESC, priority DESC
            """, (dt, di)).fetchall()
            if len(members) < 2:
                continue
            # Keep best (first), archive the rest if they have low trust
            for uuid, trust, prio in members[1:]:
                if trust < 0.4:
                    self._conn.execute(
                        "UPDATE memories SET status = 'archived', updated_at = ? WHERE uuid = ?",
                        (self._now(), uuid))
                    archived += 1

        archive_days = self._config.get("low_trust_archive_days", 365)
        # 2. Archive: very old low-trust (<0.3) non-pinned
        self._conn.execute("""
            UPDATE memories SET status = 'archived', updated_at = ?
            WHERE status = 'active'
              AND created_at < ?
              AND trust_score < 0.3
              AND priority < 2
        """, (self._now(), (now_ts - timedelta(days=archive_days)).isoformat()))

        self._conn.commit()
        return {"merged": merged, "archived": archived, "ttl_expired": ttl_expired, "cutoff": cutoff}

    # ---- Compaction (content merge) --------------------------------------

    def compact(self, similarity_threshold=0.90, topic=None, max_groups=50):
        """Merge duplicate/overlapping memories via LLM.

        Args:
            similarity_threshold: Minimum vector similarity (0.8-0.99, default 0.90)
            topic: Optional topic filter
            max_groups: Maximum groups to process (default 50)

        Returns:
            dict with groups_merged, records_compacted, details
        """
        if not self._qdrant:
            logger.warning("compact: Qdrant not available")
            return {"error": "Qdrant not available"}

        groups = self.find_duplicate_groups(similarity_threshold, topic, max_groups)
        if not groups:
            return {"groups_merged": 0, "records_compacted": 0, "details": []}

        merged = 0
        details = []
        for group in groups:
            result = self._compact_group(group)
            if result:
                merged += len(group["uuids"]) - 1  # N records -> 1 merged
                details.append(result)

        return {"groups_merged": len(details), "records_compacted": merged, "details": details}

    def find_duplicate_groups(self, similarity_threshold, topic=None, max_groups=50):
        """Find groups of similar records via Qdrant native search.

        Uses iterative Qdrant queries with score_threshold instead of O(N^2) pairwise.
        """
        if not self._qdrant:
            return []

        processed = set()
        groups = []

        # Get active records
        records = self._conn.execute(
            "SELECT uuid, data_type FROM memories WHERE status='active' AND priority < 2"
            + (" AND topic=?" if topic else ""),
            (topic,) if topic else ()
        ).fetchall()

        for record in records:
            uuid = record[0]
            if uuid in processed:
                continue

            # Query Qdrant for similar records (only filter by fields stored in payload)
            # Qdrant payload stores: data_type, data_id, session_name, profile_name
            # NOT: status, topic -- those are SQLite-only
            try:
                from qdrant_client.models import Filter, FieldCondition, MatchValue
                collection = self._get_collection(record[1] or "CUSTOM")

                query_vector = self._get_vector(uuid)
                if not query_vector:
                    continue

                conditions = []
                if self._profile_name:
                    conditions.append(FieldCondition(key="profile_name", match=MatchValue(value=self._profile_name)))

                results = self._qdrant.query_points(
                    collection_name=collection,
                    query=query_vector,
                    query_filter=Filter(must=conditions),
                    limit=10,
                    score_threshold=similarity_threshold,
                )

                # Build group from results — only include records that are still active in SQLite
                active_uuids = {r[0] for r in records}
                group_uuids = []
                for hit in results.points:
                    hit_uuid = _from_qdrant_id(hit.id)
                    if hit_uuid != uuid and hit_uuid not in processed and hit_uuid in active_uuids:
                        group_uuids.append(hit_uuid)

                if group_uuids:
                    group_uuids.insert(0, uuid)  # Include the seed record
                    processed.add(uuid)
                    processed.update(group_uuids)
                    groups.append({
                        "uuids": group_uuids,
                        "similarity": results.points[0].score if results.points else similarity_threshold
                    })

                    if len(groups) >= max_groups:
                        break

            except Exception as e:
                logger.debug("compact: search failed for %s: %s", uuid, e)
                continue

        return groups

    def _get_vector(self, record_uuid):
        """Extract embedding vector from SQLite by UUID.

        Embeddings are stored as JSON strings (json.dumps), not raw binary BLOBs.
        """
        row = self._conn.execute(
            "SELECT embedding FROM memories WHERE uuid = ?",
            (record_uuid,)
        ).fetchone()
        if row and row[0]:
            # Already a list from json.loads (SQLite stores as TEXT)
            if isinstance(row[0], list):
                return row[0]
            # JSON string
            return json.loads(row[0])
        return None

    def _embed_and_upsert(self, record_uuid, content, payload):
        """Compute embedding and upsert to Qdrant."""
        try:
            from qdrant_client.models import PointStruct
            fn = _get_embedding_fn()
            vectors = fn([content])
            if not vectors or not vectors[0]:
                return
            self._qdrant.upsert(
                collection_name=self._get_collection(payload.get("data_type", "CUSTOM")),
                points=[PointStruct(
                    id=_to_qdrant_id(record_uuid),
                    vector=vectors[0],
                    payload=payload
                )]
            )
        except Exception as e:
            logger.debug("embed_and_upsert failed for %s: %s", record_uuid, e)

    def _compact_group(self, group):
        """Merge a group of similar records via LLM.

        Returns dict with merge details, or None on failure.
        """
        uuids = group["uuids"]

        # Cap to max 10 records per merge
        if len(uuids) > 10:
            uuids = uuids[:10]

        # Fetch full records
        records = []
        placeholders = ",".join(["?" for _ in uuids])
        rows = self._conn.execute(
            "SELECT uuid, content, summary, keywords, topic, data_type, trust_score, "
            "created_at FROM memories WHERE uuid IN ({})".format(placeholders),
            uuids
        ).fetchall()

        for row in rows:
            records.append({
                "uuid": row[0], "content": row[1], "summary": row[2],
                "keywords": row[3], "topic": row[4], "data_type": row[5],
                "trust_score": row[6], "created_at": row[7]
            })

        if len(records) < 2:
            return None

        # Call LLM for merge
        try:
            merged = self._llm_merge(records)
        except Exception as e:
            logger.debug("compact: LLM merge failed: %s", e)
            return None

        if not merged or "content" not in merged:
            return None

        # Create merged record
        new_uuid = uuid_mod.uuid4().hex
        max_trust = max(r["trust_score"] for r in records)
        n_records = len(records)
        corroboration_boost = min(1.0, max_trust + 0.05 * (n_records - 1))
        first_observed = min(r["created_at"] for r in records)

        # Most common data_type
        from collections import Counter
        data_types = [r["data_type"] for r in records if r["data_type"]]
        dominant_type = Counter(data_types).most_common(1)[0][0] if data_types else "CUSTOM"

        # Add the merged record
        try:
            self._conn.execute("""
                INSERT INTO memories (uuid, content, summary, keywords, topic, data_type,
                    scope, created_at, updated_at, trust_score, first_observed_at,
                    metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                new_uuid,
                merged["content"],
                merged.get("summary", merged["content"][:100]),
                json.dumps(merged.get("keywords", [])),
                merged.get("topic", records[0]["topic"]),
                dominant_type,
                records[0].get("scope", "personal") if isinstance(records[0], dict) else "personal",
                self._now(),
                self._now(),
                corroboration_boost,
                first_observed,
                json.dumps({"compacted_from": uuids, "first_observed_at": first_observed})
            ))

            # Embed and upsert to Qdrant
            self._embed_and_upsert(new_uuid, merged["content"], {
                "data_type": dominant_type, "topic": merged.get("topic", records[0]["topic"]),
                "status": "active", "trust_score": corroboration_boost,
                "created_at": self._now(), "first_observed_at": first_observed
            })

            # Soft-delete originals
            for r in records:
                self._conn.execute(
                    "UPDATE memories SET status='deleted', compacted_into=?, updated_at=? WHERE uuid=?",
                    (new_uuid, self._now(), r["uuid"]))
                # Remove from Qdrant
                try:
                    self._qdrant.delete(
                        collection_name=self._get_collection(r["data_type"] or "CUSTOM"),
                        points_selector=[_to_qdrant_id(r["uuid"])]
                    )
                except Exception:
                    pass

            self._conn.commit()
            return {
                "merged_uuid": new_uuid,
                "original_uuids": uuids,
                "similarity": group.get("similarity", 0),
                "corroboration_boost": corroboration_boost
            }

        except Exception as e:
            logger.debug("compact: insert failed: %s", e)
            return None

    def _llm_merge(self, records):
        """Feed records to LLM for merge, parse JSON output."""
        parts = []
        parts.append("Merge the following {} overlapping memory records into a single consolidated record.".format(len(records)))
        parts.append("Preserve all unique information. Resolve contradictions using timestamps and specificity.")
        parts.append('Output JSON: {"content": "...", "summary": "...", "keywords": [...], "topic": "..."}')
        parts.append("")
        parts.append("Records:")
        for r in records:
            parts.append("[{}] created={} trust={} topic={}".format(
                r["uuid"][:8], r["created_at"], r["trust_score"], r["topic"] or "none"))
            parts.append("  content: {}".format((r["content"] or "")[:300]))
            parts.append("")
        parts.append("RULES:")
        parts.append("- TEMPORAL CONFLICTS: If records contradict on a mutable fact (location, job, project), treat the most recent record as the source of truth.")
        parts.append("- DETAIL CONFLICTS: If records do not contradict but vary in detail, absorb the subset into the most specific claim.")
        parts.append('- NEGATIVE CONSTRAINTS: You MUST explicitly preserve all negative user preferences (e.g., "Do not use lists", "Never...").')
        parts.append("- Keywords: Union of all keywords, deduplicated.")
        parts.append("- Use the most specific topic.")
        parts.append("Output ONLY valid JSON, no markdown, no explanation.")
        prompt = "\n".join(parts)

        # Call LLM (use configured layer3 endpoint)
        import httpx
        provider_cfg = self._config.get("layer3_provider_config", {})
        base_url = provider_cfg.get("base_url", self._config.get("layer3_base_url", ""))
        api_key = provider_cfg.get("api_key", self._config.get("layer3_api_key", ""))
        model = self._config.get("layer3_model", "")

        if not base_url:
            raise ValueError("layer3_base_url not configured")

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = "Bearer " + api_key

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a memory consolidation assistant. Merge overlapping records into one. Return ONLY valid JSON, no markdown, no explanation."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 1000,
        }
        # Disable reasoning for merge calls — we need direct JSON output
        reasoning_effort = self._config.get("layer3_reasoning_effort", "none")
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort

        response = httpx.post(
            "{}/chat/completions".format(base_url),
            json=payload,
            headers=headers,
            timeout=60.0
        )
        response.raise_for_status()

        msg = response.json()["choices"][0]["message"]
        content = msg.get("content") or msg.get("reasoning") or ""
        content = content.strip()
        if not content:
            raise ValueError("LLM returned empty content for merge")
        # Strip markdown code blocks if present
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        return json.loads(content)


    def purge(self, purge_deleted: bool = True, purge_archived: bool = True,
              min_age_hours: Optional[float] = None, vacuum: bool = True) -> dict:
        """Permanently delete soft-deleted and/or archived records.

        Args:
            purge_deleted: Delete records with status='deleted'
            purge_archived: Delete records with status='archived'
            min_age_hours: Minimum age in hours before purging. Defaults:
                24h for deleted, 168h (7d) for archived.
            vacuum: Vacuum SQLite after purge to reclaim disk space.

        Returns:
            dict with deleted_count, archived_count, qdrant_removed,
            space_freed_bytes, vacuumed
        """
        now_ts = datetime.now(timezone.utc)

        # Default age thresholds
        deleted_age = min_age_hours if min_age_hours is not None else 24.0
        archived_age = min_age_hours if min_age_hours is not None else 168.0  # 7 days

        deleted_cutoff = (now_ts - timedelta(hours=deleted_age)).isoformat()
        archived_cutoff = (now_ts - timedelta(hours=archived_age)).isoformat()

        deleted_count = 0
        archived_count = 0
        qdrant_removed = 0

        # 1. Purge soft-deleted records
        if purge_deleted:
            to_delete = self._conn.execute("""
                SELECT uuid, data_type FROM memories
                WHERE status = 'deleted' AND updated_at < ?
            """, (deleted_cutoff,)).fetchall()

            for uuid, data_type in to_delete:
                dt = (data_type or "CUSTOM")
                collection = self._get_collection(dt)
                # Remove from Qdrant
                try:
                    self._qdrant.delete(
                        collection_name=collection,
                        points_selector=[_to_qdrant_id(uuid)],
                    )
                    qdrant_removed += 1
                except Exception:
                    pass
                deleted_count += 1

            self._conn.execute("DELETE FROM memories WHERE status = 'deleted' AND updated_at < ?",
                               (deleted_cutoff,))

        # 2. Purge archived records
        if purge_archived:
            to_archive = self._conn.execute("""
                SELECT uuid, data_type FROM memories
                WHERE status = 'archived' AND updated_at < ?
            """, (archived_cutoff,)).fetchall()

            for uuid, data_type in to_archive:
                dt = (data_type or "CUSTOM")
                collection = self._get_collection(dt)
                # Remove from Qdrant
                try:
                    self._qdrant.delete(
                        collection_name=collection,
                        points_selector=[_to_qdrant_id(uuid)],
                    )
                    qdrant_removed += 1
                except Exception:
                    pass
                archived_count += 1

            self._conn.execute("DELETE FROM memories WHERE status = 'archived' AND updated_at < ?",
                               (archived_cutoff,))

        # 3. Commit and get space freed estimate
        self._conn.commit()

        # Estimate space freed (page count before vs after)
        space_freed = 0
        if vacuum and (deleted_count > 0 or archived_count > 0):
            # Get page count before vacuum
            before = self._conn.execute("PRAGMA page_count").fetchone()[0]
            page_size = self._conn.execute("PRAGMA page_size").fetchone()[0]

            # Vacuum to reclaim space
            try:
                self._conn.execute("VACUUM")
                self._conn.execute("VACUUM")
            except Exception as e:
                logger.debug("Vacuum failed: %s", e)

            # Get page count after vacuum
            after = self._conn.execute("PRAGMA page_count").fetchone()[0]
            space_freed = (before - after) * page_size

        logger.info("Purge complete: deleted=%d, archived=%d, qdrant_removed=%d, space_freed=%d bytes",
                    deleted_count, archived_count, qdrant_removed, space_freed)

        return {
            "deleted_count": deleted_count,
            "archived_count": archived_count,
            "qdrant_removed": qdrant_removed,
            "space_freed_bytes": max(0, space_freed),
            "vacuumed": vacuum and (deleted_count > 0 or archived_count > 0),
        }

    # ---- State.db helpers (session metadata from Hermes) ----------------

    def _state_db_path(self) -> str:
        """Resolve path to Hermes state.db for the current profile.

        state.db lives at <real_home>/.hermes/profiles/<profile>/state.db.
        Uses pwd.getpwuid() for real home — Hermes sets HOME to profile dir.
        Returns empty string if profile_name is not set.
        """
        if not self._profile_name:
            return ""
        real_home = pwd.getpwuid(os.getuid()).pw_dir
        return os.path.join(
            real_home, ".hermes", "profiles", self._profile_name, "state.db"
        )

    def _open_state_db(self):
        """Open a read-only connection to state.db. Returns None if unavailable."""
        path = self._state_db_path()
        if not path or not os.path.exists(path):
            return None
        try:
            conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            conn.execute("PRAGMA query_only = TRUE")
            return conn
        except Exception as e:
            logger.debug("Failed to open state.db: %s", e)
            return None

    def get_session_info(self, session_id: str) -> Optional[dict]:
        """Query state.db for session metadata.

        Returns dict with id, title, parent_session_id, end_reason,
        message_count, tool_call_count, output_tokens, estimated_cost_usd,
        started_at, ended_at — or None if session not found / state.db unavailable.
        """
        conn = self._open_state_db()
        if conn is None:
            return None
        try:
            row = conn.execute(
                "SELECT id, title, parent_session_id, end_reason, message_count, "
                "tool_call_count, output_tokens, estimated_cost_usd, started_at, ended_at "
                "FROM sessions WHERE id = ? LIMIT 1",
                (session_id,),
            ).fetchone()
            if row is None:
                return None
            return {
                "id": row[0],
                "title": row[1],
                "parent_session_id": row[2],
                "end_reason": row[3],
                "message_count": row[4],
                "tool_call_count": row[5],
                "output_tokens": row[6],
                "estimated_cost_usd": row[7],
                "started_at": row[8],
                "ended_at": row[9],
            }
        except Exception as e:
            logger.debug("get_session_info failed: %s", e)
            return None
        finally:
            conn.close()

    def get_session_chain(self, session_id: str) -> List[str]:
        """Walk parent_session_id chain. Returns [root, ..., current].

        Uses recursive CTE for a single query. Empty list on failure.
        """
        conn = self._open_state_db()
        if conn is None:
            return []
        try:
            rows = conn.execute(
                "WITH RECURSIVE chain AS ("
                "  SELECT id, parent_session_id, started_at FROM sessions WHERE id = ?"
                "  UNION ALL"
                "  SELECT s.id, s.parent_session_id, s.started_at "
                "  FROM sessions s JOIN chain ON s.id = chain.parent_session_id"
                ") SELECT id FROM chain ORDER BY started_at",
                (session_id,),
            ).fetchall()
            return [r[0] for r in rows]
        except Exception as e:
            logger.debug("get_session_chain failed: %s", e)
            return []
        finally:
            conn.close()

    def get_compaction_summary(self, session_id: str) -> Optional[str]:
        """Get [CONTEXT COMPACTION] message from session's first user message.

        Returns the full content string, or None if not found.
        """
        conn = self._open_state_db()
        if conn is None:
            return None
        try:
            row = conn.execute(
                "SELECT content FROM messages "
                "WHERE session_id = ? AND role = 'user' AND content LIKE '%%CONTEXT COMPACTION%%' "
                "ORDER BY id ASC LIMIT 1",
                (session_id,),
            ).fetchone()
            if row is None:
                return None
            return row[0]
        except Exception as e:
            logger.debug("get_compaction_summary failed: %s", e)
            return None
        finally:
            conn.close()

    def classify_session(self, message_count: Optional[int],
                         tool_call_count: Optional[int],
                         output_tokens: Optional[int],
                         end_reason: Optional[str]) -> str:
        """Classify session importance: skip, medium, important.

        'skip'  — trivial Q&A, health checks (<=4 msgs, <=1 tool, <=2000 tokens)
        'medium' — real conversation (>=10 msgs OR >=5 tools OR >=5000 tokens)
        'important' — compressed session or long-running
        """
        mc = message_count or 0
        tc = tool_call_count or 0
        ot = output_tokens or 0

        if mc <= 4 and tc <= 1 and ot <= 2000:
            return "skip"
        if end_reason == "compression":
            return "important"
        if mc >= 10 or tc >= 5 or ot >= 5000:
            return "important"
        return "medium"

    def build_session_name(self, session_id: str) -> str:
        """Build 'UUID :: title' session name from state.db.

        Falls back to raw UUID if state.db unavailable or title is None.
        Escapes '::' in title to preserve format.
        """
        info = self.get_session_info(session_id)
        if info and info.get("title"):
            title = info["title"].replace(" :: ", " : ")
            return f"{session_id} :: {title}"
        return session_id

    def export_memories(self, fmt: str = "json",
                        status: str = "active",
                        data_type: str = None,
                        topic: str = None,
                        scope: str = None,
                        profile_name: str = None,
                        cross_profile: bool = False) -> str:
        """Export memories as JSON or Markdown text.

        When cross_profile=True, queries all discovered profile DBs.
        When profile_name is set, queries only that profile's DB.

        Returns the serialized export string (JSON or Markdown).
        """
        where_parts = []
        params = []

        if status and status != "all":
            where_parts.append("status = ?")
            params.append(status)
        if data_type:
            where_parts.append("data_type = ?")
            params.append(data_type)
        if topic:
            where_parts.append("topic = ?")
            params.append(topic)
        if scope:
            where_parts.append("scope = ?")
            params.append(scope)

        base_where = " AND ".join(where_parts) if where_parts else "1=1"
        columns = (
            "uuid, content, summary, keywords, topic, scope, data_type, "
            "data_id, session_name, sensitivity, source, source_url, "
            "created_at, updated_at, ttl, status, trust_score, "
            "reference_count, priority, metadata"
        )

        def _query_one(db_path, prof):
            import sqlite3 as _sqlite3
            conn = _sqlite3.connect(db_path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            try:
                cursor = conn.execute(
                    f"SELECT {columns} FROM memories WHERE {base_where} "
                    f"ORDER BY created_at",
                    params,
                )
                rows = cursor.fetchall()
                cols = [d[0] for d in cursor.description]
                records = []
                for row in rows:
                    rec = dict(zip(cols, row))
                    rec["profile_name"] = prof
                    for jf in ("keywords", "metadata"):
                        try:
                            rec[jf] = json.loads(rec.get(jf, "null")) or {}
                        except (json.JSONDecodeError, TypeError):
                            rec[jf] = {}
                    records.append(rec)
                return records
            finally:
                conn.close()

        # Determine which DBs to query
        if cross_profile or (profile_name and profile_name != self._profile_name):
            all_dbs = self._discover_profile_dbs()
            if profile_name:
                all_dbs = {profile_name: all_dbs.get(profile_name, "")}
        else:
            all_dbs = {self._profile_name or "default": self._db_path}

        all_records = []
        for prof, db_path in all_dbs.items():
            if not db_path or not os.path.exists(db_path):
                continue
            try:
                all_records.extend(_query_one(db_path, prof))
            except Exception as e:
                logger.warning("Export failed for profile %s: %s", prof, e)

        # Serialize
        export_data = {
            "exported_at": self._now(),
            "source": "hermes-layered-memory",
            "version": "1.0",
            "record_count": len(all_records),
            "records": all_records,
        }

        if fmt == "md":
            lines = [
                "# Layered Memory Export",
                "",
                f"- **Exported:** {export_data['exported_at']}",
                f"- **Records:** {len(all_records)}",
                "",
            ]
            for rec in all_records:
                lines.append(f"## {rec.get('topic', 'untitled') or 'Untitled'}")
                lines.append(f"- **UUID:** {rec['uuid']}")
                lines.append(f"- **Data type:** {rec.get('data_type', 'CUSTOM')}")
                lines.append(f"- **Data ID:** {rec.get('data_id', '—')}")
                lines.append(f"- **Trust score:** {rec.get('trust_score', 0.5)}")
                lines.append(f"- **Priority:** {rec.get('priority', 0)}")
                lines.append(f"- **Created:** {rec.get('created_at', '—')}")
                lines.append(f"- **Updated:** {rec.get('updated_at', '—')}")
                lines.append(f"- **Status:** {rec.get('status', 'active')}")
                if rec.get("session_name"):
                    lines.append(f"- **Session:** {rec['session_name']}")
                if rec.get("profile_name"):
                    lines.append(f"- **Profile:** {rec['profile_name']}")
                kw = rec.get("keywords", [])
                if kw:
                    lines.append(f"- **Keywords:** {', '.join(kw)}")
                lines.append("")
                lines.append(f"_{rec.get('summary', '')}_")
                lines.append("")
                lines.append(rec.get("content", ""))
                lines.append("")
            return "\n".join(lines)

        return json.dumps(export_data, indent=2, ensure_ascii=False, default=str)

    def import_memories(self, data: str, mode: str = "skip_existing",
                        target_status: str = "active") -> dict:
        """Import memories from JSON data.

        Args:
            data: JSON string (from export) or Markdown text.
                  If a file path ending in .json is provided, reads from file.
            mode: 'skip_existing' (default) — skip records whose UUID already
                  exists; 'overwrite' — update existing records;
                  'new_uuid' — generate new UUIDs even for duplicates.
            target_status: Status for imported records (default: 'active').

        Returns dict with imported, skipped, failed counts.
        """
        results = {"imported": 0, "skipped": 0, "failed": 0, "errors": []}

        # Read from file if path provided
        if isinstance(data, str) and (
            data.endswith(".json") or data.endswith(".md")
        ):
            try:
                with open(data, "r", encoding="utf-8") as f:
                    data = f.read()
            except Exception as e:
                return {"imported": 0, "skipped": 0, "failed": 1,
                        "errors": [f"Failed to read file: {e}"]}

        # Parse JSON
        records = []
        try:
            export_data = json.loads(data)
            if isinstance(export_data, dict):
                records = export_data.get("records", [])
            elif isinstance(export_data, list):
                records = export_data
        except (json.JSONDecodeError, TypeError):
            return {"imported": 0, "skipped": 0, "failed": 1,
                    "errors": ["Input is not valid JSON"]}

        if not records:
            return results

        existing_uuids = set()
        if mode in ("skip_existing", "new_uuid", "overwrite"):
            try:
                cursor = self._conn.execute("SELECT uuid FROM memories")
                existing_uuids = {row[0] for row in cursor.fetchall()}
            except Exception:
                pass

        for rec in records:
            try:
                uuid = rec.get("uuid")
                if not uuid:
                    results["failed"] += 1
                    results["errors"].append("Record missing uuid")
                    continue

                # Skip existing
                if mode == "skip_existing" and uuid in existing_uuids:
                    results["skipped"] += 1
                    continue

                # Check if exists for overwrite
                if mode == "overwrite":
                    if uuid in existing_uuids:
                        existing = self._conn.execute(
                            "SELECT content FROM memories WHERE uuid = ?",
                            (uuid,),
                        ).fetchone()
                        if existing:
                            self._conn.execute(
                                "UPDATE memories SET content = ?, updated_at = ?, "
                                "status = ? WHERE uuid = ?",
                                (rec.get("content", ""), self._now(),
                                 target_status, uuid),
                            )
                            self._conn.commit()
                            results["imported"] += 1
                            continue

                # New UUID mode
                if mode == "new_uuid" and uuid in existing_uuids:
                    uuid = uuid_mod.uuid4().hex

                content = rec.get("content", "")
                if not content:
                    results["failed"] += 1
                    results["errors"].append(f"Record {uuid} has no content")
                    continue

                # Build keyword JSON
                kw = rec.get("keywords", [])
                if isinstance(kw, list):
                    kw_json = json.dumps(kw)
                else:
                    kw_json = "[]"

                meta = rec.get("metadata", {})
                if isinstance(meta, dict):
                    meta_json = json.dumps(meta)
                else:
                    meta_json = "{}"

                self._conn.execute(
                    """INSERT OR IGNORE INTO memories (
                        uuid, content, summary, keywords, topic, scope,
                        data_type, data_id, session_name, sensitivity,
                        source, source_url, created_at, updated_at, ttl,
                        status, trust_score, reference_count, priority,
                        metadata, embedding)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, NULL)""",
                    (uuid, content,
                     rec.get("summary", content[:100]), kw_json,
                     rec.get("topic"), rec.get("scope", "personal"),
                     rec.get("data_type", "CUSTOM"), rec.get("data_id"),
                     rec.get("session_name"), rec.get("sensitivity", 0),
                     rec.get("source", "import"), rec.get("source_url"),
                     rec.get("created_at", self._now()),
                     rec.get("updated_at", self._now()),
                     rec.get("ttl"), target_status,
                     rec.get("trust_score", 0.5),
                     rec.get("priority", 0), meta_json),
                )

                results["imported"] += 1

            except Exception as e:
                results["failed"] += 1
                err_msg = f"Failed to import {rec.get('uuid', '?')}: {e}"
                logger.debug(err_msg)
                results["errors"].append(err_msg)

        self._conn.commit()

        # Rebuild Qdrant for new records
        if results["imported"] > 0:
            try:
                self.rebuild()
            except Exception as e:
                logger.debug("Post-import rebuild failed: %s", e)

        logger.info("import: %d imported, %d skipped, %d failed",
                     results["imported"], results["skipped"], results["failed"])
        return results

    def decay(self, min_age_days: int = 30,
              max_age_days: int = 365,
              decay_rate: float = 0.05,
              min_score: float = 0.1) -> dict:
        """Apply confidence decay to memories older than min_age_days.

        Decay formula: new_score = max(min_score,
            trust_score * (1 - decay_rate * min(age_days / max_age_days, 1)))

        Records with high reference_count (>10) are decayed more slowly
        (decay_rate * 0.3) — frequently retrieved facts are reinforced.
        Records with explicit priority > 0 are not decayed.
        Records with trust_score > 0.9 are not decayed (high confidence).

        Args:
            min_age_days: Only decay memories older than this (default: 30).
            max_age_days: Age at which full decay effect applies (default: 365).
            decay_rate: Maximum decay factor per full age cycle (default: 0.05).
            min_score: Minimum trust_score after decay (default: 0.1).

        Returns dict with decayed, unchanged counts.
        """
        now = self._now()
        cutoff = (datetime.fromisoformat(now) -
                   timedelta(days=min_age_days)).isoformat()

        cursor = self._conn.execute(
            "SELECT uuid, trust_score, created_at, reference_count, priority "
            "FROM memories WHERE status = 'active' "
            "AND created_at < ? "
            "AND trust_score <= 0.9 "
            "AND priority = 0",
            (cutoff,),
        )
        rows = cursor.fetchall()

        results = {"decayed": 0, "unchanged": 0, "details": []}

        for row in rows:
            uuid, score, created_at, ref_count, priority = row

            # Age in days
            try:
                age_days = (datetime.fromisoformat(now) -
                            datetime.fromisoformat(created_at)).days
            except (ValueError, TypeError):
                continue

            if age_days <= min_age_days:
                results["unchanged"] += 1
                continue

            # Normalized age factor [0, 1]
            age_factor = min(age_days / max_age_days, 1.0)

            # Effective decay rate — reinforce frequently retrieved
            effective_rate = decay_rate
            if ref_count and ref_count > 10:
                effective_rate *= 0.3  # 70% slower decay

            # Apply decay
            new_score = max(min_score,
                            score * (1 - effective_rate * age_factor))
            new_score = round(new_score, 4)

            if new_score < score:
                self._conn.execute(
                    "UPDATE memories SET trust_score = ?, updated_at = ? "
                    "WHERE uuid = ?",
                    (new_score, now, uuid),
                )
                results["decayed"] += 1
                results["details"].append({
                    "uuid": uuid[:8],
                    "old_score": score,
                    "new_score": new_score,
                    "age_days": age_days,
                    "ref_count": ref_count,
                })
            else:
                results["unchanged"] += 1

        self._conn.commit()
        logger.info("decay: %d decayed, %d unchanged (min_age=%d days)",
                     results["decayed"], results["unchanged"], min_age_days)
        return results


