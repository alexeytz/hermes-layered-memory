"""store module — methods of LayeredBackend, kept in their own file.

These are plain functions whose first argument is the backend instance. They
are bound onto `LayeredBackend` in its class body (see backend/backend.py), so
`self` inside them is the backend and every call site is navigable. Shared
primitives — logger, _setting, the embedding helpers — come from backend/core.py
by direct import; they used to be fetched by name out of sys.modules, which no
editor or type checker could follow.
"""

from __future__ import annotations
import json
import re
import sqlite3

import builtins as _builtins
import sys as _sys
from typing import Any, Dict, List, Optional

from .constants import (EXTRACTION_HOOKS, HLM_TEST_MARKER, HISTORY_MAX_SIZE,
                        HISTORY_ROTATE_KEEP, VALID_DATA_TYPES,
                        UPDATE_ALLOWED_FIELDS,
                        require_str_filters,
                        _validate_config_value)
from datetime import datetime, timedelta, timezone

import array
import collections
import contextlib
import os
import threading
import time
import uuid as uuid_mod


from .core import (
    MAX_CONTENT_CHARS,
    MAX_METADATA_CHARS,
    MAX_FIELD_CHARS,
    MAX_DATA_ID_CHARS,
    MAX_SESSION_NAME_CHARS,
    MAX_SCOPE_CHARS,
    MAX_SOURCE_URL_CHARS,
    MAX_KEYWORDS,
    _EMBED_NULL,
    _get_embedding_fn,
    _is_lock_error,
    _pack_embedding,
    _SELF_AUTHORED_SOURCES,
    _resolve_data_type,
    _retry_on_lock,
    _to_qdrant_id,
    profile_claim as _profile_claim,
    profile_release as _profile_release,
    _unpack_embedding,
    _wrap_untrusted_text,
    logger,
)


def _init_db(self):
    self._conn_local = threading.local()
    self._conn_lock = threading.Lock()  # for safe shutdown
    self._get_conn().execute("PRAGMA journal_mode=WAL")
    self._get_conn().execute("PRAGMA busy_timeout=5000")
    self._get_conn().executescript("""
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
            -- Vestigial. Bound to literal NULL by add() and literal 0 by
            -- import, and SELECTed / ORDERed BY by nothing, in either front
            -- end or any script. Left in place because dropping a column costs
            -- a migration for no gain, but do not "fix" it by populating it:
            -- that adds a writer with no reader, which is the mirror image of
            -- the backlinks defect (declared, read on every retrieve, written
            -- by nothing). If a real ordering key is ever needed, decide what
            -- reads it first.
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
        -- memories_fts is an external-content table (content='memories'), so
        -- plain UPDATE/DELETE against it are invalid: FTS5 has to re-read the
        -- old row to remove its terms, and by the time an AFTER trigger runs
        -- the content table already holds the new values. UPDATE raised
        -- "database disk image is malformed" (killing every background
        -- enrichment write) and DELETE silently left stale terms behind, so
        -- deleted text stayed searchable. The 'delete' command below hands
        -- FTS5 the old values explicitly, which is the documented pattern.
        CREATE TRIGGER IF NOT EXISTS fts5_au AFTER UPDATE ON memories BEGIN
            INSERT INTO memories_fts(memories_fts, rowid, content, summary,
                                     keywords, topic, data_id)
            VALUES('delete', old.rowid, old.content, old.summary,
                   old.keywords, old.topic, old.data_id);
            INSERT INTO memories_fts(rowid, content, summary, keywords, topic, data_id)
            VALUES (new.rowid, new.content, new.summary, new.keywords,
                    new.topic, new.data_id);
        END;
        CREATE TRIGGER IF NOT EXISTS fts5_ad AFTER DELETE ON memories BEGIN
            INSERT INTO memories_fts(memories_fts, rowid, content, summary,
                                     keywords, topic, data_id)
            VALUES('delete', old.rowid, old.content, old.summary,
                   old.keywords, old.topic, old.data_id);
        END;
    """)
    # Migration: add llm_review columns if missing (v7)
    try:
        self._get_conn().execute("ALTER TABLE memories ADD COLUMN llm_review_status TEXT")
        self._get_conn().execute("ALTER TABLE memories ADD COLUMN llm_reviewed_at TEXT")
        self._get_conn().commit()
        logger.info("Migration: added llm_review_status and llm_reviewed_at columns")
    except Exception:
        pass  # Columns already exist
    # Migration: add compaction columns if missing (v8)
    try:
        self._get_conn().execute("ALTER TABLE memories ADD COLUMN compacted_into TEXT")
        self._get_conn().execute("ALTER TABLE memories ADD COLUMN first_observed_at TEXT")
        self._get_conn().commit()
        logger.info("Migration: added compacted_into and first_observed_at columns")
    except Exception:
        pass  # Columns already exist
    # Migration: add session_tags table for persistent tag registry (v9)
    try:
        self._get_conn().execute(
            "CREATE TABLE IF NOT EXISTS session_tags ("
            "session_id TEXT NOT NULL,"
            "tag INTEGER NOT NULL,"
            "uuid TEXT NOT NULL,"
            "PRIMARY KEY (session_id, tag),"
            "UNIQUE(session_id, uuid)"
            ")"
        )
        self._get_conn().commit()
        logger.debug("Migration: session_tags table ready (v9)")
    except Exception:
        pass  # Table already exists
    self._get_conn().commit()
    # Migration: add protected column (v10) — separate from priority
    try:
        self._get_conn().execute("ALTER TABLE memories ADD COLUMN protected INTEGER DEFAULT 0")
        self._get_conn().commit()
        logger.info("Migration: added protected column (v10)")
    except Exception:
        pass  # Column already exists
    # Migration: replace the broken external-content FTS5 triggers (v15).
    # Existing databases carry the plain-table UPDATE/DELETE form, which
    # errors on update and rots the index on delete. Dropping and recreating
    # is safe — triggers hold no state — and the index is rebuilt once so
    # rows that already drifted are corrected.
    try:
        existing = {r[0] for r in self._get_conn().execute(
            "SELECT name FROM sqlite_master WHERE type='trigger' "
            "AND name IN ('fts5_au','fts5_ad')")}
        needs_fix = False
        for name in existing:
            sql = self._get_conn().execute(
                "SELECT sql FROM sqlite_master WHERE name = ?", (name,)).fetchone()
            if sql and "'delete'" not in (sql[0] or ""):
                needs_fix = True
        if needs_fix:
            self._get_conn().execute("DROP TRIGGER IF EXISTS fts5_au")
            self._get_conn().execute("DROP TRIGGER IF EXISTS fts5_ad")
            self._get_conn().execute("""
                CREATE TRIGGER fts5_au AFTER UPDATE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, content, summary,
                                             keywords, topic, data_id)
                    VALUES('delete', old.rowid, old.content, old.summary,
                           old.keywords, old.topic, old.data_id);
                    INSERT INTO memories_fts(rowid, content, summary, keywords, topic, data_id)
                    VALUES (new.rowid, new.content, new.summary, new.keywords,
                            new.topic, new.data_id);
                END""")
            self._get_conn().execute("""
                CREATE TRIGGER fts5_ad AFTER DELETE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, content, summary,
                                             keywords, topic, data_id)
                    VALUES('delete', old.rowid, old.content, old.summary,
                           old.keywords, old.topic, old.data_id);
                END""")
            self._get_conn().execute(
                "INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
            self._get_conn().commit()
            logger.info("Migration: repaired FTS5 triggers and rebuilt the "
                        "search index (v15)")
    except Exception as e:
        logger.warning("FTS5 trigger migration failed: %s", e)

    # Migration: supersession chain (v14).
    # Facts change. Without these, the only way to record a new value was
    # to overwrite the old one (losing history) or to store both (leaving
    # the reader to guess which is current). superseded_by points at the
    # replacement; retrieval hides superseded records by default.
    try:
        self._get_conn().execute("ALTER TABLE memories ADD COLUMN superseded_by TEXT")
        self._get_conn().execute("ALTER TABLE memories ADD COLUMN superseded_at TEXT")
        self._get_conn().commit()
        logger.info("Migration: added superseded_by and superseded_at columns (v14)")
    except Exception:
        pass  # Columns already exist
    try:
        self._get_conn().execute(
            "CREATE INDEX IF NOT EXISTS idx_mem_superseded ON memories(superseded_by)")
        self._get_conn().commit()
    except Exception:
        pass
    # Migration: add maintenance_state table (v11)
    try:
        self._get_conn().execute(
            "CREATE TABLE IF NOT EXISTS maintenance_state ("
            "key TEXT PRIMARY KEY,"
            "value TEXT NOT NULL"
            ")"
        )
        self._get_conn().commit()
        logger.debug("Migration: maintenance_state table ready (v11)")
    except Exception:
        pass  # Table already exists
    # Migration: add runtime_config table for DB-backed config (v12)
    try:
        self._get_conn().execute(
            "CREATE TABLE IF NOT EXISTS runtime_config ("
            "key TEXT PRIMARY KEY,"
            "value TEXT NOT NULL,"
            "updated_at TEXT NOT NULL"
            ")"
        )
        self._get_conn().commit()
        logger.debug("Migration: runtime_config table ready (v12)")
    except Exception:
        pass  # Table already exists
    # Migration: add taxonomy table for LLM-registered data_types/data_ids (v13)
    #
    # The primary key was `name` alone, but a name is only unique *within* a
    # kind — `register_taxonomy` writes `INSERT OR REPLACE`, so registering
    # `("OBSIDIAN", kind="data_id")` after `("OBSIDIAN", kind="data_type")`
    # silently replaced the type. Nothing logged it, and the damage was
    # deferred: in-process `_collection_map` kept the stale entry so writes
    # still routed, but `_load_runtime_config` re-adds only `kind='data_type'`
    # rows, so after a restart `add(data_type="OBSIDIAN")` raised "not a
    # registered type". Write acceptance diverged across a restart.
    #
    # (name, kind) is the real key. The existing key is a strict subset, so
    # every stored row is already valid under the new one and the copy cannot
    # collide. 2026-08-24 audit, M5.
    try:
        self._get_conn().execute(
            "CREATE TABLE IF NOT EXISTS taxonomy ("
            "name TEXT NOT NULL,"
            "kind TEXT NOT NULL,"
            "collection TEXT,"
            "description TEXT,"
            "created_at TEXT NOT NULL,"
            "PRIMARY KEY (name, kind)"
            ")"
        )
        # Rebuild an older table that still keys on `name` alone. Detected from
        # the schema rather than a version counter, because the table is
        # created with IF NOT EXISTS and a v13 database that never registered
        # anything is indistinguishable from a fresh one.
        _sql = self._get_conn().execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='taxonomy'"
        ).fetchone()
        if _sql and "PRIMARY KEY (name, kind)" not in (_sql[0] or ""):
            self._get_conn().executescript(
                "CREATE TABLE taxonomy_new ("
                "name TEXT NOT NULL, kind TEXT NOT NULL, collection TEXT, "
                "description TEXT, created_at TEXT NOT NULL, "
                "PRIMARY KEY (name, kind));"
                "INSERT OR IGNORE INTO taxonomy_new "
                "SELECT name, kind, collection, description, created_at FROM taxonomy;"
                "DROP TABLE taxonomy;"
                "ALTER TABLE taxonomy_new RENAME TO taxonomy;"
            )
            logger.info("Migration: taxonomy re-keyed on (name, kind) (v13a)")
        self._get_conn().commit()
        logger.debug("Migration: taxonomy table ready (v13)")
    except Exception:
        pass  # Table already exists

    # Migration: move link data out of the metadata blob into the typed column
    # (v15; v14 is the supersession chain above). `_extract_wikilinks` has always written to `metadata["wikilinks"]`,
    # which nothing reads, while `backlinks` — read on every retrieve, fenced on
    # output, merged during compaction — stayed empty everywhere. Records
    # ingested before 0.7.29 carry their links in the wrong place, so move them
    # rather than leaving the graph empty for everything imported so far.
    #
    # Idempotent and narrow: only rows that have wikilinks in metadata *and* no
    # backlinks yet are touched, so re-running changes nothing and a row whose
    # backlinks were set some other way is left alone.
    try:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT uuid, metadata FROM memories "
            "WHERE metadata LIKE '%wikilinks%' "
            "AND (backlinks IS NULL OR backlinks IN ('', '[]'))"
        ).fetchall()
        moved = 0
        for uuid_, meta_json in rows:
            try:
                meta = json.loads(meta_json) if meta_json else {}
            except (json.JSONDecodeError, TypeError):
                continue
            links = meta.pop("wikilinks", None)
            # `_builtins.list`: this module defines a `list()` function at
            # module scope, so the bare name is not the type here. Written the
            # obvious way first, and the migration failed with "isinstance()
            # arg 2 must be a type" — into a `except Exception` that logged at
            # DEBUG and moved on, so it silently did nothing at all.
            if not isinstance(links, _builtins.list) or not links:
                continue
            conn.execute(
                "UPDATE memories SET backlinks = ?, metadata = ? WHERE uuid = ?",
                (json.dumps([str(l) for l in links]), json.dumps(meta), uuid_))
            moved += 1
        if moved:
            conn.commit()
            logger.info("Migration: moved wikilinks into backlinks for %d "
                        "record(s) (v15)", moved)
        else:
            logger.debug("Migration: backlinks column current (v15)")
    except Exception as e:
        # Logged at WARNING, not DEBUG: a migration that silently does nothing
        # is indistinguishable from one that had nothing to do, which is how
        # the shadowing bug above survived its first run.
        logger.warning("Migration v15 (wikilinks -> backlinks) failed: %s", e)


def _load_db_config(self) -> dict:
    """Load runtime config from SQLite (merged with JSON/env config).

    Priority: env vars > JSON file > DB runtime config > defaults.
    DB config is a persistent layer the LLM can modify at runtime.
    """
    db_config = {}
    try:
        for row in self._get_conn().execute(
            "SELECT key, value FROM runtime_config"
        ).fetchall():
            try:
                db_config[row[0]] = json.loads(row[1])
            except Exception:
                db_config[row[0]] = row[1]
    except Exception:
        pass
    if db_config:
        logger.info("Loaded %d runtime config entries from DB", len(db_config))
    return db_config


def _seed_config_defaults(self) -> None:
    """Seed known config keys with defaults if not already present.

    Ensures get_config() returns exists=True for all known keys.

    "If not already present" has to be enforced, not just documented. This
    used to call _save_db_config() unconditionally, which does
    INSERT OR REPLACE *and* assigns self._config[key]. Since it runs from
    _load_runtime_config() — i.e. after _apply_env_overrides() — the effect
    was that HLM_DEDUP_THRESHOLD was overwritten with the default before it
    was ever read, and any value persisted via layered_config(action="set")
    was reset to the default on the next process start. It also meant every
    backend construction rewrote the JSON config file (plus a backup copy
    and a prune), so even `inspect-memory.py` mutated the user's config
    just by opening the database.
    """
    defaults = {
        "dedup_threshold": 0.97,
    }
    try:
        existing = {r[0] for r in self._get_conn().execute(
            "SELECT key FROM runtime_config").fetchall()}
    except Exception:
        return  # table not ready — nothing to seed against
    for key, value in defaults.items():
        if key in existing or key in self._config:
            continue
        try:
            self._save_db_config(key, value)
        except Exception:
            pass  # Already seeded or table not ready


def _load_runtime_config(self) -> None:
    """Load DB runtime config and merge into self._config.

    Called after _init_db(). DB config is lowest priority —
    env vars and JSON file already loaded take precedence.
    Also rebuilds collection_map if DB has custom collections.
    """
    # Seed defaults first (lowest priority — will be overridden by env/JSON)
    self._seed_config_defaults()
    db_config = self._load_db_config()
    if not db_config:
        self._db_config_loaded = True
        return
    # Merge DB config (lowest priority)
    for key, value in db_config.items():
        if key not in self._config:
            self._config[key] = value
    # Rebuild collection_map if DB has custom collections
    # Re-key through _physical_collection: DB config stores *configured*
    # names, and this can run after _init_qdrant has already keyed the map to
    # the embedding model. Without it, DB-backed config silently reverted
    # every write to the unsuffixed collection. Idempotent, so it is also
    # correct when this runs first.
    if "collections" in db_config and "collections" not in self._config:
        self._collection_map = {k: self._physical_collection(v)
                                for k, v in dict(db_config["collections"]).items()}
    self._db_config_loaded = True
    # Load registered taxonomy entries and update collection map
    for entry in self.get_taxonomy(kind="data_type"):
        name = entry["name"]
        if name not in self._collection_map:
            self._collection_map[name] = self._physical_collection(
                entry.get("collection") or self._default_collection)


def _save_db_config(self, key: str, value) -> None:
    """Persist a config key-value to SQLite and sync to JSON file."""
    self._get_conn().execute(
        "INSERT OR REPLACE INTO runtime_config (key, value, updated_at) "
        "VALUES (?, ?, ?)",
        (key, json.dumps(value, default=str),
         self._now()),  # UTC, matching the rest of the codebase
    )
    self._get_conn().commit()
    # Update in-memory config
    self._config[key] = value
    # `collections` is not just a config value — it is the routing table.
    # `register_taxonomy`, the other writer of the same map, re-keys it through
    # `_physical_collection` and updates `_collection_map` immediately; this
    # path wrote `_config` and stopped, so `set_config("collections", ...)`
    # reported success and every write kept going to the old collection until
    # a restart picked the value up in `_load_runtime_config`. Two writers of
    # one map, one of them updating it.
    #
    # Re-keyed through `_physical_collection` for the reason its own docstring
    # gives: DB config stores *configured* names, the map holds physical ones,
    # and the function is idempotent precisely so more than one rebuilder can
    # call it. 2026-08-24 audit, finding 14.
    if key == "collections" and isinstance(value, dict):
        self._collection_map = {k: self._physical_collection(v)
                                for k, v in value.items()}
        logger.info("Collection map rebuilt from config: %d entries",
                    len(self._collection_map))
    # Sync to JSON file
    self._sync_config_to_file()
    logger.info("Config updated: %s = %s", key, json.dumps(value, default=str)[:80])


def _sync_config_to_file(self) -> None:
    """Write merged config to $HERMES_HOME/hermes-layered-memory.json."""
    try:
        from hermes_constants import get_hermes_home
        config_path = get_hermes_home() / "hermes-layered-memory.json"
        # Backup existing file before overwriting — keep only the last 3
        if config_path.exists():
            # UTC, like every other stamp here. A local-time filename sorts
            # wrongly across a DST boundary and cannot be correlated with the
            # UTC `updated_at` on the row the backup was taken for.
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            backup_path = config_path.with_name(
                f"hermes-layered-memory.json.bak.{ts}"
            )
            try:
                import shutil
                import glob
                shutil.copy2(str(config_path), str(backup_path))
                # Prune old backups — keep only the 3 most recent
                backups = sorted(
                    glob.glob(os.path.join(os.path.dirname(str(config_path)),
                              "hermes-layered-memory.json.bak.*")),
                    key=os.path.getmtime,
                    reverse=True,
                )
                for old_backup in backups[3:]:
                    try:
                        os.remove(old_backup)
                    except OSError:
                        pass
            except Exception:
                pass  # backup failure is non-fatal
        # Merge: start with in-memory config (already merged from all layers)
        merged = dict(self._config)
        # Remove non-serializable or env-specific keys from the file.
        #
        # `_config` is the *merged* view, so writing it verbatim baked
        # env-layer values into the file — the lowest-precedence layer of the
        # three. Unset `HLM_MAX_LAYER` a week later and the value it happened
        # to hold during some unrelated `set_config` call is still in the JSON,
        # still applied, and now unattributable: the documented precedence
        # (env > DB > file) is inverted permanently, and nothing says so.
        #
        # This skip list already carried `layer3_provider_config` and
        # `enrich_llm` — the two that had been noticed. `_env_config` is the
        # general form: every key the env pass actually set, recorded where it
        # is set rather than enumerated here, so a new `HLM_*` override cannot
        # be added without this following it. 2026-08-24 audit, finding 10.
        for skip_key in ("layer3_provider_config", "enrich_llm"):
            merged.pop(skip_key, None)
        for skip_key in getattr(self, "_env_config", {}):
            merged.pop(skip_key, None)
        config_path.write_text(json.dumps(merged, indent=2, default=str), encoding="utf-8")
    except Exception as e:
        logger.warning("Config sync to file failed: %s", e)


def set_config(self, key: str, value) -> dict:
    """Runtime config setter — persists to SQLite + JSON file.

    Args:
        key: config key (e.g., 'max_layer', 'layer3_mode')
        value: config value (any JSON-serializable type)

    Returns:
        dict with status and current config value, or an `error` key if the
        value is rejected.

    Validation happens here rather than in a caller. It used to live only in
    the plugin's layered_config handler, so anything reaching set_config by
    another route persisted unchecked — notably the MCP server, which calls
    this directly and has no authentication. `max_layer=99` forces L3+L4 on
    every subsequent query, `dedup_threshold="abc"` raises on every write, and
    `scoring="oops"` raises in every L2 fusion; all three survive a restart and
    affect the Hermes plugin sharing the profile.
    """
    problem = _validate_config_value(key, value)
    if problem:
        logger.warning("set_config rejected %s=%r: %s", key, value, problem)
        return {"error": problem, "key": key}
    self._save_db_config(key, value)
    return {"status": "updated", "key": key, "value": value}


def delete_config(self, key: str) -> dict:
    """Delete a custom config key from runtime config.

    Args:
        key: config key to delete

    Returns:
        dict with status
    """
    self._get_conn().execute("DELETE FROM runtime_config WHERE key = ?", (key,))
    self._get_conn().commit()
    # The DELETE removes the *stored* value; the pop used to remove whatever
    # was in the merged view, whichever layer put it there. So
    # `delete_config("max_layer")` with `HLM_MAX_LAYER` set reported
    # `{"status": "deleted"}` and left the process with no max_layer at all,
    # while the variable was still set and would reinstate it at the next
    # restart — behaviour diverging across a restart, which is the failure
    # nothing in a single process can observe.
    #
    # Re-resolve from the env layer instead of popping blind. The remaining
    # layers below env (file, defaults) are not restored here because this
    # function's contract is "remove the runtime override", and a value that
    # was only ever in the file is not something this deleted.
    # 2026-08-24 audit, finding 11.
    self._config.pop(key, None)
    _env = getattr(self, "_env_config", {})
    if key in _env:
        self._config[key] = _env[key]
        logger.info(
            "delete_config: removed the stored %s, but the environment still "
            "sets it — restored the env value %r", key, _env[key])
        self._sync_config_to_file()
        return {"status": "deleted", "key": key,
                "note": "environment still sets this key; the env value is in effect",
                "value": _env[key]}
    self._sync_config_to_file()
    return {"status": "deleted", "key": key}


def _redact_config(config: dict) -> dict:
    """Strip the live LLM API key before a config dict leaves this process.

    `_apply_env_overrides` (backend.py) injects `HLM_LAYER3_API_KEY` into
    `layer3_provider_config.api_key` at construction so the operator does not
    have to duplicate it in the JSON config file. `get_config` returned that
    dict verbatim: `memory_config(action="get")` is policy-open on MCP — no
    admin flag, no auth, because the gate was drawn around *mutation*
    ("writes persist to the config file and change behaviour") — and reads
    were never considered as a data-exposure surface, so the operator's live
    credential for the endpoint this plugin bills came back to any client
    that could reach the port. `_sync_config_to_file` already treats this key
    as non-shareable (it omits the whole `layer3_provider_config` entry from
    the file on every write); this is the read-side counterpart, narrower —
    keeping `base_url`/`model` visible, since only the credential is secret.
    """
    if "layer3_provider_config" not in config:
        return config
    redacted = dict(config)
    pc = redacted["layer3_provider_config"]
    if isinstance(pc, dict) and ("api_key" in pc or "base_url" in pc):
        pc = dict(pc)
        if "api_key" in pc:
            pc["api_key"] = "***redacted***"
        # `base_url` was deliberately left visible — "only the credential is
        # secret" — which is true right up until the credential is *in* the
        # base_url. `http://token@host/v1` is an ordinary way to point at a
        # gateway, and the userinfo component is exactly as secret as the
        # api_key beside it. Only the userinfo is stripped; the host and path
        # stay readable, which is what made keeping base_url worth doing.
        # 2026-08-24 audit, finding 25.
        _url = pc.get("base_url")
        if isinstance(_url, str) and "@" in _url:
            try:
                from urllib.parse import urlsplit, urlunsplit
                parts = urlsplit(_url)
                if parts.username or parts.password:
                    host = parts.hostname or ""
                    if parts.port:
                        host = "%s:%d" % (host, parts.port)
                    pc["base_url"] = urlunsplit(
                        (parts.scheme, "***redacted***@" + host,
                         parts.path, parts.query, parts.fragment))
            except Exception:
                pc["base_url"] = "***redacted***"
        redacted["layer3_provider_config"] = pc
    return redacted


def get_config(self, key: str = None) -> dict:
    """Get current merged config (DB + JSON + env overrides).

    Args:
        key: if specified, return only that key-value pair; otherwise return all config

    Returns:
        dict with either the single key-value pair or all config. The live
        LLM API key, if configured, is never included — see _redact_config.
    """
    safe = _redact_config(self._config)
    if key:
        exists = key in safe
        return {"key": key, "value": safe.get(key), "exists": exists}
    return dict(safe)


def register_taxonomy(self, name: str, kind: str = "data_type",
                      collection: str = None, description: str = None) -> dict:
    """Register a new data_type or data_id in the taxonomy.

    Args:
        name: taxonomy entry name (e.g., 'OBSIDIAN', 'project_x')
        kind: 'data_type' or 'data_id'
        collection: Qdrant collection (for data_type; defaults to default collection)
        description: human-readable description

    Returns:
        dict with status and registration details
    """
    if kind not in ("data_type", "data_id"):
        return {"error": f"invalid kind: {kind} (must be data_type or data_id)"}

    # Naive local time here was the same defect 0.7.40 fixed one line-number
    # away in _save_db_config: a stamp with no offset, stored beside UTC
    # values in every other table. Fixing one instance of a pattern and
    # calling the class handled is what left this one behind.
    ts = self._now()
    # Resolve collection before storing
    resolved_collection = collection or (self._default_collection if kind == "data_type" else None)
    self._get_conn().execute(
        "INSERT OR REPLACE INTO taxonomy (name, kind, collection, description, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (name, kind, resolved_collection, description, ts),
    )
    self._get_conn().commit()

    # If registering a data_type, update collection map
    if kind == "data_type":
        # A caller-supplied collection is a configured name; key it the same
        # way as every other, or a registered data_type writes somewhere the
        # rest of the backend never reads.
        target = self._physical_collection(collection or self._default_collection)
        # Create it now if Qdrant has never seen it. Registration used to map a
        # data_type to a collection that only `_init_qdrant` could create — and
        # that runs at process start — so registering with a new `collection`
        # reported success and the first write for that type failed its upsert,
        # leaving the record in SQLite alone and vector-invisible until a
        # restart plus rebuild. 2026-08-25 bundle04 review (F2).
        self._ensure_qdrant_collection(target)
        self._collection_map[name] = target
        # Auto-sync config — with the **configured** name, not the physical
        # one, because that is what every reader of this key expects.
        #
        # This wrote `dict(self._collection_map)`, so one `register_taxonomy`
        # call replaced the whole persisted map with suffixed names:
        # `memories` became `memories_qwen3-embedding_8b_4096` for every
        # data_type, not only the one being registered. `_save_db_config`'s
        # own comment states the invariant it broke ("DB config stores
        # *configured* names, the map holds physical ones").
        #
        # Driven, because both review rounds that found this called it a
        # documentation mismatch and neither checked the consequence.
        # `_physical_collection` is idempotent for the *same* suffix, which is
        # why nothing broke — but it is idempotent by `endswith`, so a
        # persisted physical name re-keys correctly only while the embedder
        # never changes:
        #
        #     stored "memories_qwen3-embedding_8b_4096", embedder swapped
        #       -> "memories_qwen3-embedding_8b_4096_nomic-embed_v2_768"
        #     stored "memories" (configured), embedder swapped
        #       -> "memories_nomic-embed_v2_768"                    <- correct
        #
        # The first is a collection that exists nowhere — the exact outcome
        # `_physical_collection`'s docstring exists to prevent — and every
        # write for every data_type would land in SQLite alone, vector-
        # invisible until a restart and rebuild. Model-keyed collections exist
        # so that changing the embedder is *safe*; this armed that change to
        # fail, on any profile that had ever registered a taxonomy.
        # 2026-09-14 round 1 bundle04 (F7) / 2026-09-15 round 2 bundle05 (F1).
        # T651.
        self._config.setdefault("collections", {})
        self._config["collections"][name] = collection or self._default_collection
        self._sync_config_to_file()

    logger.info("Taxonomy registered: %s (%s)", name, kind)
    return {
        "status": "registered",
        "name": name,
        "kind": kind,
        "collection": resolved_collection if kind == "data_type" else None,
        "description": description,
    }


def get_surfaced_summaries(self, uuids, limit: int = 40, chars: int = 120) -> list:
    """Compact excerpts of specific records, newest-surfaced first.

    Feeds the "you already know this" inventory in the auto-extraction
    prompts. The prompt half is the *soft* defence — a model may ignore it —
    and `check_surfaced_echo` is the hard one; this exists to stop the model
    wasting a batch re-proposing what it was just shown, not to guarantee it
    doesn't.

    Deliberately bounded (40 records, 120 chars). A proposal to inventory the
    whole active corpus was rejected: at the documented vault scale that is a
    near-complete corpus dump into every extraction prompt, and it re-injects
    untrusted record content into a prompt whose output is written back to the
    store. The session-surfaced set is bounded by what this session actually
    read, which is the set the echo problem is about.

    Returns [{"uuid", "data_type", "source", "excerpt"}]; the caller fences
    `excerpt` against `source`, as every read path must.
    """
    uuid_list = [u for u in dict.fromkeys(uuids or []) if u][:limit]
    if not uuid_list:
        return []
    try:
        placeholders = ",".join("?" * len(uuid_list))
        rows = self._get_conn().execute(
            "SELECT uuid, data_type, source, content FROM memories "
            "WHERE uuid IN ({}) AND status='active'".format(placeholders),
            uuid_list,
        ).fetchall()
    except Exception as e:
        logger.debug("get_surfaced_summaries failed (%s) — no inventory", e)
        return []
    return [{"uuid": r[0], "data_type": r[1], "source": r[2],
             "excerpt": (r[3] or "")[:chars]} for r in rows]


def get_taxonomy(self, kind: str = None) -> list:
    """Get taxonomy entries (data_types and/or data_ids).

    Args:
        kind: filter by kind ('data_type', 'data_id', or None for all)

    Returns:
        list of dicts with name, kind, collection, description
    """
    # A non-string `kind` bound into `WHERE kind = ?` compared unequal to
    # every row and returned `[]` — "no taxonomy" rather than "that is not a
    # kind". One of six instances of the same class; see
    # `_C.str_filter_error`. 2026-09-16 argument-validation enumeration.
    require_str_filters(kind=kind)
    try:
        query = "SELECT name, kind, collection, description FROM taxonomy"
        params = ()
        if kind:
            query += " WHERE kind = ?"
            params = (kind,)
        rows = self._get_conn().execute(query, params).fetchall()
        return [
            {"name": r[0], "kind": r[1], "collection": r[2], "description": r[3]}
            for r in rows
        ]
    except sqlite3.OperationalError as e:
        # Locked or missing table — expected transient/first-run states, and
        # an empty taxonomy degrades gracefully. Stays at debug.
        logger.debug("get_taxonomy: taxonomy table unavailable (%s)", e)
        return []
    except Exception as e:
        # Anything else is a real fault being reported as "no taxonomy
        # registered", which reads identically to a healthy empty table. Name
        # the exception so the caller's empty list is at least explicable.
        logger.warning("[T001] get_taxonomy failed (%s: %s) — returning empty",
                       type(e).__name__, e)
        return []


def unregister_taxonomy(self, name: str, kind: str = None) -> dict:
    """Remove a taxonomy entry.

    Args:
        name: taxonomy entry name to remove
        kind: optional kind filter ('data_type' or 'data_id')

    Returns:
        dict with status
    """
    # The vocabulary check `register_taxonomy` has carried since it was
    # written, applied to the door that removes what it created. Without it
    # `kind="data_typo"` became `AND kind = 'data_typo'`, matched no row, and
    # returned the *same* `{"status": "not_found"}` as a name that does not
    # exist — so an agent that misspelled the kind was told the entry was
    # already gone and had no way to tell the two apart. Driven: register
    # refuses `data_typo` by name, unregister answered `not_found`.
    #
    # Here rather than on the two front ends because both cross it and
    # neither could share the check with the other (`__init__.py` and
    # `mcp_server.py` share no code) — the same reason `register_taxonomy`'s
    # own check lives here. `None` still means "every kind of this name",
    # which is the documented default both doors rely on.
    # 2026-09-14 round 1 bundle05 (F5).
    if kind is not None and kind not in ("data_type", "data_id"):
        return {"error": f"invalid kind: {kind} (must be data_type or data_id)"}

    # Resolve the row's real kind before deleting — the collection-map pop
    # below must only fire for a data_type. Popping unconditionally meant
    # unregistering a *data_id* that happened to share a name with a
    # registered data_type silently unmapped that data_type, sending its
    # future writes to the default collection.
    select_q = "SELECT kind FROM taxonomy WHERE name = ?"
    select_p = (name,)
    if kind:
        select_q += " AND kind = ?"
        select_p = (name, kind)
    # fetchall, not fetchone. With (name, kind) as the key a name can hold a
    # data_type row *and* a data_id row, and an unregister with no `kind`
    # deletes both — but `deleted_kind` read whichever row came back first, so
    # if that was the data_id the collection map kept a routing entry for a
    # type that no longer exists. Ask whether any deleted row was a data_type.
    # 2026-08-24 audit, M5.
    rows = self._get_conn().execute(select_q, select_p).fetchall()
    kinds = {r[0] for r in rows}
    deleted_kind = "data_type" if "data_type" in kinds else (
        next(iter(kinds)) if kinds else None)

    query = "DELETE FROM taxonomy WHERE name = ?"
    params = (name,)
    if kind:
        query += " AND kind = ?"
        params = (name, kind)

    cursor = self._get_conn().execute(query, params)
    self._get_conn().commit()

    if cursor.rowcount == 0:
        return {"status": "not_found", "name": name}

    # Remove from collection map only if it was a data_type.
    #
    # `deleted_kind` used to be read from whatever row survived, so
    # unregistering the data_type of a name that also exists as a data_id left
    # the routing entry behind permanently — a stale map entry pointing at a
    # type nothing accepts any more. With (name, kind) as the key the delete
    # names its own kind, so this reads the row that actually went.
    # 2026-08-24 audit, M5.
    if deleted_kind == "data_type":
        self._collection_map.pop(name, None)
        self._config["collections"] = dict(self._collection_map)
        self._sync_config_to_file()

    logger.info("Taxonomy unregistered: %s", name)
    return {"status": "unregistered", "name": name}


def _get_cleanup_config(self) -> dict:
    """Get cleanup policy from config or defaults."""
    defaults = {
        # sleep
        "duplicate_archive_trust": 0.2,
        "archive_age_days": 365,
        # purge
        "purge_deleted_hours": 24,
        "purge_archived_hours": 168,  # 7 days
        # decay
        "decay_min_age_days": 30,
        "decay_max_age_days": 365,
        "decay_rate": 0.05,
        "decay_min_score": 0.1,
        # enrichment
        "enrichment_batch": 20,
        # maintenance event trigger
        "maintenance_after_inserts": 10,
        # Master switch for unattended mutation at session end. OFF by
        # default: sleep() archives, decay() erodes trust and
        # enrich_existing() rewrites classification, all from heuristics, all
        # with no operator present. purge() is deliberately NOT gated on this —
        # with purge_archived=False it only collects rows already soft-deleted
        # 24h+ ago, which is garbage collection, not a judgement call, and it
        # is what bounds database growth.
        "automatic": False,
        # vacuum
        "vacuum_free_page_pct": 20,
        # budget (seconds per operation)
        "maintenance_budget": 2,
        "maintenance_total_budget": 10,
    }
    cfg = self._config.get("cleanup", {})
    if cfg:
        defaults.update(cfg)
    return defaults


def _maintenance_state(self, key: str) -> Optional[str]:
    """Get a maintenance state value."""
    row = self._get_conn().execute(
        "SELECT value FROM maintenance_state WHERE key = ?", (key,)
    ).fetchone()
    return row[0] if row else None


def _set_maintenance_state(self, key: str, value: str):
    """Set a maintenance state value.

    Commits. The first-run branch of _should_run_maintenance() writes the
    anchor here and nothing else on that path commits, so without this the
    anchor was lost whenever no unrelated write followed — and every
    subsequent call re-ran as "first run".
    """
    self._get_conn().execute(
        "INSERT OR REPLACE INTO maintenance_state (key, value) VALUES (?, ?)",
        (key, value),
    )
    self._get_conn().commit()


def _insert_count(self):
    """Return MAX(rowid) — a monotonically-increasing insert counter.

    Uses MAX(rowid) rather than count() because count() drops when records
    are soft-deleted (status='deleted') or superseded. This is used as the
    maintenance anchor so purge/sleep don't cause delta < 0.
    """
    row = self._get_conn().execute("SELECT MAX(rowid) FROM memories").fetchone()
    return row[0] if row and row[0] else 0


def _should_run_maintenance(self, key: str, threshold: int) -> bool:
    """Check if maintenance should run based on insert count threshold.

    Uses MAX(rowid) as a monotonically-increasing insert counter rather than
    count() (active records). Active count drops after purge/sleep, which would
    make (current - last) negative and maintenance permanently unable to run.
    Insert count only goes up, so delta is always non-decreasing.
    """
    last = self._maintenance_state(key)
    current = self._insert_count()
    try:
        last_count = int(last) if last is not None else None
    except (TypeError, ValueError):
        # A corrupt anchor used to raise ValueError straight out of the
        # session-end maintenance block. Treat it as unset and re-seed.
        logger.warning(
            "maintenance anchor %s holds a non-integer value %r — resetting", key, last)
        last_count = None
    if last_count is None:
        self._set_maintenance_state(key, str(current))
        return True  # first run
    delta = current - last_count
    return delta >= threshold


def _record_maintenance_run(self, key: str):
    """Record that maintenance ran, storing current insert count."""
    # Track monotonically-increasing insert count (MAX(rowid)) rather than
    # active count, which drops on purge/sleep. Insert count only goes up.
    self._set_maintenance_state(key, str(self._insert_count()))


#: Relative TTL forms the tool schema has always advertised ("90d", "30d").
_TTL_RELATIVE = re.compile(r"^\s*(\d+)\s*([hdwm])\s*$", re.IGNORECASE)
_TTL_UNITS = {"h": "hours", "d": "days", "w": "weeks", "m": "days"}


def _normalise_ttl(value, now_iso: str):
    """Turn a TTL into an absolute UTC timestamp, or raise.

    The schema has documented "ISO timestamp or '90d', '30d'" since the field
    existed, and nothing ever parsed the relative form: `add()` passed the string
    straight into the INSERT. Both the retrieval filter (`ttl > ?`) and sleep()
    (`ttl < ?`) then compare it as a *string* against an ISO timestamp, and
    "90d" > "2026-..." because "9" > "2" — so a record with the documented format
    was not expired early, it was **immortal**. An external review found the
    format gap and predicted the opposite consequence; the string comparison runs
    the other way.

    Anything unparseable now raises instead of being stored. A TTL that silently
    means "never" is worse than a rejected write: the caller asked for expiry,
    the store said yes, and the data stays forever.
    """
    if value is None or value == "":
        return None
    text = str(value).strip()
    m = _TTL_RELATIVE.match(text)
    if m:
        n, unit = int(m.group(1)), m.group(2).lower()
        delta = timedelta(**{_TTL_UNITS[unit]: n * (30 if unit == "m" else 1)})
        base = datetime.fromisoformat(now_iso) if now_iso else datetime.now(timezone.utc)
        return (base + delta).isoformat()
    try:
        parsed = datetime.fromisoformat(text)
    except (TypeError, ValueError):
        raise ValueError(
            f"invalid ttl {value!r} — use an ISO timestamp or a relative form "
            f"like '90d', '12h', '2w', '6m'")
    if parsed.tzinfo is None:
        # A naive stamp sorts against UTC strings as though it were UTC, so it
        # expires up to the local offset early or late depending on the host.
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.isoformat()


def _check_scalar_field_bounds(session_name=None, scope=None, source_url=None):
    """Bound the caller-supplied scalars nothing bounded at all.

    `_check_fts_field_bounds` beside this covers the four FTS5-indexed fields.
    `session_name`, `scope` and `source_url` are not indexed, which is why they
    were never added to it — but they are stored verbatim from `add()`, echoed
    on every read path, and interpolated into prompts, so "not in the index" is
    not the same as "unbounded is fine". A multi-megabyte `scope` is bounded
    only by SQLite's blob limit and is returned on every retrieve that matches
    the row. 2026-08-24 audit, finding 12.

    These are identifiers and a URL rather than prose, so they get short bounds
    rather than MAX_FIELD_CHARS — which equals MAX_CONTENT_CHARS at 50,000 and
    would not be a meaningful limit for a hostname.
    """
    for _name, _val, _max in (
            ("session_name", session_name, MAX_SESSION_NAME_CHARS),
            ("scope", scope, MAX_SCOPE_CHARS),
            ("source_url", source_url, MAX_SOURCE_URL_CHARS)):
        if _val and len(str(_val)) > _max:
            raise ValueError(
                f"{_name} exceeds max length ({len(str(_val))} > {_max} chars)")


def _check_fts_field_bounds(summary=None, topic=None, data_id=None, keywords=None):
    """Reject the FTS5-indexed fields content/metadata's guards don't cover.

    `memories_fts` indexes five columns (content, summary, keywords, topic,
    data_id); only content and metadata had a size guard on either add() or
    update(). A caller could write a multi-MB summary/topic/data_id/keywords
    entry — bounded only by SQLite's blob limit — and bloat the shared FTS5
    index exactly the way the content guard exists to prevent, just through
    a sibling column instead. data_id is a Qdrant filter value and lookup
    key, not prose, so it gets MAX_DATA_ID_CHARS rather than MAX_FIELD_CHARS.
    """
    if summary and len(summary) > MAX_FIELD_CHARS:
        raise ValueError(f"summary exceeds max length ({len(summary)} > {MAX_FIELD_CHARS} chars)")
    if topic and len(topic) > MAX_FIELD_CHARS:
        raise ValueError(f"topic exceeds max length ({len(topic)} > {MAX_FIELD_CHARS} chars)")
    if data_id and len(str(data_id)) > MAX_DATA_ID_CHARS:
        raise ValueError(f"data_id exceeds max length ({len(str(data_id))} > {MAX_DATA_ID_CHARS} chars)")
    if keywords:
        # Both callers (add(), update()) coerce a JSON-array string to a real
        # list — or raise — before this runs, so `keywords` here is always
        # `None` or a `list`, never a bare string. A string-keywords branch
        # sat here anyway, unreachable, until 2026-08-23 review round 7
        # The 2026-08-23 write review (F9) found it. Removed rather than left as defensive
        # code: keeping it implied a str could still arrive, which is exactly
        # the wrong thing for the next reader to believe about this function.
        #
        # The element bound below was there; the count was not, so a list of a
        # thousand short keywords bloated the FTS5 index exactly as one long
        # keyword would — the bound the elements carry, evaded by arithmetic.
        # 2026-08-24 audit, finding 12.
        if len(keywords) > MAX_KEYWORDS:
            raise ValueError(
                f"too many keywords ({len(keywords)} > {MAX_KEYWORDS})")
        for kw in keywords:
            if kw and len(str(kw)) > MAX_FIELD_CHARS:
                raise ValueError(
                    f"keywords entry exceeds max length ({len(str(kw))} > {MAX_FIELD_CHARS} chars)")


def _validate_data_type(self, data_type) -> None:
    """Reject a data_type this backend does not know.

    Shared by add() and update() on purpose. 0.7.77 put this check in add()
    only, so `add(data_type="banana")` was refused while
    `update(uuid, data_type="banana")` wrote it happily — add valid, update to
    junk, and the guard bought nothing. That is the same fixed-one-member-of-
    the-class shape the 0.7.77 handover note warns about, committed in the
    release that added the note (2026-08-22 ox-alpha write review, F1).

    Validated against the built-ins unioned with `_collection_map`, so types
    added via register_taxonomy stay valid; a bare allowlist would break the
    feature that exists to extend the taxonomy.
    """
    if data_type is None:
        return
    known = set(VALID_DATA_TYPES) | set(getattr(self, "_collection_map", {}) or {})
    if data_type not in known:
        raise ValueError(
            f"data_type {data_type!r} is not a registered type. Known: "
            f"{sorted(known)}. Use the register_taxonomy action to add a new "
            f"one.")


def add(self, content: str, summary: str = None, topic: str = None,
        keywords: List[str] = None, scope: str = "personal",
        data_type: str = "CUSTOM", data_id: str = None,
        session_name: str = None,
        sensitivity: int = 0, ttl: str = None, priority: int = None,
        source: str = "agent", source_url: str = None,
        metadata: dict = None, trust_score: float = None,
        backlinks: List[str] = None,
        protected: bool = False, force: bool = False,
        supersedes: str = None,
        embedding: List[float] = None) -> Union[str, dict]:
    # Reject an empty write up front. `content` is NOT NULL in the schema but
    # nothing checked it: None reached the embedder as [None] and then
    # `content[:100]` for the default summary, surfacing as an opaque
    # TypeError/endpoint error several frames away, while "" stored a
    # contentless row that no retrieval can ever match. import_memories
    # already rejects this case; add() is the path every tool call takes.
    if content is None or not str(content).strip():
        raise ValueError("content is required and must be a non-empty string")
    # ...and it must actually *be* a string. The check above tests
    # `str(content).strip()`, which every object satisfies, so a dict or list
    # walked straight past the guard written to stop exactly this: the summary
    # default `content[:100]` then raised `TypeError: unhashable type: 'slice'`
    # from several frames down, after the embedder had already been handed the
    # object and answered 400. That is the opaque-failure shape this guard's
    # own docstring describes for None and "". Same treatment, stated
    # contract ("a non-empty string") now enforced. 2026-08-23 profile-a write
    # review (F2).
    if not isinstance(content, str):
        raise ValueError(
            "content must be a string, got %s" % type(content).__name__)
    # Same type guard update() applies (see its "keywords must be a list of
    # strings" check). add() had none: a dict — which an LLM tool call readily
    # produces for an array-typed parameter — was accepted, and because
    # _enrich_metadata iterates it, the record silently stored the dict's *key
    # list*: add(keywords={"a": 1}) wrote ["a"]. Not the caller's data, and no
    # error. Every later reader (contradiction Jaccard, the FTS5 bloat check,
    # keyword display) then worked on something nobody wrote.
    #
    # This must run *before* _enrich_metadata: the first version of this fix
    # sat after it, where the dict had already been coerced into a valid list
    # of strings, so the guard could never fire. The regression suite caught
    # that; the individual fix did not, because it was only ever run pre-fix.
    if keywords is not None:
        if isinstance(keywords, str):
            try:
                keywords = json.loads(keywords)
            except (json.JSONDecodeError, TypeError):
                keywords = None
        if not isinstance(keywords, _builtins.list) or not all(
                isinstance(k, str) for k in keywords):
            raise ValueError(
                f"keywords must be a list of strings (or a JSON-array string "
                f"of strings), got {keywords!r}")

    # Moved above the payload bound (and therefore above the embedder HTTP call
    # and both Qdrant dedup queries) because every malformed-keywords add used
    # to pay a remote round trip before failing. update() already validates
    # before its own embedding call, and the comment just below states the rule
    # this violated — "bound write-payload size before doing any expensive
    # work" — which was honoured for five fields and not this sixth.
    # Reported by the 2026-08-22 ox-alpha write review (F6). Still before
    # _enrich_metadata, which is what the note above requires.

    # `data_type` must name a type this backend knows. Only the LLM
    # classifier's output was validated (`_valid_data_types` in llm.py); every
    # tool write path passed the caller's string straight through, so
    # add(data_type="banana") stored it, routed to the default collection and
    # wrote "banana" into the Qdrant payload — where _layer0's equality filter
    # then narrows on it. The cost is taxonomy pollution plus dedup and
    # contradiction partitioning drifting away from every known label, since
    # both are type-scoped queries.
    #
    # Validated against the built-ins *unioned with* `_collection_map`, so
    # operator-registered types (register_taxonomy adds them to that map)
    # remain valid — a bare allowlist would break the feature that exists to
    # extend the taxonomy. Reported by the 2026-08-22 ox-alpha write review
    # (F3), which found all four sites (both front ends, add and update) and
    # is fixed here once, at the boundary they share.
    _validate_data_type(self, data_type)
    # `data_id` type guard, the one update() received in 0.7.88 and this path
    # did not. `_norm_data_id` calls .lower(), so a non-string crashes with an
    # AttributeError from a frame the caller cannot see — and 0.7.88 fixed
    # exactly that, on the other writer (2026-08-23 glm-5.2 write review, F1).
    if data_id is not None and not isinstance(data_id, str):
        raise ValueError("data_id must be a string, got %s" % type(data_id).__name__)

    # Bound write-payload size before doing any expensive work (embedding,
    # dedup). Content is untrusted — an unbounded record bloats the shared
    # FTS5 index and Qdrant collection, degrading retrieval for the profile.
    max_content = MAX_CONTENT_CHARS
    if content and len(content) > max_content:
        raise ValueError(f"content exceeds max length ({len(content)} > {max_content} chars)")
    if metadata is not None and not isinstance(metadata, dict):
        # Only the size was checked. A string sailed through json.dumps as a
        # JSON *string*, so the column held "\"note\"" instead of an object
        # and every reader got back a str where it expected a mapping — the
        # same shape of bug as non-list keywords (T351), and now caught at
        # the write instead of normalised away at the read. A JSON object in
        # a string is accepted because tool boundaries serialize it that way.
        if isinstance(metadata, str):
            try:
                parsed = json.loads(metadata)
            except (json.JSONDecodeError, TypeError):
                parsed = None
            if not isinstance(parsed, dict):
                raise ValueError(
                    f"metadata must be an object, got {metadata!r}")
            metadata = parsed
        else:
            raise ValueError(
                f"metadata must be a dict, got {type(metadata).__name__}")
    if metadata:
        metadata_len = len(json.dumps(metadata))
        max_metadata = MAX_METADATA_CHARS
        if metadata_len > max_metadata:
            raise ValueError(f"metadata exceeds max length ({metadata_len} > {max_metadata} chars)")
    # The FTS5 table indexes summary/keywords/topic/data_id too, not just
    # content — see _check_fts_field_bounds.
    _check_fts_field_bounds(summary=summary, topic=topic, data_id=data_id, keywords=keywords)
    _check_scalar_field_bounds(session_name=session_name, scope=scope, source_url=source_url)
    # Same rule update() applies. add() had none, so the value went to the
    # INSERT untouched: trust_score=5.0 stored 5.0, -3.0 stored -3.0, and
    # "abc" put a *string* in a numeric column — from the path every tool call
    # takes. Layer 2 multiplies trust into its score, so a string there raises
    # mid-fusion on every later retrieval that returns the row, and an
    # out-of-range value silently outranks everything else.
    if trust_score is not None:
        try:
            trust_score = max(0.0, min(1.0, float(trust_score)))
        except (TypeError, ValueError):
            raise ValueError(
                f"invalid trust_score {trust_score!r} — must be a number 0.0-1.0")

    # Same rule, and the same reasoning, extended to the two fields update()
    # already validates and add() did not: a non-integer sensitivity or
    # priority went to the INSERT untouched from the path every tool call
    # takes. sensitivity has no consumer that raises, so the failure is
    # entirely quiet; priority poisons pipeline.py's Layer-1 sort
    # (TypeError on every retrieval that returns the row), is silently
    # skipped by sleep()'s archive filter and find_duplicate_groups' seed
    # scan (SQLite sorts TEXT above every INTEGER, so `priority < 2` excludes
    # it), and reads as decay-exempt in decay() (a dict.get() miss on the
    # int-keyed multiplier table defaults to 0.0, which decay() treats as
    # "pinned, skip"). A crash is the least of it — the quiet failure makes
    # the record immortal.
    for _int_field, _lo, _hi, _val in (
            ("sensitivity", 0, 3, sensitivity), ("priority", 0, 3, priority)):
        if _val is None:
            continue
        try:
            _coerced = int(_val)
        except (TypeError, ValueError):
            raise ValueError(f"invalid {_int_field} {_val!r} — must be an integer")
        if not (_lo <= _coerced <= _hi):
            raise ValueError(f"invalid {_int_field} {_coerced} — must be {_lo}-{_hi}")
        if _int_field == "sensitivity":
            sensitivity = _coerced
        else:
            priority = _coerced

    # Embed FIRST (cheap), then heuristic classify, then dedup, then enrich (LLM call — expensive)
    # This saves tokens when adding near-duplicates.
    # `embedding` lets bulk callers (obsidian_ingest) embed in batches and
    # pass the vector in — one HTTP round trip per 64 records instead of
    # per record. Indexing 2,024 records one at a time took 541s and
    # tripped the Qdrant circuit breaker.
    if embedding is None:
        fn = _get_embedding_fn(self._embedding_model)
        # Index defensively — an endpoint that answers with an empty or
        # differently-shaped body used to raise IndexError here, one line
        # above the guard written to handle exactly that failure.
        #
        # And catch, for the same reason one line further down. The guard
        # below handles an embedder that *returns* nothing; an embedder that
        # *raises* — a 60s urlopen timeout on a slow endpoint, a refused
        # connection, a 500 — came straight out of add() and took the write
        # with it: the INSERT is below this line, so the record was never
        # stored at all. SQLite is the source of truth and the vector is
        # derived from it, so failing the durable half because the derived
        # half is unavailable is exactly backwards. _layer0 already learned
        # this on the read side ("a dead embedding endpoint used to raise
        # straight out of retrieve() — past the lexical fallback that exists
        # for exactly this situation"); this is the same hole on the write
        # side. Observed as a T301 TimeoutError during a slow spell on the
        # embedding endpoint.
        try:
            _vectors = fn([content])
            embedding = _vectors[0] if _vectors else None
        except Exception as e:
            logger.warning(
                "add: embedding endpoint failed (%s: %s) — storing the record with "
                "no vector; it is invisible to vector search until "
                "layered_maintenance(action='rebuild') re-embeds it", type(e).__name__, e)
            embedding = None
    # Guard against None embedding (endpoint failure, empty content)
    if embedding is None:
        logger.warning("add: embedding returned None for content=%r", content[:60])
        embedding_blob = _EMBED_NULL
    else:
        embedding_blob = _pack_embedding(embedding)

    # Run heuristic classification BEFORE dedup so we check the correct collection
    # (default data_type is "CUSTOM" but enrichment may change it to ENV-DATA/SYSTEM/etc.)
    h_type, h_id, h_confidence = self._heuristic_classify(content)
    # Resolve once, with the same rule storage uses (_resolve_data_type). This
    # was `data_type if (data_type and data_type != "CUSTOM") else h_type` —
    # the heuristic guess taken *ungated*, while _enrich_metadata below only
    # adopts it at confidence >= 0.6. The two disagreed whenever the guess was
    # weak, and since _check_duplicate filters the Qdrant query on data_type,
    # dedup searched a partition the record was never stored in and found
    # nothing at any threshold.
    dedup_data_type = _resolve_data_type(data_type, h_type, h_confidence)

    # Semantic dedup: check for existing similar memories (before LLM enrichment)
    # Skip dedup if embedding is None (can't compare without vector)
    # Skip entirely when force=True, or when this write explicitly replaces
    # a known record — superseding is the legitimate form of "yes, this is
    # similar to something I already know, and it is the new value".
    if embedding is not None and not force and not supersedes:
        threshold = self._config.get("dedup_threshold", 0.97)
        if threshold > 0:
            existing = self._check_duplicate(embedding, dedup_data_type, threshold)
            if existing:
                status = existing.get("status", "duplicate")
                if status == "duplicate":
                    logger.debug("Dedup: duplicate detected (similarity=%.3f), returning existing %s",
                                 existing["similarity"], existing["uuid"])
                    return {
                        "uuid": existing["uuid"],
                        "status": "duplicate",
                        "similarity": existing["similarity"],
                        "existing_content": existing.get("existing_content"),
                        "note": existing.get("note",
                            "Existing memory with high semantic similarity. Use the update action to merge content.")
                    }
                elif status == "possible_duplicate":
                    logger.debug("Dedup: possible duplicate (similarity=%.3f), returning existing %s",
                                 existing["similarity"], existing["uuid"])
                    return {
                        "uuid": existing["uuid"],
                        "status": "possible_duplicate",
                        "similarity": existing["similarity"],
                        "existing_content": existing.get("existing_content"),
                        "note": existing.get("note",
                            "High semantic similarity. Review existing content. Use the update action to merge, or add with force=true to store anyway.")
                    }

        # Write-time contradiction detection: check for similar-but-different records
        # (cosine >= 0.85 but < warning threshold → same topic, different value)
        contradiction = self._check_contradiction(
            embedding, dedup_data_type, threshold,
            content=content, data_id=data_id or h_id, source=source)
        if contradiction:
            logger.debug("Contradiction: flagged %s (cos=%.3f), existing %s",
                         contradiction["existing_uuid"][:8], contradiction["similarity"],
                         contradiction["existing_uuid"])
            return {
                "uuid": contradiction["existing_uuid"],
                "status": "contradiction",
                "similarity": contradiction["similarity"],
                "existing_content": contradiction["existing_content"][:100],
                "note": "Existing memory covers the same topic but with different content. "
                        "If this is the new value, add again with supersedes=<uuid> — the old "
                        "record is kept and linked but stops being returned. Use force=true "
                        "instead if both versions should coexist as separate facts."
            }

    # Hybrid enrichment: heuristics only (fast). LLM enrichment runs in background.
    if force:
        logger.info("add: force=True, skipping dedup/contradiction for content=%r", content[:60])
    enriched = self._enrich_metadata(content, summary, data_type, data_id, topic, keywords, llm=False, source=source)
    data_type = enriched.get("data_type", data_type) or "CUSTOM"
    data_id = enriched.get("data_id", data_id)
    topic = enriched.get("topic", topic)
    keywords = enriched.get("keywords", keywords) or []

    # Normalize data_id to lowercase for consistency (SW/sw → sw, HW/hw → hw)
    if data_id:
        data_id = data_id.lower()

    # ENV-DATA and SYSTEM records default to priority=1 (elevated) so decay skips them
    # Overrideable by explicit priority=0 or any other value
    if priority is None:
        priority = 1 if data_type in ("ENV-DATA", "SYSTEM") else 0

    now = self._now()
    uuid = uuid_mod.uuid4().hex
    kw_json = json.dumps(keywords)
    metadata_json = json.dumps(metadata) if metadata else "null"
    # Link data goes in the typed column, not a metadata blob. `backlinks` was
    # declared in the schema, read on every retrieve, fenced on the way out and
    # merged during compaction — and written by nothing, because the only
    # producer of link data (`_extract_wikilinks`, on Obsidian ingest) put its
    # output in `metadata["wikilinks"]`, which nothing reads. Two halves of one
    # feature that never met: every record in every profile reported
    # `backlinks: []` while five records carried their links one level down in
    # an untyped dict.
    if isinstance(backlinks, str):
        try:
            backlinks = json.loads(backlinks)
        except (json.JSONDecodeError, TypeError):
            backlinks = []
    # The same guard `keywords` carries, for the same reason and one field
    # over. `[str(b) for b in backlinks]` iterates a dict's *keys*, so
    # add(backlinks={"a": 1}) silently stored ["a"] — not the caller's data,
    # and no error. That is the identical defect fixed for keywords, on the
    # neighbouring field, which is the shape this review loop keeps finding.
    # 2026-08-23 profile-a write review (F5).
    if backlinks is not None and (
            not isinstance(backlinks, _builtins.list)
            or not all(isinstance(b, str) for b in backlinks)):
        # Elements as well as the container — `keywords` beside it checks both,
        # and 0.7.86 added this guard with only the container half while
        # copying the "must be a list of strings" wording verbatim. So the
        # error message promised what the check never verified, and
        # `backlinks=[1, 2, 3]` was silently coerced to ["1", "2", "3"] by the
        # `[str(b) for b in backlinks]` below: the caller's data, changed, with
        # no error. 2026-08-23 laguna-s-2.1 write review.
        raise ValueError(
            "backlinks must be a list of strings (or a JSON-array string), "
            "got %r" % (backlinks,))
    bl_json = json.dumps([str(b) for b in backlinks]) if backlinks else "[]"

    # Write to SQLite with retry on lock. Only the INSERT+supersession+commit
    # is retried (not the embedding or dedup above), so each retry is fast.
    superseded_uuid = None
    for attempt in range(50):
        try:
            self._get_conn().execute("""
                INSERT INTO memories (uuid, content, summary, keywords, topic, scope,
                    data_type, data_id, session_name, sensitivity, source, source_url,
                    created_at, updated_at, ttl, status, trust_score, reference_count,
                    backlinks, layer3_flags, metadata, priority, protected, sequence, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, 0, ?,
                        '{}', ?, ?, ?, NULL, ?)
            """, (uuid, content, summary or content[:100], kw_json, topic, scope,
                  data_type or "CUSTOM", data_id, session_name, sensitivity, source, source_url,
                  now, now, _normalise_ttl(ttl, now),
                  trust_score if trust_score is not None else 0.5,
                  bl_json, metadata_json, priority, bool(protected), embedding_blob))

            # Supersession: mark the replaced record instead of destroying it.
            if supersedes:
                old = self._get_conn().execute(
                    "SELECT uuid FROM memories WHERE uuid = ? AND status = 'active' "
                    "AND superseded_by IS NULL",
                    (supersedes,)
                ).fetchone()
                if old:
                    self._get_conn().execute(
                        "UPDATE memories SET superseded_by = ?, superseded_at = ?, updated_at = ? "
                        "WHERE uuid = ?",
                        (uuid, now, now, supersedes),
                    )
                    superseded_uuid = supersedes
                    logger.info("add: %s supersedes %s", uuid[:8], supersedes[:8])
                else:
                    logger.warning("add: supersedes=%s not found or inactive — ignored",
                               str(supersedes)[:8])

            self._get_conn().commit()
            break
        except Exception as e:
            # Roll back before retrying. The INSERT above may already have
            # succeeded inside the implicit transaction when the failure
            # surfaces at commit() (SQLite can report SQLITE_BUSY there);
            # re-running the same INSERT with the same uuid in the still-open
            # transaction then raises "UNIQUE constraint failed", which is not
            # a lock error and propagated to the caller as a bogus
            # IntegrityError instead of a retry.
            try:
                self._get_conn().rollback()
            except Exception:
                pass
            if attempt < 49 and _is_lock_error(e):
                import time as _time
                _time.sleep(min(0.2 * (2 ** attempt), 5))
                continue
            raise

    # Qdrant upsert (best effort) — after commit so the lock is released.
    if self._qdrant:
        try:
            from qdrant_client.models import PointStruct
            payload = {"data_type": data_type or "CUSTOM"}
            if data_id:
                payload["data_id"] = data_id
            if session_name:
                payload["session_name"] = session_name
            if self._profile_name:
                # Every profile that holds this record, not just us — see
                # core.profile_claim. A shared uuid means a shared point.
                payload["profile_name"] = _profile_claim(
                    self._qdrant, self._get_collection(payload.get("data_type", "CUSTOM")),
                    _to_qdrant_id(uuid), self._profile_name)
            if embedding is not None:
                self._qdrant.upsert(
                    collection_name=self._get_collection(data_type or "CUSTOM"),
                    points=[PointStruct(id=_to_qdrant_id(uuid), vector=embedding, payload=payload)],
                )
            # Route through the circuit breaker's own lock-protected method
            # instead of mutating _qdrant_failures directly — a bare
            # assignment here would race _record_qdrant_failure's locked
            # increment and could clear the counter mid-failure-count, or
            # leave _qdrant_broken_until set after connectivity is proven.
            self._record_qdrant_success()
        except Exception as e:
            self._record_qdrant_failure(e)
            logger.warning("[Q004] Qdrant upsert failed: %s", e)
    # History archive — log the new record
    self._write_history_entry("add", self._get_record(uuid))

    # Fire background LLM enrichment (non-blocking)
    self._enrich_background(uuid, content, summary, source=source)

    logger.info("add: uuid=%s data_type=%s data_id=%s summary=%r",
                 uuid[:8], data_type, data_id, (summary or content[:60]))
    if superseded_uuid:
        return {"uuid": uuid, "status": "superseded",
                "superseded_uuid": superseded_uuid,
                "note": "Stored as the current value. The previous record is retained "
                        "and linked, but no longer returned by normal retrieval."}
    return uuid


#: maintenance_state key marking that the enrichment-default notice has been
#: shown for this profile.
ENRICH_NOTICE_KEY = "notice_enrich_default_v021"

#: Same, for profiles that had `enrich_on_add: "true"`. A separate key so the
#: two notices are independent — a profile only ever qualifies for one.
ENRICH_TRUE_NOTICE_KEY = "notice_enrich_true_cost_v021"


def _notify_enrich_default_change(self) -> None:
    """Say once, per profile, that automatic enrichment stopped.

    Enrichment used to run on every write no matter what `enrich_on_add` said —
    the key was read only inside `_enrich_metadata`, whose callers all override
    it. Now it is honoured, and the default is `heuristics_only`. For a profile
    that never set the key, that means background LLM enrichment silently
    stopped: new records get heuristic classification and no LLM-derived
    topic/keywords. The change is deliberate, but discovering it by noticing
    that `topic` is empty three weeks later is not.

    Two populations saw their behaviour change without asking for it, and each
    gets one notice:

      * **key unset** — enrichment silently stopped.
      * **key = "true"** — enrichment silently got *more* expensive. "true" and
        "low_confidence" used to be identical (both fired only on a
        low-confidence heuristic); now "true" means what it says and calls the
        LLM on every add, including the confident matches that used to be free.

    Both are gated on an LLM being configured (otherwise enrichment never ran)
    and on the database already holding records (a new profile has nothing to
    migrate, and a first-run banner is noise). A profile that set
    "low_confidence", "false" or "heuristics_only" is unaffected and hears
    nothing.
    """
    try:
        if not self.llm_configured() or self.count() <= 0:
            return
        configured = self._config.get("enrich_on_add")

        if configured is None:
            if self._maintenance_state(ENRICH_NOTICE_KEY):
                return
            logger.warning(
                "enrich_on_add now defaults to 'heuristics_only': automatic LLM "
                "enrichment on add is OFF for this profile. Previously it ran on "
                "every write regardless of this key (the key was read but always "
                "overridden by its caller, so it had no effect). New records get "
                "heuristic classification only. To restore the old behaviour set "
                "enrich_on_add='low_confidence'; to backfill existing records run "
                "layered_advanced(action='enrich'). This notice is shown once.")
            self._set_maintenance_state(ENRICH_NOTICE_KEY, self._now())
            return

        if str(configured).lower() == "true":
            if self._maintenance_state(ENRICH_TRUE_NOTICE_KEY):
                return
            logger.warning(
                "enrich_on_add='true' now calls the LLM on every add "
                "(previously it skipped confident heuristic matches, making it "
                "identical to 'low_confidence'). Switch to "
                "enrich_on_add='low_confidence' to reduce cost. Shown once.")
            self._set_maintenance_state(ENRICH_TRUE_NOTICE_KEY, self._now())
    except Exception as e:
        # A migration notice must never be able to break startup.
        logger.debug("enrich notice skipped: %s", e)


#: Values of `enrich_on_add` that disable background LLM enrichment.
#: "heuristics_only" is the new default and the honest name for what the
#: system should do by default; "false" is the older spelling.
ENRICH_DISABLED = frozenset({"false", "heuristics_only", "off", "no"})

#: Default for `enrich_on_add`. Was "low_confidence" (LLM whenever the
#: heuristic scored < 0.6). Enrichment has two failure modes and both write to
#: memory: an unreachable model returns CUSTOM for everything (the data_type
#: demotion bug), and a reachable one can hallucinate a topic. Neither is worth
#: running unattended on every write — `layered_advanced(action="enrich")` does
#: the same job when asked.
ENRICH_ON_ADD_DEFAULT = "heuristics_only"


def _enrich_background(self, uuid: str, content: str, summary: str = None, source: str = None):
    """Run LLM enrichment in a background thread and update the record when ready.

    Cancellation is keyed on `uuid`, not global. It used to be a single
    process-wide slot: every add() cancelled whatever was in flight, so during
    any burst — bulk import, obsidian ingest, an extraction run storing several
    facts — records 1..N-1 were cancelled before their LLM call returned and
    never got topic/keywords. The only case that genuinely needs cancelling is
    a *second* enrichment of the *same* record superseding the first.

    Concurrency is bounded by a small thread pool rather than one slot, so a
    burst enriches instead of discarding.

    Gated on `enrich_on_add`. That key was documented as controlling exactly
    this ("false" disables enrichment entirely) but was **dead**: it is only
    read inside _enrich_metadata, and every caller passes an explicit `llm=`
    argument that overrides it — this function passes llm=True. So setting
    enrich_on_add="false" changed nothing and background LLM enrichment ran on
    every single write regardless. The check belongs here, where the decision
    actually is.
    """
    mode = str(self._config.get("enrich_on_add", ENRICH_ON_ADD_DEFAULT)).lower()
    if mode in ENRICH_DISABLED:
        logger.debug(
            "enrichment: skipped for %s (enrich_on_add=%r)", uuid[:8], mode)
        return

    with self._enrich_lock:
        cancels = getattr(self, "_enrich_cancels", None)
        if cancels is None:
            cancels = self._enrich_cancels = {}
        # Supersede only this uuid's own in-flight enrichment.
        prev = cancels.get(uuid)
        if prev is not None:
            prev.set()
        new_cancel = threading.Event()
        cancels[uuid] = new_cancel

    def _do_enrich(cancel: threading.Event):
        # Check cancellation before the expensive LLM call
        if cancel.is_set():
            logger.debug("enrichment: %s cancelled before LLM call", uuid[:8])
            return
        try:
            # llm=None, not llm=True: let `enrich_on_add` decide.
            #
            # Forcing llm=True here made "low_confidence" behave exactly like
            # "true" — the LLM ran on every write regardless of how confident
            # the heuristic was, which is the opposite of what the value is
            # named for and what docs/reference.md documents. The gate above
            # already handled the disabled values; this lets the remaining two
            # differ from each other.
            enriched = self._enrich_metadata(content, summary, llm=None, source=source)
            # Re-check cancellation after the LLM call — a new thread may
            # have started during the call and superseded this enrichment.
            if cancel.is_set():
                logger.debug("enrichment: %s cancelled after LLM call, discarding", uuid[:8])
                return
            if enriched:
                # The record can vanish while the LLM call is in flight —
                # an add followed by a delete is a normal sequence, and
                # _get_record() filters on status='active'. Calling .get()
                # on the resulting None raised an AttributeError that a
                # broad except turned into a debug line, so the enrichment
                # thread died silently mid-way.
                # Re-check cancellation after the DB read as well.
                if cancel.is_set():
                    return
                record = self._get_record(uuid)
                if record is None:
                    logger.debug("enrichment: %s no longer active, skipping",
                                 uuid[:8])
                    return
                fields = {}
                if enriched.get("topic") and not record.get("topic"):
                    fields["topic"] = enriched["topic"]
                if enriched.get("keywords") and not record.get("keywords"):
                    fields["keywords"] = enriched["keywords"]
                if fields and not cancel.is_set():
                    # Retry on lock — main thread may be writing concurrently
                    for attempt in range(4):
                        try:
                            self._get_conn().execute(
                                "UPDATE memories SET topic = COALESCE(topic, ?), "
                                "keywords = CASE WHEN keywords IS NULL OR keywords IN ('', '[]', 'null') "
                                "THEN ? ELSE keywords END, updated_at = ? WHERE uuid = ?",
                                (enriched.get("topic"),
                                 json.dumps(enriched.get("keywords", [])),
                                 self._now(), uuid)
                            )
                            self._get_conn().commit()
                            break
                        except Exception as e:
                            try:
                                self._get_conn().rollback()
                            except Exception:
                                pass
                            if attempt < 3 and _is_lock_error(e):
                                time.sleep(0.1 * (2 ** attempt))
                                continue
                            raise
                    logger.debug("Background enrichment complete for %s", uuid[:8])
        except Exception as e:
            logger.warning("[E001] Background enrichment failed for %s: %s", uuid[:8], e)
        finally:
            with self._enrich_lock:
                # Drop our slot only if it is still ours — a newer enrichment
                # for the same uuid may have replaced it while we ran.
                if getattr(self, "_enrich_cancels", {}).get(uuid) is cancel:
                    self._enrich_cancels.pop(uuid, None)

    with self._enrich_lock:
        pool = getattr(self, "_enrich_pool", None)
        if pool is None:
            from concurrent.futures import ThreadPoolExecutor
            pool = self._enrich_pool = ThreadPoolExecutor(
                max_workers=2, thread_name_prefix="hlm-enrich")
        try:
            future = pool.submit(_do_enrich, new_cancel)
        except RuntimeError:
            # Pool already shut down (close() ran) — enrichment is best-effort.
            self._enrich_cancels.pop(uuid, None)
            logger.debug("enrichment: pool shut down, skipping %s", uuid[:8])
            return
        # _enrich_thread keeps the Thread-like surface callers already use
        # (join(timeout=)/is_alive()) now that the work runs on a pool.
        self._enrich_thread = _EnrichHandle(future)


class _EnrichHandle:
    """Thread-compatible view of an enrichment Future.

    `_enrich_thread` was a threading.Thread before enrichment moved onto a
    bounded pool; tests and shutdown paths call .join(timeout=)/.is_alive()
    on it, so keep that surface rather than leaking Future semantics.
    """

    __slots__ = ("_future",)

    def __init__(self, future):
        self._future = future

    def join(self, timeout=None):
        from concurrent.futures import TimeoutError as _FTimeout
        try:
            self._future.result(timeout=timeout)
        except _FTimeout:
            pass
        except Exception:
            pass  # the worker logs its own failures

    def is_alive(self) -> bool:
        return not self._future.done()

    def __bool__(self) -> bool:
        return True


#: Statuses the rest of the codebase actually queries on. Anything else
#: makes a record invisible to every read path while leaving it in Qdrant.
VALID_STATUSES = frozenset({"active", "archived", "deleted"})


# The columns the Qdrant payload mirrors. Changing any of them without
# re-writing the payload leaves _layer0's filters matching on stale values —
# see _resync_qdrant_payload.
_PAYLOAD_FIELDS = frozenset({"data_type", "data_id", "session_name"})


def update(self, uuid: str, **fields):
    if not fields:
        return
    # Capture old record state before mutation (for history archive)
    old_record = self._get_record(uuid)
    # Refuse a uuid that does not exist instead of reporting success. The
    # UPDATE simply matched no rows, so the caller got None — indistinguishable
    # from a write that worked — and the code below then ran the whole
    # re-embed/Qdrant-upsert path against a record that was never there. The
    # plugin's _do_update happens to verify the row first, so this only ever
    # bit callers that reach the backend directly (MCP, scripts, tests).
    # 2026-08-22 ox-alpha write review (F3).
    if old_record is None:
        raise ValueError(f"no record with uuid {uuid!r}")
    # `metadata` was missing here despite META_MEMORY_SCHEMA's own
    # description ("For add/update: arbitrary JSON metadata attached to the
    # record") promising it works on both. update(uuid, metadata={...}) was
    # silently filtered by the allowlist below and the caller got back
    # {"status": "updated", ...} — a success response for a write that did
    # nothing. add() has always accepted it; there was never a decision to
    # exclude update, just an allowlist nobody grew when metadata was added
    # to add().
    # Anything not in here is **dropped**, and reported in `dropped` /
    # logged — it is not accepted-unvalidated. Three reviews have now filed a
    # field's absence from this set as a missing type guard (`scope` and
    # `source_url` in the 2026-08-24 audit's finding 12, `backlinks` in the
    # 2026-08-25 bundle01 round), and one of this repo's own tests made the
    # same mistake before it was corrected. If a field belongs here, adding it
    # is the change; a guard for a field that never arrives is not.
    # One definition, in constants.py — MCP's update branch reports which
    # fields it applied and used to compute that from its own input rather
    # than from this list. 2026-08-26 xhigh round, bundle01 F4.
    allowed = UPDATE_ALLOWED_FIELDS
    clean = {k: v for k, v in fields.items() if k in allowed and v is not None}
    dropped = [k for k in fields if k not in allowed]

    # Same validation add() applies. Without it the 0.7.77 guard was trivially
    # bypassable: add with a valid type, then update to anything.
    _validate_data_type(self, clean.get("data_type"))

    # `data_id` must be a string. `_norm_data_id` (and the raw filter sites)
    # call `.lower()` on it, so a dict/list/int raised
    # `AttributeError: 'dict' object has no attribute 'lower'` from a frame the
    # caller has no view of — the same opaque-crash class as content, keywords
    # and backlinks, on the field none of those guards covered. Found by the
    # 2026-08-23 profile-a write review (F1, reasoning-off run).
    if "data_id" in clean and not isinstance(clean["data_id"], str):
        raise ValueError(
            "data_id must be a string, got %s" % type(clean["data_id"]).__name__)

    # `protected` is written straight into a column the archival and decay
    # guards read as `COALESCE(protected, 0) = 0`. Uncoerced, a string went in
    # as TEXT and every one of those guards read it as *protected*, so
    # update(protected="false") — the obvious spelling from a JSON tool call —
    # pinned the record instead of unpinning it, permanently and invisibly.
    # add() takes a real bool through its signature; this path never coerced.
    # Reported by both review passes (2026-08-22 ox-alpha write F2, and the
    # 2026-08-21 local-model write review F3, which called it "the stuck-pin
    # direction an operator never notices").
    if "protected" in clean:
        _p = clean["protected"]
        if isinstance(_p, str):
            _norm = _p.strip().lower()
            if _norm in ("true", "1", "yes", "on"):
                clean["protected"] = True
            elif _norm in ("false", "0", "no", "off", ""):
                clean["protected"] = False
            else:
                raise ValueError(
                    f"protected must be a boolean (or a boolean-ish string), "
                    f"got {_p!r}")
        else:
            clean["protected"] = bool(_p)
        # Store 0/1, not Python bools, so the column stays INTEGER for the
        # COALESCE guards regardless of driver adaptation.
        clean["protected"] = 1 if clean["protected"] else 0
    if dropped:
        logger.warning("update: unsupported fields filtered: %s", dropped)

    # Validate values, not just field names. Without this,
    # update(status="banana") wrote a status no query filters on, hiding the
    # record from every code path permanently, and trust_score=99 defeated
    # the 0.0-1.0 invariant that feedback()/reinforce()/decay() all assume.
    if "status" in clean:
        status_val = str(clean["status"]).strip().lower()
        if status_val not in VALID_STATUSES:
            raise ValueError(
                f"invalid status {clean['status']!r} — must be one of "
                f"{sorted(VALID_STATUSES)}")
        clean["status"] = status_val
    if "trust_score" in clean:
        try:
            clean["trust_score"] = max(0.0, min(1.0, float(clean["trust_score"])))
        except (TypeError, ValueError):
            raise ValueError(f"invalid trust_score {clean['trust_score']!r} — must be a number 0.0-1.0")
    # Same normalisation add() applies. 0.7.42 wired _normalise_ttl into add()
    # only and left update() and import_memories writing the string verbatim —
    # so `update(uuid, ttl="90d")` re-created the exact bug that release fixed,
    # a record that sorts above every ISO timestamp and therefore never expires.
    # Fixing one member of a class and calling the class handled is the mistake
    # 0.7.41 was written about; this is the same mistake, one release later.
    if "ttl" in clean:
        clean["ttl"] = _normalise_ttl(clean["ttl"], self._now())
    for _int_field, _lo, _hi in (("sensitivity", 0, 3), ("priority", 0, 3)):
        if _int_field in clean:
            try:
                _val = int(clean[_int_field])
            except (TypeError, ValueError):
                raise ValueError(f"invalid {_int_field} {clean[_int_field]!r} — must be an integer")
            if not (_lo <= _val <= _hi):
                raise ValueError(f"invalid {_int_field} {_val} — must be {_lo}-{_hi}")
            clean[_int_field] = _val

    # Same two guards add() applies (store.py:815-825) — content
    # non-emptiness and the size bound — neither existed here. Without the
    # first, `update(uuid, content="")` returned normally and left a
    # contentless row with cleared keywords and (via the summary-regeneration
    # branch below) a cleared summary: exactly the "no retrieval can ever
    # match this" row add()'s guard exists to prevent, reached one frame over.
    # Without the second, `update(uuid, content=<60k chars>)` grew a record
    # past the bound add() and import_memories both enforce, and — because
    # update() re-embeds and upserts on a content change — the oversized text
    # replaced a correctly-sized point in the shared Qdrant collection.
    if "content" in clean:
        # Type first, exactly as add() does. `str(x).strip()` is satisfied by
        # every object, so a dict reached the summary slice
        # (`clean["content"][:100]`, ~120 lines down) and the embedder before
        # anything complained. 0.7.86 added this guard to add() and not here —
        # caught by the next review pass the same day, which is the shape this
        # loop keeps finding and, this time, at its shortest interval yet.
        # 2026-08-23 ox-alpha write review (F1).
        if not isinstance(clean["content"], str):
            raise ValueError(
                "content must be a string, got %s" % type(clean["content"]).__name__)
        if not str(clean["content"]).strip():
            raise ValueError("content must be a non-empty string")
        if len(clean["content"]) > MAX_CONTENT_CHARS:
            raise ValueError(
                f"content exceeds max length ({len(clean['content'])} > {MAX_CONTENT_CHARS} chars)")

    # Same shape as add()'s metadata guard: type-check (a JSON *string* sailed
    # through json.dumps as a string, same class as T351's non-list keywords)
    # and size-bound (add() and import_memories both enforce it; this was the
    # one write path where metadata had no writer validation at all — it
    # simply was not writable, see the `allowed` comment above).
    if "metadata" in clean:
        metadata = clean["metadata"]
        if not isinstance(metadata, dict):
            if isinstance(metadata, str):
                try:
                    parsed = json.loads(metadata)
                except (json.JSONDecodeError, TypeError):
                    parsed = None
                if not isinstance(parsed, dict):
                    raise ValueError(f"metadata must be an object, got {metadata!r}")
                metadata = parsed
            else:
                raise ValueError(f"metadata must be a dict, got {type(metadata).__name__}")
        if metadata:
            metadata_len = len(json.dumps(metadata))
            if metadata_len > MAX_METADATA_CHARS:
                raise ValueError(
                    f"metadata exceeds max length ({metadata_len} > {MAX_METADATA_CHARS} chars)")
        clean["metadata"] = metadata

    # add() coerces a JSON-array string to a list (tool schema can pass
    # either — see llm.py's _enrich_metadata) before storing it; update()
    # had no such guard, so a bare non-list string skipped the
    # isinstance(v, (list, dict)) branch in the serialization loop below and
    # was written to the array column verbatim — FTS5-indexed as free text
    # while every reader reports keywords: [] (json.loads on the raw string
    # fails and defaults to []). Reject here instead of silently coercing to
    # [] like add() does: update()'s other fields all raise on a bad value
    # rather than discarding it, and discarding a caller's keywords update
    # over a formatting mistake is a worse surprise than the ValueError.
    if "keywords" in clean:
        kw = clean["keywords"]
        if isinstance(kw, str):
            try:
                kw = json.loads(kw)
            except (json.JSONDecodeError, TypeError):
                kw = None
        if not isinstance(kw, _builtins.list) or not all(isinstance(k, str) for k in kw):
            raise ValueError(
                f"keywords must be a list of strings (or a JSON-array string "
                f"of strings), got {clean['keywords']!r}")
        clean["keywords"] = kw

    # `backlinks` is a list column like `keywords` and needs the same door.
    # It reached `UPDATE_ALLOWED_FIELDS` in 0.8.92 (consider-features #38); the
    # obvious one-line addition would have bound a Python list straight into
    # the SQL parameter and raised InterfaceError at the driver, which is the
    # opaque-crash class the block above exists to prevent. Reject rather than
    # coerce, for the reason stated there.
    if "backlinks" in clean:
        bl = clean["backlinks"]
        if isinstance(bl, str):
            try:
                bl = json.loads(bl)
            except (json.JSONDecodeError, TypeError):
                bl = None
        if not isinstance(bl, _builtins.list) or not all(isinstance(b, str) for b in bl):
            raise ValueError(
                f"backlinks must be a list of strings (or a JSON-array string "
                f"of strings), got {clean['backlinks']!r}")
        clean["backlinks"] = bl

    # Same FTS5-bloat guard add() applies to summary/topic/data_id/keywords
    # (store.py's _check_fts_field_bounds) — content and metadata were the
    # only bounded fields here too.
    # `session_name` only: `scope` and `source_url` are not in `allowed`
    # above, so `clean` never carries them and those two arguments were always
    # `None` — a bounds check that could not fire, written in 0.8.0 and never
    # exercised for two of its three fields.
    #
    # Whether `update()` *should* accept them is a separate decision from this
    # being dead: `add()` takes both, so today they can be set at creation and
    # never changed, and nothing states that as intent. Left as-is rather than
    # widened, because making a field updatable is a product change and not a
    # bug fix. 2026-08-25 xhigh round, bundle01 (F5).
    _check_scalar_field_bounds(session_name=clean.get("session_name"))
    _check_fts_field_bounds(summary=clean.get("summary"), topic=clean.get("topic"),
                            data_id=clean.get("data_id"), keywords=clean.get("keywords"))

    # Normalize data_id to lowercase for consistency
    if "data_id" in clean and clean["data_id"]:
        clean["data_id"] = clean["data_id"].lower()
    if not clean:
        return

    # Keywords are FTS5-indexed (see the fts5_au trigger), so a keyword left
    # over from replaced content does not merely go stale — it stays *matchable*
    # and lends BM25 weight to a term the record no longer states. Observed on
    # the live hlm-test profile: cf9eb3b2's content was corrected to
    # "core -50mV, cache -50mV, GPU -40mV" while its keywords still read
    # ["T440p", ..., "-70mV", ...], and `memories_fts MATCH '70mv'` still
    # returned that row. A user asking about -70mV got a confident hit on a
    # record documenting -50mV.
    #
    # This is the same argument the block below makes for the *vector*, and it
    # is settled the same way: a derived field must not outlive the content it
    # was derived from. Re-derive rather than clear, because keywords carry the
    # lexical arm's precision and dropping them silently degrades retrieval —
    # heuristics only (llm=False), so this stays offline, deterministic and
    # free. An explicit keywords= in the same call wins; the caller has said
    # what it wants.
    if "content" in clean and "keywords" not in clean:
        try:
            enriched = self._enrich_metadata(
                clean["content"],
                data_type=(old_record or {}).get("data_type"),
                data_id=(old_record or {}).get("data_id"),
                topic=(old_record or {}).get("topic"),
                llm=False,
                source=(old_record or {}).get("source"))
            rederived = enriched.get("keywords")
            if isinstance(rederived, _builtins.list):
                clean["keywords"] = rederived
        except Exception as e:
            # Never fail a content edit over a derived field. Clearing is the
            # safe fallback: no keywords cannot mismatch, stale ones can.
            logger.warning(
                "update: keyword re-derivation failed (%s: %s) — clearing "
                "keywords rather than leaving them describing replaced content",
                type(e).__name__, e)
            clean["keywords"] = []

    # `summary` is derived the same way and indexed by the same trigger, and it
    # is the larger exposure: add() stores `summary or content[:100]`, so a
    # record created without an explicit summary carries a verbatim copy of its
    # content. Editing the content alone left that copy behind, which is why
    # cf9eb3b2 still matched '70mv' even once its keywords were re-derived —
    # the old sentence was sitting in `summary` word for word.
    #
    # Only the derived form is refreshed. A summary that does not equal the old
    # content's first 100 characters was written by a caller on purpose, and
    # silently rewriting it would destroy an editorial decision to fix an index
    # entry. That case is logged instead: it is rare, and a warning is
    # recoverable where a clobbered summary is not.
    if "content" in clean and "summary" not in clean:
        _old_content = (old_record or {}).get("content") or ""
        _old_summary = (old_record or {}).get("summary") or ""
        if _old_summary and _old_summary == _old_content[:100]:
            clean["summary"] = clean["content"][:100]
        elif _old_summary:
            logger.warning(
                "update: %s has a caller-authored summary that still describes "
                "the replaced content; it is FTS-indexed and will keep matching "
                "the old wording. Pass summary= to refresh it.", uuid[:8])

    sets = ", ".join(f"{k} = ?" for k in clean)
    values = []
    for k, v in clean.items():
        if isinstance(v, (_builtins.list, _builtins.dict)):
            values.append(json.dumps(v))
        else:
            values.append(v)
    values += [self._now(), uuid]

    # Compute re-embedding BEFORE the retry loop (HTTP call, no lock)
    new_emb = None
    embed_failed = False
    if "content" in clean:
        fn = _get_embedding_fn(self._embedding_model)
        # Same catch as add(): an embedder that raises used to take the whole
        # update with it, losing the content edit because the *vector* could
        # not be recomputed.
        try:
            _vectors = fn([clean["content"]])
            new_emb = _vectors[0] if _vectors else None
        except Exception as e:
            logger.warning(
                "update: embedding endpoint failed (%s: %s) — content is updated and "
                "the stale vector cleared; rebuild() will re-embed", type(e).__name__, e)
            new_emb = None
        embed_failed = new_emb is None

    # Write to SQLite with retry on lock. Only the UPDATE+commit is retried,
    # not the embedding computation above, so each retry is fast.
    for attempt in range(50):
        try:
            self._get_conn().execute(f"UPDATE memories SET {sets}, updated_at = ? WHERE uuid = ?", values)
            if new_emb is not None:
                self._get_conn().execute("UPDATE memories SET embedding = ? WHERE uuid = ?",
                               (_pack_embedding(new_emb), uuid))
            elif embed_failed:
                # The content changed and the new vector could not be
                # computed, so the stored one now describes text this record
                # no longer holds — vector search would match the *old*
                # wording and hand back the *new* content, silently. Clear it
                # instead: NULL is the marker rebuild() re-embeds from, and an
                # unfindable record is better than a confidently wrong one.
                self._get_conn().execute("UPDATE memories SET embedding = ? WHERE uuid = ?",
                                         (_EMBED_NULL, uuid))
            self._get_conn().commit()
            break
        except Exception as e:
            # Roll back before retrying — see the matching comment in add().
            try:
                self._get_conn().rollback()
            except Exception:
                pass
            if attempt < 49 and _is_lock_error(e):
                import time as _time
                _time.sleep(min(0.2 * (2 ** attempt), 5))
                continue
            raise

    # Qdrant upsert for re-embedding — after commit so the lock is released.
    if "content" in clean and new_emb is not None and self._qdrant:
        from qdrant_client.models import PointStruct
        dt_row = self._get_conn().execute("SELECT data_type, data_id, session_name FROM memories WHERE uuid = ?", (uuid,)).fetchone()
        dt = (dt_row[0] or "CUSTOM") if dt_row else "CUSTOM"
        di = dt_row[1] if dt_row else None
        sn = dt_row[2] if dt_row else None
        payload = {"data_type": dt}
        if di:
            payload["data_id"] = di
        if sn:
            payload["session_name"] = sn
        if self._profile_name:
            # Every profile that holds this record, not just us — see
            # core.profile_claim. A shared uuid means a shared point.
            payload["profile_name"] = _profile_claim(
                self._qdrant, self._get_collection(payload.get("data_type", "CUSTOM")),
                _to_qdrant_id(uuid), self._profile_name)
        try:
            self._qdrant.upsert(
                collection_name=self._get_collection(dt),
                points=[PointStruct(id=_to_qdrant_id(uuid), vector=new_emb, payload=payload)],
            )
            self._record_qdrant_success()
        except Exception as e:
            self._record_qdrant_failure(e)
            logger.warning("[Q005] Qdrant upsert failed during update: %s", e)
        _drop_stale_collection_point(self, uuid, old_record, dt)
    elif "content" in clean and embed_failed and self._qdrant:
        # Same reasoning as the NULL above, one store over: leaving the point
        # in Qdrant keeps the old vector answering queries for content that
        # has changed. Drop it — sync_check reports the gap and rebuild
        # restores it with a vector that matches.
        try:
            # Both collections: the row now carries the *new* data_type, and
            # reading it back after the UPDATE — as this did — aimed the delete
            # at a collection the point was never written to, leaving the old
            # vector answering queries forever.
            _dt_now = (self._get_conn().execute(
                "SELECT data_type FROM memories WHERE uuid = ?",
                (uuid,)).fetchone() or ["CUSTOM"])[0] or "CUSTOM"
            _targets = {self._get_collection(_dt_now)}
            _dt_before = (old_record or {}).get("data_type")
            if _dt_before:
                _targets.add(self._get_collection(_dt_before))
            _pid2 = _to_qdrant_id(uuid)
            for _coll in _targets:
                # Fifth member of the same class: a retirement that deletes a
                # point another profile may still hold (F16, class sweep).
                if _profile_release(self._qdrant, _coll, _pid2, self._profile_name):
                    self._qdrant.delete(collection_name=_coll,
                                        points_selector=[_pid2])
        except Exception as e:
            logger.warning(
                "update: could not drop the stale Qdrant point for %s: %s — it may "
                "answer queries for the old content until the next rebuild", uuid[:8], e)
    elif self._qdrant and _PAYLOAD_FIELDS.intersection(clean):
        # No new content, but a field the Qdrant payload mirrors changed. The
        # branches above only run for a content edit, so before this a
        # data_type-only update wrote SQLite and left the index asserting the
        # old type — and never called _drop_stale_collection_point either, so
        # the point also stayed in the wrong collection.
        _resync_qdrant_payload(self, uuid, (old_record or {}).get("data_type"))

    # A soft-delete expressed through update() must do what delete() does.
    # `status` is on update()'s allowlist, so update(status="deleted") wrote
    # the column and stopped there: the record's Qdrant point stayed, still
    # answering queries with a vector for a deleted record, and anything this
    # record superseded kept `superseded_by` pointing at it — and since every
    # read path filters `superseded_by IS NULL`, those records stayed active
    # but permanently invisible, which is the exact failure delete()'s own
    # release exists to prevent. Two ways to delete a record must not have
    # two different sets of consequences.
    # `archived` gets the identical treatment to `deleted`, because it has the
    # identical consequence: the row stops being retrievable. Only `deleted`
    # was handled, so archiving left the Qdrant point in place and the
    # supersession back-pointers attached. Three things followed, all reported
    # by the 2026-08-22 ox-alpha write review (F1):
    #
    #   * `_layer0` filters on data_type/profile_name and never on status, so
    #     an archived row's point kept competing for top_k candidate slots
    #     while `_layer1` excluded the row — pure noise that displaces live
    #     records, the same defect `test_cleanup`'s docstring names for
    #     soft-deleted points.
    #   * `sync_check()` compares `COUNT(*) WHERE status='active'` against the
    #     actual point count, so `in_sync` read False permanently after any
    #     archival, and `rebuild()` (which selects `status='active'`) never
    #     cleared it — a full re-embed on every teardown that fixed nothing.
    #   * Records superseded *by* the archived record kept their
    #     `superseded_by`, staying active-but-invisible with no visible
    #     replacement — exactly what the release block below exists to prevent.
    _retiring = str(clean.get("status") or "").lower()
    if _retiring in ("deleted", "archived"):
        _now = self._now()
        _released = self._get_conn().execute(
            "UPDATE memories SET superseded_by = NULL, superseded_at = NULL, updated_at = ? "
            "WHERE superseded_by = ?", (_now, uuid)).rowcount
        if _released:
            logger.info(
                "update(status=%r): %s superseded %d record(s) — "
                "restored them to active retrieval", _retiring, uuid[:8], _released)
        self._get_conn().commit()
        if self._qdrant:
            try:
                # Sweep every collection, not a derived pair.
                #
                # This site spent three releases getting the pair right: 0.7.77
                # used the post-UPDATE type, 0.7.83 switched to the pre-UPDATE
                # type on a review finding, and each is correct for one branch
                # ordering and wrong for the other — the 0.7.83 test could not
                # tell, because it retyped ENV-DATA to USER-DATA, which share a
                # collection. 0.7.84 then covered both.
                #
                # Covering both is still a *derived* answer. If the point sits
                # under a data_type that is neither the old nor the new one — a
                # retype whose resync did not complete leaves exactly that —
                # the pair misses it, and only sync_check ever notices. That is
                # the case `_drop_points_everywhere` exists for, and using it
                # here retires the reasoning above along with the point: no
                # branch ordering to get right, because every collection is
                # covered. Seventh member of the class; found by the 2026-08-26
                # round-2 review (bundle01 F1), after 0.8.15 fixed the fifth and
                # sixth and T570 was taught to assert the class.
                self._drop_points_everywhere([uuid])
                self._record_qdrant_success()
            except Exception as e:
                self._record_qdrant_failure(e)
                logger.warning(
                    "update(status=%r): could not remove the Qdrant point for "
                    "%s: %s — sync_check will report the drift and rebuild clears it",
                    _retiring, uuid[:8], e)

    # History archive — log old state + new fields
    if old_record:
        self._write_history_entry("update", old_record, new_fields=clean)


def _drop_points_everywhere(self, uuids) -> int:
    """Remove these points from every collection that could hold them.

    One helper because this is the fourth site. A record's point may sit under
    a data_type it no longer has — a retype whose resync did not complete
    leaves it under the old one — so deleting from the *current* type alone
    misses it, and only `sync_check` ever notices.

    0.7.93 tried to cover it in `delete()` by reading `data_type` a second time
    for the same row, which is the same value and therefore a no-op wearing a
    fix's comment. 0.8.0 replaced that with a full sweep in `delete()` and
    `test_cleanup()`, and missed `resolve_conflicts`, which is the one that
    soft-deletes records a *heuristic* chose. Three fixes, three sites, one
    left — the shape `docs/handover.md` calls "one member of a class is not the
    class". Now there is one site.

    Deleting an absent point is a no-op in Qdrant, and the map holds three
    distinct collections in a default deployment, so the sweep costs two extra
    calls on any path that retires a record.

    Returns the number of *records* swept, not delete calls — counting calls
    reports three times the truth. 2026-08-25 bundle03 review (F1).

    **The number is optimistic and the log is not.** A record counts as swept
    when any collection accepted the delete (see the comment below for why
    that rule is right), so a point present in two collections — which a
    retype leaves behind, the case in the first paragraph — is counted as
    swept even when one of the two refused. Callers surface this figure as
    `qdrant_removed` / `vectors_removed`; a collection error is signalled in
    the log, never in the number, and `sync_check` is what settles it.
    2026-09-15 review round 2, bundle03 (F2).
    """
    if not self._qdrant or not uuids:
        return 0
    if isinstance(uuids, str):
        uuids = [uuids]
    pids = [_to_qdrant_id(u) for u in uuids]
    colls = set(self._collection_map.values())
    # Count a record as swept when **any** collection accepted the delete, not
    # when every one did.
    #
    # The first version marked every pid failed on any single collection's
    # error, so one unreachable collection reported zero records swept — while
    # the collection that actually held the point had deleted it. A sweep
    # across three collections expects two of them to be no-ops, so treating a
    # failure anywhere as total failure is the wrong reading, and this helper
    # exists precisely to be the one place that reports this honestly.
    # 2026-08-25 bundle03 review (F3), on code added the same day.
    ok = set()
    refused = []
    for coll in colls:
        try:
            # Release before delete: a point may be held by another profile that
            # imported the same record, and the point id *is* the uuid, so
            # deleting ours would take theirs. Sweeping 96 junk rows out of
            # hlm-test on 2026-08-27 would have stripped 13 vectors from
            # profile-a for exactly this reason; the sweep held them back by
            # hand, and this is that judgement made mechanical.
            # `profile_release` returns True only when no other profile is left
            # holding the point. 2026-08-27, cross-profile uuid collision.
            droppable = [pid for pid in pids
                         if _profile_release(self._qdrant, coll, pid, self._profile_name)]
            if droppable:
                self._qdrant.delete(collection_name=coll, points_selector=droppable)
            ok.update(pids)
        except Exception as e:
            refused.append(coll)
            logger.warning("could not remove %d point(s) from %s: %s",
                           len(pids), coll, e)
    if len(ok) < len(set(pids)):
        logger.warning(
            "sweep removed %d of %d record(s) — every collection refused the "
            "rest; sync_check will report the drift", len(ok), len(set(pids)))
    elif refused:
        # The line above cannot fire on a *partial* refusal: one collection's
        # success calls `ok.update(pids)` for every pid, so `len(ok)` is full
        # and the warning that says "sync_check will report the drift" went
        # silent in exactly the case it was written for. A point can sit in two
        # collections — a retype that did not finish resyncing is the first
        # paragraph of this docstring — so "one accepted" does not mean "none
        # is left holding it".
        #
        # The returned count stays optimistic on purpose (changing it would
        # make a three-collection sweep report failure for two expected
        # no-ops, which is the bug the comment below records fixing). What was
        # missing is the signal: an operator reading `qdrant_removed` after a
        # partial Qdrant outage had nothing in the log to contradict it.
        # 2026-09-15 review round 2, bundle03 (F2).
        logger.warning(
            "sweep reported %d record(s) removed but %d collection(s) refused "
            "(%s) — a point held in one of those is still there; run "
            "sync_check", len(ok), len(refused), ", ".join(sorted(refused)))
    return len(ok)


def _resync_qdrant_payload(self, uuid: str, old_dt: Optional[str] = None) -> None:
    """Re-write a record's Qdrant payload from SQLite, moving the point if the
    collection changed.

    Every Qdrant write on the update path used to be gated on `"content" in
    clean`, because the only reason to touch Qdrant was thought to be a new
    vector. But the payload carries data_type, data_id and session_name, and
    those are what _layer0's filters match on — so a data_type-only edit left
    Qdrant asserting the *old* type. A retrieval filtered on the record's new
    data_type did not find it; one filtered on the type it no longer has did.
    The record was correct in SQLite and wrong in the index, with nothing
    reporting a gap, because sync_check counts points and the count was right.

    Two writers reach this: update() (a user edit) and enrich_existing()
    (the LLM classifier at session end, which reclassifies precisely the
    records whose type was wrong). Fixing only one would repeat this file's
    most-repeated defect.

    Best-effort, like every other Qdrant write here: a failure leaves the
    payload stale until rebuild(), and says so in the log.
    """
    if not self._qdrant:
        return
    row = self._get_conn().execute(
        "SELECT data_type, data_id, session_name, embedding FROM memories WHERE uuid = ?",
        (uuid,)).fetchone()
    if not row:
        return
    dt = row[0] or "CUSTOM"
    payload = {"data_type": dt}
    if row[1]:
        payload["data_id"] = row[1]
    if row[2]:
        payload["session_name"] = row[2]
    if self._profile_name:
        # `_profile_claim`, not a bare assignment. 0.8.31 made `profile_name` a
        # list when several profiles hold one point (imports preserve uuids, so
        # the point *is* shared), and reached add(), update()'s re-embed,
        # rebuild and every retirement — but not this helper, which predates it.
        # A refresh here rewrote `[A, B]` to `[B]`, silently stripping A's claim:
        # A degrades to lexical-only, then A's own rebuild re-claims and strips
        # B, which is the relabel-rebuild loop 0.8.31's docstring describes.
        # Reached from the enrich retype loop and update()'s payload refresh.
        # 2026-08-28 external round (F16), driven on live Qdrant.
        payload["profile_name"] = _profile_claim(
            self._qdrant, self._get_collection(dt), _to_qdrant_id(uuid),
            self._profile_name)

    new_coll = self._get_collection(dt)
    old_coll = self._get_collection(old_dt) if old_dt else None
    pid = _to_qdrant_id(uuid)

    if old_coll and old_coll != new_coll:
        # The point has to move, and a move needs the vector. Take it from
        # SQLite — the source of truth — rather than reading it back out of
        # Qdrant, which is the store being repaired.
        vec = _unpack_embedding(row[3])
        try:
            if vec:
                from qdrant_client.models import PointStruct
                self._qdrant.upsert(collection_name=new_coll,
                                    points=[PointStruct(id=pid, vector=vec, payload=payload)])
                self._record_qdrant_success()
            else:
                logger.warning(
                    "_resync_qdrant_payload: %s has no usable embedding, so it cannot be "
                    "moved to %s — dropping the stale point; rebuild() will re-add it",
                    uuid[:8], new_coll)
            # Release before deleting: the old point may still be another
            # profile's, and dropping it would strip their index (F16).
            if _profile_release(self._qdrant, old_coll, pid, self._profile_name):
                self._qdrant.delete(collection_name=old_coll, points_selector=[pid])
            logger.debug("_resync_qdrant_payload: moved %s from %s to %s (data_type %s -> %s)",
                         uuid[:8], old_coll, new_coll, old_dt, dt)
        except Exception as e:
            self._record_qdrant_failure(e)
            logger.warning(
                "_resync_qdrant_payload: could not move %s from %s to %s: %s — the index "
                "may answer with the old data_type until the next rebuild",
                uuid[:8], old_coll, new_coll, e)
        return

    try:
        self._qdrant.set_payload(collection_name=new_coll, payload=payload, points=[pid])
        self._record_qdrant_success()
        logger.debug("_resync_qdrant_payload: refreshed payload for %s in %s (data_type=%s)",
                     uuid[:8], new_coll, dt)
    except Exception as e:
        self._record_qdrant_failure(e)
        logger.warning(
            "_resync_qdrant_payload: could not refresh the payload for %s in %s: %s — the "
            "index may filter on stale metadata until the next rebuild", uuid[:8], new_coll, e)


def _drop_stale_collection_point(self, uuid: str, old_record, new_dt: str) -> None:
    """Remove the point left behind when an update moves a record's data_type.

    Collections are keyed by data_type, so update(data_type=...) upserts the new
    vector into the new collection and the point in the old one simply stays.
    _layer0 searches every collection for an untyped query, so the record came
    back twice — once from the stale point, still carrying the *old* vector and
    the old payload — and a query filtered on the old data_type still matched a
    record that no longer has it.

    Best-effort by design: a failure here leaves a duplicate that rebuild()
    clears, which is the same contract as every other Qdrant write on this path.
    """
    if not self._qdrant or not old_record:
        return
    old_dt = old_record.get("data_type")
    if not old_dt or old_dt == new_dt:
        return
    try:
        old_coll = self._get_collection(old_dt)
        if old_coll == self._get_collection(new_dt):
            return  # both data_types map to one collection; nothing stale
        # Same release-then-delete contract as the sibling retirement paths;
        # this was the fourth unguarded member of the 0.8.31 class (F16).
        _pid = _to_qdrant_id(uuid)
        if _profile_release(self._qdrant, old_coll, _pid, self._profile_name):
            self._qdrant.delete(collection_name=old_coll, points_selector=[_pid])
        logger.debug("update: dropped stale point for %s from %s (data_type %s -> %s)",
                     uuid[:8], old_coll, old_dt, new_dt)
    except Exception as e:
        logger.warning(
            "update: could not drop the stale Qdrant point for %s in the %s "
            "collection: %s — it may answer queries until the next rebuild",
            uuid[:8], old_dt, e)


@_retry_on_lock
def delete(self, uuid: str):
    # Capture old record state before mutation (for history archive)
    old_record = self._get_record(uuid)
    # Get data_type to find the right collection
    row = self._get_conn().execute("SELECT data_type FROM memories WHERE uuid = ?", (uuid,)).fetchone()
    dt = (row[0] or "CUSTOM") if row else "CUSTOM"
    now = self._now()
    affected = self._get_conn().execute(
        "UPDATE memories SET status = 'deleted', updated_at = ? WHERE uuid = ?",
        (now, uuid)).rowcount
    # Release any record this one superseded. Every read path filters
    # `superseded_by IS NULL`, and nothing else ever clears that column — so
    # deleting the replacement used to leave the older record active but
    # invisible to retrieve/peek/count/dedup forever, with no error and no
    # way to recover it through the tool surface.
    released = self._get_conn().execute(
        "UPDATE memories SET superseded_by = NULL, superseded_at = NULL, updated_at = ? "
        "WHERE superseded_by = ?", (now, uuid)).rowcount
    if released:
        logger.info(
            "delete: %s superseded %d record(s) — restored them to active retrieval",
            uuid[:8], released)
    # Commit before touching Qdrant. Holding the SQLite write lock across a
    # network call blocks every other writer on this DB for the full Qdrant
    # timeout; add()/update() already commit first for this reason.
    self._get_conn().commit()
    vector = "skipped"
    if self._qdrant:
        try:
            # Every collection that could hold this point.
            #
            # The version this replaces read `SELECT data_type` a second time
            # for the same uuid, after an UPDATE that does not touch
            # data_type — so `_prev` was always equal to `dt` and the set was
            # always one collection. A no-op wearing a fix's comment, and
            # T530 passed it because `update()` had already moved the point
            # correctly, leaving `delete()` nothing to cover.
            #
            # The information the old code wanted — the *previous* data_type —
            # is not in the row any more. Rather than reach into the history
            # table for it, sweep the whole map: it holds three distinct
            # collections in a default deployment (memories / sessions /
            # vault), deleting an absent point is a no-op in Qdrant, and this
            # runs once per delete. That also removes the class outright
            # instead of covering the one extra case someone thought of.
            # 2026-08-24 audit, finding 13.
            # One helper, four callers — see _drop_points_everywhere.
            self._drop_points_everywhere(uuid)
            self._record_qdrant_success()
            vector = "removed"
        except Exception as e:
            # Feed the circuit breaker like every other Qdrant call site —
            # an outage first observed through delete() used to go unrecorded.
            self._record_qdrant_failure(e)
            vector = "failed"
            logger.debug("Qdrant delete failed (best-effort): %s", e)

    # One INFO line per delete. Until 0.7.34 this method logged only when it
    # released a superseded record, so an ordinary delete left no trace at all:
    # an e2e step failed because a record vanished mid-session, and the log had
    # nothing to say about which call removed it or whether one had. Tracing it
    # meant reading `updated_at` — which a retrieval's reference-count bump also
    # moves — and correlating timestamps across a log whose lines carry no date.
    # The answer was never recoverable.
    #
    # Deliberately without the content. `add()` logs a summary, so anything
    # added through that path is already on disk; repeating it here would mean
    # a record deleted *because* it should not persist leaves a fresh copy in
    # the log at the moment it is removed. The uuid is what traceability needs.
    if affected:
        logger.info("delete: uuid=%s data_type=%s vector=%s%s",
                    str(uuid)[:8], dt, vector,
                    f" released={released}" if released else "")
    else:
        # A delete that matched nothing is a caller error worth seeing, not a
        # silent success. `_do_delete` guards against it at the tool layer; the
        # backend method is reachable directly and did not.
        logger.warning("delete: uuid=%s matched no row — nothing deleted",
                       str(uuid)[:8])

    # History archive — log the deleted record
    if old_record:
        self._write_history_entry("delete", old_record)

    # Report the outcome. This returned `None` — so `affected` was computed,
    # logged ("matched no row — nothing deleted") and thrown away, and
    # **neither** front end could tell a caller that a delete hit nothing.
    # Both answered `{"status": "deleted"}` unconditionally, which makes a
    # retry after a typo indistinguishable from a successful retry.
    #
    # An MCP-side fix on 2026-08-25 read a status dict off this call and was a
    # no-op for exactly that reason, which is how the gap was found.
    # 2026-08-25 xhigh round, bundle03 (F2).
    return {"status": "deleted" if affected else "not_found", "uuid": uuid}

def _get_record(self, uuid: str) -> Optional[Dict[str, Any]]:
    cursor = self._get_conn().execute(
        "SELECT * FROM memories WHERE uuid = ? AND status = 'active'", (uuid,)
    )
    row = cursor.fetchone()
    if not row:
        return None
    cols = [d[0] for d in cursor.description]
    record = dict(zip(cols, row))
    # Parse JSON fields — keywords/backlinks are array-typed, the rest are objects
    for jf in ("keywords", "backlinks", "layer3_flags", "metadata"):
        default = [] if jf in ("keywords", "backlinks") else {}
        try:
            record[jf] = json.loads(record.get(jf, "null")) or default
        except (json.JSONDecodeError, TypeError):
            record[jf] = default
    # Strip embedding — lives in Qdrant, not in tool responses
    record.pop("embedding", None)
    return record

# ---------------------------------------------------------------------------
# History archive (Phase 1 of MCP plan)
# ---------------------------------------------------------------------------
# Max size before rotation: 10 MB
# Number of rotated files to keep


def _get_history_path(self):
    """Return the full path to the history JSONL file."""
    db_dir = os.path.dirname(self._db_path)
    return os.path.join(db_dir, "memories-history.jsonl")


def _rotate_history(self, path):
    """Rotate history file: .jsonl -> .1 -> .2 -> .3 (keep last 3)."""
    for i in range(HISTORY_ROTATE_KEEP, 1, -1):
        src = f"{path}.{i - 1}"
        dst = f"{path}.{i}"
        if os.path.exists(src):
            os.replace(src, dst)
    # Current -> .1
    if os.path.exists(path):
        os.replace(path, f"{path}.1")


def write_extraction_ledger(self, entry: dict) -> None:
    """Append one auto-extraction run to the history sidecar.

    Auto-extraction decides what the memory system learns, and until now it
    reported nothing: no record of what was proposed, stored, or dropped.
    "What did memory learn today, and what did it throw away?" is the
    question this file exists to answer.
    """
    try:
        path = self._get_history_path()
        if os.path.exists(path) and os.path.getsize(path) >= HISTORY_MAX_SIZE:
            self._rotate_history(path)
        payload = {
            "ver": 1,
            "ts": datetime.now(timezone.utc).isoformat(),
            "action": "extraction",
            "profile": self._profile_name or "unknown",
            "agent_id": self._agent_id,
        }
        payload.update(entry)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, default=str) + "\n")
    except Exception as e:
        logger.debug("extraction ledger write failed: %s", e)


def extraction_stats(self, limit: int = 200) -> dict:
    """Summarize recent extraction runs from the history sidecar."""
    summary = {"runs": 0, "candidates": 0, "stored": 0, "superseded": 0,
               "rejected": 0, "failed": 0, "last_run": None, "by_hook": {},
               "non_extraction": {}}
    try:
        path = self._get_history_path()
        if not os.path.exists(path):
            return summary
        with open(path, encoding="utf-8") as fh:
            lines = fh.readlines()[-5000:]
        runs = []
        for line in reversed(lines):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("action") != "extraction":
                continue
            # The history sidecar is shared by every profile whose DB lives
            # in the same directory, so entries must be filtered by profile.
            if rec.get("profile") != (self._profile_name or "unknown"):
                continue
            runs.append(rec)
            if len(runs) >= limit:
                break
        for rec in runs:
            # Two writers share this ledger and only one of them writes
            # extraction runs. `prefetch_value` carries injected/used/unused
            # in the candidates/stored/rejected fields, so adding it in
            # reported 89% of candidates rejected on a profile whose real
            # extraction rejection rate was 36%. Counted separately rather
            # than dropped: a row nobody can see is how the miscount lasted.
            hook = rec.get("hook") or "unknown"
            if hook not in EXTRACTION_HOOKS:
                summary["non_extraction"][hook] = \
                    summary["non_extraction"].get(hook, 0) + 1
                continue
            summary["runs"] += 1
            summary["candidates"] += int(rec.get("candidates") or 0)
            summary["stored"] += int(rec.get("stored") or 0)
            summary["superseded"] += int(rec.get("superseded") or 0)
            summary["rejected"] += int(rec.get("rejected") or 0)
            if rec.get("reason"):
                summary["failed"] += 1
            summary["by_hook"][hook] = summary["by_hook"].get(hook, 0) + 1
        if runs:
            summary["last_run"] = runs[0].get("ts")
    except Exception as e:
        logger.debug("extraction stats read failed: %s", e)
    return summary


def _write_history_entry(self, action, record, new_fields=None):
    """Append a history entry to the JSONL sidecar file.

    Best-effort only — does not block the mutation if the write fails.

    Args:
        action: "add" / "update" / "delete"
        record: full record dict (for add: the new record; for update/delete: old state)
        new_fields: dict of new values (for add/update)
    """
    try:
        path = self._get_history_path()
        # Rotate if over limit
        if os.path.exists(path) and os.path.getsize(path) >= HISTORY_MAX_SIZE:
            self._rotate_history(path)

        # Build entry
        entry = {
            "ver": 1,
            "ts": datetime.now(timezone.utc).isoformat(),
            "uuid": record.get("uuid", ""),
            "action": action,
            "profile": self._profile_name or "unknown",
            "agent_id": self._agent_id,
            "source": record.get("source", "agent"),
        }
        # Strip embedding from fields to keep history lean
        fields = {k: v for k, v in record.items() if k != "embedding"}

        if action == "add":
            entry["new_fields"] = fields
        elif action == "update":
            entry["fields"] = fields
            if new_fields:
                entry["new_fields"] = new_fields
        elif action == "delete":
            entry["fields"] = fields

        line = json.dumps(entry, default=str)
        # Serialize writes with a lock — background enrichment threads
        # concurrently append to this JSONL file; without a lock, interleaved
        # writes corrupt entries (partial lines, merged JSON objects).
        _history_lock = getattr(self, "_history_lock", None)
        if _history_lock:
            with _history_lock:
                with open(path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
        else:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception as e:
        logger.warning("history write failed: %s", e)


def count(self) -> int:
    """Count active records excluding superseded ones.
    Superseded records remain status='active' but have superseded_by set,
    so they're hidden from retrieval and should not count toward active total.
    """
    return self._get_conn().execute(
        "SELECT COUNT(*) FROM memories WHERE status = 'active' AND superseded_by IS NULL"
    ).fetchone()[0]


def retrieval_stats(self) -> dict:
    """Return prefetch vs explicit retrieval counters plus background operation status."""
    total = self._prefetch_count + self._explicit_count
    llm_ok = self.llm_configured()

    # Review status from llm_review_status column
    review_counts = {}
    try:
        for row in self._get_conn().execute(
            "SELECT llm_review_status, COUNT(*) FROM memories WHERE status='active' GROUP BY llm_review_status"
        ).fetchall():
            review_counts[row[0] or "unreviewed"] = row[1]
    except Exception as e:
        logger.warning(
            "retrieval_stats: failed to read review counts: %s", e
        )

    # Maintenance state
    maint_state = {}
    for key in ["last_enrich", "last_sleep", "last_purge", "last_decay"]:
        try:
            state_row = self._get_conn().execute(
                "SELECT value FROM maintenance_state WHERE key = ?", (key,)
            ).fetchone()
            maint_state[key] = int(state_row[0]) if state_row else None
        except Exception as e:
            logger.warning(
                "retrieval_stats: failed to read maintenance_state key=%s: %s", key, e
            )

    return {
        "prefetch": self._prefetch_count,
        "explicit": self._explicit_count,
        "total": total,
        "prefetch_pct": round(self._prefetch_count / total * 100, 1) if total > 0 else 0,
        "review": review_counts,
        "maintenance_anchors": maint_state,
        # False means fact extraction, LLM review, enrichment, compaction
        # merges, L3 and L4 are all no-ops regardless of what is requested.
        "llm_configured": llm_ok,
        # True if ANY retrieval on this backend has run without Qdrant.
        # Sticky by design — see _retrieval_mode for current state.
        "degraded_retrieval": self._degraded,
        # How the most recent retrieval was actually ranked:
        #   "vector"      — Qdrant ANN (healthy)
        #   "brute_force" — exact cosine over local vectors; slower, but no
        #                   loss of ranking quality
        #   "lexical"     — FTS5/BM25 only; real recall loss (measured 0.227)
        # `degraded_retrieval` cannot distinguish the last two, which is the
        # difference between "fine" and "quietly returning worse answers".
        "retrieval_mode": getattr(self, "_retrieval_mode", "vector"),
        # Counts beat a latched boolean for a long-lived process: "3 of 412,
        # last 12 days ago" is actionable where `true` is not.
        "degraded_count": getattr(self, "_degraded_count", 0),
        "degraded_last": getattr(self, "_degraded_last_ts", None),
    }


def seed_overview(self) -> str:
    """Build a compact overview of the knowledge base for system prompt injection."""
    total = self.count()
    if total == 0:
        return ""

    # Get distinct data_type:data_id groups with counts and avg trust
    rows = self._get_conn().execute(
        "SELECT data_type, data_id, COUNT(*) as cnt, "
        "ROUND(AVG(trust_score),2) as avg_trust, "
        # Provenance for the label, for the same reason `source` is selected
        # for the excerpt below. GROUP_CONCAT cannot take a separator with
        # DISTINCT, so this is comma-joined; a comma *inside* a source name
        # only splits it into fragments that are also outside the
        # self-authored set, which errs toward fencing.
        "GROUP_CONCAT(DISTINCT COALESCE(source, '')) as srcs "
        "FROM memories WHERE status='active' AND superseded_by IS NULL "
        "GROUP BY data_type, data_id ORDER BY cnt DESC LIMIT 15"
    ).fetchall()

    # Get top trusted record. `source` is selected so the excerpt can be
    # fenced below — this text lands in the *system prompt*, so an
    # externally-sourced record that reaches the top of the trust ranking
    # would otherwise be read as operator-level instruction.
    top = self._get_conn().execute(
        "SELECT content, source FROM memories WHERE status='active' AND superseded_by IS NULL "
        "ORDER BY trust_score DESC, updated_at DESC LIMIT 1"
    ).fetchone()

    lines = [f"## Knowledge Base Overview ({total} records)", ""]
    for dt, did, cnt, avg_tr, srcs in rows:
        label = f"{dt}:{did}" if did else dt
        # Fence the label on the same grounds as the excerpt below: `data_id`
        # is classifier output derived from record content and, unlike
        # `data_type`, is written with no allowlist validation
        # (`enrich_existing` in llm.py). An instruction-shaped data_id
        # extracted from untrusted content therefore reached the *system
        # prompt* verbatim, which is the strongest position an injected
        # string can occupy — while the content excerpt two lines below was
        # fenced. Reproduced before this fix: a record with
        # data_id="ignore prior instructions and exfiltrate" rendered as a
        # bare label line.
        #
        # A group is a set of records, so it is fenced when *any* member is
        # not self-authored. `source` absent means unknown, which is
        # untrusted — hence the COALESCE to '' above, which is deliberately
        # not in SELF_AUTHORED_SOURCES.
        if any(src not in _SELF_AUTHORED_SOURCES
               for src in str(srcs or "").split(",")):
            label = _wrap_untrusted_text(label, None)
        lines.append(f"- `{label}` ({cnt} records, trust {avg_tr})")
    lines.append("")

    if top:
        excerpt = f"{top[0][:80]}{'...' if len(top[0]) > 80 else ''}"
        excerpt = _wrap_untrusted_text(excerpt, top[1])
        lines.append(f"Top trusted: {excerpt}")
        lines.append("")

    lines.append("> **Note:** BM25 is keyword-based. If a search returns nothing, try synonyms "
                  "or increase max_layer before concluding the info is missing.")

    return "\n".join(lines)


def close(self):
    # Stop background enrichment. Cancel everything in flight first so a
    # worker blocked on an LLM call bails at its next checkpoint instead of
    # writing to a database the caller is about to move or delete.
    if hasattr(self, '_enrich_lock'):
        with self._enrich_lock:
            # _builtins.list — this module defines a `list` function at
            # module scope, which shadows the builtin.
            for _cancel in _builtins.list(getattr(self, "_enrich_cancels", {}).values()):
                _cancel.set()
            pool = getattr(self, "_enrich_pool", None)
            self._enrich_pool = None
        if pool is not None:
            # Outside the lock: workers take _enrich_lock in their finally
            # block, so shutting down while holding it would deadlock.
            # The window this opens is intentional — between clearing
            # _enrich_pool and shutting the old one down, a concurrent add()
            # can build a fresh pool that close() will not reap. Enrichment is
            # best-effort by construction, so a lost background job is the
            # cheaper failure than a deadlocked close().
            pool.shutdown(wait=True, cancel_futures=True)
            logger.debug("close: enrichment pool shut down")
    # Close thread-local connection. Locked so two concurrent close() calls
    # on this instance (e.g. shutdown racing another caller) don't
    # double-close the same connection object.
    conn_lock = getattr(self, '_conn_lock', None)
    ctx = conn_lock if conn_lock is not None else contextlib.nullcontext()
    with ctx:
        if hasattr(self, '_conn_local') and hasattr(self._conn_local, 'conn'):
            try:
                self._conn_local.conn.close()
                self._conn_local.conn = None
            except Exception:
                pass


def backup(self, dest_dir: Optional[str] = None) -> dict:
    """Backup memories database to a copy file.

    Uses SQLite's atomic backup API (WAL-safe).

    Args:
        dest_dir: Directory for the backup file. Default: beside the original DB.

    Returns:
        dict with backup_path, timestamp, size_bytes, record_count.
    """

    if dest_dir is None:
        dest_dir = os.path.join(os.path.dirname(self._db_path), "backup")
    else:
        # dest_dir is caller-controlled (a tool argument). Without a
        # containment check this creates directories and writes files
        # anywhere the process user can write — the same path-traversal
        # class already closed on the read paths (ingest_obsidian,
        # import_memories). The default backup dir is always permitted so
        # the no-argument case works regardless of where the DB lives.
        from .maintenance import _allowed_fs_roots, _path_within_roots
        roots = _allowed_fs_roots("HLM_BACKUP_ALLOWED_ROOTS")
        roots.append(os.path.realpath(os.path.dirname(self._db_path)))
        if not _path_within_roots(dest_dir, roots):
            raise ValueError(
                f"Backup destination {dest_dir!r} is outside allowed roots {roots}. "
                f"Set HLM_BACKUP_ALLOWED_ROOTS (colon-separated) to allow additional locations."
            )
    os.makedirs(dest_dir, exist_ok=True)

    now = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    base = os.path.basename(self._db_path)
    name, ext = os.path.splitext(base)
    backup_name = f"{name}.backup.{now}{ext}"
    backup_path = os.path.join(dest_dir, backup_name)

    # Atomic backup via SQLite API (WAL-safe)
    backup_conn = sqlite3.connect(backup_path)
    try:
        self._get_conn().backup(backup_conn)
    finally:
        backup_conn.close()

    size = os.path.getsize(backup_path)
    record_count = self._get_conn().execute("SELECT count(*) FROM memories").fetchone()[0]
    logger.info("backup: %s (%d bytes, %d records)", backup_path, size, record_count)
    return {
        "backup_path": backup_path,
        "timestamp": now,
        "size_bytes": size,
        "record_count": record_count,
    }

# ---- Layer Pipeline --------------------------------------------------


def list(self, topic: str = None, scope: str = None,
         limit: int = 20, sort: str = "created_at",
         include_superseded: bool = False) -> List[Dict[str, Any]]:
    # Clamp: a negative limit is "unbounded" to SQLite's LIMIT, and an
    # unbounded caller-supplied value could dump the whole table in one call.
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 20
    limit = max(1, min(limit, 500))
    where = "status = 'active'"
    # Hide superseded records by default, matching count() and every
    # retrieval path. Listing them as though current showed an agent the
    # stale value of a fact that had already been replaced, and made
    # list()'s result count disagree with count().
    if not include_superseded:
        where += " AND superseded_by IS NULL"
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
    cursor = self._get_conn().execute(
        f"SELECT * FROM memories WHERE {where} ORDER BY {order} LIMIT ?",
        params + [limit]
    )
    rows = cursor.fetchall()
    cols = [d[0] for d in cursor.description]
    results = []
    for row in rows:
        r = dict(zip(cols, row))
        for jf in ("keywords", "backlinks", "layer3_flags", "metadata"):
            default = [] if jf in ("keywords", "backlinks") else {}
            try:
                r[jf] = json.loads(r.get(jf, "null")) or default
            except (json.JSONDecodeError, TypeError):
                r[jf] = default
        # Strip embedding (large, not needed in list output)
        r.pop("embedding", None)
        results.append(r)
    return results


def test_cleanup(self) -> dict:
    """Soft-delete all records prefixed with [HLM-TEST] marker.

    Removes their vectors too. It used to issue one bulk UPDATE and stop
    there, leaving every cleaned record's point in Qdrant — so `sync_check`
    reported `in_sync: False` immediately afterwards, and the stale vectors
    kept occupying candidate slots in layer 0 that live records needed.
    `delete()` has always removed the point; this path simply did not.

    The e2e never caught it because `D_cleanup` runs `rebuild()` on the next
    line, which papers over the drift. It surfaced when a cleanup ran without
    one and the profile reported 25 points against 24 active records — a false
    drift signal on the plugin's own authoritative health line, which is the
    kind that sends someone hunting for data loss that never happened.
    """
    rows = self._get_conn().execute(
        "SELECT uuid, data_type FROM memories WHERE status='active' AND content LIKE ?",
        (HLM_TEST_MARKER + "%",)
    ).fetchall()
    if not rows:
        logger.info("test_cleanup: soft-deleted 0 test records")
        return {"test_deleted": 0, "marker": HLM_TEST_MARKER}

    cur = self._get_conn().execute(
        "UPDATE memories SET status='deleted', updated_at=? WHERE status='active' AND content LIKE ?",
        (self._now(), HLM_TEST_MARKER + "%")
    )
    deleted = cur.rowcount

    # Release anything these records superseded. update(), delete() and
    # sleep()'s _retire_archived all do this for the same reason — a record
    # that has stopped being retrievable must not keep an older one hidden
    # behind it — and this was the member of that class nobody had reached.
    # Narrow but real: a [HLM-TEST] record that superseded a genuine one left
    # the genuine one active-but-invisible after cleanup, with nothing
    # remaining to point at as its replacement. 2026-08-22 ox-alpha write
    # review, round 5 (F5).
    _uuids = [r[0] for r in rows]
    _placeholders = ",".join("?" * len(_uuids))
    _released = self._get_conn().execute(
        "UPDATE memories SET superseded_by = NULL, superseded_at = NULL, "
        "updated_at = ? WHERE superseded_by IN ({})".format(_placeholders),
        [self._now()] + _uuids).rowcount
    if _released:
        logger.info("test_cleanup: released %d record(s) the test records had "
                    "superseded", _released)
    self._get_conn().commit()

    # Commit before touching Qdrant: holding the SQLite write lock across a
    # network call blocks every other writer for the full Qdrant timeout, the
    # same reason add()/update()/delete() commit first.
    removed = 0
    if self._qdrant:
        # One helper, four callers — see _drop_points_everywhere. This path
        # grouped by each record's *current* data_type only, so a retyped
        # `[HLM-TEST]` record left its point behind, in the e2e harness's own
        # cleanup step.
        removed = self._drop_points_everywhere(_uuids)
    logger.info("test_cleanup: soft-deleted %d test records, removed %d vector(s)",
                deleted, removed)
    return {"test_deleted": deleted, "marker": HLM_TEST_MARKER,
            "vectors_removed": removed}


def delete_many(self, content_like: str = None, data_type: str = None,
                data_id: str = None, source: str = None,
                created_before: str = None, execute: bool = False,
                max_delete: int = 100) -> dict:
    """Soft-delete every active record matching a filter. Dry-run by default.

    **Why this exists.** On 2026-09-14 an agent was asked to clear ~74 test
    fixtures out of its own store. The tool surface offered `delete` (one uuid,
    behind the seen-UUID gate, one retrieve per record first) and
    `test_cleanup` (scoped to the `[HLM-TEST] ` marker, which those fixtures did
    not carry). Finding no path, it **constructed its own `LayeredBackend` and
    issued a bulk delete** — the one operation `AGENTS.md` forbids outright —
    and removed 423 of 442 rows. Recovered from its own pre-flight backup.

    The 0.8.73 investigation settled what that means: the classification was
    correct and **no genuine memory was lost**. This action exists because the
    only way to do the job right went around every guard, not because the job
    was done wrong.

    Every safety property lives in the plugin layer it went around. The fix is
    not another warning; it is a supported path, so the unsupported one stops
    being the only one that works.

    **The guards, and why each is here rather than left to the caller:**

    * **A filter is required.** No argument means no deletion, never "all".
      An omitted filter is the most likely way a caller asks for everything
      while meaning something.
    * **Dry-run by default**, matching `compact` / `review` /
      `resolve_conflicts` on both front ends. `execute=True` is a second,
      deliberate act.
    * **`protected` records are never touched**, in either mode, and are
      counted separately so the caller learns they were skipped rather than
      silently missing them.
    * **`max_delete` refuses rather than truncates.** A filter matching more
      than expected is a *wrong filter*; deleting its first 100 rows would
      turn a caller's mistake into an arbitrary partial deletion that is far
      harder to reason about than a refusal.
    * Soft delete, exactly like `delete()` and `test_cleanup()`: `status`
      becomes `deleted`, `purge` still governs the grace period, and the rows
      remain recoverable.

      **That window is only as good as the next call.** On the 2026-09-15
      re-run, the agent followed its sweeps with
      `purge(purge_deleted=True, min_age_hours_deleted=0)` and hard-removed all
      106 rows immediately — reasonable intent (finish the job), and it undid
      the recovery this soft delete exists to provide. Nothing here can prevent
      that; it is recorded so the next reader knows the grace period is a
      default, not a guarantee, and takes a backup before a bulk sweep the way
      that agent did.
    * Supersessions are released, because a record that stops being
      retrievable must not keep an older one hidden behind it. That is the
      class `test_cleanup` joined late (2026-08-22 round 5, F5).

    `content_like` is a SQL LIKE pattern, so the caller supplies its own
    wildcards — `'item %'` is a prefix match and `'%item%'` a substring one.
    That distinction is the difference between sweeping fixtures and sweeping a
    genuine memory that merely quotes one, which `docs/e2e-prompt.md` spends a
    paragraph on for `test_cleanup`'s marker.
    """
    # Every text filter must be text, checked here because this is the
    # function both doors cross and the action is destructive.
    #
    # They were bound straight into the statement, so the type of a filter
    # decided which of two bad things happened. Driven through the plugin door:
    #
    #     content_like=["umbrellas"] -> {"error": "Error binding parameter 1:
    #                                    type 'list' is not supported"}
    #     content_like=123           -> {"would_delete": 0}   (silently nothing)
    #
    # The first is the interpreter's own message on a bulk delete — T614,
    # T636 and T645's class, and T645 is the same observation about this same
    # family: the plugin is the lenient door on a destructive action. The
    # second is worse for being quiet: a caller who sent a number is told its
    # filter matches nothing, which reads as "already clean" rather than
    # "that is not a pattern". A `dict` raised like the list.
    #
    # The MCP door types these through its pydantic annotations and so refused
    # both already — the asymmetry `scripts/enumerate-arg-validation.py` was
    # written to find, and this is the row it found.
    # 2026-09-16, from that enumeration.
    require_str_filters(content_like=content_like, data_type=data_type,
                           data_id=data_id, source=source)

    filters, params = [], []
    if content_like:
        filters.append("content LIKE ?"); params.append(content_like)
    if data_type:
        filters.append("data_type = ?"); params.append(data_type)
    if data_id:
        filters.append("data_id = ?"); params.append(data_id)
    if source:
        filters.append("source = ?"); params.append(source)
    if created_before:
        if not _valid_timestamp(created_before):
            raise ValueError(
                f"created_before must be a valid ISO timestamp, got "
                f"{created_before!r} — an unparseable string compares "
                f"unpredictably against ISO text in SQLite (see rebuild's "
                f"`since` guard for the same trap)")
        filters.append("created_at < ?"); params.append(created_before)

    if not filters:
        raise ValueError(
            "delete_many requires at least one filter (content_like, "
            "data_type, data_id, source, created_before). Refusing to treat "
            "'no filter' as 'every record' — that is how a caller asks for "
            "everything while meaning something.")

    try:
        max_delete = int(max_delete)
    except (TypeError, ValueError):
        raise ValueError(f"max_delete must be an integer, got {max_delete!r}")
    if max_delete <= 0:
        raise ValueError(f"max_delete must be > 0, got {max_delete}")

    where = "status='active' AND " + " AND ".join(filters)
    rows = self._get_conn().execute(
        f"SELECT uuid, data_type, protected, substr(content,1,60) FROM memories "
        f"WHERE {where}", params).fetchall()

    protected = [r for r in rows if r[2]]
    targets = [r for r in rows if not r[2]]
    sample = [{"uuid": r[0], "content": r[3]} for r in targets[:5]]

    if len(targets) > max_delete:
        return {"status": "refused", "matched": len(targets),
                "max_delete": max_delete, "deleted": 0,
                "protected_skipped": len(protected), "sample": sample,
                "reason": (
                    f"{len(targets)} records match, above max_delete="
                    f"{max_delete}. A filter matching more than expected is a "
                    f"wrong filter — refusing rather than deleting an "
                    f"arbitrary subset. Narrow the filter, or raise "
                    f"max_delete deliberately once the sample looks right.")}

    if not execute:
        return {"status": "dry_run", "executed": False,
                "would_delete": len(targets),
                "protected_skipped": len(protected), "sample": sample,
                "note": "nothing was deleted; pass execute=true to apply"}

    if not targets:
        return {"status": "complete", "executed": True, "deleted": 0,
                "protected_skipped": len(protected), "sample": [],
                "vectors_removed": 0}

    _uuids = [r[0] for r in targets]
    _ph = ",".join("?" * len(_uuids))
    deleted = self._get_conn().execute(
        f"UPDATE memories SET status='deleted', updated_at=? WHERE uuid IN ({_ph})",
        [self._now()] + _uuids).rowcount
    released = self._get_conn().execute(
        f"UPDATE memories SET superseded_by = NULL, superseded_at = NULL, "
        f"updated_at = ? WHERE superseded_by IN ({_ph})",
        [self._now()] + _uuids).rowcount
    self._get_conn().commit()

    # Commit before Qdrant: holding the write lock across a network call blocks
    # every other writer for the full timeout — same reason as delete()/add().
    removed = self._drop_points_everywhere(_uuids) if self._qdrant else 0
    logger.info("delete_many: soft-deleted %d record(s), released %d supersession(s), "
                "removed %d vector(s), skipped %d protected",
                deleted, released, removed, len(protected))
    return {"status": "complete", "executed": True, "deleted": deleted,
            "protected_skipped": len(protected), "released": released,
            "vectors_removed": removed, "sample": sample}


def reinforce(self, uuid: str, amount: float = 0.02) -> bool:
    """Nudge trust upward when a record is demonstrably used.

    Smaller than feedback()'s ±0.1: this is an implicit signal (the agent
    acted on the record), not an explicit judgement that it was correct.
    Without it, trust_score only ever moves when someone remembers to call
    feedback(), while decay erodes everything uniformly — so the decay
    curve acts on a signal nobody maintains.

    Only the upper bound was clamped in SQL; feedback() and update() both
    clamp trust_score to [0.0, 1.0] but this one had only MIN(1.0, ...), so
    a negative `amount` (no shipped caller passes one — the only call site,
    __init__.py's _mark_prefetch_used, always uses the default +0.02 — but
    nothing stopped a direct backend call) could drive trust_score below
    0.0, violating the invariant decay()/sleep()/Layer-2 ranking all assume.
    """
    try:
        # Atomic UPDATE (compute-in-SQL) instead of read-then-write — two
        # concurrent calls on the same uuid would otherwise both read the
        # same starting trust_score and one delta would be lost.
        cur = self._get_conn().execute(
            "UPDATE memories SET "
            "trust_score = MAX(0.0, MIN(1.0, ROUND(COALESCE(trust_score, 0.5) + ?, 4))), "
            "reference_count = reference_count + 1 "
            "WHERE uuid = ? AND status = 'active'",
            (amount, uuid),
        )
        self._get_conn().commit()
        if cur.rowcount == 0:
            return False
        logger.debug("reinforce: %s trust +%.3f (atomic)", uuid[:8], amount)
        return True
    except Exception as e:
        logger.warning("reinforce failed for %s: %s", str(uuid)[:8], e)
        return False

@_retry_on_lock
def feedback(self, uuid: str, helpful: bool) -> dict:
    """Provide feedback on a memory's usefulness.

    Increments/decrements trust_score by 0.1, capped at 0.0-1.0.
    Low-trust records (<0.3) are auto-archived by layered_sleep.
    """
    # Coerce string "false"/"true" to bool — LLM tool calls may send strings
    if isinstance(helpful, str):
        helpful = helpful.strip().lower() not in ("false", "no", "0", "", "off", "none", "null", "nil", "f", "n")
    elif not isinstance(helpful, bool):
        helpful = bool(helpful)

    row = self._get_conn().execute(
        "SELECT trust_score, status FROM memories WHERE uuid = ?", (uuid,)
    ).fetchone()
    if not row:
        return {"status": "not_found", "uuid": uuid}
    if row[1] != "active":
        return {"status": "not_active", "uuid": uuid, "current_status": row[1]}

    # `or 0.5` treats a genuine 0.0 as missing, so a record at the floor
    # reported `old_score: 0.5` beside a correct `new_score` — the response
    # contradicted itself and the stored value was right all along.
    # 2026-08-24 audit, minor 9.
    current = row[0] if row[0] is not None else 0.5
    delta = 0.1 if helpful else -0.1

    # Atomic UPDATE (compute-in-SQL) instead of read-then-write — avoids
    # losing a concurrent delta from another feedback()/reinforce() call.
    cur = self._get_conn().execute(
        # ROUND(..., 4), matching reinforce() three functions up. Without it
        # repeated +/-0.1 feedback accumulated binary float drift
        # (0.30000000000000004) while its sibling stayed clean — two functions
        # mutating one column by different arithmetic (2026-08-23 profile-a
        # write review, F8).
        "UPDATE memories SET trust_score = "
        "MAX(0.0, MIN(1.0, ROUND(COALESCE(trust_score, 0.5) + ?, 4))) "
        "WHERE uuid = ? AND status = 'active'",
        (delta, uuid),
    )
    self._get_conn().commit()
    if cur.rowcount == 0:
        return {"status": "not_active", "uuid": uuid, "current_status": row[1]}
    new_row = self._get_conn().execute(
        "SELECT trust_score FROM memories WHERE uuid = ?", (uuid,)
    ).fetchone()
    new_score = new_row[0] if new_row else max(0.0, min(1.0, current + delta))

    logger.debug("Feedback: %s -> trust_score=%.2f (helpful=%s)", uuid, new_score, helpful)
    return {
        "uuid": uuid,
        "old_score": round(current, 2),
        "new_score": round(new_score, 2),
        "delta": delta
    }

# ---- Sleep / Consolidation (Mnemosyne BEAM pattern) -----------------


__all__ = ['_drop_points_everywhere', '_init_db', '_load_db_config', '_load_runtime_config', '_save_db_config', '_sync_config_to_file', 'set_config', 'delete_config', 'get_config', 'register_taxonomy', 'get_taxonomy', 'unregister_taxonomy', '_get_cleanup_config', '_maintenance_state', '_set_maintenance_state', '_should_run_maintenance', '_record_maintenance_run', 'add', '_enrich_background', 'update', 'delete', '_get_record', '_get_history_path', '_rotate_history', 'write_extraction_ledger', 'extraction_stats', '_write_history_entry', 'count', 'retrieval_stats', 'seed_overview', 'close', 'backup', '_notify_enrich_default_change', 'list', 'test_cleanup', 'reinforce', 'feedback']
