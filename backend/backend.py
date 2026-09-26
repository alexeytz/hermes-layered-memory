"""Layered memory backend — SQLite + Qdrant + layer pipeline.

This is the engine. The MemoryProvider in __init__.py is a thin wrapper.

Shared primitives (logger, _setting, embeddings, encodings) live in
backend/core.py and are re-exported here: `backend.backend.logger` and
friends are load-bearing for tests, mcp_server.py and __init__.py.

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
import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Dict

from . import constants as _C
from . import store as _store
from . import llm as _llm
from . import index as _index
from . import pipeline as _pipeline
from . import maintenance as _maintenance
from .core import (  # noqa: F401  (re-exported for external importers)
    _EMBED_NULL,
    _HLM_HANDLER_FLAG,
    _LiveStderrHandler,
    _SELF_AUTHORED_SOURCES,
    _UNTRUSTED_CLOSE,
    _UNTRUSTED_OPEN,
    MAX_CONTENT_CHARS,
    MAX_METADATA_CHARS,
    _embed_fn_cache_maxsize,
    _embedding_fn_cache,
    _embedding_fn_cache_lock,
    _EMBED_FN_CACHE_MAXSIZE,
    _from_qdrant_id,
    _get_embedding_fn,
    _get_secret_getter,
    _hlm_handlers_attached,
    _is_lock_error,
    _is_process_poisoned_db_error,
    _pack_embedding,
    _retry_on_lock,
    _setting,
    _setup_logger,
    _to_qdrant_id,
    _unpack_embedding,
    _wrap_untrusted_text,
    logger,
)

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
    Default (no "collections" key in config): SYSTEM/USER-DATA/ENV-DATA/CUSTOM
    map to the configured qdrant_collection; SESSION-DATA maps to "sessions"
    and OBSIDIAN to "vault". Any data_type not present in the map falls back
    to the configured qdrant_collection at lookup time (see index.py
    _get_collection).
    """

    def __init__(self, db_path: str, qdrant_url: str = "http://localhost:6333",
                 qdrant_collection: str = "memories", embedding_model: str = "all-MiniLM-L6-v2",
                 layer0_top_k: int = 30, config: dict = None, profile_name: str = None,
                 agent_id: str = "hermes"):
        # Expand ~ in db_path — use real home, not profile HOME
        real_home = _C.real_home()
        if db_path.startswith("~"):
            db_path = db_path.replace("~", real_home, 1)
        if "$" in db_path:
            db_path = os.path.expandvars(db_path)
        self._db_path = db_path
        self._qdrant_url = qdrant_url
        self._default_collection = qdrant_collection
        # The same name, kept in its *configured* spelling and never rewritten.
        # `_init_qdrant` rewrites `_default_collection` in place to the
        # model-keyed physical name (`index.py`, `_physical_collection`), which
        # is right for routing and wrong for the three paths that *persist* it:
        # the `taxonomy.collection` column, `_config["collections"]`, and the
        # JSON file behind it. Every reader re-keys what it loads through
        # `_physical_collection`, so a stored physical name double-suffixes the
        # moment the embedder changes and points at a collection that exists
        # nowhere — `T651`'s failure, which was fixed for the branch that
        # copied the whole map and left the default path writing the same value
        # one line further on. 0.8.113, T715.
        self._configured_default_collection = qdrant_collection
        self._embedding_model = embedding_model
        self._layer0_top_k = layer0_top_k
        self._profile_name = profile_name
        self._agent_id = agent_id
        self._qdrant = None
        self._config = config or {}
        # Env var overrides (higher priority than JSON config)
        self._apply_env_overrides()
        # Cap layer0_top_k — excessive candidates hurt performance
        if self._layer0_top_k > 200:
            logger.warning(
                "layer0_top_k=%d capped at 200 (excessive candidates hurt "
                "performance)", self._layer0_top_k
            )
            self._layer0_top_k = 200
        # Ensure the DB parent directory exists — _apply_env_overrides may have
        # replaced self._db_path with a path whose parent doesn't exist yet.
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)

        # Load DB-backed runtime config (lowest priority — env/JSON override it).
        # Actual order: the default _collection_map below is built first (from
        # JSON config / qdrant_collection), then _init_db()/_load_runtime_config()
        # runs later and mutates self._collection_map in place if DB-backed
        # config overrides it — so DB config still wins, just via a mutate-
        # after-default pass rather than being consulted before this default
        # is built.
        self._db_config_loaded = False
        # Physical collection names are keyed to the embedding model, but the
        # model is only known once _init_qdrant probes it — so this is empty
        # until then, and _physical_collection() is a no-op meanwhile.
        self._collection_suffix = ""
        # Collection mapping: data_type -> qdrant_collection
        self._collection_map = self._config.get("collections", {})
        if not self._collection_map:
            # Default: all data_types use the configured collection
            self._collection_map = {
                "SYSTEM": qdrant_collection,
                "USER-DATA": qdrant_collection,
                "ENV-DATA": qdrant_collection,
                "SESSION-DATA": "sessions",
                "OBSIDIAN": "vault",
                "CUSTOM": qdrant_collection,
            }
        # Circuit breaker state (YantrikDB pattern)
        self._qdrant_failures = 0
        self._qdrant_broken_until = None
        self._circuit_lock = threading.Lock()
        # Lock for Qdrant initialization — prevents double-init if two
        # threads race on _init_qdrant (unlikely from __init__, but a
        # defensive guard against future callers).
        self._init_lock = threading.Lock()
        # Retrieval source tracking (prefetch vs explicit)
        self._prefetch_count = 0
        self._explicit_count = 0
        # Hard switch: disable Qdrant entirely (SQLite-only mode)
        # Read through _setting() so a multiplex gateway resolves the active
        # profile's value rather than the root-level one (AGENTS.md: never
        # read profile-scoped env vars via os.environ directly).
        self._qdrant_enabled = (_setting("HLM_QDRANT_ENABLED", "true")).lower() not in ("false", "0", "no")
        # ── Retrieval health ────────────────────────────────────────────
        # `_degraded` is sticky for the life of this backend: True once any
        # retrieval has been served without Qdrant. That is the right shape
        # for the per-session provider, but on its own it answers the wrong
        # question — it cannot distinguish a brute-force scan (exact,
        # semantic, just slower) from a lexical BM25 fallback (measured
        # recall@5 0.227), and in a long-lived process (mcp_server.py) a
        # single blip leaves it True for weeks.
        #
        # So it is kept, unchanged, for callers that ask "did this ever
        # degrade?" — and `_retrieval_mode` answers "how is it ranking right
        # now?" while the counters give an operator something to judge with.
        self._degraded = False
        # Mode of the most recent retrieval:
        #   "vector"      — Qdrant ANN (normal)
        #   "brute_force" — exact cosine over locally stored vectors
        #   "lexical"     — FTS5/BM25 only; real recall loss
        self._retrieval_mode = "vector"
        self._degraded_count = 0
        self._degraded_last_ts = None
        self._enrich_thread = None
        self._enrich_lock = threading.Lock()
        self._history_lock = threading.Lock()
        #: Guards the retrieval counters. `n += 1` is LOAD/ADD/STORE, not
        #: atomic, so concurrent retrievals on a shared backend (the MCP
        #: server serves several at once) silently lost increments.
        self._stats_lock = threading.Lock()
        self._max_layer_from_env = False  # set True by _apply_env_overrides if HLM_MAX_LAYER is set
        # Set before _init_db(), which makes the first _get_conn() call —
        # _is_process_poisoned_db_error's condition below must be checkable
        # from the very first connection attempt, not just later ones.
        self._db_fatal_error = None
        self._init_db()
        # Load DB-backed runtime config (after DB is initialized)
        self._load_runtime_config()
        if self._qdrant_enabled:
            self._init_qdrant()
        else:
            logger.info("Qdrant disabled by HLM_QDRANT_ENABLED=false — SQLite-only mode")
        self._warn_if_llm_unconfigured()
        # One-time, per-profile: automatic enrichment stopped being automatic.
        self._notify_enrich_default_change()

    def llm_configured(self) -> bool:
        """True when _call_llm() can actually reach a provider."""
        provider_cfg = self._config.get("layer3_provider_config") or {}
        return bool(self._config.get("layer3_model") and provider_cfg.get("base_url"))

    def _warn_if_llm_unconfigured(self) -> None:
        """Say so, once, when every LLM-backed feature is a no-op.

        `_call_llm()` returns "" when `layer3_model` or
        `layer3_provider_config.base_url` is missing, and six features route
        through it: L3 rerank, L4 gap detection, fact extraction, LLM review,
        enrichment classification and compaction merges. Degrading to a silent
        no-op made auto-extraction stop writing on profiles that had relied on
        the old subprocess route, with nothing in the logs to say why.
        Missing config is a setting; failing invisibly is a defect.
        """
        if self.llm_configured():
            return
        missing = []
        if not self._config.get("layer3_model"):
            missing.append("layer3_model")
        if not (self._config.get("layer3_provider_config") or {}).get("base_url"):
            missing.append("layer3_provider_config.base_url")
        logger.warning(
            "LLM provider not configured (missing: %s) — fact extraction, LLM review, "
            "enrichment, compaction merges, L3 rerank and L4 gap detection are all "
            "disabled. Retrieval up to max_layer=2 is unaffected. Set these in "
            "hermes-layered-memory.json or via HLM_LAYER3_MODEL / "
            "HLM_LAYER3_BASE_URL.",
            ", ".join(missing),
        )

    def _apply_env_overrides(self):
        """Apply env var overrides to config (higher priority than JSON).

        Reads through _setting() rather than os.environ directly so the active
        profile's values are honoured under a multiplex gateway, where the
        profile .env lives in an isolated secret scope instead of os.environ.
        """
        db_path_env = _setting("HLM_DB_PATH")
        if db_path_env:
            # Hermes sets HOME to profile dir, breaking ~ expansion.
            # Use the actual user home directory from passwd.
            real_home = _C.real_home()
            dbp = db_path_env
            if dbp.startswith("~"):
                dbp = dbp.replace("~", real_home, 1)
            if "$" in dbp:
                dbp = os.path.expandvars(dbp)
            # Say so when this overrides a path the caller passed explicitly.
            #
            # An explicit keyword argument losing to an environment variable is
            # backwards from least-surprise, and it is the single mechanism
            # behind every harness incident this repo has recorded: the
            # 2026-08-25 sweep that deleted 44 of hlm-test's points, the
            # 2026-08-26 one that deleted ~90, CLAUDE.md's "never point a
            # backend at a copy of a database while using the real profile
            # name", and a review's own probe scripts writing to the live
            # profile-b store because the shell exported HLM_DB_PATH.
            # `tests/conftest.py` unsets the variable for exactly this reason —
            # the suite works around the behaviour rather than relying on it.
            #
            # Changing the precedence is a behaviour change and is deliberately
            # NOT made here: the env layer overriding *config* is documented and
            # load-bearing. Overriding an explicit *argument* silently is the
            # part that surprises, so it is now loud. One WARNING at construction
            # would have made all four incidents visible on the spot instead of
            # via a vector count days later.
            # 2026-08-26 ox-alpha round 2, out-of-scope note on bundle02.
            if self._db_path and os.path.abspath(dbp) != os.path.abspath(self._db_path):
                logger.warning(
                    "HLM_DB_PATH=%s overrides the db_path passed to LayeredBackend "
                    "(%s). The environment wins here; if you meant to isolate this "
                    "backend, unset HLM_DB_PATH — an explicit argument does not "
                    "protect you.", dbp, self._db_path)
            self._db_path = dbp

        # Which keys this env pass owns. The config layers are documented as
        # env > DB > file > defaults, but nothing recorded *which* layer a
        # given value came from once it was merged into `self._config` — so
        # two paths downstream could not tell an env override from a stored
        # setting, and both got it wrong:
        #
        #   * `_sync_config_to_file` wrote `dict(self._config)` to the JSON
        #     file on every config change, baking env values into the *static*
        #     layer. Unset the variable later and the value it had at the time
        #     of some unrelated `set_config` call survives forever, which
        #     inverts the documented precedence permanently and silently. The
        #     tell that this was already known: the write skips exactly two
        #     keys, `layer3_provider_config` and `enrich_llm` — a fix applied
        #     to the two that had been noticed.
        #   * `delete_config` popped the key from `_config` whatever its
        #     origin, so deleting a key that env supplies wiped a live
        #     override in-process while reporting success, and it came back at
        #     the next restart. Behaviour diverging across a restart, with
        #     nothing logged — the same shape as the taxonomy key in 0.7.96.
        #
        # 2026-08-24 audit, findings 10 and 11.
        self._env_config = {}

        qdrant_url = _setting("HLM_QDRANT_URL")
        if qdrant_url:
            self._qdrant_url = qdrant_url

        max_layer = _setting("HLM_MAX_LAYER")
        if max_layer:
            try:
                self._config["max_layer"] = int(max_layer)
                self._env_config["max_layer"] = self._config["max_layer"]
                self._max_layer_from_env = True  # explicit env var override
            except ValueError:
                logger.warning("invalid HLM_MAX_LAYER=%r, ignoring", max_layer)

        top_k = _setting("HLM_LAYER0_TOP_K")
        if top_k:
            try:
                self._layer0_top_k = int(top_k)
            except ValueError:
                logger.warning("invalid HLM_LAYER0_TOP_K=%r, ignoring", top_k)

        query_instruction = _setting("HLM_QUERY_INSTRUCTION")
        if query_instruction:
            self._config["query_instruction"] = query_instruction
            self._env_config["query_instruction"] = self._config["query_instruction"]

        dedup_threshold = _setting("HLM_DEDUP_THRESHOLD")
        if dedup_threshold:
            try:
                val = float(dedup_threshold)
                if 0.0 <= val <= 1.0:
                    self._config["dedup_threshold"] = val
                    self._env_config["dedup_threshold"] = self._config["dedup_threshold"]
                else:
                    logger.warning("HLM_DEDUP_THRESHOLD must be 0.0-1.0, ignoring")
            except ValueError:
                logger.warning("invalid HLM_DEDUP_THRESHOLD=%r, ignoring", dedup_threshold)

        layer3_mode = _setting("HLM_LAYER3_MODE")
        if layer3_mode:
            self._config["layer3_mode"] = layer3_mode
            self._env_config["layer3_mode"] = self._config["layer3_mode"]

        layer3_model = _setting("HLM_LAYER3_MODEL")
        if layer3_model:
            self._config["layer3_model"] = layer3_model
            self._env_config["layer3_model"] = self._config["layer3_model"]

        base_url = _setting("HLM_LAYER3_BASE_URL")
        if base_url:
            self._config.setdefault("layer3_provider_config", {})["base_url"] = base_url

        api_key = _setting("HLM_LAYER3_API_KEY")
        if api_key:
            self._config.setdefault("layer3_provider_config", {})["api_key"] = api_key

        # Conflict detection thresholds (env var overrides JSON)
        thresholds = _setting("HLM_CONFLICT_THRESHOLDS")
        if thresholds:
            try:
                self._config["conflict_thresholds"] = json.loads(thresholds)
                self._env_config["conflict_thresholds"] = self._config["conflict_thresholds"]
            except (json.JSONDecodeError, TypeError):
                logger.warning("invalid HLM_CONFLICT_THRESHOLDS, using defaults")

        reasoning = _setting("HLM_REASONING_EFFORT")
        if reasoning:
            self._config["layer3_reasoning_effort"] = reasoning
            self._env_config["layer3_reasoning_effort"] = self._config["layer3_reasoning_effort"]

        # `layer3_reasoning_style`'s env form. It was documented as a full
        # `HLM_*` variable in `docs/reference.md` — with a default, four
        # spellings and two separate "set this to pin it" instructions —
        # advertised by `scripts/measure-reasoning-suppression.py` as the
        # alternative to writing the DB key, and named by that script's failure
        # path as the thing that "wins over the DB". Nothing read it. Driven:
        # with `HLM_REASONING_STYLE=openai` exported,
        # `config["layer3_reasoning_style"]` came back `None`.
        #
        # Every sibling has an env form (MODE, MODEL, BASE_URL, API_KEY,
        # EFFORT), so the outlier was the missing override rather than the
        # documentation, and retracting a documented, defaulted, tested knob
        # from the reference would have been a regression wearing a doc fix's
        # clothes. Found in the 2026-09-15 documentation-drift pass. T656.
        #
        # Validated, unlike EFFORT above, because this key has a closed
        # vocabulary that `set_config` already enforces: an unrecognised value
        # would otherwise reach `_reasoning_kwargs` and fall through every
        # branch to auto-detection, which is the silent wrong answer the whole
        # measurement script exists to eliminate.
        style = _setting("HLM_REASONING_STYLE")
        if style:
            _valid = _C._CONFIG_VALUE_CHOICES["layer3_reasoning_style"]
            _norm = str(style).strip().lower()
            if _norm in _valid:
                self._config["layer3_reasoning_style"] = _norm
                self._env_config["layer3_reasoning_style"] = _norm
            else:
                logger.warning(
                    "HLM_REASONING_STYLE=%r is not one of %s — ignoring, "
                    "auto-detection stays in effect", style, ", ".join(_valid))

        enrich = _setting("HLM_ENRICH_LLM")
        if enrich:
            self._config["enrich_llm"] = enrich.lower() in ("true", "1", "yes")
            self._env_config["enrich_llm"] = self._config["enrich_llm"]

        archive_days = _setting("HLM_LOW_TRUST_ARCHIVE_DAYS")
        if archive_days:
            try:
                self._config["low_trust_archive_days"] = int(archive_days)
                self._env_config["low_trust_archive_days"] = self._config["low_trust_archive_days"]
            except ValueError:
                logger.warning("invalid HLM_LOW_TRUST_ARCHIVE_DAYS=%r, ignoring",
                               archive_days)

        if (_setting("HLM_TRACING") or "").lower() in ("true", "1", "yes"):
            self._config["tracing"] = True
            self._env_config["tracing"] = self._config["tracing"]

        if (_setting("HLM_SEED_OVERVIEW") or "").lower() in ("true", "1", "yes"):
            self._config["seed_overview"] = True
            self._env_config["seed_overview"] = self._config["seed_overview"]

        # Instance-level heuristic reason (not class-level — avoids shared mutable state)
        self._layer_suggest_reason = ""
        self._entity_patterns_cache = None










    def _get_new_conn(self) -> sqlite3.Connection:
        """Create a new SQLite connection with standard pragmas.

        The `journal_mode=WAL` pragma is the first real touch of the file —
        `sqlite3.connect()` itself doesn't validate anything — so it is
        where `_is_process_poisoned_db_error`'s condition actually surfaces.
        A caught instance means every future call to this method, in this
        process, against this path, will fail the exact same way forever
        (see that function's docstring), so it is recorded on the instance
        rather than just raised: `_get_conn()` checks `_db_fatal_error`
        before ever calling this method again, so the process stops
        repeating a doomed connection attempt — and the raw traceback that
        came with it — on every subsequent tool call.
        """
        try:
            conn = sqlite3.connect(self._db_path)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            return conn
        except Exception as e:
            if _is_process_poisoned_db_error(e):
                # First (and only) time this instance sees the condition:
                # build the message once and log it once. A concurrent
                # thread racing in here before this assignment lands would
                # log a second time — cosmetic, not worth a lock over.
                self._db_fatal_error = (
                    f"SQLite connection to {self._db_path} is permanently "
                    f"unusable in this process ({e}). The database file "
                    f"itself is very likely fine — verify from a "
                    f"*different* process with `sqlite3 {self._db_path} "
                    f"'PRAGMA integrity_check'` — this is process-local "
                    f"SQLite state left over from another process deleting "
                    f"or replacing the -wal/-shm files while this process "
                    f"held them open. Restart this session/process to "
                    f"recover; reconnecting from inside it will not help."
                )
                logger.error(self._db_fatal_error)
                raise RuntimeError(self._db_fatal_error) from e
            raise

    def _get_conn(self) -> sqlite3.Connection:
        """Get a thread-local SQLite connection (creates one if needed)."""
        if self._db_fatal_error is not None:
            raise RuntimeError(self._db_fatal_error)
        if not hasattr(self._conn_local, 'conn') or self._conn_local.conn is None:
            self._conn_local.conn = self._get_new_conn()
        return self._conn_local.conn


















    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()



    def _parse_keywords(self, kw_json):
        """Parse keywords from JSON string."""
        if not kw_json:
            return []
        try:
            # json available at module level
            return json.loads(kw_json)
        except (TypeError, ValueError):
            return []


































    def _discover_profile_dbs(self) -> Dict[str, str]:
        """Discover profiles that have HLM enabled via their config.

        Strategy:
        1. Scan ~/.hermes/profiles/ for profile directories
        2. Check config.yaml for memory.provider == hermes-layered-memory
        3. Read HLM_DB_PATH from .env, or assume default
        """
        dbs = {}
        _yaml = None
        try:
            import yaml as _yaml
        except ImportError:
            pass

        actual_home = _C.real_home()
        profiles_dir = os.path.join(actual_home, ".hermes", "profiles")
        if not os.path.isdir(profiles_dir):
            return dbs

        for entry in sorted(os.listdir(profiles_dir)):
            profile_dir = os.path.join(profiles_dir, entry)
            if not os.path.isdir(profile_dir):
                continue

            # 1. Check config.yaml for memory.provider
            config_yaml = os.path.join(profile_dir, "config.yaml")
            has_hlm = False
            if os.path.isfile(config_yaml) and _yaml is not None:
                try:
                    with open(config_yaml) as f:
                        cfg = _yaml.safe_load(f) or {}
                    has_hlm = cfg.get("memory", {}).get("provider") == "hermes-layered-memory"
                except Exception:
                    pass

            if not has_hlm:
                continue

            # 2. Read HLM_DB_PATH from .env
            env_file = os.path.join(profile_dir, ".env")
            db_path = None
            if os.path.isfile(env_file):
                try:
                    with open(env_file) as f:
                        for line in f:
                            line = line.strip()
                            if line.startswith("HLM_DB_PATH="):
                                db_path = line.split("=", 1)[1].strip().strip("\"'")
                                break
                except Exception:
                    pass

            # 3. Default: ~/.hermes/hermes-layered-memory-dbs/<profile>.db
            if not db_path:
                db_path = os.path.join(actual_home, ".hermes",
                                       "hermes-layered-memory-dbs", f"{entry}.db")

            # Expand ~ and env vars in path
            if db_path.startswith("~"):
                db_path = db_path.replace("~", actual_home, 1)
            if "$" in db_path:
                db_path = os.path.expandvars(db_path)

            dbs[entry] = db_path

        return dbs

    # ── Methods defined in the sibling modules ───────────────────────────
    #
    # These were attached after the class with
    #     for _m in (...): setattr(LayeredBackend, _m.__name__, _m)
    # which no editor or type checker can follow. `self._compact_group(...)`
    # resolved to nothing, which is part of why an off-by-one column read
    # inside it survived six review passes. Bound in the class body instead:
    # one line each, every call site navigable, and a typo is an ImportError
    # at start-up rather than an AttributeError on the first call to reach it.
    #
    # `self` is unchanged — these are plain functions taking the backend as
    # their first argument, which is why they bind directly without wrappers.

    # store
    _init_db = _store._init_db
    _seed_config_defaults = _store._seed_config_defaults
    _load_db_config = _store._load_db_config
    _load_runtime_config = _store._load_runtime_config
    _save_db_config = _store._save_db_config
    _sync_config_to_file = _store._sync_config_to_file
    set_config = _store.set_config
    delete_config = _store.delete_config
    get_config = _store.get_config
    register_taxonomy = _store.register_taxonomy
    get_surfaced_summaries = _store.get_surfaced_summaries
    get_taxonomy = _store.get_taxonomy
    unregister_taxonomy = _store.unregister_taxonomy
    _get_cleanup_config = _store._get_cleanup_config
    _maintenance_state = _store._maintenance_state
    _set_maintenance_state = _store._set_maintenance_state
    _insert_count = _store._insert_count
    _should_run_maintenance = _store._should_run_maintenance
    _record_maintenance_run = _store._record_maintenance_run
    add = _store.add
    _enrich_background = _store._enrich_background
    update = _store.update
    _resync_qdrant_payload = _store._resync_qdrant_payload
    _drop_points_everywhere = _store._drop_points_everywhere
    delete = _store.delete
    _get_record = _store._get_record
    _get_history_path = _store._get_history_path
    _rotate_history = _store._rotate_history
    write_extraction_ledger = _store.write_extraction_ledger
    extraction_stats = _store.extraction_stats
    _write_history_entry = _store._write_history_entry
    count = _store.count
    retrieval_stats = _store.retrieval_stats
    seed_overview = _store.seed_overview
    close = _store.close
    backup = _store.backup
    _notify_enrich_default_change = _store._notify_enrich_default_change
    list = _store.list
    test_cleanup = _store.test_cleanup
    delete_many = _store.delete_many
    reinforce = _store.reinforce
    feedback = _store.feedback

    # llm
    _load_entity_patterns = _llm._load_entity_patterns
    _load_all_patterns_dir = _llm._load_all_patterns_dir
    _ensure_entity_patterns_dir = _llm._ensure_entity_patterns_dir
    _load_json_patterns = _llm._load_json_patterns
    _merge_patterns = _llm._merge_patterns
    _parse_llm_json = _llm._parse_llm_json
    re_enrich = _llm.re_enrich
    _llm_classify = _llm._llm_classify
    _llm_classify_batch = _llm._llm_classify_batch
    _enrich_metadata = _llm._enrich_metadata
    enrich_existing = _llm.enrich_existing
    _call_llm = _llm._call_llm
    _llm_merge = _llm._llm_merge
    _state_db_path = _llm._state_db_path
    _open_state_db = _llm._open_state_db
    get_session_info = _llm.get_session_info
    get_session_chain = _llm.get_session_chain
    get_compaction_summary = _llm.get_compaction_summary
    classify_session = _llm.classify_session
    build_session_name = _llm.build_session_name

    # index
    _init_qdrant = _index._init_qdrant
    _ensure_qdrant_collection = _index._ensure_qdrant_collection
    _physical_collection = _index._physical_collection
    _record_qdrant_failure = _index._record_qdrant_failure
    _record_qdrant_success = _index._record_qdrant_success
    _get_collection = _index._get_collection
    _check_duplicate = _index._check_duplicate
    _check_duplicate_sqlite = _index._check_duplicate_sqlite
    _check_contradiction = _index._check_contradiction
    check_surfaced_echo = _index.check_surfaced_echo
    _embed_batch = _index._embed_batch
    _layer0 = _index._layer0
    _set_retrieval_mode = _index._set_retrieval_mode
    _degraded_semantic_fallback = _index._degraded_semantic_fallback
    _brute_force_search = _index._brute_force_search
    _has_profile_column = _index._has_profile_column
    _fallback_uuids = _index._fallback_uuids
    _detect_orphans = _index._detect_orphans
    _fts5_fallback = _index._fts5_fallback
    _get_vector = _index._get_vector
    _embed_and_upsert = _index._embed_and_upsert
    _cosine_similarity = _index._cosine_similarity
    _jaccard_similarity = _index._jaccard_similarity

    # pipeline
    retrieve = _pipeline.retrieve
    discover = _pipeline.discover
    _write_trace = _pipeline._write_trace
    _rotate_traces = _pipeline._rotate_traces
    get_traces = _pipeline.get_traces
    peek = _pipeline.peek
    _run_pipeline = _pipeline._run_pipeline
    _suggest_max_layer = _pipeline._suggest_max_layer
    _layer1 = _pipeline._layer1
    _add_bm25 = _pipeline._add_bm25
    _add_bm25_conn = _pipeline._add_bm25_conn
    _get_weight = _pipeline._get_weight
    _heuristic_classify = _pipeline._heuristic_classify
    _extract_entities = _pipeline._extract_entities
    _layer2 = _pipeline._layer2
    _conflict_thresholds = _pipeline._conflict_thresholds
    _is_conflict_worth_resolving = _pipeline._is_conflict_worth_resolving
    _detect_conflicts = _pipeline._detect_conflicts
    _layer3 = _pipeline._layer3
    _format_conflict_alert = _pipeline._format_conflict_alert
    _layer4 = _pipeline._layer4
    list_profiles = _pipeline.list_profiles
    _parse_temporal = _pipeline._parse_temporal

    # maintenance
    rebuild = _maintenance.rebuild
    graph_health = _maintenance.graph_health
    resolve_conflicts = _maintenance.resolve_conflicts
    sync_check = _maintenance.sync_check
    sleep = _maintenance.sleep
    compact = _maintenance.compact
    find_duplicate_groups = _maintenance.find_duplicate_groups
    _compact_group = _maintenance._compact_group
    purge = _maintenance.purge
    export_memories = _maintenance.export_memories
    import_memories = _maintenance.import_memories
    decay = _maintenance.decay
    _parse_frontmatter = _maintenance._parse_frontmatter
    _extract_wikilinks = _maintenance._extract_wikilinks
    _embed_query_text = _maintenance._embed_query_text
    ingest_obsidian = _maintenance.ingest_obsidian
