"""Layered Memory Provider — MemoryProvider ABC implementation.

Thin wrapper around the backend (backend.py). Handles lifecycle,
config, tool dispatch, and Hermes hooks.

Config in $HERMES_HOME/hermes-layered-memory.json or config.yaml:
  plugins:
    layered:
      db_path: $HERMES_HOME/hermes-layered-memory.db
      qdrant_url: http://localhost:6333
      qdrant_collection: memories
      embedding_model: all-MiniLM-L6-v2
      max_layer: 2
      layer0_top_k: 30
      layer3_mode: inline
      layer3_model: qwen36-fp8
"""

from __future__ import annotations

import json
import os
import pwd
import shlex
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

from .backend import LayeredBackend, logger
from .summaries import SummariesBackend

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def _load_config() -> dict:
    from hermes_constants import get_hermes_home
    config_path = get_hermes_home() / "hermes-layered-memory.json"
    if config_path.exists():
        try:
            return json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


# ---------------------------------------------------------------------------
# Tool Schemas
# ---------------------------------------------------------------------------

RETRIEVE_SCHEMA = {
    "name": "layered_retrieve",
    "description": (
        "Multi-layer memory retrieval. Casts wide net then filters for precision. "
        "Pass max_layer to control depth (2=fast, 3=precise, 4=thorough). "
        "Default is 2 (fused scores, no LLM). Escalate to 3 only for ambiguous queries, "
        "multi-concept searches, or when the user explicitly needs precision. "
        "Layer 3 adds LLM reranking (~90s) — avoid for routine lookups."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "max_layer": {"type": "integer", "default": 2, "description": "Pipeline depth: 2=fused, 3=reranked, 4=with gap detection"},
            "rerank": {"type": "boolean", "default": False, "description": "Explicit LLM reranking. Overrides max_layer to at least 3. Use for genuinely ambiguous multi-concept queries where semantic reranking adds value."},
            "scope": {"type": "string", "description": "Filter by scope: personal, work, project:X"},
            "data_type": {"type": "string", "description": "Filter by data_type: USER-DATA (user prefs, habits, style), ENV-DATA (hardware, software, tools, network), SESSION-DATA (per-session context), SYSTEM (agent identity, rules, constraints), CUSTOM (free-form). See layered_add for mapping."},
            "data_id": {"type": "string", "description": "Sub-partition: HW (hardware), SW (software), NET (network), workflow (operational patterns), conventions (naming/structure), troubleshooting (fixes), preferences, habits, style, identity, rules, session_id. Use alongside data_type for precise filtering."},
            "session_name": {"type": "string", "description": "Filter by session name (human-readable, e.g. 'GPU tuning', 'code review'). Linked to data_id for cross-referencing. Use for per-session memory."},
            "profile_name": {"type": "string", "description": "Search a specific profile's memories. Default: current profile. Set to None or cross_profile=true to search all profiles."},
            "cross_profile": {"type": "boolean", "default": False, "description": "Search across all profiles instead of just the current one. Useful for kanban orchestrator or cross-profile knowledge sharing."},
            "limit": {"type": "integer", "default": 5, "description": "Max results to return (default 5)"},
        },
        "required": ["query"],
    },
}

PEEK_SCHEMA = {
    "name": "layered_peek",
    "description": (
        "Peek at a specific layer's output without running the full pipeline. "
        "Use to inspect what was dropped at a given layer or compare layer outputs."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "layer": {"type": "integer", "enum": [1, 2, 3, 4], "description": "Which layer to peek at"},
        },
        "required": ["query", "layer"],
    },
}

ADD_SCHEMA = {
    "name": "layered_add",
    "description": (
        "Store a new memory. For important facts the user would expect you to remember. "
        "Set sensitivity=2+ for private data. Set ttl to auto-expire. "
        "Use priority=3 for pinned memories that always surface first."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "Full memory content"},
            "summary": {"type": "string", "description": "One-line summary"},
            "topic": {"type": "string", "description": "Topic/category"},
            "keywords": {"type": "array", "items": {"type": "string"}, "description": "Searchable keywords"},
            "scope": {"type": "string", "default": "personal", "description": "personal, work, project:X"},
            "data_type": {"type": "string", "default": "CUSTOM", "description": "USER-DATA (user prefs, habits, communication style), ENV-DATA (hardware, software, tools, network config, workflows, conventions, troubleshooting), SESSION-DATA (per-session context), SYSTEM (agent identity, rules, constraints), CUSTOM (free-form notes). Use data_id for sub-partitioning."},
            "data_id": {"type": "string", "description": "Sub-partition within data_type. ENV-DATA: HW (hardware), SW (software), NET (network), workflow (operational patterns), conventions (naming/structure), troubleshooting (fixes/workarounds). USER-DATA: preferences, habits, conventions, style. SYSTEM: identity, rules. SESSION-DATA: session_id. CUSTOM: omit or set folder name."},
            "session_name": {"type": "string", "description": "Human-readable session label. Stored alongside data_id for named lookups."},
            "sensitivity": {"type": "integer", "default": 0, "description": "0=public, 1=internal, 2=private, 3=restricted"},
            "ttl": {"type": "string", "description": "ISO timestamp for auto-expiry, or omit for permanent"},
            "priority": {"type": "integer", "default": 0, "description": "0=normal, 1=elevated, 2=high, 3=pinned"},
        },
        "required": ["content"],
    },
}

UPDATE_SCHEMA = {
    "name": "layered_update",
    "description": "Update an existing memory by UUID. Partial update — only provided fields change. Supports: content, summary, priority, status, trust_score, ttl, sensitivity, topic, data_type, data_id, session_name, keywords.",
    "parameters": {
        "type": "object",
        "properties": {
            "uuid": {"type": "string", "description": "Memory UUID"},
            "content": {"type": "string", "description": "New content"},
            "summary": {"type": "string", "description": "New summary"},
            "priority": {"type": "integer", "description": "New priority (0-3)"},
            "status": {"type": "string", "enum": ["active", "archived", "deleted"], "description": "New status"},
            "trust_score": {"type": "number", "description": "New trust score (0.0-1.0)"},
            "ttl": {"type": "string", "description": "New TTL ISO timestamp"},
            "sensitivity": {"type": "string", "description": "New sensitivity label"},
            "topic": {"type": "string", "description": "New topic"},
            "data_type": {"type": "string", "description": "New data_type"},
            "data_id": {"type": "string", "description": "New data_id"},
            "session_name": {"type": "string", "description": "New session_name"},
            "keywords": {"type": "array", "items": {"type": "string"}, "description": "New keywords array"},
        },
        "required": ["uuid"],
    },
}

DELETE_SCHEMA = {
    "name": "layered_delete",
    "description": "Soft-delete a memory by UUID. Marks as deleted in SQLite, removes from Qdrant. Use `layered_purge` to permanently remove from disk.",
    "parameters": {
        "type": "object",
        "properties": {
            "uuid": {"type": "string", "description": "Memory UUID"},
        },
        "required": ["uuid"],
    },
}

REBUILD_SCHEMA = {
    "name": "layered_rebuild",
    "description": "Rebuild Qdrant index from SQLite. Full or incremental (since timestamp).",
    "parameters": {
        "type": "object",
        "properties": {
            "since": {"type": "string", "description": "ISO timestamp — only rebuild records updated after this. Omit for full rebuild."},
        },
        "required": [],
    },
}

PURGE_SCHEMA = {
    "name": "layered_purge",
    "description": "Permanently delete soft-deleted and/or archived records. Purges from SQLite, removes from Qdrant, and vacuums the database. Use to reclaim storage space.",
    "parameters": {
        "type": "object",
        "properties": {
            "purge_deleted": {"type": "boolean", "description": "Purge soft-deleted records (status=deleted). Default: true."},
            "purge_archived": {"type": "boolean", "description": "Purge archived records (status=archived). Default: true."},
            "min_age_hours": {"type": "number", "description": "Minimum age in hours before purging. Default: 24 for deleted, 168 (7 days) for archived."},
            "vacuum": {"type": "boolean", "description": "Vacuum SQLite after purge to reclaim disk space. Default: true."},
        },
        "required": [],
    },
}

SYNC_SCHEMA = {
    "name": "layered_sync_check",
    "description": "Check sync between Qdrant index and SQLite store.",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}

LIST_SCHEMA = {
    "name": "layered_list",
    "description": "List recent memories with optional filters. Does NOT run vector search.",
    "parameters": {
        "type": "object",
        "properties": {
            "topic": {"type": "string", "description": "Filter by topic"},
            "scope": {"type": "string", "description": "Filter by scope"},
            "limit": {"type": "integer", "default": 20, "description": "Max results"},
            "sort": {"type": "string", "enum": ["created_at", "priority", "trust_score"], "default": "created_at", "description": "Sort order"},
        },
        "required": [],
    },
}

SLEEP_SCHEMA = {
    "name": "layered_sleep",
    "description": "Run memory consolidation: archive low-trust duplicates in same data_type:data_id group, archive old low-trust memories. Mnemosyne BEAM pattern.",
    "parameters": {
        "type": "object",
        "properties": {
            "max_items": {"type": "integer", "default": 100, "description": "Max groups to process"},
            "min_age_hours": {"type": "number", "default": 24.0, "description": "Minimum age before consolidation"},
        },
        "required": [],
    },
}

OBSIDIAN_INGEST_SCHEMA = {
    "name": "obsidian_ingest",
    "description": "Ingest an Obsidian vault into the memory index. Parses frontmatter, extracts tags and wikilinks, embeds all notes.",
    "parameters": {
        "type": "object",
        "properties": {
            "vault_path": {"type": "string", "description": "Path to the Obsidian vault directory"},
            "exclude": {"type": "array", "items": {"type": "string"}, "description": "Directories to exclude (default: .obsidian, _resources, Templates, Excalidraw)"},
        },
        "required": ["vault_path"],
    },
}

ENRICH_SCHEMA = {
    "name": "layered_enrich",
    "description": "Batch-enrich existing memories with missing metadata (data_type, data_id, topic, keywords). Targets records with empty metadata. Uses heuristics first, LLM fallback for uncertain cases.",
    "parameters": {
        "type": "object",
        "properties": {
            "since": {"type": "string", "description": "ISO timestamp — only enrich records updated after this. Omit for all eligible records."},
            "max_items": {"type": "integer", "default": 100, "description": "Max records to enrich in one batch"},
        },
        "required": [],
    },
}

FEEDBACK_SCHEMA = {
    "name": "layered_feedback",
    "description": "Provide feedback on a memory's usefulness. Increments/decrements trust_score (0.0-1.0). Low-trust records (<0.3) are auto-archived by layered_sleep.",
    "parameters": {
        "type": "object",
        "properties": {
            "uuid": {"type": "string", "description": "Memory UUID"},
            "helpful": {"type": "boolean", "description": "true=helpful (+0.1), false=not helpful (-0.1). Trust score capped at 0.0-1.0."},
        },
        "required": ["uuid", "helpful"],
    },
}

# Summaries tools
SUMMARIZE_SCHEMA = {
    "name": "layered_summarize",
    "description": "Summarize a source (YouTube video, git repo, documentation, web page). Stores highlights in DB, full summary as .md file. Idempotent by source_url (canonicalized) — returns existing UUID if already summarized unless force=True. Rejects empty highlights+full_text.",
    "parameters": {
        "type": "object",
        "properties": {
            "source": {"type": "string", "description": "Source URL (canonicalized automatically)"},
            "source_type": {"type": "string", "description": "Source type (youtube, git, docs, web)"},
            "title": {"type": "string", "description": "Summary title"},
            "highlights": {"type": "array", "items": {"type": "string"}, "description": "Key bullet points (5-10)"},
            "full_text": {"type": "string", "description": "Full summary text (saved as .md)"},
            "tags": {"type": "array", "items": {"type": "string"}, "description": "Tags for categorization"},
            "force": {"type": "boolean", "description": "If true, create new summary even if source_url exists"},
        },
        "required": ["source", "source_type"],
    },
}

LIST_SUMMARIES_SCHEMA = {
    "name": "layered_list_summaries",
    "description": "List stored summaries with optional filters. Returns highlights and metadata. Use source_url to check if a URL was already summarized (idempotency). Tag search via tag= or FTS5 tag:foo syntax.",
    "parameters": {
        "type": "object",
        "properties": {
            "source_type": {"type": "string", "description": "Filter by source type"},
            "source_url": {"type": "string", "description": "Filter by source URL (check if already summarized)"},
            "tag": {"type": "string", "description": "Filter by tag"},
            "profile": {"type": "string", "description": "Filter by profile name"},
            "limit": {"type": "integer", "description": "Max results (default 20)"},
            "offset": {"type": "integer", "description": "Pagination offset"},
            "sort_by": {"type": "string", "description": "Sort field (created_at, updated_at). Default: created_at DESC"},
        },
    },
}

GET_SUMMARY_SCHEMA = {
    "name": "layered_get_summary",
    "description": "Get a summary by UUID. Returns highlights (small) + path to .md file. Agent reads .md file itself via read_file (offset/limit for chunking) or delegates to subagent with file path.",
    "parameters": {
        "type": "object",
        "properties": {
            "uuid": {"type": "string", "description": "Summary UUID"},
        },
        "required": ["uuid"],
    },
}

SEARCH_SUMMARIES_SCHEMA = {
    "name": "layered_search_summaries",
    "description": "Search summaries via FTS5 (Porter stemmer enabled). Returns matching snippet + rank alongside highlights and path.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "limit": {"type": "integer", "description": "Max results (default 10)"},
        },
        "required": ["query"],
    },
}

DELETE_SUMMARY_SCHEMA = {
    "name": "layered_delete_summary",
    "description": "Delete a summary and its .md file. Defaults to profile-scoped (only deletes own profile records). Use profile='all' to delete any profile's record.",
    "parameters": {
        "type": "object",
        "properties": {
            "uuid": {"type": "string", "description": "Summary UUID"},
            "profile": {"type": "string", "enum": ["own", "all"], "default": "own", "description": "If 'own' (default), only delete own profile. If 'all', delete any profile."},
        },
        "required": ["uuid"],
    },
}

UPDATE_SUMMARY_SCHEMA = {
    "name": "layered_update_summary",
    "description": "Update a summary (title, highlights, tags, metadata, full_text). Keeps the same UUID. Metadata stores source_fingerprint for staleness detection.",
    "parameters": {
        "type": "object",
        "properties": {
            "uuid": {"type": "string", "description": "Summary UUID"},
            "title": {"type": "string", "description": "New title"},
            "highlights": {"type": "array", "items": {"type": "string"}, "description": "New highlights (replaces existing, recalculates coverage_status)"},
            "tags": {"type": "array", "items": {"type": "string"}, "description": "New tags (replaces existing)"},
            "metadata": {"type": "object", "description": "New metadata (source_fingerprint, etc., replaces existing)"},
            "full_text": {"type": "string", "description": "New full text (re-writes .md, recomputes hash/token_estimate)"},
        },
        "required": ["uuid"],
    },
}

DELETE_MULTIPLE_SUMMARIES_SCHEMA = {
    "name": "layered_delete_multiple_summaries",
    "description": "Delete multiple summaries at once. Defaults to profile-scoped (only deletes own profile records). Use profile='all' to delete any profile's records.",
    "parameters": {
        "type": "object",
        "properties": {
            "uuids": {"type": "array", "items": {"type": "string"}, "description": "List of summary UUIDs to delete"},
            "profile": {"type": "string", "enum": ["own", "all"], "default": "own", "description": "If 'own' (default), only delete own profile. If 'all', delete any profile."},
        },
        "required": ["uuids"],
    },
}

LIST_EXPIRING_SUMMARIES_SCHEMA = {
    "name": "layered_list_expiring_summaries",
    "description": "List summaries older than max_age_days. Use to identify stale summaries for deletion. Store source_fingerprint in metadata for staleness detection.",
    "parameters": {
        "type": "object",
        "properties": {
            "max_age_days": {"type": "integer", "description": "Age threshold in days (default 30)"},
        },
    },
}

SYNC_SUMMARIES_SCHEMA = {
    "name": "layered_sync_summaries",
    "description": "Sync summaries DB with filesystem. Finds orphaned records where .md file was deleted by user. Deletes them.",
    "parameters": {
        "type": "object",
        "properties": {},
    },
}

REVIEW_SCHEMA = {
    "name": "layered_review",
    "description": "Review memories for garbage/test records. Spawns a background session that feeds unreviewed records to an LLM which marks them as 'keep' or 'delete'. Records marked 'delete' are soft-deleted. Use min_age_hours to control how old records must be before review (default: 1 hour to avoid reviewing fresh entries). Force=true reviews ALL records regardless of llm_review_status. After review, run layered_purge(min_age_hours=24) to clean soft-deleted records.",
    "parameters": {
        "type": "object",
        "properties": {
            "min_age_hours": {"type": "number", "description": "Minimum age in hours (default: 1. Avoids reviewing fresh entries)"},
            "force": {"type": "boolean", "description": "If true, review ALL records regardless of llm_review_status. Default: false (only reviews unreviewed records)"},
        },
    },
}

COMPACT_SCHEMA = {
    "name": "layered_compact",
    "description": (
        "Merge duplicate or overlapping memories into a single consolidated record. "
        "Finds records with similar content (same topic, high vector similarity), "
        "feeds them to LLM for merge, creates a new merged record, soft-deletes originals. "
        "Run after layered_review() to clean up redundant facts. "
        "Parameters: similarity_threshold (0.8-0.99, default 0.90), topic (optional, limit to one topic), "
        "max_groups (default 50), dry_run (true to preview without merging)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "similarity_threshold": {"type": "number", "description": "Minimum vector similarity (0.8-0.99, default 0.90)"},
            "topic": {"type": "string", "description": "Optional: only compact records matching this topic"},
            "max_groups": {"type": "integer", "description": "Maximum duplicate groups to process (default 50)"},
            "dry_run": {"type": "boolean", "description": "If true, preview without merging"},
        },
    },
}

BACKUP_SCHEMA = {
    "name": "layered_backup",
    "description": (
        "Create a backup of both the memories database and summaries database. "
        "Uses SQLite's atomic backup API (WAL-safe). "
        "Returns backup paths and metadata. "
        "Keep last N days by default, older backups are deleted."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "dest_dir": {"type": "string", "description": "Backup directory. Default: beside the original DB files."},
            "keep_days": {"type": "integer", "default": 7, "description": "Days to keep old backups. Older backups are deleted (default: 7). Set 0 to keep all."},
        },
        "required": [],
    },
}

EXPORT_SCHEMA = {
    "name": "layered_export",
    "description": (
        "Export memories as JSON or Markdown. Supports filtering by status, data_type, topic, scope. "
        "Use cross_profile=true to export from all profiles. Output can be saved to a file for backup "
        "or migration between hosts."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "format": {"type": "string", "description": "Output format: 'json' (default) or 'md'"},
            "status": {"type": "string", "description": "Filter by status (default: 'active'). Use 'all' for everything."},
            "data_type": {"type": "string", "description": "Filter by data_type (USER-DATA, ENV-DATA, SESSION-DATA, SYSTEM, CUSTOM)"},
            "topic": {"type": "string", "description": "Filter by topic"},
            "scope": {"type": "string", "description": "Filter by scope: personal, work, project:X"},
            "profile_name": {"type": "string", "description": "Export from a specific profile only"},
            "cross_profile": {"type": "boolean", "default": False, "description": "Export from all profiles"},
        },
    },
}

IMPORT_SCHEMA = {
    "name": "layered_import",
    "description": (
        "Import memories from a JSON file or JSON string (from layered_export). "
        "Modes: 'skip_existing' (default, skip duplicate UUIDs), "
        "'overwrite' (update existing records), 'new_uuid' (generate new UUIDs for duplicates). "
        "Triggers Qdrant rebuild after import."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "data": {"type": "string", "description": "JSON string or file path (.json/.md). For file paths, provides the path to the exported JSON file."},
            "mode": {"type": "string", "description": "Import mode: 'skip_existing' (default), 'overwrite', 'new_uuid'"},
            "target_status": {"type": "string", "description": "Status for imported records (default: 'active')"},
        },
        "required": ["data"],
    },
}

DECAY_SCHEMA = {
    "name": "layered_decay",
    "description": (
        "Apply confidence decay to stale memories. Memories older than min_age_days have their "
        "trust_score reduced based on age. Frequently retrieved records (reference_count > 10) "
        "decay slower. Records with priority > 0 or trust_score > 0.9 are not decayed. "
        "Run this periodically (e.g., at session end) to prevent stale facts from accumulating noise."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "min_age_days": {"type": "integer", "description": "Only decay memories older than this (default: 30)"},
            "max_age_days": {"type": "integer", "description": "Age at which full decay effect applies (default: 365)"},
            "decay_rate": {"type": "number", "description": "Maximum decay factor per full age cycle (default: 0.05)"},
            "min_score": {"type": "number", "description": "Minimum trust_score after decay (default: 0.1)"},
        },
    },
}

ALL_TOOLS = [RETRIEVE_SCHEMA, PEEK_SCHEMA, ADD_SCHEMA, UPDATE_SCHEMA,
             DELETE_SCHEMA, REBUILD_SCHEMA, PURGE_SCHEMA, SYNC_SCHEMA, LIST_SCHEMA,
             SLEEP_SCHEMA, OBSIDIAN_INGEST_SCHEMA, ENRICH_SCHEMA, FEEDBACK_SCHEMA,
             SUMMARIZE_SCHEMA, LIST_SUMMARIES_SCHEMA, GET_SUMMARY_SCHEMA,
             SEARCH_SUMMARIES_SCHEMA, DELETE_SUMMARY_SCHEMA, UPDATE_SUMMARY_SCHEMA,
             DELETE_MULTIPLE_SUMMARIES_SCHEMA, LIST_EXPIRING_SUMMARIES_SCHEMA,
             SYNC_SUMMARIES_SCHEMA, REVIEW_SCHEMA, COMPACT_SCHEMA,
             EXPORT_SCHEMA, IMPORT_SCHEMA, DECAY_SCHEMA, BACKUP_SCHEMA]


# ---------------------------------------------------------------------------
# MemoryProvider Implementation
# ---------------------------------------------------------------------------

class LayeredMemoryProvider(MemoryProvider):
    """Layered funnel memory: Qdrant → SQLite → Fusion → Reranker."""

    def __init__(self, config: dict | None = None):
        self._config = config or _load_config()
        self._backend = None
        self._summaries = None
        self._summaries_initialized = False
        self._session_id = None
        self._max_layer = int(self._config.get("max_layer", 2))
        # Auto-ingestion state (per-turn + session-end extraction)
        self._profile_name = None
        self._turn_count = 0
        self._last_extract_time = 0
        self._extract_lock = threading.Lock()
        # Session chain tracking (repo-local sessions.txt)
        # Use agent's start cwd as the immutable project root — not __file__ path.
        # This is set once at session boot and never changes.
        self._project_dir = os.getcwd()
        self._plugin_dir = os.path.dirname(os.path.abspath(__file__))

    @property
    def name(self) -> str:
        return "hermes-layered-memory"

    # ---- Required ABC methods -------------------------------------------

    def is_available(self) -> bool:
        # SQLite is always available. Qdrant deps are optional at startup.
        return True

    def initialize(self, session_id: str, **kwargs) -> None:
        from hermes_constants import get_hermes_home
        profile_name = kwargs.get("profile_name") or kwargs.get("agent_identity") or "default"
        logger.info(
            "initialize: starting (session=%s, profile=%s)",
            session_id[:16], profile_name,
        )
        hermes_home = str(get_hermes_home())
        self._profile_name = profile_name
        self._turn_count = 0
        self._last_extract_time = 0

        db_path = self._config.get("db_path", f"{hermes_home}/hermes-layered-memory-dbs/{profile_name}.db")
        db_path = db_path.replace("$HERMES_HOME", hermes_home)
        db_path = os.path.expanduser(db_path)

        # Ensure DB directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        self._backend = LayeredBackend(
            db_path=db_path,
            qdrant_url=self._config.get("qdrant_url", "http://localhost:6333"),
            qdrant_collection=self._config.get("qdrant_collection", "memories"),
            embedding_model=self._config.get("embedding_model", "all-MiniLM-L6-v2"),
            layer0_top_k=self._config.get("layer0_top_k", 30),
            config=self._config,
            profile_name=kwargs.get("profile_name") or kwargs.get("agent_identity"),
        )

        # Initialize summaries backend (shared across all profiles)
        summaries_dir = os.environ.get("HERMES_LAYERED_SUMMARIES_DIR")
        if not summaries_dir:
            # Shared location: real Hermes home, not profile dir
            real_home = pwd.getpwuid(os.getuid()).pw_dir
            summaries_dir = os.path.join(real_home, "Documents", "hlm-summaries/")
        summaries_db = os.environ.get("HERMES_LAYERED_SUMMARIES_DB")
        if not summaries_db:
            # Shared location: real Hermes home, not profile dir
            real_home = pwd.getpwuid(os.getuid()).pw_dir
            summaries_db = os.path.join(real_home, ".hermes", "hermes-layered-memory-dbs", "digests.db")
        try:
            self._summaries = SummariesBackend(summaries_db, summaries_dir)
        except Exception as e:
            logger.exception("Failed to initialize SummariesBackend: %s", e)
            self._summaries = None

        self._session_id = session_id

        # Extract from compaction summary if this is a post-compression session
        try:
            compaction = self._backend.get_compaction_summary(session_id)
            if compaction:
                session_name = self._backend.build_session_name(session_id)
                chain = self._backend.get_session_chain(session_id)
                lineage = ", ".join(chain) if chain else session_id
                prompt = (
                    "Extract durable facts from the following context compaction summary. "
                    "This is a structured summary of a previous session that was compressed. "
                    "Pay attention to sections like '## Active Task', '## Goal', "
                    "'## Completed Actions', '## Remaining Work', '## Constraints'. "
                    "For each fact found, call layered_add(content=\"...\", topic=\"...\", "
                    "keywords=[...], session_name=\"" + session_name + "\", "
                    "data_id=\"" + session_id +
                    "\"). Do NOT store ephemeral conversation or task progress. "
                    "If no durable facts exist, do nothing and stop.\n\n"
                    + compaction
                )
                cmd = f"hermes -p {shlex.quote(profile_name)} -z {shlex.quote(prompt)}"
                subprocess.Popen(
                    cmd, shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                logger.debug("initialize: spawned compaction summary extraction (chain: %s)", lineage)
        except Exception as e:
            logger.debug("initialize: compaction extraction failed: %s", e)

        # Auto-rebuild Qdrant index if stale — SQLite is the single source of truth
        try:
            sync = self._backend.sync_check()
            sqlite_n = sync.get("sqlite_active", 0)
            qdrant_n = sync.get("qdrant_profile", 0)
            in_sync = sync.get("in_sync", True)
            logger.info("initialize: sync check — SQLite=%d, Qdrant=%d, in_sync=%s", sqlite_n, qdrant_n, in_sync)
            if sqlite_n > 0 and not in_sync:
                logger.info(
                    "initialize: Qdrant stale (SQLite=%d, Qdrant=%d) — rebuilding",
                    sqlite_n, qdrant_n,
                )
                result = self._backend.rebuild()
                logger.info("initialize: rebuild complete: %s", result)
            else:
                logger.debug("initialize: no rebuild needed")
        except Exception as e:
            logger.info("initialize: auto-rebuild check failed: %s", e)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return ALL_TOOLS

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if not self._backend:
            return tool_error("Layered memory not initialized")

        dispatch = {
            "layered_retrieve": self._do_retrieve,
            "layered_peek": self._do_peek,
            "layered_add": self._do_add,
            "layered_update": self._do_update,
            "layered_delete": self._do_delete,
            "layered_rebuild": self._do_rebuild,
            "layered_purge": self._do_purge,
            "layered_sync_check": self._do_sync_check,
            "layered_list": self._do_list,
            "layered_sleep": self._do_sleep,
            "layered_enrich": self._do_enrich,
            "layered_feedback": self._do_feedback,
            "obsidian_ingest": self._do_obsidian_ingest,
            # Summaries tools
            "layered_summarize": self._do_summarize,
            "layered_list_summaries": self._do_list_summaries,
            "layered_get_summary": self._do_get_summary,
            "layered_search_summaries": self._do_search_summaries,
            "layered_delete_summary": self._do_delete_summary,
            "layered_update_summary": self._do_update_summary,
            "layered_delete_multiple_summaries": self._do_delete_multiple_summaries,
            "layered_list_expiring_summaries": self._do_list_expiring_summaries,
            "layered_sync_summaries": self._do_sync_summaries,
            "layered_review": self._do_review,
            "layered_compact": self._do_compact,
            "layered_export": self._do_export,
            "layered_import": self._do_import,
            "layered_decay": self._do_decay,
            "layered_backup": self._do_backup,
        }
        handler = dispatch.get(tool_name)
        if not handler:
            return tool_error(f"Unknown tool: {tool_name}")
        try:
            return json.dumps(handler(args))
        except Exception as e:
            logger.exception("Layered tool %s failed: %s", tool_name, e)
            return tool_error(str(e))

    # ---- Optional hooks -------------------------------------------------

    def system_prompt_block(self) -> str:
        if not self._backend:
            return ""
        total = self._backend.count()
        return (
            f"# Layered Memory\n"
            f"Active. {total} memories with multi-layer retrieval. "
            f"Use layered_retrieve for search (max_layer controls depth). "
            f"Use layered_add to store facts. Use layered_peek to inspect layer outputs.\n"
            f"Depth presets: 1=fast filtered, 2=fused scores, 3=LLM reranked, 4=with gap detection.\n"
            f"\n"
            f"**Using retrieved data:** When layered_retrieve returns results, use them to answer "
            f"the user's question before running additional tools. If you reference specific memory "
            f"content, cite the UUID. Do not ignore HLM results and re-discover the same facts "
            f"via other means (shell commands, file reads, etc.).\n"
            f"\n"
            f"**Auto-capture:** When the user shares facts, preferences, environment details, rules, or conventions — "
            f"store them with layered_add(content=\"...user's fact...\"). Do NOT require the user to structure the call "
            f"or know data types — capture from natural language. The plugin enriches metadata automatically.\n"
            f"**Do NOT store:** ephemeral conversation, task progress, completed work logs, PR numbers, commit SHAs, "
            f"or anything stale in a week. Memory is for durable facts only.\n"
            f"\n"
            f"**Summaries:** When the user asks to summarize a YouTube video, web page, article, documentation, or any long-form content — "
            f"after producing the summary, ALWAYS save it with:\n"
            f"  Call layered_summarize(source=\"<original URL>\", source_type=\"youtube\"|\"web\"|\"docs\"|\"article\", "
            f"title=\"<title>\", highlights=[\"key point 1\", \"key point 2\", ...], full_text=\"<summary text>\", tags=[\"topic1\", \"topic2\"])\n"
            f"Use 5-10 concise highlights. Tags: include the content type plus 1-2 topic tags. "
            f"Skip only if the user explicitly says not to save."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not self._backend or not query:
            return ""
        try:
            results = self._backend.retrieve(query, max_layer=self._max_layer)
            if not results:
                return ""
            lines = [f"- [{r.get('score', 0):.2f}] {r.get('summary', r.get('content', ''))[:120]}"
                     for r in results[:5]]
            return "## Layered Memory Recall\n" + "\n".join(lines)
        except Exception as e:
            logger.debug("Layered prefetch failed: %s", e)
            return ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        # Could cache the result for the next turn. No-op for now.
        pass

    def _record_session_in_chain(self, session_id: Optional[str]) -> None:
        """Append session ID to project-local sessions.txt if not already present.

        Gate: only records sessions when running from a git project directory.
        On first write, creates the file with a PROJECT_HOME header line.
        Safe to call from any hook.
        """
        if not session_id:
            return

        sessions_file = os.path.join(self._project_dir, "sessions.txt")

        # Fast path: file exists → append (current behavior)
        if os.path.exists(sessions_file):
            try:
                with open(sessions_file, "r") as f:
                    existing = f.read()
                if session_id in existing:
                    return

                title = ""
                if self._backend:
                    info = self._backend.get_session_info(session_id)
                    if info and info.get("title"):
                        title = f" :: {info['title']}"

                from datetime import datetime
                line = f"{session_id}{title} ({datetime.now():%Y-%m-%d})\n"
                with open(sessions_file, "a") as f:
                    f.write(line)
                logger.debug("session chain: recorded %s", session_id[:16])
            except Exception as e:
                logger.debug("session chain: failed to record %s: %s", session_id[:16], e)
            return

        # No sessions.txt → gate on .git existence
        if not os.path.exists(os.path.join(self._project_dir, '.git')):
            return  # Not a project directory, skip tracking entirely

        # First time: create with PROJECT_HOME header + first session record
        try:
            title = ""
            if self._backend:
                info = self._backend.get_session_info(session_id)
                if info and info.get("title"):
                    title = f" :: {info['title']}"

            from datetime import datetime
            header = "# Session chain — project development lineage\n"
            header += "# Format: SESSION_ID :: short description (YYYY-MM-DD)\n"
            header += "# Source: echo $HERMES_SESSION_ID (from env, no state.db access)\n"
            header += f"PROJECT_HOME: {self._project_dir}\n"
            line = f"{session_id}{title} ({datetime.now():%Y-%m-%d})\n"
            with open(sessions_file, "w") as f:
                f.write(header + line)
            logger.debug("session chain: created %s (project: %s)", sessions_file, self._project_dir)
        except Exception as e:
            logger.debug("session chain: failed to create %s: %s", sessions_file, e)

    def _extract_facts_background(self, text: str, label: str,
                                   session_name: Optional[str] = None,
                                   session_id: Optional[str] = None) -> None:
        """Spawn a fire-and-forget hermes session to extract facts from text.

        Runs `hermes -p <profile> -z "extract facts..."` in a daemon thread.
        The spawned session loads HLM, reads the system_prompt_block instructions,
        calls layered_add() for each fact it finds, then exits.

        Args:
            text: The text to extract facts from (capped at 8000 chars).
            label: Label for logging (e.g., "sync_turn", "pre_compress", "session_end").
            session_name: Pre-built session name from state.db. Falls back to self._session_id.
            session_id: Raw session UUID. Used as data_id for SESSION-DATA records.
        """
        if not self._profile_name:
            return
        # Throttle: max 1 extraction per 30s, every 5 turns
        with self._extract_lock:
            self._turn_count += 1
            now = time.time()
            if now - self._last_extract_time < 30 or self._turn_count % 5 != 0:
                return
            self._last_extract_time = now

        profile = self._profile_name
        sname = session_name  # capture for closure
        sid = session_id

        def _run():
            # Resolve session_name: provided > state.db lookup > raw UUID
            sn = sname
            if not sn:
                if self._backend and self._session_id:
                    sn = self._backend.build_session_name(self._session_id)
                else:
                    sn = getattr(self, '_session_id', '')

            # Resolve session_id: provided > raw UUID
            s = sid
            if not s:
                s = getattr(self, '_session_id', '')

            prompt = (
                "Extract durable facts, preferences, environment details, rules, or conventions "
                "from the following text. For each fact found, call layered_add(content=\"...\", "
                "topic=\"...\", keywords=[...], session_name=\"" + sn + "\", "
                "data_id=\"" + s +
                "\"). Do NOT store ephemeral conversation, task progress, "
                "or anything stale in a week. If no durable facts exist, do nothing and stop.\n\n"
                + text[:8000]
            )
            cmd = f"hermes -p {shlex.quote(profile)} -z {shlex.quote(prompt)}"
            try:
                subprocess.Popen(
                    cmd, shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                logger.debug("[%s] spawned fact extraction (fire-and-forget)", label)
            except Exception as e:
                logger.debug("[%s] fact extraction spawn failed: %s", label, e)

        t = threading.Thread(target=_run, daemon=True, name=f"hlm-extract-{label}")
        t.start()

    def sync_turn(self, user_content: str, assistant_content: str,
                  *, session_id: str = "", messages: Optional[List[Dict[str, Any]]] = None) -> None:
        """Per-turn auto-ingestion: extract facts from the conversation exchange.

        Fires a background hermes session (non-blocking). Throttled to every 5 turns,
        min 30s between extractions. Matches mem0/hindsight per-turn extraction.
        """
        turn_text = f"User: {user_content}\n\nAssistant: {assistant_content}"
        self._extract_facts_background(turn_text, "sync_turn")

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """Session end: extract facts from full history, then run maintenance."""
        if not self._backend:
            return

        # Record session in chain (fallback for sessions without compression)
        self._record_session_in_chain(self._session_id)
        # Extract facts from full message history (one LLM call, fire-and-forget)
        if messages:
            # Build a compact representation of the conversation
            turn_parts = []
            for m in messages:
                role = m.get("role", "unknown")
                content = m.get("content", "")
                if role in ("user", "assistant") and content:
                    turn_parts.append(f"{role}: {content[:2000]}")  # cap per message
            history = "\n\n".join(turn_parts[-40:])  # last 40 messages
            if history:
                self._extract_facts_background(history, "session_end")

        # Maintenance (blocking — runs at session end, user isn't waiting)
        try:
            self._backend.enrich_existing(max_items=20)
            self._backend.sleep(max_items=50, min_age_hours=0)
            self._backend.purge(purge_deleted=True, purge_archived=False,
                               min_age_hours=24.0, vacuum=True)
            # Confidence decay — prevent stale facts from accumulating noise
            self._backend.decay(min_age_days=30, max_age_days=365,
                               decay_rate=0.05, min_score=0.1)
            summaries = getattr(self, '_summaries', None)
            if summaries:
                summaries.sync()
            logger.debug("on_session_end: extraction + maintenance complete")
        except Exception as e:
            logger.debug("on_session_end maintenance failed: %s", e)

        # Rebuild Qdrant index if maintenance caused drift
        try:
            sync = self._backend.sync_check()
            sqlite_n = sync.get("sqlite_active", 0)
            qdrant_n = sync.get("qdrant_profile", 0)
            in_sync = sync.get("in_sync", True)
            logger.info("on_session_end: sync check — SQLite=%d, Qdrant=%d, in_sync=%s", sqlite_n, qdrant_n, in_sync)
            if sqlite_n > 0 and not in_sync:
                logger.info(
                    "on_session_end: Qdrant stale after maintenance (SQLite=%d, Qdrant=%d) — rebuilding",
                    sqlite_n, qdrant_n,
                )
                result = self._backend.rebuild()
                logger.info("on_session_end: rebuild complete: %s", result)
            else:
                logger.debug("on_session_end: no rebuild needed")
        except Exception as e:
            logger.info("on_session_end: auto-rebuild check failed: %s", e)

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        """Extract facts before context compression discards conversation history.

        Queries state.db for current session metadata (UUID + title),
        builds session_name, then spawns a background extraction session.
        """
        if not self._backend or not messages or not self._session_id:
            return ""

        # Record session in chain (compression happened = meaningful work)
        self._record_session_in_chain(self._session_id)

        # Build session_name from state.db
        session_name = self._backend.build_session_name(self._session_id)
        session_id = self._session_id

        # Classify session importance — skip trivial sessions
        info = self._backend.get_session_info(session_id)
        if info:
            classification = self._backend.classify_session(
                info.get("message_count"),
                info.get("tool_call_count"),
                info.get("output_tokens"),
                info.get("end_reason"),
            )
            if classification == "skip":
                logger.debug("on_pre_compress: skipping trivial session %s", session_id[:16])
                return ""
        else:
            classification = "medium"

        # Build a compact representation of the conversation
        turn_parts = []
        for m in messages[-40:]:
            role = m.get("role", "unknown")
            content = m.get("content", "")
            if role in ("user", "assistant") and content:
                turn_parts.append(f"{role}: {content[:2000]}")
        history = "\n\n".join(turn_parts)
        if history:
            self._extract_facts_background(
                history, "pre_compress",
                session_name=session_name,
                session_id=session_id,
            )

        return ""

    def on_memory_write(self, action: str, target: str, content: str,
                        metadata: Optional[Dict[str, Any]] = None) -> None:
        if action == "add" and self._backend and content:
            try:
                scope = "personal" if target == "user" else "general"
                self._backend.add(content, summary=content[:100], scope=scope)
            except Exception as e:
                logger.debug("Layered memory_write mirror failed: %s", e)

    def shutdown(self) -> None:
        if self._backend:
            self._backend.close()
            self._backend = None

    # ---- Config schema --------------------------------------------------

    def get_config_schema(self) -> List[Dict[str, Any]]:
        from hermes_constants import display_hermes_home
        _default_db = f"{display_hermes_home()}/hermes-layered-memory.db"
        return [
            {"key": "db_path", "description": "SQLite database path", "default": _default_db},
            {"key": "qdrant_url", "description": "Qdrant server URL", "default": "http://localhost:6333"},
            {"key": "max_layer", "description": "Default retrieval depth (1-4)", "default": "2",
             "choices": ["1", "2", "3", "4"]},
            {"key": "layer3_mode", "description": "Reranker mode", "default": "inline",
             "choices": ["inline", "delegate", "self"]},
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        config_path = f"{hermes_home}/hermes-layered-memory.json"
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(values, f, indent=2)
        except Exception:
            pass

    # ---- Tool dispatch helpers ------------------------------------------

    def _do_retrieve(self, args: dict) -> dict:
        query = args["query"]
        max_layer = args.get("max_layer", self._max_layer)
        scope = args.get("scope")
        data_type = args.get("data_type")
        data_id = args.get("data_id")
        session_name = args.get("session_name")
        profile_name = args.get("profile_name")
        cross_profile = args.get("cross_profile", False)
        limit = args.get("limit", 5)
        rerank = args.get("rerank", False)
        return self._backend.retrieve(query, max_layer=max_layer, scope=scope,
                                      data_type=data_type, data_id=data_id,
                                      session_name=session_name, profile_name=profile_name,
                                      cross_profile=cross_profile, limit=limit, rerank=rerank)

    def _do_peek(self, args: dict) -> dict:
        return {"layer": args["layer"], "results": self._backend.peek(args["query"], args["layer"])}

    def _do_add(self, args: dict) -> dict:
        result = self._backend.add(**args)
        if isinstance(result, dict):
            # Duplicate detected — return the dict directly
            return result
        return {"uuid": result}

    def _do_update(self, args: dict) -> dict:
        self._backend.update(args.pop("uuid"), **args)
        return {"status": "updated"}

    def _do_delete(self, args: dict) -> dict:
        self._backend.delete(args["uuid"])
        return {"status": "deleted"}

    def _do_rebuild(self, args: dict) -> dict:
        since = args.get("since")
        return self._backend.rebuild(since=since)

    def _do_purge(self, args: dict) -> dict:
        return self._backend.purge(
            purge_deleted=args.get("purge_deleted", True),
            purge_archived=args.get("purge_archived", True),
            min_age_hours=args.get("min_age_hours"),
            vacuum=args.get("vacuum", True))

    def _do_sync_check(self, args: dict) -> dict:
        return self._backend.sync_check()

    def _do_list(self, args: dict) -> dict:
        return {"results": self._backend.list(**args)}

    def _do_sleep(self, args: dict) -> dict:
        return self._backend.sleep(
            max_items=args.get("max_items", 100),
            min_age_hours=args.get("min_age_hours", 24.0))

    def _do_obsidian_ingest(self, args: dict) -> dict:
        vault_path = args["vault_path"]
        exclude = args.get("exclude")
        return self._backend.ingest_obsidian(vault_path, exclude=exclude)

    def _do_enrich(self, args: dict) -> dict:
        return self._backend.enrich_existing(
            since=args.get("since"),
            max_items=args.get("max_items", 100))

    def _do_feedback(self, args: dict) -> dict:
        return self._backend.feedback(args["uuid"], args["helpful"])

    def _ensure_summaries(self):
        """Lazy-initialize summaries backend if not yet done."""
        if self._summaries is not None:
            return self._summaries
        # Lazy init - replicate what initialize() does
        try:
            from hermes_constants import get_hermes_home
            hermes_home = str(get_hermes_home())
            profile_name = getattr(self, '_profile_name', None) or "default"
            db_path = self._config.get("db_path", f"{hermes_home}/hermes-layered-memory-dbs/{profile_name}.db")
            db_path = db_path.replace("$HERMES_HOME", hermes_home)
            db_path = os.path.expanduser(db_path)
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            summaries_dir = os.environ.get("HERMES_LAYERED_SUMMARIES_DIR")
            if not summaries_dir:
                real_home = pwd.getpwuid(os.getuid()).pw_dir
                summaries_dir = os.path.join(real_home, "Documents", "hlm-summaries/")
            summaries_db = os.environ.get("HERMES_LAYERED_SUMMARIES_DB")
            if not summaries_db:
                real_home = pwd.getpwuid(os.getuid()).pw_dir
                summaries_db = os.path.join(real_home, ".hermes", "hermes-layered-memory-dbs", "digests.db")
            self._summaries = SummariesBackend(summaries_db, summaries_dir)
        except Exception as e:
            logger.exception("Failed to lazy-init SummariesBackend: %s", e)
        return self._summaries

    # ---- Summaries tools -------------------------------------------------

    def _do_summarize(self, args: dict) -> dict:
        summaries = self._ensure_summaries()
        if not summaries:
            return tool_error("Summaries not initialized")
        return summaries.add(
            source_url=args["source"],
            source_type=args["source_type"],
            title=args.get("title"),
            highlights=args.get("highlights"),
            full_text=args.get("full_text"),
            tags=args.get("tags"),
            metadata=args.get("metadata"),
            force=args.get("force", False),
            profile_name=self._profile_name,
        )

    def _do_list_summaries(self, args: dict) -> dict:
        summaries = self._ensure_summaries()
        if not summaries:
            return tool_error("Summaries not initialized")
        return summaries.list_summaries(
            source_type=args.get("source_type"),
            source_url=args.get("source_url"),
            tag=args.get("tag"),
            profile=args.get("profile"),
            limit=args.get("limit", 20),
            offset=args.get("offset", 0),
            sort_by=args.get("sort_by"),
        )

    def _do_get_summary(self, args: dict) -> dict:
        summaries = self._ensure_summaries()
        if not summaries:
            return tool_error("Summaries not initialized")
        result = summaries.get(args["uuid"])
        if not result:
            return {"status": "not_found", "uuid": args["uuid"]}
        return result

    def _do_search_summaries(self, args: dict) -> dict:
        summaries = self._ensure_summaries()
        if not summaries:
            return tool_error("Summaries not initialized")
        return summaries.search(args["query"], limit=args.get("limit", 10))

    def _do_delete_summary(self, args: dict) -> dict:
        summaries = self._ensure_summaries()
        if not summaries:
            return tool_error("Summaries not initialized")
        deleted = summaries.delete(
            args["uuid"],
            profile_name=self._profile_name,
            profile=args.get("profile", "own"),
        )
        return {"status": "deleted" if deleted else "not_found", "uuid": args["uuid"]}

    def _do_update_summary(self, args: dict) -> dict:
        summaries = self._ensure_summaries()
        if not summaries:
            return tool_error("Summaries not initialized")
        updated = summaries.update(
            args["uuid"],
            title=args.get("title"),
            highlights=args.get("highlights"),
            tags=args.get("tags"),
            metadata=args.get("metadata"),
            full_text=args.get("full_text"),
        )
        if not updated:
            return {"status": "not_found", "uuid": args["uuid"]}
        return {"status": "updated", "uuid": args["uuid"]}

    def _do_delete_multiple_summaries(self, args: dict) -> dict:
        summaries = self._ensure_summaries()
        if not summaries:
            return tool_error("Summaries not initialized")
        return summaries.delete_multiple(
            args["uuids"],
            profile_name=self._profile_name,
            profile=args.get("profile", "own"),
        )

    def _do_list_expiring_summaries(self, args: dict) -> dict:
        summaries = self._ensure_summaries()
        if not summaries:
            return tool_error("Summaries not initialized")
        records = summaries.list_expiring(
            max_age_days=args.get("max_age_days", 30))
        return {"records": records, "total": len(records)}

    def _do_sync_summaries(self, args: dict) -> dict:
        summaries = self._ensure_summaries()
        if not summaries:
            return tool_error("Summaries not initialized")
        return summaries.sync(profile_name=self._profile_name, profile="own")

    def _do_review(self, args: dict) -> dict:
        """Review memories for garbage/test records via background LLM session.

        Spawns a fire-and-forget hermes session that:
        1. Queries all unreviewed records (or all if force=true)
        2. Feeds them to LLM for keep/delete classification
        3. Marks records as 'keep' or 'delete' in llm_review_status
        4. Soft-deletes records marked 'delete'
        """
        if not self._backend or not self._profile_name:
            return tool_error("Backend not initialized")

        min_age = args.get("min_age_hours", 1)
        force = args.get("force", False)

        # Build the review prompt
        prompt = (
            "Review the following memory records and decide if each is worth keeping.\n"
            "CRITERIA: Keep records with useful facts, preferences, environment details, rules, or conventions. "
            "Delete test records, garbage data, ephemeral task progress, or anything stale/useless.\n\n"
            "For each record, output exactly: KEEP <uuid> or DELETE <uuid>\n\n"
            "Records to review:\n"
        )

        # Get unreviewed records from the DB
        # Validate min_age is numeric to prevent SQL injection
        try:
            min_age = float(min_age)
        except (TypeError, ValueError):
            return tool_error("min_age_hours must be a number")

        try:
            review_filter_param = "1=1" if force else "llm_review_status IS NULL"
            records = self._backend._conn.execute(
                "SELECT uuid, content, topic, data_type, created_at FROM memories "
                "WHERE status='active' AND ({}) AND created_at < datetime('now', ?) "
                "ORDER BY created_at DESC LIMIT 100".format(review_filter_param),
                (f"-{min_age} hours",)
            ).fetchall()

            if not records:
                return {"status": "complete", "reviewed": 0, "message": "No records to review"}

            for r in records:
                prompt += f"[{r[0]}] topic={r[2]} type={r[3]} created={r[4]}\n"
                prompt += f"  content: {r[1][:200]}\n\n"

        except Exception as e:
            return tool_error(f"Failed to query records: {e}")

        def _run_review():
            cmd = f"hermes -p {shlex.quote(self._profile_name)} -z {shlex.quote(prompt)}"
            try:
                subprocess.Popen(
                    cmd, shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                logger.debug("[review] spawned LLM review session (%d records)", len(records))
            except Exception as e:
                logger.debug("[review] spawn failed: %s", e)

        t = threading.Thread(target=_run_review, daemon=True, name="hlm-review")
        t.start()

        return {
            "status": "queued",
            "records_to_review": len(records),
            "message": f"Background review spawned for {len(records)} records. "
                       f"Check back later. Run layered_purge(min_age_hours=24) after review to clean soft-deleted records."
        }

    def _do_compact(self, args: dict) -> dict:
        """Merge duplicate/overlapping memories via LLM."""
        if not self._backend:
            return tool_error("Backend not initialized")

        threshold = args.get("similarity_threshold", 0.90)
        topic = args.get("topic")
        max_groups = args.get("max_groups", 50)
        dry_run = args.get("dry_run", False)

        if dry_run:
            groups = self._backend.find_duplicate_groups(threshold, topic, max_groups)
            return {
                "status": "dry_run",
                "groups_found": len(groups),
                "groups": [
                    {
                        "record_uuids": g["uuids"],
                        "similarity": g.get("similarity", 0)
                    }
                    for g in groups
                ]
            }

        return self._backend.compact(threshold, topic, max_groups)

    def _do_export(self, args: dict) -> str:
        """Export memories as JSON or Markdown."""
        if not self._backend:
            return tool_error("Backend not initialized")

        fmt = args.get("format", "json")
        status = args.get("status", "active")
        if status == "all":
            status = None
        data_type = args.get("data_type")
        topic = args.get("topic")
        scope = args.get("scope")
        profile_name = args.get("profile_name")
        cross_profile = args.get("cross_profile", False)

        return self._backend.export_memories(
            fmt=fmt,
            status=status,
            data_type=data_type,
            topic=topic,
            scope=scope,
            profile_name=profile_name,
            cross_profile=cross_profile,
        )

    def _do_import(self, args: dict) -> dict:
        """Import memories from JSON data."""
        if not self._backend:
            return tool_error("Backend not initialized")

        data = args.get("data", "")
        mode = args.get("mode", "skip_existing")
        target_status = args.get("target_status", "active")

        if not data:
            return tool_error("No data provided. Pass a JSON string or file path.")

        return self._backend.import_memories(
            data=data,
            mode=mode,
            target_status=target_status,
        )

    def _do_decay(self, args: dict) -> dict:
        """Apply confidence decay to stale memories."""
        if not self._backend:
            return tool_error("Backend not initialized")

        min_age_days = args.get("min_age_days", 30)
        max_age_days = args.get("max_age_days", 365)
        decay_rate = args.get("decay_rate", 0.05)
        min_score = args.get("min_score", 0.1)

        return self._backend.decay(
            min_age_days=min_age_days,
            max_age_days=max_age_days,
            decay_rate=decay_rate,
            min_score=min_score,
        )

    def _do_backup(self, args: dict) -> dict:
        """Backup both memories and summaries databases."""
        if not self._backend:
            return tool_error("Backend not initialized")

        dest_dir = args.get("dest_dir")
        keep_days = args.get("keep_days", 7)

        result = {"memories": None, "summaries": None, "cleaned_old": 0}

        # Backup memories DB
        mem_result = self._backend.backup(dest_dir=dest_dir)
        result["memories"] = mem_result

        # Backup summaries DB
        summaries = self._ensure_summaries()
        if summaries:
            summ_result = summaries.backup(dest_dir=mem_result.get("backup_path") and os.path.dirname(mem_result["backup_path"]))
            result["summaries"] = summ_result

        # Clean old backups
        if keep_days > 0 and dest_dir:
            import glob
            import time
            now = time.time()
            cutoff = now - (keep_days * 86400)
            backup_dir = dest_dir if dest_dir else os.path.dirname(self._backend._db_path)
            for f in glob.glob(os.path.join(backup_dir, "*.backup.*.db")):
                if os.path.getmtime(f) < cutoff:
                    os.remove(f)
                    result["cleaned_old"] += 1

        return result


def register(ctx) -> None:
    ctx.register_memory_provider(LayeredMemoryProvider())

