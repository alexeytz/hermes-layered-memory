"""Summaries backend — digest storage with highlights + full MD files.

Separate from layered memory. Stores summaries of YouTube videos, git repos,
documentation, etc. Highlights in SQLite (FTS5-searchable), full text in MD files.
"""
from __future__ import annotations

import json
import os
import re
import uuid as _uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import sqlite3
import logging
logger = logging.getLogger(__name__)

SCHEMA_VERSION = 5


class SummariesBackend:
    """Summaries database backend.

    SQLite with FTS5 for searchable highlights. Full summaries saved as
    .md files in a configurable directory (default: ~/Documents/hlm-summaries/).
    """

    def __init__(self, db_path: str, summaries_dir: str = None):
        """Initialize summaries backend.

        Args:
            db_path: Path to SQLite database.
            summaries_dir: Directory for .md summary files.
                          Default: ~/Documents/hlm-summaries/
        """
        self._db_path = os.path.expanduser(db_path)
        # Resolve summaries_dir — use real home, not profile HOME
        if summaries_dir:
            self._summaries_dir = summaries_dir
        else:
            import pwd
            real_home = pwd.getpwuid(os.getuid()).pw_dir
            self._summaries_dir = os.path.join(real_home, "Documents", "hlm-summaries/")

        # Ensure directories exist
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        os.makedirs(self._summaries_dir, exist_ok=True)

        self._conn = sqlite3.connect(self._db_path, timeout=30, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.row_factory = sqlite3.Row
        self._init_schema()
        logger.info("summaries: db=%s dir=%s", self._db_path, self._summaries_dir)

    def _init_schema(self):
        """Create summaries tables if they don't exist."""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS summaries (
                uuid TEXT PRIMARY KEY,
                source_url TEXT NOT NULL,
                source_type TEXT NOT NULL,
                title TEXT,
                highlights TEXT,
                summary_path TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT,
                tags TEXT,
                metadata TEXT,
                content_hash TEXT,
                status TEXT NOT NULL DEFAULT 'complete',
                coverage_status TEXT,
                token_estimate INTEGER,
                full_text TEXT
            )
        """)

        # FTS5 virtual table for searchable highlights + full_text
        # Porter stemmer + Unicode support. Indexes full_text for deep search.
        self._conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS summaries_fts
            USING fts5(uuid, highlights, title, tags, full_text,
                      content='summaries', content_rowid='rowid',
                      tokenize='porter unicode61 remove_diacritics 2')
        """)

        # Triggers to keep FTS5 in sync
        # Tags stored as space-separated string in FTS5 (not JSON) for individual tag indexing
        self._conn.execute("""
            CREATE TRIGGER IF NOT EXISTS summaries_ai AFTER INSERT ON summaries
            BEGIN
                INSERT INTO summaries_fts(uuid, highlights, title, tags, full_text)
                SELECT new.uuid, new.highlights, new.title,
                    CASE WHEN new.tags IS NOT NULL
                        THEN lower(replace(replace(replace(replace(new.tags, '[', ''), ']', ''), '"', ''), ',', ' '))
                        ELSE '' END,
                    new.full_text;
            END
        """)

        self._conn.execute("""
            CREATE TRIGGER IF NOT EXISTS summaries_au AFTER UPDATE ON summaries
            BEGIN
                UPDATE summaries_fts SET
                    highlights = new.highlights,
                    title = new.title,
                    tags = CASE WHEN new.tags IS NOT NULL
                        THEN lower(replace(replace(replace(replace(new.tags, '[', ''), ']', ''), '"', ''), ',', ' '))
                        ELSE '' END,
                    full_text = new.full_text
                WHERE uuid = new.uuid;
            END
        """)

        self._conn.execute("""
            CREATE TRIGGER IF NOT EXISTS summaries_ad AFTER DELETE ON summaries
            BEGIN
                DELETE FROM summaries_fts WHERE uuid = old.uuid;
            END
        """)

        # Schema version tracking
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_info (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)

        # Migrate existing schema
        current_version_row = self._conn.execute(
            "SELECT value FROM schema_info WHERE key = 'version'").fetchone()
        current_version = int(current_version_row[0]) if current_version_row else 0

        if current_version < 2:
            # Add content_hash column
            try:
                self._conn.execute("ALTER TABLE summaries ADD COLUMN content_hash TEXT")
                logger.info("summaries: migrated to schema v2 (added content_hash)")
            except Exception:
                pass  # Column already exists

        if current_version < 3:
            # Add lifecycle metadata columns
            try:
                self._conn.execute("ALTER TABLE summaries ADD COLUMN status TEXT NOT NULL DEFAULT 'complete'")
                self._conn.execute("ALTER TABLE summaries ADD COLUMN coverage_status TEXT")
                self._conn.execute("ALTER TABLE summaries ADD COLUMN token_estimate INTEGER")
                logger.info("summaries: migrated to schema v3 (added lifecycle metadata)")
            except Exception:
                pass  # Columns already exist

        if current_version < 4:
            # Add full_text column for FTS5 deep search
            try:
                self._conn.execute("ALTER TABLE summaries ADD COLUMN full_text TEXT")
                logger.info("summaries: migrated to schema v4 (added full_text)")
            except Exception:
                pass  # Column already exists

            # Rebuild FTS5 index to sync with current data
            try:
                self._conn.execute("INSERT INTO summaries_fts(summaries_fts) VALUES('rebuild')")
                logger.info("summaries: rebuilt FTS5 index (v3→v4)")
            except Exception as e:
                logger.warning("summaries: FTS5 rebuild failed: %s", e)

        if current_version < 5:
            # Add profile_name column for profile-scoped summaries
            try:
                self._conn.execute("ALTER TABLE summaries ADD COLUMN profile_name TEXT")
                logger.info("summaries: migrated to schema v5 (added profile_name)")
            except Exception:
                pass  # Column already exists

        # Verify migration succeeded before advancing version
        columns = [row[1] for row in self._conn.execute("PRAGMA table_info(summaries)").fetchall()]
        if "profile_name" in columns:
            self._conn.execute(
                "INSERT OR REPLACE INTO schema_info(key, value) VALUES ('version', ?)",
                (str(SCHEMA_VERSION),))
            logger.info("summaries: schema version set to %s", SCHEMA_VERSION)
        else:
            logger.error("summaries: schema v5 migration FAILED — profile_name column missing. "
                         "Version not advanced; will retry on next init.")

        # Always rebuild FTS5 index on init to prevent stale data
        # This is a small cost and ensures consistency after schema changes
        try:
            self._conn.execute("INSERT INTO summaries_fts(summaries_fts) VALUES('rebuild')")
        except Exception as e:
            logger.warning("summaries: FTS5 rebuild on init failed: %s", e)

        # Health check: verify FTS5 row count matches content table
        try:
            fts_count = self._conn.execute("SELECT COUNT(*) FROM summaries_fts").fetchone()[0]
            db_count = self._conn.execute("SELECT COUNT(*) FROM summaries").fetchone()[0]
            if fts_count != db_count:
                logger.warning("summaries: FTS5 count mismatch (fts=%d, db=%d), rebuilding", fts_count, db_count)
                self._conn.execute("INSERT INTO summaries_fts(summaries_fts) VALUES('rebuild')")
        except Exception as e:
            logger.debug("summaries: FTS5 health check skipped: %s", e)

        self._conn.commit()

    def _now(self) -> str:
        """Current UTC timestamp."""
        return datetime.now(timezone.utc).isoformat()

    def _canonicalize_url(self, url: str) -> str:
        """Canonicalize URL for idempotency.

        Normalizes domain aliases, strips query params (except core),
        removes trailing .git, lowercases.
        """
        from urllib.parse import urlparse, urlencode, parse_qs

        parsed = urlparse(url)
        scheme = parsed.scheme.lower()
        netloc = parsed.netloc.lower()

        # YouTube domain aliases
        if netloc == 'youtu.be':
            # youtu.be/VIDEO_ID -> youtube.com/watch?v=VIDEO_ID
            video_id = parsed.path.lstrip('/')
            return f"https://www.youtube.com/watch?v={video_id}"
        if 'youtube.com' in netloc:
            netloc = 'www.youtube.com'
            # Strip tracking params, keep v, list, index
            qs = parse_qs(parsed.query)
            keep = {k: v for k, v in qs.items() if k in ('v', 'list', 'index', 't')}
            path = parsed.path
            if keep:
                path += '?' + urlencode(keep, doseq=True)
            return f"{scheme}://{netloc}{path}"

        # GitHub: https vs ssh, trailing .git
        if 'github.com' in netloc:
            netloc = 'github.com'
            path = parsed.path.lower().rstrip('/')
            if path.endswith('.git'):
                path = path[:-4]
            return f"https://{netloc}{path}"

        # General: lowercase netloc, strip trailing slash
        path = parsed.path.rstrip('/') or '/'
        return f"{scheme}://{netloc}{path}"

    def add(self, source_url: str, source_type: str, title: str,
            highlights: List[str] = None, full_text: str = None,
            tags: List[str] = None, metadata: Dict[str, Any] = None,
            force: bool = False, profile_name: Optional[str] = None) -> dict:
        """Add a new summary.

        Args:
            source_url: URL or path to the source (canonicalized automatically).
            source_type: Type of source (youtube, git, docs, web, etc.).
            title: Title of the summary.
            highlights: List of key bullet points (5-10).
            full_text: Full summary text (saved as .md file).
            tags: List of tags for categorization.
            metadata: Additional metadata dict.
            force: If True, create new summary even if source_url exists.
            profile_name: Profile that created this summary (for scoped deletes).

        Returns:
            dict with uuid, summary_path, already_exists, canonical_url.
        """

        # Canonicalize URL for idempotency
        canonical_url = self._canonicalize_url(source_url)

        # Idempotency check: if source_url exists, return existing UUID
        if not force:
            existing = self._conn.execute(
                "SELECT uuid FROM summaries WHERE source_url = ?", (canonical_url,)
            ).fetchone()
            if existing:
                logger.info("summaries: duplicate source_url %s (existing: %s)", canonical_url, existing["uuid"])
                return {"uuid": existing["uuid"], "already_exists": True, "canonical_url": canonical_url}

        # Empty validation: reject empty highlights + full_text
        if not full_text and not highlights:
            raise ValueError("summaries: either full_text or highlights must be non-empty")

        summary_uuid = str(_uuid.uuid4())
        now = self._now()

        # Normalize tags: lowercase, trim, replace spaces with hyphens
        if tags:
            tags = [re.sub(r'\s+', '-', t.strip().lower()) for t in tags if t.strip()]

        # Generate filename from title or uuid (heavy slugification for portability)
        if title:
            safe_title = re.sub(r'[^\w\s-]', '', title)[:50].strip().lower()
            safe_title = re.sub(r'[\s_]+', '-', safe_title) or summary_uuid[:8]
        else:
            safe_title = summary_uuid[:8]

        filename = f"{safe_title}-{summary_uuid[:8]}.md"
        # Store relative path in DB (resolve at runtime for portability)
        relative_path = filename
        summary_path = os.path.join(self._summaries_dir, filename)

        # Save full text as .md file FIRST (crash safety: write file before DB row)
        # Crash between file and DB write leaves harmless orphaned file (sync handles it)
        if full_text:
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(f"# {title or 'Summary'}\n\n")
                f.write(f"**Source:** {canonical_url}\n")
                f.write(f"**Type:** {source_type}\n")
                f.write(f"**Created:** {now}\n\n")
                if highlights:
                    f.write("## Highlights\n\n")
                    for h in highlights:
                        f.write(f"- {h}\n")
                    f.write("\n")
                f.write("## Full Summary\n\n")
                f.write(full_text)
                f.write("\n")
            logger.debug("summaries: saved %s", summary_path)

        # Compute content hash for freshness detection
        content_hash = None
        if full_text:
            import hashlib
            content_hash = hashlib.sha256(full_text.encode('utf-8')).hexdigest()[:16]

        # Compute lifecycle metadata
        token_estimate = None
        coverage_status = None
        if full_text:
            token_estimate = len(full_text.split())  # rough word count
            coverage_status = "full" if len(highlights or []) >= 3 else "partial"

        # Store in database AFTER file is written (crash safety)
        # Store relative path for portability (resolve at runtime)
        self._conn.execute("""
            INSERT INTO summaries (uuid, source_url, source_type, title,
                                   highlights, summary_path, created_at,
                                   updated_at, tags, metadata, content_hash,
                                   status, coverage_status, token_estimate, full_text,
                                   profile_name)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (summary_uuid, canonical_url, source_type, title,
              json.dumps(highlights or []), relative_path if full_text else None,
              now, None, json.dumps(tags or []), json.dumps(metadata or {}),
              content_hash, "complete", coverage_status, token_estimate,
              full_text, profile_name))

        self._conn.commit()

        return {
            "uuid": summary_uuid,
            "summary_path": summary_path,
            "source_url": canonical_url,
            "canonical_url": canonical_url,
            "already_exists": False,
            "source_type": source_type,
            "title": title,
            "created_at": now,
        }

    def _resolve_path(self, relative_path: str) -> str:
        """Resolve relative path to absolute path.

        Args:
            relative_path: Relative path from DB (e.g., "title-uuid.md").

        Returns:
            Absolute path.
        """
        if not relative_path:
            return ""
        # Already absolute (legacy data)
        if os.path.isabs(relative_path):
            return relative_path
        return os.path.join(self._summaries_dir, relative_path)

    def get(self, summary_uuid: str) -> Optional[dict]:
        """Get a summary by UUID.

        Returns highlights from DB (small) + path to .md file.
        Validates .md file exists before returning (validation-on-access).
        Agent reads .md file itself via read_file (offset/limit for
        chunking) or delegates to subagent with file path.

        Returns:
            dict with summary data, or None if not found.
        """
        row = self._conn.execute(
            "SELECT * FROM summaries WHERE uuid = ?", (summary_uuid,)
        ).fetchone()

        if not row:
            return None

        result = dict(row)
        result["highlights"] = json.loads(result.get("highlights") or "[]")
        result["tags"] = json.loads(result.get("tags") or "[]")
        result["metadata"] = json.loads(result.get("metadata") or "{}")

        # Don't leak full_text to agent — return path only.
        # Agent reads .md via read_file(offset, limit) for context budget control.
        result.pop("full_text", None)

        # Resolve relative path to absolute (for portability)
        if result.get("summary_path"):
            result["summary_path"] = self._resolve_path(result["summary_path"])

        # Validate .md file exists (validation-on-access)
        if result.get("summary_path"):
            if not os.path.exists(result["summary_path"]):
                # Orphaned — delete the row entirely (not mark-and-null)
                self._conn.execute("DELETE FROM summaries WHERE uuid = ?", (summary_uuid,))
                self._conn.commit()
                logger.warning("summaries: deleted orphaned record %s (md missing)", summary_uuid)
                return None  # Not found

        return result

    def list_summaries(self, source_type: Optional[str] = None, source_url: Optional[str] = None,
                       tag: Optional[str] = None, limit: int = 20, offset: int = 0,
                       sort_by: Optional[str] = None, profile: Optional[str] = None) -> dict:
        """List summaries with optional filters.

        Args:
            source_type: Filter by source type.
            source_url: Filter by source URL (for idempotency — check if already summarized).
            tag: Filter by tag.
            limit: Max results to return.
            offset: Pagination offset.
            sort_by: Sort field (created_at, updated_at). Default: created_at DESC.
            profile: Filter by profile name.

        Returns:
            dict with records and total count.
        """
        query = "SELECT * FROM summaries WHERE 1=1"
        params = []

        if source_type:
            query += " AND source_type = ?"
            params.append(source_type)

        if source_url:
            query += " AND source_url = ?"
            params.append(source_url)

        if tag:
            query += " AND tags LIKE ?"
            params.append(f"%\"{tag}\"%")

        if profile:
            query += " AND profile_name = ?"
            params.append(profile)

        # Sort: default created_at DESC, allow updated_at DESC
        if sort_by == "updated_at":
            query += " ORDER BY updated_at DESC"
        else:
            query += " ORDER BY created_at DESC"
        query += " LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        rows = self._conn.execute(query, params).fetchall()

        records = []
        for row in rows:
            r = dict(row)
            r["highlights"] = json.loads(r.get("highlights") or "[]")
            r["tags"] = json.loads(r.get("tags") or "[]")
            r.pop("full_text", None)  # Consistent: path only, not full text
            # Resolve relative path to absolute
            if r.get("summary_path"):
                r["summary_path"] = self._resolve_path(r["summary_path"])
            records.append(r)

        # Get filtered count (not DB total)
        count_query = "SELECT COUNT(*) FROM summaries WHERE 1=1"
        count_params = []
        if source_type:
            count_query += " AND source_type = ?"
            count_params.append(source_type)
        if source_url:
            count_query += " AND source_url = ?"
            count_params.append(source_url)
        if tag:
            count_query += " AND tags LIKE ?"
            count_params.append(f"%\"{tag}\"%")
        if profile:
            count_query += " AND profile_name = ?"
            count_params.append(profile)

        total = self._conn.execute(count_query, count_params).fetchone()[0]

        return {"records": records, "total": total, "limit": limit, "offset": offset}

    def search(self, query: str, limit: int = 10, sort_by: str = None) -> List[dict]:
        """Search summaries by FTS5 on highlights, title, tags, full_text.

        Args:
            query: Search query.
            limit: Max results.
            sort_by: Sort field (created_at, updated_at). Default: rank (relevance).

        Returns:
            List of matching summaries with snippet and rank.
        """
        # Escape FTS5 special characters to prevent syntax errors
        # Hyphen (-) is FTS5 NOT operator, quotes wrap phrases, * is wildcard
        escaped = query.replace('"', '""').replace('-', ' ')

        # Try with snippet first, fall back to rank only if snippet not supported
        try:
            rows = self._conn.execute("""
                SELECT s.*, snippet(summaries_fts, highlights, '<b>', '</b>', '...', 48) AS snippet, rank
                FROM summaries_fts
                JOIN summaries s ON summaries_fts.rowid = s.rowid
                WHERE summaries_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            """, (escaped, limit)).fetchall()
        except Exception:
            # If snippet fails, try rank only. If that also fails, rebuild FTS5.
            try:
                rows = self._conn.execute("""
                    SELECT s.*, '' AS snippet, rank
                    FROM summaries_fts
                    JOIN summaries s ON summaries_fts.rowid = s.rowid
                    WHERE summaries_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                """, (escaped, limit)).fetchall()
            except Exception as inner_e:
                # FTS5 corruption detected - try to rebuild
                logger.warning("summaries: FTS5 search failed (%s), rebuilding index", inner_e)
                try:
                    self._conn.execute("INSERT INTO summaries_fts(summaries_fts) VALUES('rebuild')")
                    self._conn.commit()
                    # Retry search after rebuild
                    rows = self._conn.execute("""
                        SELECT s.*, '' AS snippet, rank
                        FROM summaries_fts
                        JOIN summaries s ON summaries_fts.rowid = s.rowid
                        WHERE summaries_fts MATCH ?
                        ORDER BY rank
                        LIMIT ?
                    """, (escaped, limit)).fetchall()
                except Exception as rebuild_e:
                    logger.error("summaries: FTS5 rebuild failed (%s), returning empty results", rebuild_e)
                    rows = []

        results = []
        for row in rows:
            r = dict(row)
            r["highlights"] = json.loads(r.get("highlights") or "[]")
            r["tags"] = json.loads(r.get("tags") or "[]")
            r["metadata"] = json.loads(r.get("metadata") or "{}")
            r.pop("full_text", None)  # Don't leak full text
            # Resolve relative path to absolute
            if r.get("summary_path"):
                r["summary_path"] = self._resolve_path(r["summary_path"])
            results.append(r)

        return results
    def delete(self, summary_uuid: str, profile_name: Optional[str] = None, profile: str = "own") -> bool:
        """Delete a summary and its .md file.

        Args:
            summary_uuid: UUID to delete.
            profile_name: Current profile name (used for profile-scoped delete).
            profile: If "own" (default), only delete if profile matches. If "all", delete any profile.

        Returns:
            True if deleted, False if not found or profile mismatch.
        """
        row = self._conn.execute(
            "SELECT summary_path, profile_name FROM summaries WHERE uuid = ?",
            (summary_uuid,)
        ).fetchone()

        if not row:
            return False

        # Profile scope check
        if profile == "own" and profile_name:
            if row["profile_name"] and row["profile_name"] != profile_name:
                logger.debug("summaries: refused delete %s (profile mismatch: %s vs %s)",
                           summary_uuid, row["profile_name"], profile_name)
                return False

        # Delete .md file (resolve relative path to absolute)
        md_path = self._resolve_path(row["summary_path"]) if row["summary_path"] else ""
        if md_path and os.path.exists(md_path):
            os.remove(md_path)
            logger.debug("summaries: deleted %s", md_path)

        # Delete from database
        self._conn.execute("DELETE FROM summaries WHERE uuid = ?", (summary_uuid,))
        self._conn.commit()

        return True

    def update(self, summary_uuid: str, **fields) -> bool:
        """Update a summary.

        Args:
            summary_uuid: UUID to update.
            **fields: Fields to update (title, highlights, tags, metadata, full_text).

        Returns:
            True if found (updated or unchanged), False if not found.
        """
        allowed = {"title", "highlights", "tags", "metadata", "full_text"}

        # Check exists first
        existing = self._conn.execute(
            "SELECT * FROM summaries WHERE uuid = ?", (summary_uuid,)
        ).fetchone()

        if not existing:
            return False

        clean = {k: v for k, v in fields.items() if k in allowed and v is not None}

        if not clean:
            return True  # Found but nothing to update

        now = self._now()

        # Normalize tags: lowercase, trim, replace spaces with hyphens
        if "tags" in clean and clean["tags"]:
            clean["tags"] = [re.sub(r'\s+', '-', t.strip().lower()) for t in clean["tags"] if t.strip()]

        # Recalculate coverage_status and token_estimate if highlights changed
        if "highlights" in clean:
            highlights = clean["highlights"]
            if isinstance(highlights, list):
                clean["coverage_status"] = "full" if len(highlights) >= 3 else "partial"

        # Update full_text with hash and estimate
        if "full_text" in clean:
            import hashlib
            text = clean["full_text"]
            clean["content_hash"] = hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]
            clean["token_estimate"] = len(text.split())
            # Update/create .md file (handle partial-to-full transition)
            relative_path = existing["summary_path"]
            md_path = self._resolve_path(relative_path) if relative_path else ""
            title = existing["title"]
            created = existing["created_at"] or now
            source_url = existing["source_url"]
            source_type = existing["source_type"]
            highlights = clean.get("highlights") or json.loads(existing["highlights"] or "[]")
            if isinstance(highlights, str):
                highlights = json.loads(highlights)

            if not relative_path:
                # Partial-to-full transition: create new .md file
                safe_title = re.sub(r'[^\w\s-]', '', title)[:100].strip().lower()
                safe_title = re.sub(r'[\s_]+', '-', safe_title) or existing["uuid"][:8]
                relative_path = f"{safe_title}-{existing['uuid'][:8]}.md"
                md_path = os.path.join(self._summaries_dir, relative_path)
                clean["summary_path"] = relative_path

            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(f"# {title or 'Summary'}\n\n")
                f.write(f"**Source:** {source_url}\n")
                f.write(f"**Type:** {source_type}\n")
                f.write(f"**Created:** {created}\n\n")
                if highlights:
                    f.write("## Highlights\n\n")
                    f.write("\n".join(f"- {h}" for h in highlights) + "\n\n")
                f.write("## Full Summary\n\n")
                f.write(text + "\n")

        # Serialize to JSON
        for k, v in clean.items():
            if isinstance(v, (list, dict)):
                clean[k] = json.dumps(v)

        sets = ", ".join(f"{k} = ?" for k in clean)
        params = list(clean.values()) + [now, summary_uuid]
        self._conn.execute(
            f"UPDATE summaries SET {sets}, updated_at = ? WHERE uuid = ?", params
        )

        # Update .md highlights if highlights changed (but full_text not)
        if "highlights" in clean and "full_text" not in clean:
            md_path = existing["summary_path"]
            if md_path and os.path.exists(md_path):
                with open(md_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                new_highlights = clean["highlights"]
                if isinstance(new_highlights, str):
                    new_highlights = json.loads(new_highlights)
                highlights_section = "\n".join(f"- {h}" for h in new_highlights)
                pattern = r"(## Highlights\n\n)(.*?)(\n\n## Full Summary)"
                replacement = rf"\g<1>{highlights_section}\3"
                content = re.sub(pattern, replacement, content, flags=re.DOTALL)
                with open(md_path, 'w', encoding='utf-8') as f:
                    f.write(content)

        self._conn.commit()
        return True

    def delete_multiple(self, uuids: List[str], profile_name: Optional[str] = None, profile: str = "own") -> dict:
        """Delete multiple summaries and their .md files.

        Args:
            uuids: List of UUIDs to delete.
            profile_name: Current profile name (used for profile-scoped delete).
            profile: If "own" (default), only delete own profile. If "all", delete any profile.

        Returns:
            dict with deleted count and not_found count.
        """
        deleted = 0
        not_found = 0
        for uuid in uuids:
            if self.delete(uuid, profile_name=profile_name, profile=profile):
                deleted += 1
            else:
                not_found += 1
        return {"deleted": deleted, "not_found": not_found}

    def list_expiring(self, max_age_days: int = 30) -> List[dict]:
        """List summaries older than max_age_days.

        Args:
            max_age_days: Age threshold in days.

        Returns:
            List of old summaries.
        """
        from datetime import datetime, timedelta, timezone
        cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        rows = self._conn.execute(
            "SELECT * FROM summaries WHERE created_at < ? ORDER BY created_at ASC",
            (cutoff.isoformat(),)
        ).fetchall()

        results = []
        for row in rows:
            r = dict(row)
            r["highlights"] = json.loads(r.get("highlights") or "[]")
            r["tags"] = json.loads(r.get("tags") or "[]")
            r.pop("full_text", None)
            results.append(r)
        return results

    def close(self):
        """Close database connection."""
        if self._conn:
            self._conn.close()

    def backup(self, dest_dir: Optional[str] = None) -> dict:
        """Backup summaries database to a copy file.

        Uses SQLite's atomic backup API (WAL-safe).

        Args:
            dest_dir: Directory for the backup file. Default: beside the original DB.

        Returns:
            dict with backup_path, timestamp, size_bytes.
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
        logger.info("summaries: backed up to %s (%d bytes)", backup_path, size)
        return {
            "backup_path": backup_path,
            "timestamp": now,
            "size_bytes": size,
        }

    def sync(self, profile_name: Optional[str] = None, profile: str = "own") -> dict:
        """Sync summaries DB with filesystem.

        Checks if .md files referenced in DB still exist. Deletes records
        where the .md file was deleted by the user.

        Args:
            profile_name: Current profile name (for scoped sync).
            profile: If "own" (default), only sync own profile records. If "all", sync all.

        Returns:
            dict with orphaned count and cleaned records.
        """
        where = "summary_path IS NOT NULL"
        params = []
        if profile == "own" and profile_name:
            where += " AND (profile_name = ? OR profile_name IS NULL)"
            params = [profile_name]

        orphaned = 0
        rows = self._conn.execute(
            f"SELECT uuid, summary_path FROM summaries WHERE {where}", params
        ).fetchall()

        for row in rows:
            md_path = self._resolve_path(row["summary_path"])
            if not os.path.exists(md_path):
                self._conn.execute("DELETE FROM summaries WHERE uuid = ?", (row["uuid"],))
                orphaned += 1
                logger.warning("summaries: deleted orphaned record %s (md file missing: %s)",
                               row["uuid"], md_path)

        self._conn.commit()
        return {"orphaned": orphaned, "total_checked": len(rows)}
