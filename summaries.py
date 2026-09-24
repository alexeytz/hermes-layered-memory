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
from typing import Any, Dict, List, Optional, Tuple

import sqlite3
import logging
import pwd
import threading
logger = logging.getLogger(__name__)

# `backend.constants` is a stdlib-only leaf module, so importing it here adds
# no cycle: it is the only place both this file and `backend/` can share a
# definition, and the string-filter refusal is one defect found in six places.
from backend.constants import require_str_filters  # noqa: E402

#: The two words a caller may use for `profile`/`scope` on a summaries call.
#: One definition, because the vocabulary was enforced on exactly one of nine
#: backend doors: `sync` refused an unknown word (2026-08-22 ox-alpha F1) and
#: the other eight — everything reached through `_profile_sql` or
#: `_row_profile_ok` — treated anything that is not the literal "all" as if
#: the caller had said "own". Driven on the pre-fix tree:
#:
#:     list(profile="all")  -> 2 records   (both profiles)
#:     list(profile="al")   -> 1 record    (silently own, no error, no log)
#:     sync(profile="al")   -> {"error": "profile must be 'own' or 'all' ..."}
#:
#: The direction is safe — a junk word can only *narrow* to the caller's own
#: rows, never widen — which is exactly why it survives review: a caller who
#: meant "all" and typed "al" is told nothing and reads a short answer as a
#: complete one. A frozenset rather than a repeated tuple literal so a third
#: scope word cannot be added to one door and not the others.
#: 2026-09-14 round 1 bundle05 (F1).
PROFILE_SCOPES = frozenset({"own", "all"})


def _valid_profile_scope(profile: Any) -> bool:
    """Whether `profile` is one of the two words every summaries door accepts."""
    return profile in PROFILE_SCOPES


SCHEMA_VERSION = 6


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
        # Resolve ~ to real home (Hermes remaps $HOME)
        real_home = pwd.getpwuid(os.getuid()).pw_dir
        if db_path.startswith("~"):
            db_path = db_path.replace("~", real_home, 1)
        elif "$" in db_path:
            db_path = os.path.expandvars(db_path)
        self._db_path = db_path
        # Resolve summaries_dir — use real home, not profile HOME
        if summaries_dir:
            self._summaries_dir = summaries_dir
        else:
            real_home = pwd.getpwuid(os.getuid()).pw_dir
            self._summaries_dir = os.path.join(real_home, "Documents", "hlm-summaries/")

        # Ensure directories exist
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        os.makedirs(self._summaries_dir, exist_ok=True)

        # One connection per thread. A single shared connection lets a commit()
        # on any thread commit whatever another thread has in flight — the same
        # defect fixed in backend.py. Callers reach it through the _conn
        # property, so every call site resolves to its own thread's handle.
        self._conn_local = threading.local()
        self._init_schema()
        logger.info("summaries: db=%s dir=%s", self._db_path, self._summaries_dir)

    @property
    def _conn(self) -> sqlite3.Connection:
        """Thread-local SQLite connection (created on first use per thread)."""
        conn = getattr(self._conn_local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._db_path, timeout=30)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=30000")  # shared DB, multiple profiles
            conn.row_factory = sqlite3.Row
            self._conn_local.conn = conn
        return conn

    def _init_schema(self):
        """Create summaries tables if they don't exist."""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS summaries (
                uuid TEXT PRIMARY KEY,
                source_url TEXT NOT NULL UNIQUE,
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

        # Triggers to keep FTS5 in sync.
        #
        # summaries_fts is an external-content table (content='summaries',
        # content_rowid='rowid'), so every trigger must pass the *content
        # rowid* explicitly. All three omitted it, and both halves broke:
        #
        #   INSERT without rowid — FTS5 assigns its own autoincrement rowid, so
        #     index rowids drift away from content rowids. Any join on
        #     f.rowid = s.rowid then returns the wrong record: a live database
        #     was found where MATCH 'Horthy' returned a different summary and
        #     MATCH 'IPython' returned that one, the two swapped.
        #   'delete' without rowid — FTS5 cannot locate the entry, so the delete
        #     is a no-op and the index keeps a stale entry for every row ever
        #     removed. Measured: 86 content rows against 87 docsize entries.
        #
        # `CREATE TRIGGER IF NOT EXISTS` will not replace an existing wrong
        # trigger, so they are dropped first — otherwise every database created
        # before this fix keeps the broken pair forever.
        #
        # Tags are flattened to a space-separated string so individual tags are
        # indexed; the same expression must be used on the delete side or FTS5
        # computes a different term list and the removal misses.
        _tags_expr = ("CASE WHEN {r}.tags IS NOT NULL THEN "
                      "lower(replace(replace(replace(replace({r}.tags, '[', ''), "
                      "']', ''), '\"', ''), ',', ' ')) ELSE '' END")
        for _t in ("summaries_ai", "summaries_au", "summaries_ad"):
            self._conn.execute(f"DROP TRIGGER IF EXISTS {_t}")

        self._conn.execute(f"""
            CREATE TRIGGER summaries_ai AFTER INSERT ON summaries
            BEGIN
                INSERT INTO summaries_fts(rowid, uuid, highlights, title, tags, full_text)
                VALUES (new.rowid, new.uuid, new.highlights, new.title,
                        {_tags_expr.format(r='new')}, new.full_text);
            END
        """)

        self._conn.execute(f"""
            CREATE TRIGGER summaries_au AFTER UPDATE ON summaries
            BEGIN
                INSERT INTO summaries_fts(summaries_fts, rowid, uuid, highlights, title, tags, full_text)
                VALUES('delete', old.rowid, old.uuid, old.highlights, old.title,
                       {_tags_expr.format(r='old')}, old.full_text);
                INSERT INTO summaries_fts(rowid, uuid, highlights, title, tags, full_text)
                VALUES (new.rowid, new.uuid, new.highlights, new.title,
                        {_tags_expr.format(r='new')}, new.full_text);
            END
        """)

        self._conn.execute(f"""
            CREATE TRIGGER summaries_ad AFTER DELETE ON summaries
            BEGIN
                INSERT INTO summaries_fts(summaries_fts, rowid, uuid, highlights, title, tags, full_text)
                VALUES('delete', old.rowid, old.uuid, old.highlights, old.title,
                       {_tags_expr.format(r='old')}, old.full_text);
            END
        """)

        # Reconcile the FTS index before any migration that deletes rows.
        #
        # Ordering is load-bearing. The v6 de-duplication below removes rows,
        # which now fires a delete trigger that actually works. FTS5 documents
        # that a 'delete' whose supplied values are not present in the index
        # leaves that index corrupt — and a database arriving here with an
        # inconsistent index is precisely the population this repair exists
        # for, since the pre-fix triggers wrote entries at the wrong rowids and
        # removed nothing. Deleting first and repairing afterwards therefore
        # turned a stale index into a malformed one: observed as
        # "database disk image is malformed" on the very next open.
        #
        # A rebuild is cheap relative to a migration and idempotent, so it runs
        # whenever the shadow table disagrees with the content table.
        try:
            _shadow = self._conn.execute(
                "SELECT COUNT(*) FROM summaries_fts_docsize").fetchone()[0]
            _content = self._conn.execute("SELECT COUNT(*) FROM summaries").fetchone()[0]
            if _shadow != _content:
                logger.warning("summaries: FTS index disagrees with content "
                               "(docsize=%d, rows=%d) — rebuilding before migrating",
                               _shadow, _content)
                self._conn.execute("INSERT INTO summaries_fts(summaries_fts) VALUES('rebuild')")
                self._conn.commit()
        except sqlite3.Error as e:
            logger.warning("summaries: could not reconcile the FTS index before "
                           "migrating: %s", e)

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

        if current_version < 6:
            # Add UNIQUE constraint on source_url to prevent duplicate sources.
            # SQLite cannot add constraints via ALTER TABLE, so this is a unique
            # index instead.
            #
            # The index cannot be created while the table already violates it,
            # and the original code wrapped the attempt in `except Exception:
            # pass` commented "Index already exists or conflict" — which treated
            # a real conflict as a no-op. The verification below then correctly
            # found the index missing, declined to advance the version, and
            # logged an error. Every subsequent init repeated the whole cycle:
            # observed 30 times on a production database stuck at v5 since
            # 2026-08-06, with the documented global UNIQUE on source_url simply
            # not present the entire time.
            #
            # So resolve the duplicates rather than tripping over them. Newest
            # per URL wins (created_at, uuid as a deterministic tie-break) and
            # the rest are removed, which matches this module's own delete()
            # semantics — summaries are hard-deleted, there is no soft state to
            # preserve. Each removal is logged at WARNING with the uuid, title
            # and file path, so the decision is auditable after the fact and the
            # .md files are left on disk rather than destroyed.
            dupes = [r[0] for r in self._conn.execute(
                "SELECT source_url FROM summaries GROUP BY source_url "
                "HAVING COUNT(*) > 1").fetchall()]
            removed = 0
            for url in dupes:
                rows = self._conn.execute(
                    "SELECT uuid, title, summary_path, created_at FROM summaries "
                    "WHERE source_url = ? ORDER BY created_at DESC, uuid DESC",
                    (url,)).fetchall()
                logger.warning(
                    "summaries: %d rows share source_url %r — keeping the newest "
                    "(%s, %s) for the v6 UNIQUE index",
                    len(rows), url, rows[0][0][:8], str(rows[0][3])[:19])
                for uuid_, title, path, created in rows[1:]:
                    self._conn.execute("DELETE FROM summaries WHERE uuid = ?", (uuid_,))
                    removed += 1
                    logger.warning(
                        "summaries:   removed duplicate %s (%r, created %s); "
                        "its file is left on disk at %s",
                        uuid_[:8], (title or "")[:60], str(created)[:19], path)
            if removed:
                # Rebuild the FTS index, do not leave it to the count check
                # below. summaries_fts is an external-content table, so deleting
                # rows out from under it leaves the inverted index describing
                # content that has moved: observed in production after this
                # migration ran, where MATCH 'Horthy' returned the *other*
                # summary and MATCH 'IPython' returned this one — the two were
                # swapped while both counts still agreed. A count comparison
                # cannot see that, which is why it went unnoticed.
                self._conn.execute("INSERT INTO summaries_fts(summaries_fts) VALUES('rebuild')")
                self._conn.commit()
                logger.warning("summaries: removed %d duplicate row(s) to allow the "
                               "v6 UNIQUE index, and rebuilt the FTS index", removed)
            try:
                self._conn.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS idx_summaries_source_url ON summaries(source_url)"
                )
                logger.info("summaries: migrated to schema v6 (added UNIQUE on source_url)")
            except Exception as e:
                # Never silent again: this is the failure that went unnoticed.
                logger.error("summaries: v6 UNIQUE index could not be created: %s: %s",
                             type(e).__name__, e)

        # Verify migration succeeded before advancing version
        columns = [row[1] for row in self._conn.execute("PRAGMA table_info(summaries)").fetchall()]
        indexes = [row[1] for row in self._conn.execute("PRAGMA index_list(summaries)").fetchall()]
        if "profile_name" in columns and ("idx_summaries_source_url" in indexes or current_version >= 6):
            self._conn.execute(
                "INSERT OR REPLACE INTO schema_info(key, value) VALUES ('version', ?)",
                (str(SCHEMA_VERSION),))
            logger.info("summaries: schema version set to %s", SCHEMA_VERSION)
        else:
            # Say which half failed. The original message named both and left
            # the reader to work out which, on a line that then repeated on
            # every init forever.
            missing = []
            if "profile_name" not in columns:
                missing.append("profile_name column")
            if "idx_summaries_source_url" not in indexes:
                dupes = self._conn.execute(
                    "SELECT COUNT(*) FROM (SELECT source_url FROM summaries "
                    "GROUP BY source_url HAVING COUNT(*) > 1)").fetchone()[0]
                missing.append(
                    f"idx_summaries_source_url ({dupes} duplicate source_url(s) block it)"
                    if dupes else "idx_summaries_source_url")
            logger.error("summaries: schema migration FAILED — missing: %s. "
                         "Version not advanced; will retry on next init.",
                         "; ".join(missing))

        # Only rebuild FTS5 if schema changed, the counts disagree, or the index
        # no longer describes the rows it points at (not every init).
        #
        # Counting is not enough on its own. summaries_fts is an external-content
        # table: its rows are *read through* to `summaries`, so COUNT(*) reports
        # the content table's size no matter what the inverted index holds. A
        # production database was found with both counts at 86 while the index
        # was offset by one — MATCH on one summary's distinctive word returned a
        # different summary. So probe the content too: take the newest row and
        # ask the index to find it by its own uuid.
        try:
            fts_count = self._conn.execute("SELECT COUNT(*) FROM summaries_fts").fetchone()[0]
            db_count = self._conn.execute("SELECT COUNT(*) FROM summaries").fetchone()[0]
            misaligned = False
            # COUNT(*) on an external-content table reads through to `summaries`,
            # so it can never disagree — the shadow docsize table is where a
            # leak actually shows. Every row the old delete trigger failed to
            # remove is still counted here, which is the cheapest true signal
            # that the index and the content have diverged.
            try:
                shadow = self._conn.execute(
                    "SELECT COUNT(*) FROM summaries_fts_docsize").fetchone()[0]
                if shadow != db_count:
                    misaligned = True
            except sqlite3.Error:
                pass  # older FTS5 build without the docsize shadow table
            if not misaligned and fts_count == db_count and db_count:
                newest = self._conn.execute(
                    "SELECT rowid, uuid FROM summaries ORDER BY rowid DESC LIMIT 1").fetchone()
                if newest:
                    found = self._conn.execute(
                        "SELECT f.rowid FROM summaries_fts f WHERE summaries_fts MATCH ? "
                        "AND f.rowid = ?", (f'uuid:"{newest[1]}"', newest[0])).fetchone()
                    misaligned = found is None
            if fts_count != db_count or misaligned:
                logger.warning("summaries: FTS5 %s (fts=%d, db=%d), rebuilding",
                               "index is misaligned with its content" if misaligned
                               else "count mismatch", fts_count, db_count)
                # A rebuild reindexes from the content table using the real
                # rowids, which is also what repairs a database whose entries
                # were written by the pre-fix triggers.
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

        # The canonical form *is* the identity of a summary — it carries the
        # UNIQUE constraint and decides already_exists. Two degenerate cases
        # used to produce identities that were actively wrong:
        #
        #   ""  ->  ":///"   every blank input collapsed onto one shared key, so
        #                    unrelated summaries collided. Two such rows exist in
        #                    the wild and are what blocked the v6 migration.
        #   "example.com/x" -> "://example.com/x"
        #   "/local/path.md" -> ":///local/path.md"
        #
        # urlparse puts a schemeless string entirely in `path`, and the generic
        # branch below then formats `scheme://netloc` from two empty strings.
        # The last case matters because this method's own docstring accepts a
        # "URL or path".
        raw = (url or "").strip()
        if not raw:
            raise ValueError(
                "source_url is blank — it is the identity of a summary and "
                "cannot be empty (a blank value used to canonicalize to ':///', "
                "which every other blank value also mapped to)")

        parsed = urlparse(raw)
        if not parsed.scheme and not parsed.netloc:
            head = raw.split("/", 1)[0]
            if "." in head and not raw.startswith("/"):
                # Bare host: "example.com/x". Assume https rather than emitting
                # a string with an empty scheme.
                parsed = urlparse("https://" + raw)
            else:
                # A local path or an opaque identifier. Not URL-shaped, so leave
                # it verbatim: distinct inputs stay distinct, which is the only
                # property the caller actually depends on.
                return raw

        scheme = parsed.scheme.lower()
        netloc = parsed.netloc.lower()
        # IDNA-normalise so an internationalised domain and its Punycode form
        # canonicalise to one key. Without this, https://example.jp/ typed in
        # unicode and the same host as xn--… are two different `source_url`
        # values, and digests.db has a GLOBAL unique constraint on that column —
        # so the same page can be summarised twice under two spellings.
        # Best-effort: a netloc that will not encode (already punycode, or
        # malformed) is left exactly as it was rather than dropped.
        if netloc and any(ord(ch) > 127 for ch in netloc):
            try:
                host, _, port = netloc.rpartition(":") if ":" in netloc else (netloc, "", "")
                encoded = (host or netloc).encode("idna").decode("ascii")
                netloc = f"{encoded}:{port}" if port else encoded
            except (UnicodeError, ValueError):
                pass

        # YouTube domain aliases (exact/suffix match to prevent typosquat spoofing)
        if netloc == 'youtu.be':
            # youtu.be/VIDEO_ID -> youtube.com/watch?v=VIDEO_ID
            video_id = parsed.path.lstrip('/')
            return f"https://www.youtube.com/watch?v={video_id}"
        if netloc == 'youtube.com' or netloc.endswith('.youtube.com'):
            netloc = 'www.youtube.com'
            # Normalize /shorts/VIDEO_ID and /embed/VIDEO_ID to /watch?v=VIDEO_ID
            path = parsed.path
            video_id = None
            if path.startswith('/shorts/'):
                video_id = path.split('/shorts/')[1].split('?')[0].split('/')[0]
            elif path.startswith('/embed/'):
                video_id = path.split('/embed/')[1].split('?')[0].split('/')[0]
            if video_id:
                return f"https://{netloc}/watch?v={video_id}"
            # Strip tracking params, keep v, list, index
            qs = parse_qs(parsed.query)
            keep = {k: v for k, v in qs.items() if k in ('v', 'list', 'index', 't')}
            if keep:
                path += '?' + urlencode(keep, doseq=True)
            return f"{scheme}://{netloc}{path}"

        # GitHub: https vs ssh, trailing .git (exact/suffix match)
        if netloc == 'github.com' or netloc.endswith('.github.com'):
            netloc = 'github.com'
            path = parsed.path.lower().rstrip('/')
            if path.endswith('.git'):
                path = path[:-4]
            return f"https://{netloc}{path}"

        # General: lowercase netloc, preserve query params (only strip tracking)
        path = parsed.path.rstrip('/') or '/'
        # Strip known tracking params but preserve content-distinguishing ones
        from urllib.parse import parse_qs, urlencode as _urlencode
        keep = {}
        tracking = {'utm_source', 'utm_medium', 'utm_campaign', 'utm_term',
                    'utm_content', 'utm_id', 'fbclid', 'gclid', 'mc_eid'}
        for k, v in parse_qs(parsed.query).items():
            if k.lower() not in tracking:
                keep[k] = v
        qs = ''
        if keep:
            qs = '?' + _urlencode(keep, doseq=True)
        return f"{scheme}://{netloc}{path}{qs}"

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
            force: Skip the idempotency pre-check. **It cannot create a second
                summary for the same source_url** — `source_url` carries a
                UNIQUE index (v6), so the INSERT is rejected and the
                IntegrityError handler below converges to the existing row,
                returning its uuid with `already_exists: True`. The parameter's
                original contract ("create new summary even if source_url
                exists") stopped being achievable when that index landed, and
                the wording outlived it. What force still does is bypass the
                pre-check query, which changes the failure mode (constraint
                violation rather than early return) and nothing else.
                2026-08-22 ox-alpha maintenance review (F2).
            profile_name: Profile that created this summary (for scoped deletes).

        Returns:
            dict with uuid, summary_path, already_exists, canonical_url.
        """

        # Canonicalize URL for idempotency
        canonical_url = self._canonicalize_url(source_url)
        # Keep what the caller actually supplied when canonicalization changed
        # it. The canonical form is the identity — it carries the UNIQUE index
        # and decides already_exists — but it is not always what the source
        # *is*: youtu.be/X becomes youtube.com/watch?v=X, tracking parameters
        # are dropped, a bare host gains https. Recording only the transformed
        # value means the .md file cannot tell you where the summary came from,
        # and when canonicalization was broken it recorded ':///' with no way
        # back. Two summaries lost their provenance that way and had to be
        # recovered from session logs.
        raw_source = (source_url or "").strip()
        original_source = raw_source if raw_source != canonical_url else None
        if original_source:
            metadata = dict(metadata or {})
            metadata.setdefault("source_url_original", original_source)

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
        # A crash between the file write and the DB insert leaves an orphan
        # .md with no row. That is harmless — nothing reads a file without a
        # row — but it is *not* cleaned up: sync() handles the opposite case
        # (a row whose .md is missing, which it deletes), and nothing deletes
        # a file that no row points at. The comment here used to say "sync
        # handles it", which claimed a cleanup that does not exist.
        if full_text:
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(f"# {title or 'Summary'}\n\n")
                f.write(f"**Source:** {canonical_url}\n")
                if original_source:
                    f.write(f"**Original:** {original_source}\n")
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

        # A non-list `highlights` (a caller-supplied JSON string that decodes
        # to a plain string rather than an array, or any other stray type)
        # reached `len(highlights or [])` below uncaught — for a string, that
        # counts *characters*, not highlights, so a 3+ character string
        # silently reported "full" coverage regardless of the actual count.
        # `_do_summarize` (plugin) and MCP's typed schema both guard the
        # common shape (a JSON-array *string*), but neither catches a decoded
        # non-list value, and this is the one write path — direct backend
        # callers included. Coerced rather than raised: every other malformed
        # optional field in this function fails soft the same way.
        # 2026-08-23 review round 6 maintenance F6.
        if highlights is not None and not isinstance(highlights, list):
            logger.warning(
                "summaries.add: highlights was %s, not a list — discarding",
                type(highlights).__name__)
            highlights = None

        # Compute lifecycle metadata
        token_estimate = None
        coverage_status = None
        if full_text:
            token_estimate = len(full_text.split())  # rough word count
            coverage_status = "full" if len(highlights or []) >= 3 else "partial"

        # Store in database AFTER file is written (crash safety)
        # Store relative path for portability (resolve at runtime)
        try:
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
        except sqlite3.IntegrityError:
            # TOCTOU: another add() for the same canonical_url won the race
            # between our pre-check (line ~333) and this INSERT — the v6
            # UNIQUE index on source_url rejects us. Converge to the winner's
            # UUID (same idempotent contract the pre-check provides in the
            # non-concurrent case) and clean up our now-orphaned .md file.
            #
            # The failed INSERT leaves this connection's implicit transaction
            # open (nothing to commit, but sqlite3 doesn't auto-close it on
            # error) — since this connection is thread-local and long-lived,
            # every subsequent write on this thread would block behind the
            # dangling lock until something else on this thread commits.
            # rollback() releases it; there's nothing to lose since the
            # INSERT never took effect.
            self._conn.rollback()
            if full_text and os.path.exists(summary_path):
                try:
                    os.remove(summary_path)
                except OSError:
                    pass
            winner = self._conn.execute(
                "SELECT uuid FROM summaries WHERE source_url = ?", (canonical_url,)
            ).fetchone()
            if winner:
                logger.info("summaries: lost add() race for %s, using winner %s", canonical_url, winner["uuid"])
                return {"uuid": winner["uuid"], "already_exists": True, "canonical_url": canonical_url}
            raise
        except Exception:
            # Any other INSERT failure (disk full, locked DB, schema drift)
            # leaves the .md we wrote a moment ago with no row pointing at it
            # — an orphan that list/get can never reach and that no cleanup
            # path knows about, because every one of them starts from a row.
            # Only the IntegrityError branch above used to sweep it up. Roll
            # back for the same dangling-lock reason, drop the file, re-raise
            # so the caller still sees the real failure.
            try:
                self._conn.rollback()
            except sqlite3.Error:
                pass
            if full_text and os.path.exists(summary_path):
                try:
                    os.remove(summary_path)
                except OSError:
                    pass
            raise

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

    def get(self, summary_uuid: str, profile_name: Optional[str] = None,
            profile: str = "own") -> Optional[dict]:
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

        # Ownership, not just existence. This read returned any row in the
        # shared digests.db by uuid alone, so an MCP client with one profile's
        # credentials could read another profile's title, highlights, tags and
        # metadata — all of it summarised from web pages the reader never saw.
        # delete() and update() have refused this for releases; the read beside
        # them did not, which is the same class as every other "one member of
        # the class was fixed" defect in this file's history.
        if not self._row_profile_ok(row["profile_name"], profile_name, profile,
                                    "get", summary_uuid):
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

        # Report a missing .md file; do not delete on a read.
        #
        # This used to DELETE the row and return None. The row carries the
        # full_text column, so that destroyed the only surviving copy of the
        # summary whenever the file was gone for *any* reason — a wrong
        # summaries_dir, an unmounted volume, or an EACCES (os.path.exists
        # returns False for all three), not just a genuine user deletion.
        # Destructive cleanup belongs in the explicit sync() action, which
        # now also preserves rows that still hold full_text.
        if result.get("summary_path") and not os.path.exists(result["summary_path"]):
            logger.warning("summaries: %s references a missing .md file (%s)",
                           summary_uuid, result["summary_path"])
            result["file_missing"] = True

        return result

    def list_summaries(self, source_type: Optional[str] = None, source_url: Optional[str] = None,
                       tag: Optional[str] = None, limit: int = 20, offset: int = 0,
                       sort_by: Optional[str] = None, profile_name: Optional[str] = None,
                       profile: str = "own") -> dict:
        """List summaries with optional filters.

        Args:
            source_type: Filter by source type.
            source_url: Filter by source URL (for idempotency — check if already summarized).
            tag: Filter by tag.
            limit: Max results to return.
            offset: Pagination offset.
            sort_by: Sort field (created_at, updated_at). Default: created_at DESC.
            profile_name: The calling profile, matched exactly under scope "own".
            profile: "own" (default) or "all" — the scope word every other
                entry point uses. See below.

        Returns:
            dict with records and total count.

        `profile` used to be a *literal profile name filter*, which is why this
        was the one summaries read 0.7.51 did not fix. It broke in both
        directions at once: the plugin passed no argument at all, so the
        default listed every profile's rows out of the shared digests.db, and
        the escape hatch the release documented — profile="all" — appended
        `AND profile_name = 'all'` and returned only rows belonging to a
        profile literally named "all", i.e. nothing. Measured before the fix,
        two rows owned by two profiles:

            list_summaries()                     -> both rows
            list_summaries(profile="all")        -> []

        Now identical in shape to get/search/list_expiring. A caller that
        passes a bare profile *name* here (the old calling convention) is
        refused by _profile_sql with a message naming what to pass, rather
        than silently filtering on a scope word.
        """
        # A non-string filter reached `source_type = ?` / `source_url = ?` /
        # the tag LIKE and raised `sqlite3.ProgrammingError: type 'list' is
        # not supported` out of the plugin door, or — for an int — returned
        # `{"records": [], "total": 0}`, which is indistinguishable from
        # "no summaries match". One of six instances of the same class; see
        # `backend.constants.str_filter_error` for the whole set and for why
        # the MCP twin never had it. 2026-09-16 enumeration.
        require_str_filters(source_type=source_type, source_url=source_url,
                            tag=tag, sort_by=sort_by)
        # `limit`/`offset` reached `LIMIT ? OFFSET ?` with no validation —
        # SQLite treats a negative LIMIT as "unlimited" (the same footgun
        # documented on backend/pipeline.py's retrieve()), and an unbounded
        # limit could return the whole shared digests.db table in one call.
        # Clamped the same way retrieve() clamps its own limit, rather than
        # raising: this is a read-side pagination parameter, not a
        # destructive-window age, and every other listing door in this
        # codebase fails soft on malformed input rather than erroring.
        # 2026-08-23 review round 6 F7 / round 7 maintenance F10.
        try:
            limit = max(1, min(int(limit), 200))
        except (TypeError, ValueError):
            limit = 20
        try:
            offset = max(0, int(offset))
        except (TypeError, ValueError):
            offset = 0

        scope = self._profile_sql(profile_name, profile, "list")
        if scope is None:
            return {"records": [], "total": 0, "limit": limit, "offset": offset}
        scope_sql, scope_params = scope

        query = "SELECT * FROM summaries WHERE 1=1"
        params = []

        if source_type:
            query += " AND source_type = ?"
            params.append(source_type)

        if source_url:
            query += " AND source_url = ?"
            params.append(source_url)

        if tag:
            # Normalise the caller's tag the same way `add`/`update` normalise
            # on write (lowercase, trim, spaces to hyphens). Matching the raw
            # value meant a filter could never match what was stored:
            # list_summaries(tag="Machine Learning") looked for `"Machine
            # Learning"` while the row held `"machine-learning"`, and returned
            # nothing while reporting success. 2026-08-22 ox-alpha maintenance
            # review (F4).
            _tag = re.sub(r'\s+', '-', str(tag).strip().lower())
            query += " AND tags LIKE ?"
            params.append(f"%\"{_tag}\"%")

        query += scope_sql
        params.extend(scope_params)

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
            # metadata too, like get() and search(). Leaving it as the raw JSON
            # string made the plugin's fence double-encode it: 0.7.45 added
            # metadata to _wrap_summary_fields, which json.dumps() whatever it
            # is handed, so a string came back quoted and brace-escaped.
            r["metadata"] = json.loads(r.get("metadata") or "{}")
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
            # `_tag`, not `tag` — 0.7.79 normalised the SELECT and left this
            # one raw, so `total` counted against a pattern the rows no longer
            # matched: records came back while total said 0. The comment below
            # states the invariant that broke.
            count_query += " AND tags LIKE ?"
            count_params.append(f"%\"{_tag}\"%")
        # The same scope as the SELECT — `total` is what pagination trusts, so
        # a count over a wider set than the rows would report pages that do
        # not exist.
        count_query += scope_sql
        count_params.extend(scope_params)

        total = self._conn.execute(count_query, count_params).fetchone()[0]

        return {"records": records, "total": total, "limit": limit, "offset": offset}

    def search(self, query: str, limit: int = 10, sort_by: str = None,
               profile_name: Optional[str] = None, profile: str = "own") -> List[dict]:
        """Search summaries by FTS5 on highlights, title, tags, full_text.

        Args:
            query: Search query.
            limit: Max results.
            sort_by: Sort field (created_at, updated_at). Default: rank (relevance).

        Returns:
            List of matching summaries with snippet and rank.
        """
        # Tokenize and quote each term, the same way backend/pipeline.py's
        # _add_bm25_conn does.
        #
        # The previous approach escaped quotes with a backslash and quoted
        # only hyphenated fragments. FTS5 escapes a quote by doubling it, not
        # with a backslash, and an unquoted query is parsed as FTS5 *syntax* —
        # so "what's this?" was `syntax error near "'"`, 'setup (v2)' was a
        # syntax error, and 'cost: 5' was read as a column filter
        # (`no such column: cost`). Four of six ordinary queries failed, and
        # each failure fell through to the "FTS5 corruption" branch below,
        # rebuilding the entire index — a full-table write on a DB shared by
        # every profile — before returning nothing.
        import re as _re
        tokens = _re.findall(r'\w+', query.lower())
        if not tokens:
            logger.debug("summaries: search query %r has no searchable tokens", query[:40])
            return []
        escaped = " OR ".join('"' + t.replace('"', '""') + '"' for t in tokens)

        # Scope before LIMIT — see _profile_sql. Unscoped, this returned other
        # profiles' summaries verbatim, and it is the widest of the three read
        # leaks because it needs no uuid: any query term reaches every row in
        # the shared digests.db.
        scope = self._profile_sql(profile_name, profile, "search")
        if scope is None:
            return []
        scope_sql, scope_params = scope

        # Declared in the signature and docstring since search() was written
        # but never consumed — all three branches below hardcoded
        # `ORDER BY rank`, so a caller sorting by recency silently got
        # relevance order with no error. Same sort_by contract as
        # list_summaries() just above: created_at/updated_at DESC, default
        # rank (relevance).
        if sort_by == "updated_at":
            order_clause = "ORDER BY s.updated_at DESC"
        elif sort_by == "created_at":
            order_clause = "ORDER BY s.created_at DESC"
        else:
            order_clause = "ORDER BY rank"

        # snippet()'s second argument is a column *index*, not a name. Passing
        # the bare identifier `highlights` made SQLite read that column's value
        # and coerce the text to integer 0 — column 0 is `uuid` — so every
        # search silently returned the record's own uuid as its "snippet"
        # instead of a matched excerpt. Index 1 is `highlights` in the
        # summaries_fts column list (uuid, highlights, title, tags, full_text).
        # Try with snippet first, fall back to rank only if snippet not supported
        try:
            rows = self._conn.execute(f"""
                SELECT s.*, snippet(summaries_fts, 1, '<b>', '</b>', '...', 48) AS snippet, rank
                FROM summaries_fts
                JOIN summaries s ON summaries_fts.rowid = s.rowid
                WHERE summaries_fts MATCH ?{scope_sql.replace("profile_name", "s.profile_name")}
                {order_clause}
                LIMIT ?
            """, (escaped, *scope_params, limit)).fetchall()
        except Exception:
            # If snippet fails, try rank only. If that also fails, rebuild FTS5.
            try:
                rows = self._conn.execute(f"""
                    SELECT s.*, '' AS snippet, rank
                    FROM summaries_fts
                    JOIN summaries s ON summaries_fts.rowid = s.rowid
                    WHERE summaries_fts MATCH ?{scope_sql.replace("profile_name", "s.profile_name")}
                    {order_clause}
                    LIMIT ?
                """, (escaped, *scope_params, limit)).fetchall()
            except Exception as inner_e:
                # Only rebuild for genuine index corruption. A syntax error
                # is always the caller's query and never the index, so
                # rebuilding on it meant a full-table write (on a DB shared
                # by every profile) for every punctuated search — then
                # retrying the same broken query and returning nothing.
                msg = str(inner_e).lower()
                corrupt = ("malformed" in msg or "corrupt" in msg
                           or "no such table" in msg or "database disk image" in msg)
                if not corrupt:
                    logger.warning("summaries: FTS5 search failed for query %r: %s",
                                   query[:60], inner_e)
                    return []
                logger.warning("summaries: FTS5 index corrupt (%s), rebuilding", inner_e)
                try:
                    self._conn.execute("INSERT INTO summaries_fts(summaries_fts) VALUES('rebuild')")
                    self._conn.commit()
                    # Retry search after rebuild
                    rows = self._conn.execute(f"""
                        SELECT s.*, '' AS snippet, rank
                        FROM summaries_fts
                        JOIN summaries s ON summaries_fts.rowid = s.rowid
                        WHERE summaries_fts MATCH ?{scope_sql.replace("profile_name", "s.profile_name")}
                        {order_clause}
                        LIMIT ?
                    """, (escaped, *scope_params, limit)).fetchall()
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
            # `rank` is FTS5's own relevance number (lower is better), an
            # index implementation detail this query selects to order by. It
            # rode out to the model on both front ends' search results while
            # neither get() nor list() exposes anything equivalent — payload
            # shape only, no fence gap (`snippet` beside it is fenced).
            # 2026-08-26 xhigh round, bundle03 F8.
            r.pop("rank", None)
            # Resolve relative path to absolute
            if r.get("summary_path"):
                r["summary_path"] = self._resolve_path(r["summary_path"])
            results.append(r)

        return results
    def _profile_sql(self, profile_name: Optional[str], profile: str,
                     op: str) -> Optional[Tuple[str, list]]:
        """The SQL half of _row_profile_ok, for reads that filter in the query.

        Returns (clause, params) to append to a WHERE, or None meaning the
        caller must refuse. Filtering in SQL rather than over the returned
        rows is deliberate: LIMIT is applied by SQLite, so a post-filter would
        silently return fewer than `limit` rows — or none at all — while
        reporting success.

        Note the exact match. `profile_name = ? OR profile_name IS NULL` was
        the rule here and in sync(), and it made every pre-v5 row owned by
        every profile at once. Legacy rows need profile="all", stated
        deliberately, exactly as _row_profile_ok requires.
        """
        if not _valid_profile_scope(profile):
            logger.warning(
                "summaries: %s refused (unknown profile scope %r — use 'own' or 'all')",
                op, profile)
            return None
        if profile == "all":
            return "", []
        if not profile_name:
            logger.warning(
                "summaries: %s refused (profile='own' but profile_name not provided) — "
                "pass profile_name, or profile='all' to read every profile deliberately", op)
            return None
        return " AND profile_name = ?", [profile_name]

    def _row_profile_ok(self, row_profile: Optional[str], profile_name: Optional[str],
                        profile: str, op: str, summary_uuid: str) -> bool:
        """Whether a caller in `profile_name` may act on a row owned by `row_profile`.

        One expression of the rule, called by every row-level write. `delete`
        carried it inline and `update` carried nothing at all, which is how an
        unauthenticated MCP client could overwrite any summary in any profile by
        uuid while the identical delete was refused (0.7.51). Two copies of a
        rule drift; one copy and a second caller cannot.

        `profile="all"` is the deliberate cross-profile escape hatch and is
        honoured as-is. Under `"own"`:

        - no `profile_name` supplied is a refusal, not a wildcard — the caller
          has not said who it is.
        - a row whose `profile_name` is NULL or empty belongs to *nobody*, not
          to everybody. The predicate used to read
          `profile_name = ? OR profile_name IS NULL`, so every profile owned
          every pre-v5 row: any client could delete or sync away legacy rows
          another profile created. Reaching those rows now takes an explicit
          `profile="all"`, which is a decision rather than an accident.
        """
        if not _valid_profile_scope(profile):
            logger.warning(
                "summaries: %s refused %s (unknown profile scope %r — use 'own' or 'all')",
                op, summary_uuid, profile)
            return False
        if profile == "all":
            return True
        if not profile_name:
            logger.warning(
                "summaries: %s refused %s (profile='own' but profile_name not provided)",
                op, summary_uuid)
            return False
        if not row_profile:
            logger.debug(
                "summaries: refused %s %s (row has no profile_name — a legacy row "
                "belongs to no profile; pass profile='all' to act on it deliberately)",
                op, summary_uuid)
            return False
        if row_profile != profile_name:
            logger.debug("summaries: refused %s %s (profile mismatch: %s vs %s)",
                         op, summary_uuid, row_profile, profile_name)
            return False
        return True

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

        # Profile scope check — one shared expression, see _row_profile_ok.
        if not self._row_profile_ok(row["profile_name"], profile_name, profile,
                                    "delete", summary_uuid):
            return False

        # Delete .md file (resolve relative path to absolute).
        #
        # Only when the row is ours. `summary_path` is stored *relative* to a
        # summaries directory, and `_resolve_path` joins it to **this**
        # instance's directory — which is the wrong one for a row belonging to
        # a profile configured with a different HLM_SUMMARIES_DIR. Under
        # `profile="all"` that meant the row was deleted while its file lived
        # on somewhere else, orphaned and invisible: nothing reads a file with
        # no row, and `sync()` only cleans the opposite case. Say so instead of
        # guessing at a path we cannot know.
        _row_profile = row["profile_name"] if "profile_name" in row.keys() else None
        _foreign = bool(_row_profile) and bool(profile_name) and _row_profile != profile_name
        md_path = self._resolve_path(row["summary_path"]) if row["summary_path"] else ""
        if _foreign and md_path and not os.path.isabs(row["summary_path"] or ""):
            logger.warning(
                "summaries: deleting row %s owned by profile %r but leaving its .md "
                "file alone — the stored path (%r) is relative to that profile's "
                "summaries directory, not this server's (%s). Remove it from that "
                "profile if it should go.",
                summary_uuid[:8], _row_profile, row["summary_path"], self._summaries_dir)
        elif md_path and os.path.exists(md_path):
            os.remove(md_path)
            logger.debug("summaries: deleted %s", md_path)

        # Delete from database
        self._conn.execute("DELETE FROM summaries WHERE uuid = ?", (summary_uuid,))
        self._conn.commit()

        return True

    def update(self, summary_uuid: str, profile_name: Optional[str] = None,
               profile: str = "own", **fields) -> bool:
        """Update a summary, scoped to a profile.

        Args:
            summary_uuid: UUID to update.
            profile_name: The calling profile. Required under profile="own".
            profile: "own" (default) checks row ownership; "all" crosses profiles.
            **fields: Fields to update (title, highlights, tags, metadata, full_text).

        Returns:
            True if found and owned (updated or unchanged), False otherwise.

        The scope arguments are not optional decoration. digests.db is one file
        shared by every profile, and this method selected the row and then wrote
        `WHERE uuid = ?` without ever reading its `profile_name`. `delete` beside
        it did the check; `update` did not, and the MCP handler passed it no
        profile at all — so any client reaching that unauthenticated port could
        overwrite the title, highlights, tags, metadata and on-disk .md of any
        summary in any profile, by uuid alone. `_check_write_profile_allowed`
        did not help: it validates the *named* `profile` argument, and a caller
        who names none passes it untouched.
        """
        allowed = {"title", "highlights", "tags", "metadata", "full_text"}

        # Check exists first — and read profile_name, which is the whole point.
        existing = self._conn.execute(
            "SELECT * FROM summaries WHERE uuid = ?", (summary_uuid,)
        ).fetchone()

        if not existing:
            return False

        if not self._row_profile_ok(existing["profile_name"], profile_name, profile,
                                    "update", summary_uuid):
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
                # Partial-to-full transition: create new .md file.
                #
                # Unlike delete()/sync(), which already refuse to touch a
                # foreign row's .md file because `summary_path` resolves
                # against *this* instance's `_summaries_dir` — not
                # necessarily the row's owning profile's — this branch
                # creates a *new* file the same way, for a row `profile="all"`
                # deliberately let through cross-profile. In the common
                # configuration (every profile pointed at the same
                # HLM_SUMMARIES_DIR) that's harmless; a profile configured
                # with its own directory would get the new file created
                # somewhere the owning profile never looks. Not refused
                # outright — the DB row and its metadata still update
                # correctly regardless — but worth a loud line so a real
                # divergence is diagnosable rather than a silent surprise.
                # 2026-08-23 recovered finding (round 7 maintenance
                # F9 / round 10 glm-5.2 maintenance F3, same defect).
                if existing["profile_name"] and profile_name and existing["profile_name"] != profile_name:
                    logger.warning(
                        "summaries: creating .md for row %s owned by profile %r "
                        "from profile %r's summaries directory (%s) — if the two "
                        "profiles use different HLM_SUMMARIES_DIR, the owning "
                        "profile will not find this file",
                        summary_uuid, existing["profile_name"], profile_name,
                        self._summaries_dir)
                # `title` is nullable (add() explicitly supports title=None),
                # and re.sub() on None raises TypeError out of the tool call.
                safe_title = re.sub(r'[^\w\s-]', '', title or "")[:100].strip().lower()
                safe_title = re.sub(r'[\s_]+', '-', safe_title) or existing["uuid"][:8]
                relative_path = f"{safe_title}-{existing['uuid'][:8]}.md"
                md_path = os.path.join(self._summaries_dir, relative_path)
                clean["summary_path"] = relative_path

            # update() has no raw input of its own, so recover the original
            # from metadata rather than dropping the line on every edit.
            #
            # Subscript, not .get(). `existing` is a sqlite3.Row (row_factory
            # is set in __init__), so `isinstance(existing, dict)` was always
            # False and this whole block was dead: _original was always None
            # and the **Original:** line was dropped from the .md on the first
            # edit, which is precisely what the comment above says it prevents.
            # add() writes that line for any URL canonicalization rewrites
            # (youtu.be/X -> youtube.com/watch?v=X), so the provenance the
            # file existed to carry was lost on the next full_text update.
            _meta = existing["metadata"]
            if isinstance(_meta, str):
                try:
                    _meta = json.loads(_meta)
                except (json.JSONDecodeError, TypeError):
                    _meta = {}
            _original = (_meta or {}).get("source_url_original") \
                if isinstance(_meta, dict) else None

            # Build the content in memory; the write itself is deferred until
            # after the UPDATE below, in the same DB-write-then-file-write
            # order (and same rollback-on-file-failure) as the highlights-only
            # branch further down. This branch used to open(md_path, 'w') and
            # write immediately, with no try/except and *before* the UPDATE
            # even ran: an UPDATE failure (or the process dying between the
            # write and commit()) left the .md file holding new content while
            # the row still carried the old content_hash/token_estimate and,
            # on a partial-to-full transition, the old (empty) summary_path —
            # file and row diverged with no error raised anywhere. The
            # highlights-only branch beside it already gets this right; this
            # one just never matched it. 2026-08-23 recovered finding F-V.
            _full_text_md_parts = [f"# {title or 'Summary'}\n\n",
                                    f"**Source:** {source_url}\n"]
            if _original and _original != source_url:
                _full_text_md_parts.append(f"**Original:** {_original}\n")
            _full_text_md_parts.append(f"**Type:** {source_type}\n")
            _full_text_md_parts.append(f"**Created:** {created}\n\n")
            if highlights:
                _full_text_md_parts.append("## Highlights\n\n")
                _full_text_md_parts.append("\n".join(f"- {h}" for h in highlights) + "\n\n")
            _full_text_md_parts.append("## Full Summary\n\n")
            _full_text_md_parts.append(text + "\n")
            _full_text_md_content = "".join(_full_text_md_parts)
        else:
            md_path = None
            _full_text_md_content = None

        # Serialize to JSON
        for k, v in clean.items():
            if isinstance(v, (list, dict)):
                clean[k] = json.dumps(v)

        sets = ", ".join(f"{k} = ?" for k in clean)
        params = list(clean.values()) + [now, summary_uuid]
        self._conn.execute(
            f"UPDATE summaries SET {sets}, updated_at = ? WHERE uuid = ?", params
        )

        if _full_text_md_content is not None:
            try:
                with open(md_path, 'w', encoding='utf-8') as f:
                    f.write(_full_text_md_content)
            except Exception:
                # The UPDATE above is staged but not yet committed — same
                # reasoning as the highlights-only branch's rollback below.
                try:
                    self._conn.rollback()
                except sqlite3.Error:
                    pass
                raise

        # Update .md highlights if highlights changed (but full_text not)
        if "highlights" in clean and "full_text" not in clean:
            # summary_path is stored *relative* to _summaries_dir (see add()).
            # Using it raw made os.path.exists() resolve against the process
            # CWD, which is never the summaries directory — so this whole
            # block was dead and editing highlights left the on-disk .md
            # (the copy the user is told to read back) showing the old bullets.
            md_path = self._resolve_path(existing["summary_path"])
            if md_path and os.path.exists(md_path):
                try:
                    with open(md_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    new_highlights = clean["highlights"]
                    if isinstance(new_highlights, str):
                        new_highlights = json.loads(new_highlights)
                    highlights_section = "\n".join(f"- {h}" for h in new_highlights)
                    pattern = r"(## Highlights\n\n)(.*?)(\n\n## Full Summary)"
                    # A callable replacement, not an rf-string template.
                    # re.sub parses a *string* replacement for backreferences
                    # (backslash-g-angle-bracket-1, backslash-1 through
                    # backslash-9) — interpolating caller-supplied highlight
                    # text into that template meant a backslash in the
                    # highlight decided what the rewrite did, instead of
                    # being written verbatim. Measured both directions: a
                    # highlight containing a Windows path raised
                    # `re.error: bad escape` out of update(); one containing
                    # a backreference sequence raised nothing and silently
                    # duplicated the section header into the bullet on disk
                    # — success reported, the DB row correct, the .md the
                    # caller is told to read back wrong, with no way to
                    # detect it short of reading the file. A callable's
                    # return value is always inserted literally, so this
                    # closes both failure modes.
                    content = re.sub(
                        pattern,
                        lambda m: m.group(1) + highlights_section + m.group(3),
                        content, flags=re.DOTALL)
                    with open(md_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                except Exception:
                    # The UPDATE above is staged but not yet committed. This
                    # connection is thread-local and long-lived, and
                    # digests.db is one file shared by every profile — an
                    # uncommitted write left open here blocks every later
                    # writer on the shared file with "database is locked"
                    # until this thread happens to commit something else.
                    # Same reasoning as add()'s rollback on INSERT failure.
                    try:
                        self._conn.rollback()
                    except sqlite3.Error:
                        pass
                    raise

        self._conn.commit()
        return True

    def delete_multiple(self, uuids: List[str], profile_name: Optional[str] = None, profile: str = "own") -> dict:
        """Delete multiple summaries and their .md files.

        Args:
            uuids: List of UUIDs to delete.
            profile_name: Current profile name (used for profile-scoped delete).
            profile: If "own" (default), only delete own profile. If "all", delete any profile.

        Returns:
            dict with deleted, not_found, refused and invalid counts.
        """
        # An element that is not a string is `invalid`, not `not_found`.
        #
        # The container was validated on both front ends and its elements were
        # not, so `uuids=[123, "nope"]` reached SQLite as a bound integer,
        # compared unequal to the TEXT `uuid` column, and came back
        # `{"deleted": 0, "not_found": 2}` — a type error wearing the response
        # shape of two uuids that simply do not exist. An LLM that serialised
        # its uuids as numbers is told its rows are gone.
        #
        # Counted here rather than raising, because a batch call's contract is
        # per-element outcomes: one bad element must not refuse the good ones
        # beside it. Here rather than on the two front ends for the usual
        # reason — they share no code, and this is the single function both
        # cross. 2026-09-14 round 1 bundle05 (F3).
        #
        # delete()'s bool return collapses "no such uuid" and "uuid exists but
        # belongs to another profile" into the same False — a deliberate,
        # documented choice for that single-uuid door (its docstring says so).
        # Batch-deleting through it inherited the collapse and reported a
        # caller's own rows it wasn't allowed to touch identically to uuids
        # that were simply typos or already gone — a real distinction for
        # anyone auditing a batch delete or debugging why a count came up
        # short. Checked here instead, ahead of the call, rather than
        # widening delete()'s contract for its one other caller.
        # 2026-08-23 review round 7 maintenance F3.
        deleted = 0
        not_found = 0
        refused = 0
        invalid = 0
        for uuid in uuids:
            if not isinstance(uuid, str) or not uuid:
                invalid += 1
                logger.warning(
                    "summaries.delete_multiple: ignoring a non-string uuid %r "
                    "(%s) — a batch element must be a uuid string",
                    uuid, type(uuid).__name__)
                continue
            row = self._conn.execute(
                "SELECT profile_name FROM summaries WHERE uuid = ?", (uuid,)
            ).fetchone()
            if not row:
                not_found += 1
                continue
            if not self._row_profile_ok(row["profile_name"], profile_name, profile,
                                        "delete", uuid):
                refused += 1
                continue
            if self.delete(uuid, profile_name=profile_name, profile=profile):
                deleted += 1
            else:
                # Row vanished between the check above and delete()'s own —
                # a race, not a refusal or a pre-existing absence.
                not_found += 1
        return {"deleted": deleted, "not_found": not_found, "refused": refused,
                "invalid": invalid}

    def list_expiring(self, max_age_days: int = 30, profile_name: Optional[str] = None,
                      profile: str = "own") -> List[dict]:
        """List summaries older than max_age_days.

        Args:
            max_age_days: Age threshold in days.

        Returns:
            List of old summaries.
        """
        from datetime import datetime, timedelta, timezone
        # A negative max_age_days puts the cutoff in the future — harmless
        # here (created_at < a future cutoff still just matches every row,
        # not more), but "every row" is not what a caller asking for a small
        # or negative age threshold meant, and clamping matches this
        # function's own read-side pagination guard in list_summaries above.
        # 2026-08-23 review round 6 F7 / round 7 maintenance F10.
        try:
            max_age_days = max(0, int(max_age_days))
        except (TypeError, ValueError):
            max_age_days = 30
        cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        # Scoped like get and search. This one is the retention view an agent
        # is told to act on, so leaking another profile's rows into it invites
        # a delete of records the caller never owned.
        scope = self._profile_sql(profile_name, profile, "list_expiring")
        if scope is None:
            return []
        scope_sql, scope_params = scope
        rows = self._conn.execute(
            f"SELECT * FROM summaries WHERE created_at < ?{scope_sql} "
            "ORDER BY created_at ASC",
            (cutoff.isoformat(), *scope_params)
        ).fetchall()

        results = []
        for row in rows:
            r = dict(row)
            r["highlights"] = json.loads(r.get("highlights") or "[]")
            r["tags"] = json.loads(r.get("tags") or "[]")
            # See list_summaries: unparsed metadata is double-encoded by the
            # plugin's fence.
            r["metadata"] = json.loads(r.get("metadata") or "{}")
            r.pop("full_text", None)
            results.append(r)
        return results

    def close(self):
        """Close this thread's database connection.

        Connections opened by other threads are released when those threads
        end and their thread-local storage is collected.
        """
        conn = getattr(self._conn_local, "conn", None)
        if conn is not None:
            try:
                conn.close()
            finally:
                self._conn_local.conn = None

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
        else:
            # Same containment its memories-DB twin applies (backend/store.py
            # backup()). dest_dir is caller-controlled, and without this the
            # call creates directories and writes a database copy anywhere the
            # process user can write. The summaries DB holds full page text, so
            # the leak is of content, not just structure. The default location
            # is always permitted so the no-argument case works wherever the DB
            # lives. Reported by the 2026-08-22 ox-alpha maintenance review
            # (F5), which found it by diffing against the twin.
            from backend.maintenance import _allowed_fs_roots, _path_within_roots
            roots = _allowed_fs_roots("HLM_BACKUP_ALLOWED_ROOTS")
            roots.append(os.path.realpath(os.path.dirname(self._db_path)))
            if not _path_within_roots(dest_dir, roots):
                raise ValueError(
                    f"Backup destination {dest_dir!r} is outside allowed roots "
                    f"{roots}. Set HLM_BACKUP_ALLOWED_ROOTS (colon-separated) "
                    f"to allow additional locations.")
        os.makedirs(dest_dir, exist_ok=True)

        now = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        base = os.path.basename(self._db_path)
        name, ext = os.path.splitext(base)
        backup_name = f"{name}.backup.{now}{ext}"
        backup_path = os.path.join(dest_dir, backup_name)

        # Atomic backup via SQLite API (WAL-safe).
        #
        # try/finally: a failing self._conn.backup() used to leak the
        # destination connection, leaving a half-written file open for the life
        # of the process — and on Windows, unremovable.
        backup_conn = sqlite3.connect(backup_path)
        try:
            self._conn.backup(backup_conn)
        finally:
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
        # Fail closed on a scope word this function does not know. `sync` tests
        # `profile == "own"`, so anything else — a typo, "Own", "" — fell into
        # the unscoped sweep, inferring the cross-profile escape hatch that
        # docs/security.md says "must be said, not inferred". The sibling
        # helpers `_profile_sql` and `_row_profile_ok` test `== "all"` and so
        # already fail closed; this one tested the other end of the pair.
        #
        # The 2026-08-22 ox-alpha maintenance review filed this as a Critical
        # cross-profile deletion. Measured against the tree it reviewed, it is
        # not: 0.7.76's `skipped_foreign` guard in the loop below already keeps
        # foreign rows, and a junk scope only widened the *scan*. What was
        # genuinely wrong is this — a scoping decision taken from an
        # unvalidated string.
        if not _valid_profile_scope(profile):
            logger.warning(
                "summaries.sync: refusing unknown profile scope %r — use 'own' "
                "or 'all'", profile)
            return {"orphaned": 0, "preserved": 0, "skipped_foreign": 0,
                    "total_checked": 0,
                    "error": f"profile must be 'own' or 'all', got {profile!r}"}
        if profile == "own":
            if not profile_name:
                # Refuse rather than silently sweeping globally. The guard
                # used to be `if profile == "own" and profile_name:`, so a
                # caller that omitted profile_name — as on_session_end did —
                # fell through to no filter at all and hard-deleted rows
                # across every profile in the shared digests.db. Because
                # summary_path is relative and resolved against the *current*
                # profile's directory, a profile with its own
                # HLM_SUMMARIES_DIR resolved everyone else's paths into its
                # own tree, found nothing, and deleted their records —
                # full_text included, which is the only remaining copy.
                logger.warning(
                    "summaries: sync refused (profile='own' but profile_name not provided) — "
                    "pass profile_name, or profile='all' to sweep every profile deliberately")
                return {"orphaned": 0, "total_checked": 0, "skipped": "no profile_name"}
            # Exact match, not `OR profile_name IS NULL`. The NULL branch made
            # every pre-v5 row owned by every profile, so an "own"-scoped sweep
            # hard-deleted legacy rows another profile created — and this
            # function's whole reason for existing is that it deletes
            # `full_text`, the only remaining copy. Legacy rows now need an
            # explicit profile="all", matching _row_profile_ok.
            where += " AND profile_name = ?"
            params = [profile_name]

        orphaned = 0
        rows = self._conn.execute(
            f"SELECT uuid, summary_path, full_text, profile_name FROM summaries "
            f"WHERE {where}", params
        ).fetchall()

        preserved = 0
        skipped_foreign = 0
        for row in rows:
            # A foreign row's `summary_path` is relative to *its* profile's
            # summaries directory, and `_resolve_path` joins against this
            # instance's. Under profile="all" that made every foreign row look
            # like its .md had been deleted: the row was then hard-deleted (or,
            # with full_text, had its path cleared), destroying a perfectly
            # valid record on the strength of a path this instance cannot
            # resolve. `delete()` was taught to refuse exactly this and says so
            # in its own comment — sync() is the same hazard through the other
            # door, and it is the more dangerous one because it runs
            # unattended.
            row_profile = row["profile_name"] if "profile_name" in row.keys() else None
            # `not profile_name` counts as foreign, not as a free pass. The
            # guard used to require `profile_name` to be truthy, so an
            # "all"-scoped sweep called without one — which is exactly the
            # shape that reaches the unscoped branch — skipped the check for
            # every row and swept the whole shared digests.db. Without a
            # caller identity there is no row we can prove is ours, so the
            # only safe reading of an owned row is "someone else's".
            # 2026-08-22 ox-alpha maintenance review (F4), on the guard added
            # in 0.7.76.
            # A row whose `profile_name` is NULL is *unowned*, not ours. The
            # 0.7.81 guard read `if row_profile and ...`, so a falsy
            # profile_name skipped the ownership check altogether and the row
            # was swept — and legacy rows predating the column are precisely
            # the ones carrying NULL. Two holes in one condition: 0.7.81 closed
            # the `profile_name` half and left this one.
            #
            # Skipped under **every** scope, "all" included. The reason foreign
            # rows are skipped is that `summary_path` is relative to the owning
            # profile's directory and `_resolve_path` joins against ours — for a
            # row with no profile_name that directory is not merely different,
            # it is unknown, so the missing-file test cannot mean anything. The
            # deliberate "all" escape hatch is about reaching *another named
            # profile's* rows, not about deleting unattributable ones.
            #
            # The first attempt at this fix guarded only when `profile != "all"`
            # — which is the one path where the SQL WHERE already excludes NULL
            # rows, so it fired nowhere and would have read as protection.
            # 2026-08-23 profile-a maintenance review (F1).
            if not row_profile:
                skipped_foreign += 1
                logger.debug(
                    "summaries.sync: skipped %s — no profile_name, so its "
                    "summary_path cannot be resolved against any known "
                    "directory", row["uuid"])
                continue
            if row_profile and row_profile != profile_name:
                skipped_foreign += 1
                logger.debug(
                    "summaries.sync: skipped %s (profile %s) — its summary_path "
                    "resolves against that profile's directory, not ours",
                    row["uuid"], row_profile)
                continue
            md_path = self._resolve_path(row["summary_path"])
            if os.path.exists(md_path):
                continue
            # Never destroy the last copy. full_text lives in the DB, so a
            # row whose .md is missing may still hold the entire summary —
            # and os.path.exists() also returns False for an unmounted
            # volume or an EACCES, neither of which means "user deleted it".
            keeps_text = bool(row["full_text"]) if "full_text" in row.keys() else False
            if keeps_text:
                self._conn.execute(
                    "UPDATE summaries SET summary_path = NULL WHERE uuid = ?", (row["uuid"],))
                preserved += 1
                logger.warning(
                    "summaries: %s lost its .md file (%s) but still holds full_text — "
                    "cleared the path, kept the record", row["uuid"], md_path)
                continue
            self._conn.execute("DELETE FROM summaries WHERE uuid = ?", (row["uuid"],))
            orphaned += 1
            logger.warning("summaries: deleted orphaned record %s (md file missing: %s)",
                           row["uuid"], md_path)

        self._conn.commit()
        return {"orphaned": orphaned, "preserved": preserved,
                "skipped_foreign": skipped_foreign, "total_checked": len(rows)}
