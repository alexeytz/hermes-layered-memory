"""HLM shared primitives — logging, settings, embeddings, encodings.

Everything in this module is used by *both* backend/backend.py and the method
modules (store/index/pipeline/llm/maintenance). It deliberately imports none of
them, which is what makes the dependency graph a DAG and lets the method
modules import these names directly.

Before this split they could not: backend/backend.py imported the method
modules at the bottom of the file, so the method modules could not import it
back at module level. They reached these names through a
`_get_backend_name("logger")` lookup in sys.modules instead — 261 call sites,
201 of them just fetching the logger, none of them navigable by an editor or
checkable by a type checker.
"""
from __future__ import annotations
import threading
import array
from functools import wraps
import collections
import json
import logging
import os
import pwd
import re
import sqlite3
import sys as _sys
import time
import uuid as uuid_mod
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Union

# Retry wrapper for SQLite write operations (handles concurrent Hermes sessions)
def _is_lock_error(e: Exception) -> bool:
    """True only for a genuine SQLite busy/locked condition.

    The old test was `"locked" in str(e) or "lock" in str(e)`. The second
    clause is a substring match that also fires on "blocked", "Blocked by
    policy", "deadlock" and "clock skew" — so a permanent, non-retryable
    error (an LLM content-filter rejection, say) burned the full retry
    budget: 50 attempts with backoff capped at 5s is ~4 minutes of blocking
    on a call that could never succeed.

    SQLite reports this condition as OperationalError with "database is
    locked" / "database table is locked", so match the type and the word
    "locked" specifically. sqlite3.OperationalError covers the DB-API
    surface; the string check alone remains as a fallback for wrappers that
    re-raise a plain Exception carrying the same message.
    """
    msg = str(e).lower()
    if isinstance(e, sqlite3.OperationalError):
        # SQLite's exact error messages: "database is locked", "database
        # table X is locked", "database is busy". Match those specifically
        # to avoid "clock"/"clocked"/"blocked"/"deadlock"/"blocked by"
        # false positives that would retry permanent errors.
        return ("database is locked" in msg or "database table is locked" in msg
                or "database is busy" in msg)
    return "database is locked" in msg or "database table is locked" in msg


def _is_process_poisoned_db_error(e: Exception) -> bool:
    """True for a SQLite failure that is permanent for the life of THIS
    process — retrying, or even closing and reopening the connection, will
    keep failing identically, unlike `_is_lock_error`'s condition.

    Root-caused 2026-08-17 against a real E2E run: a long-lived Hermes
    session's `-wal`/`-shm` sidecar files were deleted while its connection
    was open (the deleter was never pinned — ruled out purge/VACUUM by log
    inspection, ruled out cron/systemd/dmesg — but the trigger doesn't
    matter for this check). Every subsequent `layered_memory` call in that
    process then failed with "database disk image is malformed", and it
    never recovered.

    Reproduced directly: open connection A (WAL mode, keep it open), delete
    the sidecars externally, then in the SAME process — closing A first
    makes no difference — every fresh `sqlite3.connect()` fails identically
    forever, including with `journal_mode=DELETE` (so it is not WAL-specific
    either). A brand-new *process* opens the same file immediately, and
    `PRAGMA integrity_check` from one confirms the file itself is fine: this
    is SQLite's per-process VFS bookkeeping (the shared-memory node it caches
    per inode) getting out of sync with what's on disk, not real corruption.

    So a caught instance of this is not "try again" — it is "stop trying
    against this file, in this process, permanently." SQLite reports it as
    both `sqlite3.DatabaseError` ("database disk image is malformed") and
    `sqlite3.OperationalError` ("disk I/O error"), observed from the exact
    same reproduction depending on timing, so both are matched.
    """
    msg = str(e).lower()
    if not isinstance(e, (sqlite3.DatabaseError, sqlite3.OperationalError)):
        return False
    return "disk image is malformed" in msg or "disk i/o error" in msg


def _retry_on_lock(func):
    """Retry func on 'database is locked' with exponential backoff.

    Used by delete() and feedback() which are simple operations.
    add() and update() use inline retry (see backend/store.py) to avoid
    retrying expensive embedding calls.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        for attempt in range(20):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt < 19 and _is_lock_error(e):
                    time.sleep(min(0.2 * (2 ** attempt), 5))
                    continue
                raise
    return wrapper

logger = logging.getLogger("hermes-layered-memory")

# StreamHandler that re-resolves sys.stderr on each emit.
# Hermes CLI's patch_stdout() replaces sys.stderr with a StdoutProxy
# during app.run(). When patch_stdout() exits on shutdown, the proxy is
# torn down — a standard StreamHandler holding a stale reference writes
# to a dead object. This subclass re-reads sys.stderr every time.
class _LiveStderrHandler(logging.StreamHandler):
    @property
    def stream(self):
        return _sys.stderr
    @stream.setter
    def stream(self, value):
        pass  # ignored — always live

# File-based logging — writes to profile logs dir
_log_initialized = False

#: Marker on the handlers this module installs. The duplicate guard has to
#: distinguish "we already set this up" from "somebody else attached a handler
#: to our logger", because those need opposite answers and the old check
#: (`if logger.handlers`) could not tell them apart.
_HLM_HANDLER_FLAG = "_hlm_owned"


def _hlm_handlers_attached() -> bool:
    return any(getattr(h, _HLM_HANDLER_FLAG, False) for h in logger.handlers)


def _setup_logger():
    """Install HLM's own file + stderr handlers, exactly once.

    The guard used to be `if logger.handlers: return`, meaning: if *anything*
    had already attached a handler to this logger, bail. Under `hermes -z`
    (one-shot) the CLI configures logging before the plugin imports, so that
    branch was taken and HLM never installed its file handler, never called
    setLevel(), and left propagate=True. Its records then fell through to the
    host's root handler and were filtered out at WARNING — so every INFO and
    DEBUG line vanished. Verified: a one-shot `add` wrote the row to SQLite and
    produced no log line in the profile log, the root log, or agent.log.

    That silently breaks the E2E log gate, which reads
    `profiles/<profile>/logs/hermes-layered-memory.log`: a scripted run leaves
    that file untouched, so `check-e2e-log.py --last-run` scans an *older* run
    and can report a pass for a run it never saw.

    Now the guard keys on our own marker, so a foreign handler no longer stops
    HLM owning its log. `propagate = False` is kept, matching what interactive
    mode has always done — the HLM log file is the documented source of truth
    for the gate, and duplicating every line into the host log is not wanted.
    Handlers the host attached directly to this logger still fire; propagate
    only governs ancestor loggers.
    """
    global _log_initialized
    if _log_initialized and _hlm_handlers_attached():
        return
    if _hlm_handlers_attached():
        # Re-imported into a fresh module object (plugin reload) but the logger
        # object is process-global and already carries our handlers.
        _log_initialized = True
        return
    _log_initialized = True

    # Resolve log path — env var overrides default
    log_path = os.environ.get("HLM_LOG_FILE")
    if log_path:
        real_home = _C.real_home()
        if log_path.startswith("~"):
            log_path = log_path.replace("~", real_home, 1)
        if "$" in log_path:
            log_path = os.path.expandvars(log_path)
        log_file = log_path
        dirname = os.path.dirname(log_file)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
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
    setattr(fh, _HLM_HANDLER_FLAG, True)

    # Console handler — also keep stderr for debugging
    ch = _LiveStderrHandler()
    ch.setFormatter(logging.Formatter("[h-l-m @ %(asctime)s] %(levelname)s %(message)s",
                                       datefmt="%H:%M:%S"))
    setattr(ch, _HLM_HANDLER_FLAG, True)

    # Set level from env or default to INFO
    _env_level = os.environ.get("HLM_LOG", "INFO").upper()
    level = getattr(logging, _env_level, logging.INFO)

    logger.setLevel(level)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.propagate = False

# Auto-setup on first use
# _C must be imported before _setup_logger() because real_home() is needed
# when HLM_LOG_FILE is set (conftest.py sets it before importing backend).
from . import constants as _C

# ── Module-level aliases (backward compat for external imports) ─────────
_SELF_AUTHORED_SOURCES = _C.SELF_AUTHORED_SOURCES
_UNTRUSTED_OPEN = _C.UNTRUSTED_OPEN
_UNTRUSTED_CLOSE = _C.UNTRUSTED_CLOSE
_EMBED_NULL = _C.EMBED_NULL
MAX_CONTENT_CHARS = _C.MAX_CONTENT_CHARS
MAX_METADATA_CHARS = _C.MAX_METADATA_CHARS
MAX_FIELD_CHARS = _C.MAX_FIELD_CHARS
MAX_DATA_ID_CHARS = _C.MAX_DATA_ID_CHARS
MAX_SESSION_NAME_CHARS = _C.MAX_SESSION_NAME_CHARS
MAX_SCOPE_CHARS = _C.MAX_SCOPE_CHARS
MAX_SOURCE_URL_CHARS = _C.MAX_SOURCE_URL_CHARS
MAX_KEYWORDS = _C.MAX_KEYWORDS

_setup_logger()

# UUID format conversion (SQLite stores 32-char hex, Qdrant wants standard UUID)

def profile_claim(qdrant, collection, point_id, own):
    """The `profile_name` payload for a point we are about to write.

    A Qdrant point's id **is** the record uuid, and `import` preserves uuids, so
    two profiles holding the same record share **one point** — which, with a
    scalar `profile_name`, can name only one owner. Measured on the dev box:
    21 records imported from `hlm-test` into `profile-a` in two batches left
    `hlm-test` with 24 active rows and 3 points it could see. Each profile's
    `initialize` then found itself short, rebuilt, relabelled the shared points
    to itself, and handed the other the same condition: 443 full-corpus rebuilds
    across the two logs, each correct alone and each undoing the other.

    So a point records **every** profile that holds the record. Qdrant's scalar
    `MatchValue` already matches a value inside an array, verified against a
    live instance, so every read filter — `_layer0`'s, rebuild's orphan scroll,
    `_detect_orphans` — keeps working unchanged: "my name is in the list" reads
    as mine, which is exactly the ownership question they mean to ask.

    Returns a bare string while only one profile holds the record, so the common
    case writes the payload it always wrote and existing points need no
    migration; a list appears only where a record genuinely is shared.
    2026-08-27, option 3 of the cross-profile uuid collision.
    """
    own = own or "default"
    existing = []
    if qdrant is not None and collection and point_id is not None:
        try:
            pts = qdrant.retrieve(collection_name=collection, ids=[point_id],
                                  with_payload=True, with_vectors=False)
            if pts:
                cur = (pts[0].payload or {}).get("profile_name")
                if isinstance(cur, list):
                    existing = [p for p in cur if p]
                elif cur:
                    existing = [cur]
        except Exception as e:
            # A read failure must not silently drop another profile's claim, but
            # it must not block the write either: log and fall back to our own
            # name, which is what the code did before this existed.
            logger.debug("profile_claim: could not read existing owners of %s "
                         "in %s (%s) — claiming for %s only", point_id, collection, e, own)
    names = sorted(set(existing) | {own})
    return names[0] if len(names) == 1 else names


def profile_release(qdrant, collection, point_id, own):
    """Drop `own`'s claim on a point. Returns True if the point is now unowned.

    Deleting a record must not delete a point another profile still holds. That
    is not hypothetical: sweeping 96 junk rows out of `hlm-test` would have
    stripped 13 vectors from `profile-a`, because `delete()` retires the point
    by uuid and the uuid is shared. The sweep held those 13 back by hand; this
    is that judgement made mechanical.

    True means "no other profile claims it, delete the point". False means the
    claim was removed and the point must stay.
    2026-08-27, option 3 of the cross-profile uuid collision.
    """
    own = own or "default"
    if qdrant is None or not collection or point_id is None:
        return True
    try:
        pts = qdrant.retrieve(collection_name=collection, ids=[point_id],
                              with_payload=True, with_vectors=False)
    except Exception as e:
        logger.debug("profile_release: could not read %s in %s (%s) — deleting",
                     point_id, collection, e)
        return True
    if not pts:
        return True
    cur = (pts[0].payload or {}).get("profile_name")
    owners = [p for p in cur if p] if isinstance(cur, list) else ([cur] if cur else [])
    remaining = [p for p in owners if p != own]
    if not remaining:
        return True
    try:
        qdrant.set_payload(collection_name=collection,
                           payload={"profile_name": remaining[0] if len(remaining) == 1
                                    else sorted(remaining)},
                           points=[point_id])
        logger.info("point %s kept: still held by %s", point_id, ", ".join(sorted(remaining)))
    except Exception as e:
        # Could not rewrite the claim. Keeping the point is the safe failure:
        # a stale extra owner costs a duplicate in that profile's results, while
        # deleting costs another profile its vector with no way to notice.
        logger.warning("profile_release: could not rewrite owners of %s in %s: %s",
                       point_id, collection, e)
    return False


def orphan_sweep_refused(conn, profile_name, seen: int, orphaned: int,
                         where: str, db_path: str = None,
                         overlap: int = None) -> bool:
    """True when an orphan sweep must not run because we may not be authoritative.

    Both sweep sites compute the same thing — "every point our SQLite does not
    know about" — and both delete the result. That is correct when our SQLite
    really is this profile's database, and catastrophic when it is not: a
    backend opened over the wrong file knows *none* of the uuids, so every point
    looks orphaned and the whole profile's index is deleted.

    That is not hypothetical. It happened twice in two days, and the mechanism
    was the same both times: `_apply_env_overrides()` lets `HLM_DB_PATH` beat an
    explicitly passed `db_path`, so a script that carefully passed the right
    path still opened the wrong database while carrying the right profile name.
    44 points on 2026-08-25, ~90 on 2026-08-26. `_detect_orphans`' own docstring
    records a third: "saw 83 candidates, declared 78 orphaned, removed them".

    The existing guards compare profile *names*, and a name match is exactly
    what a wrong-database backend still has. This asks a question a name cannot
    answer: **is our SQLite plausibly the authority for these points?**

    The signature of the wrong database is a *total* mismatch — we would delete
    everything we saw — *and* a database too small to be that collection's
    authority. Emptiness is the extreme case, and it was the only case checked
    until 0.8.40: on 2026-08-28 (cycle 21) a two-row shadow database passed the
    `own > 0` waiver and cost a live profile all 30 of its points, five times in
    one run. Holding *some* rows is not authority; holding fewer rows than the
    points you are about to delete, while sharing none of them, is the same
    signal emptiness was.

    `own < seen` is only a heuristic, and 0.8.42 records where it fails: a
    backend carrying profile B's *name* over profile A's **live** database has
    plenty of rows — more than the points it is about to delete — so it clears
    every check above and sweeps B's entire index. Reported as F13 by the
    2026-08-28 external round and driven at own=500/seen=30.

    `overlap` is the exact answer the heuristics approximate: **how many points
    in this collection does this database actually claim?** Zero, when the
    collection has points for us, means we are not the authority for them,
    whatever our row count says. Callers that can compute it pass it and the
    heuristic is skipped; callers that cannot pass nothing and keep the old
    behaviour. It is deliberately not computed here — this function takes a
    SQLite connection, and the answer lives in Qdrant.

    The overlap must be counted *under our own profile claim*, not merely by
    uuid: shared collections hold several profiles' points, so "this uuid is
    present" is true for the wrong database too. "This uuid is present and
    carries our profile name" is not.
    Genuine orphans are a minority of a healthy collection. Refusing on that
    signature costs a real but empty profile its stranded points until someone
    looks; `sync_check` reports the drift and says so. Deleting a live profile's
    vectors costs a rebuild and an hour of confusion, which is the trade this
    picks, and it is the direction the rest of this codebase already prefers.
    """
    if orphaned <= 0 or orphaned < seen:
        return False
    # No `profile_name` filter, and fail CLOSED.
    #
    # The first version of this guard asked for
    # `... AND (profile_name IS NULL OR profile_name = ?)` and wrapped the query
    # in `except Exception: return False`. Both were wrong, and together they
    # made the guard a no-op against the very case it exists for: a database old
    # enough to predate the `profile_name` column raised `no such column`, the
    # except swallowed it, and the sweep proceeded to delete everything. A
    # safety check that cannot establish safety must refuse, not permit — the
    # whole point is that we do not know whose data this is.
    #
    # The count is deliberately unfiltered. If this database holds no active
    # rows AT ALL, it cannot be the authority for a collection that has points,
    # whatever the profile column says or whether it exists.
    try:
        own = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE status='active'").fetchone()[0]
    except Exception as e:
        logger.warning(
            "orphan sweep REFUSED in %s: could not count this database's active "
            "rows (%s: %s), so authority for %d point(s) cannot be established. "
            "Refusing is the safe direction — a sweep that cannot prove "
            "ownership must not delete.", where, type(e).__name__, e, seen)
        return True
    # A total mismatch is refused whether or not this database holds rows.
    #
    # Until 0.8.40 there was an `if own: return False` here: a non-empty
    # database was taken as proof of authority. It is not. On 2026-08-28
    # (cycle 21) a grading shell running under the *driver* profile built a
    # backend with an explicit db_path to hlm-test's real database;
    # `_apply_env_overrides` resolved `HLM_DB_PATH` through the driver
    # profile's secret scope to a 2-row shadow file and overrode the argument.
    # `own` was 2, not 0 — so this returned False, and every one of hlm-test's
    # 30 points was deleted. Five times in one run.
    #
    # Zero overlap is the signal, and `own` does not change what it means: none
    # of the points we examined for this profile is known to the database we are
    # holding. Either it is the wrong database, or SQLite and Qdrant share no
    # uuids at all — and the answer to both is a rebuild, not a delete. Genuine
    # orphans are a minority of a healthy collection; that is what makes
    # `orphaned >= seen` the wrong-database signature rather than ordinary drift.
    #
    # This does not strand the points. rebuild() re-upserts every active row
    # *before* its cleanup scrolls, so overlap is restored and the genuinely
    # stale minority sweeps normally on the next pass.
    if overlap is not None:
        if overlap > 0:
            return False
        logger.warning(
            "orphan sweep REFUSED in %s: every one of the %d point(s) for "
            "profile %r looked orphaned, and this database claims none of the "
            "points in that collection (db=%s, %d active row(s)). Whatever it "
            "holds, it is not the authority for these points — that is a "
            "backend over the wrong file, or an index to rebuild.",
            where, seen, profile_name, db_path or "<unknown>", own)
        return True
    if not own:
        # The original signature (0.8.20), kept on a reachable path and with its
        # own wording: "no rows at all" and "fewer rows than points" are
        # different diagnoses and the operator acts on them differently. When
        # the `overlap` branch above was added this message ended up after a
        # `return True` and became unreachable, so an empty database was
        # reported as holding "only 0 active row(s)" by the message below.
        # Reported by the 0.8.42 re-validation as dead code; it was that, and
        # also a lost diagnosis.
        logger.warning(
            "orphan sweep REFUSED in %s: every one of the %d point(s) for profile "
            "%r looked orphaned, and this database holds no active rows for it. "
            "That is the signature of a backend opened over the wrong file (db=%s) "
            "— deleting them would strip a live profile's index. If this profile "
            "really is empty, its points are stranded and rebuild will clear them "
            "once it has rows to compare against.",
            where, seen, profile_name, db_path or "<unknown>")
        return True
    if own >= seen:
        # We hold at least as many active rows as points we examined. Zero
        # overlap here is an index that needs rebuilding, not evidence that we
        # are reading the wrong database — and blocking it would disable
        # ordinary orphan cleanup on small candidate batches, which is what
        # `_detect_orphans` passes (a query-scoped subset, where "all of this
        # batch is stale" is unremarkable).
        return False
    logger.warning(
        "orphan sweep REFUSED in %s: every one of the %d point(s) for profile "
        "%r looked orphaned, and this database holds only %d active row(s) — "
        "fewer rows than points we are about to delete, sharing none of them "
        "(db=%s). A database that is the authority for a collection does not "
        "hold an order of magnitude fewer records than the collection has "
        "points for it. Refusing; rebuild is the repair.",
        where, seen, profile_name, own, db_path or "<unknown>")
    return True


def claimed_overlap(qdrant, collection, uuids, own_profile, limit: int = 64) -> int:
    """How many of `uuids` exist in `collection` carrying `own_profile`'s claim.

    The question `orphan_sweep_refused` needs and cannot ask: is this database
    the authority for the points it is about to delete? Presence by uuid is not
    enough — collections are shared, so a backend holding profile A's database
    while carrying profile B's name finds A's points present and concludes it
    belongs. The claim is what distinguishes them.

    Bounded to `limit` uuids: this runs only on the total-mismatch path, and a
    single hit is all the answer requires. Fails CLOSED — an error counting is
    reported as zero overlap, which refuses the sweep, because a check that
    cannot establish authority must not grant it.
    """
    if not qdrant or not uuids or not own_profile:
        return 0
    sample = [_to_qdrant_id(u) for u in list(uuids)[:limit]]
    try:
        points = qdrant.retrieve(collection_name=collection, ids=sample,
                                 with_payload=True, with_vectors=False)
    except Exception as e:
        logger.debug("claimed_overlap: could not read %s (%s) — reporting 0, "
                     "which refuses the sweep", collection, e)
        return 0
    hits = 0
    for pt in points or []:
        claim = (getattr(pt, "payload", None) or {}).get("profile_name")
        if claim == own_profile or (isinstance(claim, (list, tuple))
                                    and own_profile in claim):
            hits += 1
    return hits


def _to_qdrant_id(uuid_hex: str) -> str:
    """Convert a UUID to the canonical dashed form Qdrant accepts as a point id.

    **Idempotent**, which the slicing version was not. It assumed 32-char hex
    and sliced blindly, so an already-dashed 36-char uuid came out as
    `b2a86b04--23d-3-46-fc-9-c82-6d2a0e7bb8d5` — right hex, dashes in the wrong
    places. Python's `uuid.UUID()` accepts that (it is lenient about
    separators); **Qdrant does not**, and rejects it with
    `400 ... data did not match any variant of untagged enum PointsSelector`.

    Point deletion is batched, so one mangled id takes the whole batch with it:
    `_drop_points_everywhere` logged "could not remove 1 point(s)" for each
    collection and "sweep removed 0 of 1 record(s)", and every *other* record in
    that batch kept its point as an orphan. Observed 2026-09-05 sweeping a
    profile that held one dashed uuid; the shared summaries database holds 98.

    Same class as `_physical_collection()`, which AGENTS.md already requires to
    stay idempotent for the same reason — a transform applied twice.
    """
    if not uuid_hex:
        return uuid_hex
    hexpart = uuid_hex.replace("-", "")
    if len(hexpart) != 32:
        # Not a uuid this function can canonicalise. Hand it back untouched
        # rather than emit a shape Qdrant will reject: the caller's own
        # validation should decide, and a silent mangle is what this fixes.
        return uuid_hex
    return (f"{hexpart[:8]}-{hexpart[8:12]}-{hexpart[12:16]}-"
            f"{hexpart[16:20]}-{hexpart[20:]}")

def _from_qdrant_id(qid) -> str:
    """Convert Qdrant ID (str or int) back to 32-char hex UUID."""
    if isinstance(qid, int):
        return format(qid, '032x')[:32]
    return str(qid).replace('-', '')

# Embedding model — lazy import, cached per resolved config (see
# _get_embedding_fn for why this isn't a single global).
_embedding_fn_cache: "collections.OrderedDict" = collections.OrderedDict()
_embedding_fn_cache_lock = threading.Lock()
# Max distinct embedding configs to keep in memory. Local models (FastEmbed,
# sentence-transformers) are large (200-300MB each) — an unbounded cache
# under a multiplex gateway with many profiles rotating through would leak
# memory. Cap and evict the least recently used.
# Clamped to >= 1 and parsed defensively: this value gates a
# `while len(cache) >= MAXSIZE: cache.popitem()` loop, so 0 or a negative
# number made popitem() raise KeyError off an empty OrderedDict on the
# add()/retrieve() hot path, and a non-numeric value raised ValueError at
# import time — both from a tuning knob that should never be able to break
# retrieval.
def _embed_fn_cache_maxsize() -> int:
    raw = os.environ.get("HLM_EMBED_FN_CACHE_MAXSIZE", "8")
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        logger.warning("invalid HLM_EMBED_FN_CACHE_MAXSIZE=%r, using 8", raw)
        return 8


_EMBED_FN_CACHE_MAXSIZE = _embed_fn_cache_maxsize()
# Test-only fault-injection seam: if set (not None), _get_embedding_fn
# returns this directly instead of consulting the cache, regardless of
# resolved config. Tests monkeypatch backend.backend._embedding_fn to
# inject a dead/flaky embedder; production code never sets this.
_embedding_fn = None

# ── Profile-scoped settings ─────────────────────────────────────────────────
# Under a multiplex gateway (gateway.multiplex_profiles), one process serves
# many profiles and the active profile's .env is loaded into an isolated
# mapping rather than os.environ — see docs/profile-isolation.md. A direct
# os.environ read there silently returns the root-level value or nothing, so
# every HLM_* tuning var reverts to its default without a word.
# Tests install a fake scope by assigning to _secret_getter.
_secret_getter = None   # None = not resolved yet, False = unavailable, else callable


def _get_secret_getter():
    """Resolve agent.secret_scope.get_secret once, if it exists."""
    global _secret_getter
    if _secret_getter is None:
        try:
            from agent.secret_scope import get_secret as _gs
            _secret_getter = _gs
            logger.debug("secret scope available — profile vars will be read through it")
        except Exception:
            _secret_getter = False
    return _secret_getter


def _setting(name: str, default: Optional[str] = None) -> Optional[str]:
    """Read a profile setting: secret scope first, then os.environ.

    An unscoped-secret error means "not set for this profile", not a failure —
    multiplex raises rather than leaking the root value, so it is treated as
    absent and os.environ is consulted as the non-multiplex fallback.
    """
    getter = _get_secret_getter()
    if getter:
        try:
            value = getter(name)
            if value not in (None, ""):
                return value
        except Exception:
            pass
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def _pack_embedding(vec: Optional[List[float]]):
    """Serialize an embedding for SQLite as packed float32.

    A 4096-dim vector is ~16KB packed against ~40-80KB as JSON text, and
    unpacking is a memcpy rather than a parse. Legacy JSON rows are still
    readable — see _unpack_embedding — so no migration is required.
    """
    if not vec:
        return _EMBED_NULL
    return sqlite3.Binary(array.array("f", vec).tobytes())


def _unpack_embedding(blob, expected_dim: int = None) -> Optional[List[float]]:
    """Read an embedding written in either format.

    Rows written before float32 packing hold a JSON string; rows written
    since hold packed bytes. Both are read transparently.

    If `expected_dim` is provided and the vector dimension doesn't match,
    a WARNING is logged — this is how the operator knows that records were
    stored with a different embedding model and need re-embedding via
    rebuild(). Without this, retrieval silently degrades (garbage cosine
    scores, wrong nearest neighbours) with no diagnostic trail.
    """
    if blob is None:
        return None
    if isinstance(blob, list):
        if expected_dim is not None and len(blob) != expected_dim:
            logger.warning("_unpack_embedding: dimension mismatch %d vs expected %d — "
                           "records were embedded with a different model; run "
                           "layered_maintenance(action='rebuild') to fix",
                           len(blob), expected_dim)
        return blob
    if isinstance(blob, (bytes, bytearray, memoryview)):
        raw = bytes(blob)
        if not raw or raw == b"null":
            return None
        # A legacy JSON row can surface as bytes depending on the driver.
        # Only attempt JSON parse if it looks like valid JSON — if it fails,
        # fall through to binary unpacking (a coincidental first byte of '['
        # or 'n' in float32 data is possible and must not cause a silent drop).
        result = None
        if raw[:1] in (b"[", b"n"):
            try:
                result = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                pass  # Not JSON, try binary below
        if result is None:
            arr = array.array("f")
            arr.frombytes(raw)
            result = arr.tolist()
        if expected_dim is not None and len(result) != expected_dim:
            logger.warning("_unpack_embedding: dimension mismatch %d vs expected %d — "
                           "records were embedded with a different model; run "
                           "layered_maintenance(action='rebuild') to fix",
                           len(result), expected_dim)
        return result
    if isinstance(blob, str):
        if not blob or blob == _EMBED_NULL:
            return None
        try:
            result = json.loads(blob)
        except json.JSONDecodeError:
            return None
        if expected_dim is not None and len(result) != expected_dim:
            logger.warning("_unpack_embedding: dimension mismatch %d vs expected %d — "
                           "records were embedded with a different model; run "
                           "layered_maintenance(action='rebuild') to fix",
                           len(result), expected_dim)
        return result
    return None


def _embedding_identity(embedding_model: Optional[str] = None) -> str:
    """Name the embedding model that _get_embedding_fn would actually pick.

    Mirrors that function's tier order exactly — remote endpoint, then local
    FastEmbed, then sentence-transformers — because the two must agree or a
    collection gets named for a model that never produced its vectors.

    Vectors from different models are not comparable, so this is the identity
    that collections are keyed on (see _collection_suffix).
    """
    embed_url = _setting("HLM_EMBED_URL")
    embed_model = _setting("HLM_EMBED_MODEL")
    local_model = _setting("HLM_LOCAL_EMBED_MODEL")
    st_model = embedding_model or "all-MiniLM-L6-v2"
    if embed_url and embed_model:
        return embed_model
    if local_model:
        return local_model
    return st_model


def _slug(text: str) -> str:
    """Reduce a model name to something Qdrant accepts in a collection name.

    Model names carry characters collection names should not: `qwen3-embedding:8b`
    and `intfloat/multilingual-e5-large` are both normal. Collapse anything
    outside [A-Za-z0-9_-] to a single underscore.
    """
    return re.sub(r"_+", "_", re.sub(r"[^A-Za-z0-9_-]+", "_", text or "")).strip("_")


def _collection_suffix(embedding_model: Optional[str], dim: Optional[int]) -> str:
    """The `<model>_<dim>` tag appended to every collection this backend uses.

    A Qdrant collection has one fixed vector size, so switching embedders used
    to mean every upsert failed with a dimension error until a full rebuild —
    _init_qdrant could only warn about it. Keying the name on the model means
    the new model gets a new collection and the old one keeps serving reads
    until it is rebuilt or dropped.

    Keyed on the model, not the dimension alone. Two different 4096-dim models
    share a dimension but not an embedding space: putting them in one
    collection returns confident nonsense from cosine similarity, which is
    strictly worse than the loud dimension error it would replace. The
    dimension is carried along for legibility, not identity.
    """
    ident = _slug(_embedding_identity(embedding_model))
    return f"{ident}_{dim}" if dim else ident


def _suffix_enabled() -> bool:
    """Model-keyed collection names, on unless explicitly disabled.

    The escape hatch exists for an operator pointing HLM at a Qdrant instance
    whose collections are named by something else (an external indexer, a
    shared deployment). `HLM_QDRANT_COLLECTION_SUFFIX=false` keeps the flat
    names, and with them the dimension-collision failure mode.
    """
    return str(_setting("HLM_QDRANT_COLLECTION_SUFFIX", "true")).strip().lower() not in (
        "false", "0", "no", "off")


#: A heuristic guess this confident or better replaces the caller's default
#: type. Below it the guess is advisory and the record keeps CUSTOM.
HEURISTIC_CONFIDENCE_MIN = 0.6


def _resolve_data_type(data_type, h_type, confidence) -> str:
    """The single rule deciding a record's `data_type`.

    Caller's explicit type wins; otherwise a *confident* heuristic guess; and
    failing both, CUSTOM. It lives here because it used to be expressed twice —
    `_enrich_metadata` applied the confidence gate, `add()` did not — and the
    two disagreed exactly when the heuristic was unsure.

    The consequence was not a wrong label but a silently disabled check.
    `add()` passed its ungated guess to `_check_duplicate`, which filters the
    Qdrant query on `data_type`, so a record the heuristic called ENV-DATA at
    0.33 confidence was *searched for* among ENV-DATA while being *stored* as
    CUSTOM. Both map to the same collection, so the query ran, matched nothing,
    and returned no duplicate — at any threshold. Measured: two records at
    cosine 0.9507, comfortably inside the 0.95 possible_duplicate band, stored
    silently with no warning; the same pair with `data_type="ENV-DATA"` passed
    explicitly returned `possible_duplicate` as designed. Dedup and
    contradiction detection were both off for every write that omitted a
    data_type and drew a low-confidence guess — the ordinary path for an agent
    storing a fact.
    """
    if data_type and data_type != "CUSTOM":
        return data_type
    try:
        confident = float(confidence) >= HEURISTIC_CONFIDENCE_MIN
    except (TypeError, ValueError):
        confident = False
    if h_type and h_type != "CUSTOM" and confident:
        return h_type
    return data_type or "CUSTOM"


def _conflict_partners(value) -> set:
    """Read `layer3_flags.conflict_with` in either shape it can be on disk.

    It used to hold one uuid, overwritten on every later detection, so a record
    re-paired with something else silently dropped its earlier edge. It holds a
    list now — but rows written before that change still carry a bare string,
    and resolve_conflicts unions over whatever it finds, so both shapes have to
    parse. Anything else (a dict, a number, a stray null) yields no edges
    rather than raising: the flag is advisory and a malformed one must not take
    a maintenance pass down with it.
    """
    if isinstance(value, str):
        return {value} if value else set()
    if isinstance(value, (list, tuple, set)):
        return {v for v in value if isinstance(v, str) and v}
    return set()


def _norm_data_id(data_id):
    """Normalize a `data_id` filter to the form the column actually stores.

    The write path lowercases `data_id` ("SW/sw -> sw"), so every read-side
    comparison has to lowercase too. Four sites build a `data_id = ?` filter
    (`_layer1`, `_fts5_fallback`, `_brute_force_search`, `_fallback_uuids`)
    and only `_layer1` did, so a capitalized `data_id` matched nothing in the
    other three: measured, `_fts5_fallback(data_id="hw")` returned 1 candidate
    and `data_id="HW"` returned 0. `retrieve()` concealed it, because the
    vector arm's candidates hydrate through `_layer1` — the visible symptom
    was not an empty result but the silent loss of the lexical arm.

    Normalizing at `retrieve()` alone would leave each arm wrong in isolation,
    which is how the class re-forms; this is one expression the four sites
    share.
    """
    return str(data_id).lower() if data_id else data_id


def _wrap_untrusted_text(text: str, source: Optional[str]) -> str:
    """Fence external content, stripping any delimiter already in the payload.

    Without the strip, a document containing the literal closing tag ends the
    fence early and everything after it reads as agent-directed instruction.
    """
    if not text or source in _SELF_AUTHORED_SOURCES:
        return text
    cleaned = text.replace(_UNTRUSTED_OPEN, "").replace(_UNTRUSTED_CLOSE, "")
    return f"{_UNTRUSTED_OPEN}{cleaned}{_UNTRUSTED_CLOSE}"


def _get_embedding_fn(embedding_model: Optional[str] = None):
    """Get embedding function. Priority: remote endpoint → local FastEmbed → local sentence-transformers.

    Cached per resolved (embed_url, embed_model, local_model, st_model) key,
    not as a single process-wide global. Under a multiplex gateway one
    process can serve multiple profiles, each with its own HLM_EMBED_URL/
    HLM_EMBED_MODEL resolved via _setting()'s profile scope — a single
    shared cache would let whichever profile calls this first permanently
    decide the embedding model/endpoint for every other profile in that
    process, silently mixing embedding spaces or breaking Qdrant dimension
    consistency for profiles configured differently.

    `embedding_model` is LayeredBackend's per-instance `_embedding_model`
    (default "all-MiniLM-L6-v2") — it only selects the local
    sentence-transformers model in the final fallback tier; it does not
    affect the remote-endpoint or FastEmbed tiers, which are controlled by
    HLM_EMBED_URL/HLM_EMBED_MODEL/HLM_LOCAL_EMBED_MODEL.
    """
    if _embedding_fn is not None:
        # Test fault-injection override — see module-level _embedding_fn.
        return _embedding_fn
    # Read through _setting() so a multiplex gateway resolves the active
    # profile's embedding endpoint rather than the root-level one.
    embed_url = _setting("HLM_EMBED_URL")
    embed_model = _setting("HLM_EMBED_MODEL")
    local_model = _setting("HLM_LOCAL_EMBED_MODEL")
    st_model = embedding_model or "all-MiniLM-L6-v2"

    cache_key = (embed_url, embed_model, local_model, st_model)
    with _embedding_fn_cache_lock:
        cached = _embedding_fn_cache.get(cache_key)
        if cached is not None:
            # Move to end (most recently used)
            _embedding_fn_cache.move_to_end(cache_key)
            return cached

        if embed_url and embed_model:
            # Remote embedding endpoint (Ollama, llama.cpp, etc.)
            import urllib.request
            def remote_embed(texts):
                payload = json.dumps({"model": embed_model, "input": texts}).encode()
                req = urllib.request.Request(
                    embed_url,
                    data=payload,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=60) as resp:
                    result = json.loads(resp.read())
                # Validate the response shape here rather than letting a
                # bare `[0]` at the call site raise IndexError. The common
                # real-world misconfiguration is HLM_EMBED_URL pointing at
                # Ollama's /api/embeddings (which answers with a singular
                # {"embedding": [...]}) instead of /api/embed (which answers
                # with {"embeddings": [[...]]}) — that used to surface as a
                # bare IndexError out of layered_memory(action="add") with
                # nothing naming the endpoint.
                vectors = result.get("embeddings")
                if vectors is None and isinstance(result.get("embedding"), list):
                    # Singular form: one vector for one input.
                    vectors = [result["embedding"]]
                if not isinstance(vectors, list) or len(vectors) != len(texts):
                    raise ValueError(
                        f"embedding endpoint {embed_url} (model={embed_model}) returned "
                        f"{type(vectors).__name__} with "
                        f"{len(vectors) if isinstance(vectors, list) else 'n/a'} vectors for "
                        f"{len(texts)} input(s); expected a list of {len(texts)}. "
                        f"Check that HLM_EMBED_URL points at the batch embedding route "
                        f"(Ollama: /api/embed, not /api/embeddings)."
                    )
                return vectors
            fn = remote_embed
            logger.info("Using remote embedding: %s with %s", embed_url, embed_model)
        elif local_model:
            # Local FastEmbed (multilingual-e5-large, 1024-dim)
            from fastembed import TextEmbedding
            model = TextEmbedding(model_name=local_model)
            def fastembed_fn(texts):
                return [v.tolist() for v in model.embed(texts)]
            fn = fastembed_fn
            logger.info("Using local FastEmbed: %s (1024-dim)", local_model)
        else:
            # Local sentence-transformers (default all-MiniLM-L6-v2, 384-dim)
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(st_model, device="cpu")
            def st_embed_fn(texts):
                return model.encode(texts).tolist()
            fn = st_embed_fn
            logger.info("Using local sentence-transformers: %s", st_model)

        # Evict least recently used if cache is full
        while len(_embedding_fn_cache) >= _EMBED_FN_CACHE_MAXSIZE:
            evicted_key, evicted_fn = _embedding_fn_cache.popitem(last=False)
            logger.info("embedding_fn_cache: evicted LRU entry %s",
                        evicted_key[:2])  # safe repr of config key
        _embedding_fn_cache[cache_key] = fn
        return fn


# ---------------------------------------------------------------------------
# LayeredBackend
# ---------------------------------------------------------------------------
