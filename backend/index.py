"""index module — methods of LayeredBackend, kept in their own file.

These are plain functions whose first argument is the backend instance. They
are bound onto `LayeredBackend` in its class body (see backend/backend.py), so
`self` inside them is the backend and every call site is navigable. Shared
primitives — logger, _setting, the embedding helpers — come from backend/core.py
by direct import; they used to be fetched by name out of sys.modules, which no
editor or type checker could follow.
"""

from __future__ import annotations
import json
import sqlite3
import os

import builtins as _builtins
import sys as _sys
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta, timezone

import collections
import time
import re
import uuid as uuid_mod


from . import constants as _C
from .core import (
    orphan_sweep_refused,
    claimed_overlap as _claimed_overlap,
    profile_release as _profile_release,
    _collection_suffix,
    _from_qdrant_id,
    _get_embedding_fn,
    _setting,
    _suffix_enabled,
    _to_qdrant_id,
    _pack_embedding,
    _unpack_embedding,
    logger,
    _norm_data_id,
    _wrap_untrusted_text,
)


def _physical_collection(self, name: str) -> str:
    """Configured collection name -> the collection actually written to.

    Idempotent, because the map it serves is rebuilt in more than one place:
    _load_runtime_config replaces it wholesale from DB config and
    register_taxonomy adds entries to it, both of which can run after
    _init_qdrant has already keyed the names. Suffixing twice would send
    writes to `memories_model_dim_model_dim`, which exists nowhere.
    """
    suffix = getattr(self, "_collection_suffix", "")
    if not suffix or not name or name.endswith(f"_{suffix}"):
        return name
    return f"{name}_{suffix}"


def _ensure_qdrant_collection(self, coll: str) -> bool:
    """Create `coll` if Qdrant does not have it yet. Returns True if it exists after.

    Collections are otherwise created only by `_init_qdrant`, which runs once at
    process start over the collections configured *then*. `register_taxonomy`
    adds a name to `_collection_map` at runtime, so registering a data_type with
    a new `collection` mapped it to something Qdrant had never heard of: the
    call reported success, and the first `add()` for that type failed its upsert
    (logged as a Q004 warning), landed the row in SQLite alone, and left the
    record unreachable by vector until someone restarted and rebuilt.

    Same shape and indexes as `_init_qdrant`'s creation path, because a
    collection missing the payload indexes filters nothing and would fail
    differently. 2026-08-25 bundle04 review (F2).
    """
    if not self._qdrant or not coll:
        return False
    try:
        self._qdrant.get_collection(coll)
        return True
    except Exception:
        pass
    try:
        from qdrant_client.models import (VectorParams, Distance,
                                          PayloadSchemaType)
        # Size it from a collection that already exists rather than re-probing
        # the embedder. `_init_qdrant` learns the dimension from a sample
        # vector and does not keep it on the instance, and a second probe here
        # would be a network round trip on a path that runs during a config
        # write. The default collection is created at init and is correct by
        # construction.
        dim = None
        for _c in (self._get_collection("CUSTOM"), self._default_collection):
            try:
                dim = self._qdrant.get_collection(_c).config.params.vectors.size
                break
            except Exception:
                continue
        if not dim:
            logger.warning(
                "cannot size a new Qdrant collection %s: no existing collection "
                "to read the dimension from — it will be created at the next "
                "start-up instead", coll)
            return False
        self._qdrant.create_collection(
            collection_name=coll,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        for field in ("data_type", "data_id", "session_name", "profile_name"):
            self._qdrant.create_payload_index(coll, field_name=field,
                                              field_schema=PayloadSchemaType.KEYWORD)
        logger.info("Created Qdrant collection on demand: %s (dim=%d)", coll, dim)
        return True
    except Exception as e:
        logger.warning(
            "could not create Qdrant collection %s: %s — writes routed there "
            "will fail their upsert and the records will be retrievable by "
            "BM25 only until a rebuild", coll, e)
        return False


def _init_qdrant(self):
    # Idempotent guard: safe against double-init from future callers.
    with self._init_lock:
        if self._qdrant is not None:
            return
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
        except ImportError:
            self._qdrant = None
            logger.warning("qdrant_client not installed — vector search disabled")
            return

        # Detect embedding dimension from configured model first, in its own
        # try/except. This call hits HLM_EMBED_URL, not Qdrant — but it used
        # to sit inside the same try block as the Qdrant client, so an
        # embedder DNS failure or timeout was logged as "Qdrant initialization
        # failed", sending whoever read it to check the vector store instead
        # of the embedder. A misconfigured/unreachable ollama.<domain> host
        # produced exactly that misleading line in practice.
        try:
            fn = _get_embedding_fn(self._embedding_model)
            _sample_vectors = fn(["dimension detection"])
            if not _sample_vectors or not _sample_vectors[0]:
                raise ValueError(
                    "embedding endpoint returned no vector during dimension detection — "
                    "cannot size Qdrant collections")
            sample = _sample_vectors[0]
        except Exception as e:
            self._qdrant = None
            logger.warning(
                "Embedding endpoint unreachable, needed to size Qdrant collections: %s — "
                "vector search disabled, retrieval will use brute-force cosine over "
                "locally stored vectors", e)
            return
        dim = len(sample)
        logger.info("Embedding dimension: %d", dim)

        try:
            # Explicit timeout — without one, the underlying httpx client's
            # implicit default applies (version-dependent, currently 5s), which
            # is invisible to a maintainer and may be too short for large batch
            # upserts (rebuild) or too generous for fail-fast-into-circuit-breaker.
            # Read via _setting(), not os.environ directly — under a multiplex
            # gateway a profile's own .env is loaded through the secret scope,
            # not os.environ, so a direct read here would silently revert to
            # the root-level value (or the default) for that profile.
            qdrant_timeout = float(_setting("HLM_QDRANT_TIMEOUT", "30"))
            # Build into a local and only publish to self._qdrant once
            # initialization has fully succeeded. Assigning first meant that any
            # later failure in this block (create_collection denied) left a
            # live client behind while logging "vector search disabled" —
            # every `if self._qdrant:` guard still passed, _layer0's
            # `self._qdrant is None` degraded branch was skipped, and
            # retrieval hammered a broken client instead of using the
            # brute-force fallback built for this case.
            client = QdrantClient(url=self._qdrant_url, timeout=qdrant_timeout)
            # Create all configured collections
            collections = set(self._collection_map.values())
            # Key the collection names on the model that just produced that
            # vector. Done here rather than in __init__ because this is where
            # the dimension is known — the probe above already paid for it.
            #
            # Before this, every profile wrote into a flat `memories` sized by
            # whichever model reached it first. Pointing a profile at a
            # different embedder made every upsert fail with a dimension error
            # and dropped retrieval into brute-force mode, and the only remedy
            # was a full rebuild. Now the new model simply gets its own
            # collection.
            if _suffix_enabled():
                self._collection_suffix = _collection_suffix(self._embedding_model, dim)
                self._collection_map = {k: self._physical_collection(v)
                                        for k, v in self._collection_map.items()}
                self._default_collection = self._physical_collection(self._default_collection)
                collections = set(self._collection_map.values())
                logger.info("Qdrant collections keyed to embedding model: %s",
                            self._collection_suffix)
            for coll in collections:
                try:
                    info = client.get_collection(coll)
                    # Check dimension mismatch
                    existing_dim = info.config.params.vectors.size
                    if existing_dim != dim:
                        logger.warning(
                            "Qdrant collection '%s' has dimension %d but model uses %d — "
                            "upserts and searches will fail and retrieval will run in "
                            "degraded (brute-force) mode. Run layered_maintenance(action='rebuild') "
                            "or delete and recreate the collection.",
                            coll, existing_dim, dim)
                except Exception:
                    client.create_collection(
                        collection_name=coll,
                        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                    )
                    # Create keyword indexes for efficient filtering
                    from qdrant_client.models import PayloadSchemaType
                    for field in ("data_type", "data_id", "session_name", "profile_name"):
                        client.create_payload_index(coll, field_name=field,
                            field_schema=PayloadSchemaType.KEYWORD)
                    logger.info("Created Qdrant collection: %s (dim=%d, indexes: data_type, data_id, session_name, profile_name)", coll, dim)
            # Fully initialized — publish it.
            self._qdrant = client
        except Exception as e:
            self._qdrant = None
            logger.warning(
                "Qdrant initialization failed: %s — vector search disabled, retrieval will use "
                "brute-force cosine over locally stored vectors", e)

# ---- CRUD ------------------------------------------------------------


def _record_qdrant_failure(self, e):
    """Circuit breaker: CLOSED -> OPEN -> HALF-OPEN -> CLOSED/OPEN.

    - CLOSED: count consecutive failures; at 5 the circuit opens.
    - OPEN: every further failure returns immediately, so the "OPEN"
      warning is emitted exactly once per incident rather than on each
      failure.
    - HALF-OPEN: once the cooldown elapses the next request is a trial.
      If it fails, go straight back to OPEN — do not spend four more
      requests on a backend that is still down, which is the whole point
      of having a half-open state.
    """
    with self._circuit_lock:
        now = time.time()
        if self._qdrant_broken_until and now < self._qdrant_broken_until:
            return  # already open; caller falls back

        if self._qdrant_broken_until is not None:
            # Cooldown elapsed and the trial request failed.
            self._qdrant_broken_until = now + 120
            self._qdrant_failures = 0
            logger.warning("[Q001] Qdrant circuit re-OPENED (half-open probe failed), "
                       "120s cooldown: %s", e)
            return

        self._qdrant_failures += 1
        if self._qdrant_failures >= 5:
            self._qdrant_broken_until = now + 120
            self._qdrant_failures = 0
            logger.warning("[Q002] Qdrant circuit OPEN after 5 consecutive failures, "
                       "120s cooldown: %s", e)
        else:
            logger.warning("[Q003] Qdrant failure (%d/5): %s", self._qdrant_failures, e)


def _record_qdrant_success(self):
    """Close the circuit after a successful call."""
    with self._circuit_lock:
        self._qdrant_failures = 0
        self._qdrant_broken_until = None

# ---- Database init --------------------------------------------------


def _get_collection(self, data_type: str) -> str:
    """Get Qdrant collection for a data_type."""
    return self._collection_map.get(data_type, self._default_collection)


def _check_duplicate(self, embedding: List[float], data_type: str, threshold: float):
    """Check for semantically similar existing memories.

    Returns None if no duplicate found, or dict with uuid/similarity/note.
    The dict includes 'status': 'duplicate' (>= threshold) or 'possible_duplicate'
    (>= warning_threshold but < threshold) so the caller can decide.
    """
    # Warning threshold: warn about high similarity before hard block
    warning_threshold = self._config.get("dedup_warning_threshold", 0.95)

    if not self._qdrant:
        # No vector index at all — the exact scan is the only dedup available.
        return self._check_duplicate_sqlite(embedding, data_type, threshold,
                                           warning_threshold)

    qdrant_answered = False
    try:
        coll = self._get_collection(data_type or "CUSTOM")
        from qdrant_client.models import Filter, FieldCondition, MatchValue as MatchValueQ
        scroll_filter = Filter(must=[])
        if data_type:
            scroll_filter.must.append(
                FieldCondition(key="data_type", match=MatchValueQ(value=data_type)))
        if self._profile_name:
            scroll_filter.must.append(
                FieldCondition(key="profile_name", match=MatchValueQ(value=self._profile_name))
            )

        resp = self._qdrant.query_points(
            collection_name=coll,
            query=embedding,
            query_filter=scroll_filter,
            limit=3,
        )
        qdrant_answered = True

        for hit in resp.points:
            similarity = hit.score if hasattr(hit, 'score') else 1.0 - hit.distance
            if similarity >= threshold:
                uuid = _from_qdrant_id(hit.id)
                # status + superseded_by, matching _check_duplicate_sqlite. The
                # Qdrant arm resolved hits by uuid alone, and Qdrant keeps the
                # point for a superseded record (add(supersedes=) only writes
                # SQLite) and for a soft-deleted one whose best-effort point
                # removal failed. Re-adding a superseded fact then matched the
                # hidden original and told the caller to update a record no read
                # path returns — while blocking the write.
                row = self._get_conn().execute(
                    "SELECT content, topic FROM memories "
                    "WHERE uuid = ? AND status = 'active' AND superseded_by IS NULL",
                    (uuid,)
                ).fetchone()
                if not row:
                    continue  # stale point: superseded, deleted, or gone
                return {
                    "uuid": uuid,
                    "similarity": similarity,
                    "status": "duplicate",
                    "existing_content": row[0][:200] if row else None,
                    "note": f"High semantic similarity ({similarity:.3f}). Use the update action to merge content."
                }
            elif similarity >= warning_threshold:
                uuid = _from_qdrant_id(hit.id)
                # Same qualification as the duplicate branch above: warning the
                # caller about a near-duplicate they cannot see is no better
                # than blocking them on one.
                row = self._get_conn().execute(
                    "SELECT content, topic FROM memories "
                    "WHERE uuid = ? AND status = 'active' AND superseded_by IS NULL",
                    (uuid,)
                ).fetchone()
                if not row:
                    continue
                return {
                    "uuid": uuid,
                    "similarity": similarity,
                    "status": "possible_duplicate",
                    "existing_content": row[0][:200] if row else None,
                    "note": f"High semantic similarity ({similarity:.3f}). Review existing content. Use the update action to merge, or add with force=true to store anyway."
                }
    except Exception as e:
        logger.warning("Dedup check failed: %s", e)

    # Exact-scan fallback — only when the vector index could not answer.
    #
    # This used to run whenever Qdrant returned nothing above the warning
    # threshold, i.e. on every ordinary (non-duplicate) write. It unpacks the
    # embedding of every active record of this data_type and runs a matmul:
    # ~8 MB of blob reads per write at the documented vault scale, growing
    # linearly with the corpus, on the hot write path. That is a large part of
    # why bulk ingest was slow enough to trip the Qdrant circuit breaker
    # (541s for 2,024 records).
    #
    # What it was there for — catching a duplicate that Qdrant's index had not
    # caught up on yet — is a narrow window against a permanent per-write cost.
    # A duplicate that slips through is recoverable (compact() merges it, and
    # sync_check/rebuild reconcile the index); an O(corpus) write path is not
    # something the caller can opt out of. Set `dedup_exact_scan: true` to
    # restore the old always-scan behaviour.
    if qdrant_answered and not self._config.get("dedup_exact_scan", False):
        return None
    return self._check_duplicate_sqlite(embedding, data_type, threshold, warning_threshold)


def _check_duplicate_sqlite(self, embedding: List[float], data_type: str,
                            threshold: float, warning_threshold: float) -> Optional[dict]:
    """Fallback dedup: compare against SQLite stored embeddings.

    Used when Qdrant ANN returns nothing above the warning threshold —
    typically because the index hasn't caught up after a recent update.
    """
    try:
        rows = self._get_conn().execute(
            "SELECT uuid, embedding FROM memories "
            "WHERE status='active' AND data_type=? AND embedding IS NOT NULL "
            "AND superseded_by IS NULL",
            (data_type or "CUSTOM",),
        ).fetchall()
    except Exception:
        logger.debug("SQLite dedup fallback: failed to query embeddings")
        return None

    import numpy as np
    q = _unpack_embedding(embedding) if isinstance(embedding, bytes) else embedding
    if not q:
        return None

    # Score in one matmul rather than a Python loop allocating two numpy
    # arrays and a norm per row. This runs on the common (non-duplicate)
    # add() path — Qdrant only short-circuits it when it finds a hit above
    # the warning threshold — so at the documented vault scale (2,000+
    # records) the per-row form was a full vector scan per write.
    uuids, vectors = [], []
    qlen = len(q)
    for uuid, emb_blob in rows:
        if emb_blob is None:
            continue
        try:
            existing = _unpack_embedding(emb_blob)
        except Exception:
            continue
        if not existing or len(existing) != qlen:
            continue
        uuids.append(uuid)
        vectors.append(existing)
    if not vectors:
        return None

    matrix = np.asarray(vectors, dtype=np.float32)
    query_vec = np.asarray(q, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query_vec)
    norms[norms == 0] = np.inf  # zero-norm rows score 0, never selected
    sims = (matrix @ query_vec) / norms
    best_idx = int(np.argmax(sims))
    best_sim = float(sims[best_idx])
    best_uuid = uuids[best_idx]

    # `existing_content` is what makes the verdict actionable: both notes below
    # tell the caller to merge into the existing record, and this arm used to
    # answer with None — the Qdrant arm returns a 200-char excerpt for exactly
    # this reason. The gap was invisible because this arm only answers when
    # `dedup_exact_scan` is on or Qdrant is degraded.
    #
    # Returned **raw**, matching the Qdrant arm one screen up. Fencing happens
    # at the front ends, which know the caller: `mcp_server.py` and
    # `__init__.py`'s add handler both wrap it against the record's source.
    # Fencing here as well would double-wrap it on every path that works.
    def _excerpt(u):
        row = self._get_conn().execute(
            "SELECT content FROM memories WHERE uuid = ?", (u,)).fetchone()
        return (row[0] or "")[:200] if row else None

    if best_sim >= threshold and best_uuid:
        return {
            "uuid": best_uuid,
            "similarity": best_sim,
            "status": "duplicate",
            "existing_content": _excerpt(best_uuid),
            "note": f"High semantic similarity ({best_sim:.3f}). Use the update action to merge content."
        }
    elif best_sim >= warning_threshold and best_uuid:
        return {
            "uuid": best_uuid,
            "similarity": best_sim,
            "status": "possible_duplicate",
            "existing_content": _excerpt(best_uuid),
            "note": f"High semantic similarity ({best_sim:.3f}). Review existing content. Use the update action to merge, or add with force=true to store anyway."
        }
    return None


def check_surfaced_echo(self, content: str, uuids, threshold: float = None,
                        embedding: List[float] = None):
    """Is `content` a restatement of a record this session already surfaced?

    The self-loop auto-extraction could not see: retrieval injects a record
    into the conversation, extraction reads the conversation, and the model
    dutifully "learns" the fact HLM told it. The store then grows a paraphrase
    of a record it already holds, and the paraphrase competes with the
    original for retrieval slots.

    Neither dedup arm caught this, and fixing the type does not fully fix it
    either — `_check_duplicate` is scoped `WHERE data_type = ?`, so it only
    compares when *this* extraction picks the same label a different writer
    picked months ago. Nothing enforces that agreement. This check does not
    depend on it: the caller passes the uuids retrieval actually surfaced this
    session (the plugin's `_uuid_to_tag`, which `_register_uuid` fills from
    prefetch and from every explicit retrieve/peek/list), and the comparison
    is text-to-text via the stored vectors.

    Deliberately *not* a dedup threshold. See EXTRACTION_ECHO_THRESHOLD for
    the measured bands: at 0.95 a reworded echo (0.9643) is caught while a
    real mutation (0.9021 for a version bump) passes through to add(), where
    the contradiction path turns it into a supersession. Tightening this
    constant is how you would make HLM unable to learn that a fact changed.

    Fails **open** — no vector, no embedder, no rows, anything raising — and
    returns None, because a shield that cannot run must not silently swallow
    writes. The worst case of failing open is the duplicate this exists to
    prevent; the worst case of failing closed is losing what the session
    learned.

    Returns {"uuid", "similarity"} for an echo, else None.
    """
    if not content or not uuids:
        return None
    if threshold is None:
        threshold = _C.EXTRACTION_ECHO_THRESHOLD

    # `embedding` lets the caller hand in a vector it already has. The
    # extraction path embeds each candidate to run this check and then calls
    # add(), which embeds again — one HTTP round trip per fact became two, on
    # a path that runs a whole batch at session end. add() takes the same
    # parameter for the same reason (obsidian_ingest), so the caller computes
    # once and both consumers reuse it.
    q = embedding
    if q is None:
        try:
            fn = _get_embedding_fn(self._embedding_model)
            vectors_out = fn([content])
            q = vectors_out[0] if vectors_out else None
        except Exception as e:
            logger.debug("[X001] echo shield: embedding failed, allowing write (%s)", e)
            return None
    if not q:
        return None

    # Chunk the IN clause: SQLITE_MAX_VARIABLE_NUMBER is 999 on stock builds
    # and the caller's set is capped at _MAX_SEEN_UUIDS (5000), so a long
    # session would otherwise raise "too many SQL variables" — which, failing
    # open, would quietly disable the shield exactly on the sessions that
    # retrieve the most and therefore echo the most.
    uuid_list = [u for u in dict.fromkeys(uuids) if u]
    rows = []
    try:
        for start in range(0, len(uuid_list), 500):
            chunk = uuid_list[start:start + 500]
            placeholders = ",".join("?" * len(chunk))
            rows.extend(self._get_conn().execute(
                "SELECT uuid, embedding FROM memories "
                "WHERE uuid IN ({}) AND status='active' "
                "AND embedding IS NOT NULL".format(placeholders),
                chunk,
            ).fetchall())
    except Exception as e:
        logger.debug("[X002] echo shield: query failed, allowing write (%s)", e)
        return None
    if not rows:
        return None

    import numpy as np
    qlen = len(q)
    cand_uuids, vectors = [], []
    for uuid, blob in rows:
        try:
            existing = _unpack_embedding(blob)
        except Exception:
            continue
        if not existing or len(existing) != qlen:
            continue
        cand_uuids.append(uuid)
        vectors.append(existing)
    if not vectors:
        return None

    # One matmul, matching _check_duplicate_sqlite — this runs per extracted
    # candidate, and an extraction batch is tens of facts against a session
    # that may have surfaced hundreds of records.
    matrix = np.asarray(vectors, dtype=np.float32)
    query_vec = np.asarray(q, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query_vec)
    norms[norms == 0] = np.inf
    sims = (matrix @ query_vec) / norms
    best_idx = int(np.argmax(sims))
    best_sim = float(sims[best_idx])
    if best_sim >= threshold:
        return {"uuid": cand_uuids[best_idx], "similarity": best_sim}
    return None


def _check_contradiction(self, embedding: List[float], data_type: str,
                          dedup_threshold: float, content: str = "",
                          data_id: str = None, source: str = "agent"):
    """Check for write-time contradictions.

    Finds records with high semantic similarity (cosine >= 0.85) but
    below the dedup threshold — meaning same topic but different content.
    Uses Jaccard keyword overlap to distinguish contradiction from duplicate.

    Returns None if no contradiction found, or dict with existing_uuid,
    similarity, existing_content if contradiction detected.
    """
    if not self._qdrant:
        # No SQLite fallback here, unlike _check_duplicate — which drops to
        # _check_duplicate_sqlite and keeps working. So while Qdrant is down,
        # contradiction detection does not merely degrade, it stops: a changed
        # fact is stored as a second coexisting record instead of being
        # flagged, and nothing downstream ever revisits it.
        #
        # Made loud rather than fixed. Adding an exact-scan arm here would put
        # a second full embedding scan on the write path during exactly the
        # period the system is already struggling, and _check_duplicate's own
        # comment records what that cost when it ran unconditionally. The
        # honest interim is that the caller can see it happened.
        # 2026-08-22 ox-alpha write review (F3).
        logger.warning(
            "[C002] contradiction detection skipped for this write — Qdrant "
            "unavailable and this check has no exact-scan fallback; a changed "
            "value will be stored as a separate record rather than flagged")
        return None
    t = self._conflict_thresholds()
    cosine_min = t.get("cosine_min", 0.85)
    jaccard_min = t.get("jaccard_min", 0.7)
    jaccard_max = t.get("jaccard_max", 1.0)

    try:
        coll = self._get_collection(data_type or "CUSTOM")
        from qdrant_client.models import Filter, FieldCondition, MatchValue as MatchValueQ
        scroll_filter = Filter(must=[])
        if data_type:
            scroll_filter.must.append(
                FieldCondition(key="data_type", match=MatchValueQ(value=data_type)))
        if self._profile_name:
            scroll_filter.must.append(
                FieldCondition(key="profile_name", match=MatchValueQ(value=self._profile_name))
            )

        # Search for similar vectors (use top 5 for broader check)
        resp = self._qdrant.query_points(
            collection_name=coll,
            query=embedding,
            query_filter=scroll_filter,
            limit=5,
        )

        for hit in resp.points:
            similarity = hit.score if hasattr(hit, 'score') else 1.0 - hit.distance
            # Below dedup threshold but above conflict threshold
            if similarity >= dedup_threshold:
                continue  # Too similar — would be caught by dedup
            if similarity < cosine_min:
                # `break`, not `continue`, and that is a claim about Qdrant:
                # query_points returns points sorted by score descending, so the
                # first hit under the floor guarantees the rest are too. If that
                # ordering ever changes, this stops finding conflict candidates
                # silently rather than failing — which is why the assumption is
                # written down here instead of living in the reader's head.
                break

            uuid_hex = _from_qdrant_id(hit.id)
            # Get full record from SQLite
            row = self._get_conn().execute(
                "SELECT content, data_id, created_at, source, keywords, status, "
                "superseded_by FROM memories WHERE uuid = ?",
                (uuid_hex,)
            ).fetchone()
            # A superseded record keeps status='active' — supersession hides it
            # via superseded_by, not via status — so the status check alone let
            # contradictions be raised against records no read path returns.
            if not row or row[5] != 'active' or row[6]:
                continue

            # Build record dicts for guard checks
            existing_rec = {
                'uuid': uuid_hex, 'content': row[0], 'data_id': row[1],
                'created_at': row[2], 'source': row[3], 'keywords': self._parse_keywords(row[4])
            }
            # The new record doesn't have created_at yet; use now for guard checks.
            # data_id is the caller's real classification. It used to be
            # _heuristic_classify("dummy"), which returns None — that did not
            # "match existing" as the old comment claimed, it disabled guard 1
            # entirely by short-circuiting the `and`.
            #
            # source was hardcoded 'agent' here regardless of the write's
            # actual source. Guard 3 (below) suppresses a contradiction
            # between sequential same-source writes — but with this hardcode
            # it could only ever match an *existing* record whose source is
            # also literally 'agent'. An MCP client (source='mcp-client') or
            # import (source='import') writing the same fact twice within
            # source_guard_days got a false contradiction instead of the
            # suppression Guard 3 exists to provide, because the equality
            # check compared the existing record's real source against a
            # constant instead of against what this write actually is.
            new_rec = {
                'data_id': data_id,
                'created_at': self._now(), 'source': source
            }

            # Run behavioral guards
            if not self._is_conflict_worth_resolving(existing_rec, new_rec):
                continue

            # Jaccard on keywords: if very high overlap → duplicate, not
            # contradiction. LLM enrichment is async so the new record has no
            # stored keywords yet, but entity extraction is cheap and runs on
            # raw content — enough to compare. Both branches used to assign
            # None, so the documented Jaccard step never ran at write time.
            kw_existing = set(k.lower() for k in (existing_rec.get('keywords') or []))
            kw_new = set(k.lower() for k in self._extract_entities(content)) if content else set()
            if kw_existing and kw_new:
                jaccard = self._jaccard_similarity(kw_existing, kw_new)
            else:
                # No basis for comparison — defer to cosine, as before.
                jaccard = None

            # If Jaccard available and in conflict range, flag it
            # If no keywords, defer to embedding (conservative)
            if jaccard is not None:
                if jaccard >= jaccard_max:
                    continue  # Too similar → duplicate territory
                if jaccard < jaccard_min:
                    continue  # Too different → not a contradiction

            # Flag as contradiction
            return {
                "existing_uuid": uuid_hex,
                "similarity": similarity,
                "existing_content": row[0][:200],
            }

        return None
    except Exception as e:
        logger.warning("Contradiction check failed: %s", e)
        return None


def _embed_batch(self, texts: List[str], batch_size: int = 64) -> List[Optional[List[float]]]:
    """Embed many texts with one request per `batch_size`, order preserved.

    Falls back to per-text embedding for a batch that fails, so one bad
    record cannot lose the other 63.
    """
    fn = _get_embedding_fn(self._embedding_model)
    out: List[Optional[List[float]]] = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start:start + batch_size]
        try:
            vectors = fn(chunk)
            if len(vectors) != len(chunk):
                raise ValueError(
                    f"embedder returned {len(vectors)} vectors for {len(chunk)} texts")
            # A correctly-sized batch can still contain a null/empty entry
            # for one text without the embedder raising (a real partial-
            # failure mode) — retry just that text instead of accepting a
            # silent None into an otherwise-good batch.
            for i, v in enumerate(vectors):
                if v:
                    out.append(v)
                    continue
                try:
                    out.append(fn([chunk[i]])[0])
                except Exception:
                    out.append(None)
        except Exception as e:
            logger.warning("batch embed failed (%d texts): %s — falling back to per-text",
                           len(chunk), e)
            for text in chunk:
                try:
                    out.append(fn([text])[0])
                except Exception:
                    out.append(None)
    return out


def _set_retrieval_mode(self, mode: str) -> None:
    """Record how the most recent retrieval was actually ranked.

    `_degraded` alone conflated two very different outcomes: a brute-force
    scan is exact and semantic (only slower), while a lexical BM25 fallback
    loses most of the recall (measured 0.227). An operator seeing
    `degraded_retrieval: true` could not tell which had happened. The mode
    says so directly; `_degraded` keeps its documented sticky meaning for
    existing callers, and the counters give a long-lived process something
    better than a boolean that latches True forever after one blip.
    """
    self._retrieval_mode = mode
    if mode != "vector":
        self._degraded = True
        self._degraded_count = getattr(self, "_degraded_count", 0) + 1
        self._degraded_last_ts = self._now()


def _degraded_semantic_fallback(self, query: str, data_type: str = None,
                                 data_id: str = None, cross_profile: bool = False,
                                 profile_name: str = None) -> List[tuple]:
    """Brute-force cosine over packed SQLite vectors, falling back to lexical.

    Shared by both "Qdrant disabled" and "circuit breaker cooldown" paths.
    Measured recall@5 fell to 0.227 during a cooldown when this fell straight
    to lexical-only (see _fallback_uuids) — brute-force keeps real semantic
    ranking available while Qdrant itself is unreachable.

    Sets the retrieval mode itself, because only here is it known whether the
    brute-force scan actually produced hits or fell through to lexical.

    `_brute_force_search` below is still not `profile_name`-aware — it always
    scans this backend's own packed vectors. Since 0.7.64 that no longer means
    a targeted-other read gets answered from the wrong profile: the arm is
    skipped entirely for any read it cannot serve, and such reads go straight
    to the profile-aware lexical tail with an `[R010]` line saying so (see the
    comment below). What is still open is making the scan itself multi-profile,
    which means opening each target's database and reading its embedding
    column — a decision about what a degraded path may cost, not a bug fix.
    """
    # The brute-force arm scans *this* backend's packed vectors and nothing
    # else, so it can only answer a question about this profile. Asked for
    # another profile — or for all of them — it used to answer anyway: it
    # returned own-profile uuids, which is a non-empty result, so it returned
    # early and the profile-aware lexical tail below never ran. `_layer1` then
    # hydrated those uuids against the *target* profile's database, where they
    # do not exist, and dropped every one.
    #
    # The caller therefore got a confident empty list: not another profile's
    # data (hydration is per-database, so nothing leaked) but no data at all,
    # with no indication that the profile targeting had been silently
    # unsatisfiable. Measured before this change: profile B with Qdrant
    # disabled, reading with profile_name=A, returned 0 results while A held a
    # matching record — and `_fallback_uuids` below, which opens the target
    # profile's DB, would have found it.
    #
    # So: use the arm only for the read it can serve, and let the lexical tail
    # take the rest. Making the *brute-force* arm multi-profile is a different
    # and much larger decision about what a degraded path should cost; it is
    # deliberately not taken here.
    _own = self._profile_name or "default"
    _serves_own_profile = not cross_profile and (
        not profile_name or profile_name == _own)
    if _serves_own_profile:
        try:
            fn = _get_embedding_fn(self._embedding_model)
            _qvs = fn([self._embed_query_text(query)])
            qv = _qvs[0] if _qvs else None
            if qv:
                hits = self._brute_force_search(qv, data_type, data_id,
                                                limit=self._layer0_top_k)
                if hits:
                    self._set_retrieval_mode("brute_force")
                    return hits
        except Exception as e:
            logger.debug("brute-force path unavailable (%s), using lexical", e)
    else:
        logger.info(
            "[R010] vector search is down and the brute-force arm scans only "
            "this profile's vectors, so it cannot serve a %s read — answering "
            "from the lexical fallback, which opens the target profile's "
            "database. Semantic ranking is unavailable for this query.",
            "cross-profile" if cross_profile else f"profile_name={profile_name!r}")
    self._set_retrieval_mode("lexical")
    return self._fallback_uuids(data_type, data_id, cross_profile=cross_profile,
                                query=query, profile_name=profile_name)


def _layer0(self, query: str, data_type: str = None,
            data_id: str = None, profile_name: str = None,
            cross_profile: bool = False) -> List[tuple]:
    """Return [(uuid, distance), ...] from Qdrant ANN. Circuit breaker protected.

    Searches across all relevant collections and merges results.
    When cross_profile=True, requests more candidates to cover all profiles.
    """
    # Hard switch: skip Qdrant entirely if disabled. Still do real
    # semantic search — brute force over the packed vectors — rather than
    # dropping to lexical-only.
    if not self._qdrant_enabled:
        return self._degraded_semantic_fallback(query, data_type, data_id, cross_profile,
                                                profile_name=profile_name)
    # Circuit breaker: YantrikDB pattern (5 failures → 120s cooldown)
    if self._qdrant_broken_until:
        with self._circuit_lock:
            now = time.time()
            if now < self._qdrant_broken_until:
                # Still cooling down — skip Qdrant, but keep real semantic
                # ranking via brute force rather than dropping to lexical-only.
                # The fallback sets the mode (brute_force vs lexical).
                return self._degraded_semantic_fallback(query, data_type, data_id, cross_profile,
                                                        profile_name=profile_name)
            # Cooldown expired — allow one trial request (half-open).
            # Reset the timestamp atomically so other threads also block.
            self._qdrant_broken_until = now + 120
            logger.debug("Qdrant half-open: single trial request")
    if self._qdrant is None:
        # _qdrant_enabled=True but _init_qdrant() failed (Qdrant not
        # running, connection refused, ImportError) — the most common
        # real-world "Qdrant unavailable" case per AGENTS.md. Use the same
        # brute-force-then-lexical fallback as the disabled/cooldown paths
        # rather than dropping straight to lexical-only (measured recall@5
        # 0.227 regression, see _fallback_uuids).
        return self._degraded_semantic_fallback(query, data_type, data_id, cross_profile,
                                                profile_name=profile_name)

    # Embedding the query sits outside the Qdrant try/except below, so a
    # dead embedding endpoint used to raise straight out of retrieve() —
    # past the lexical fallback that exists for exactly this situation.
    try:
        fn = _get_embedding_fn(self._embedding_model)
        query_vec = fn([self._embed_query_text(query)])[0]
    except Exception as e:
        logger.warning("Embedding unavailable (%s) — degraded retrieval "
                       "(lexical FTS5 ranking only)", e)
        self._set_retrieval_mode("lexical")
        return self._fallback_uuids(data_type, data_id,
                                    cross_profile=cross_profile, query=query,
                                    profile_name=profile_name)
    if query_vec is None:
        logger.warning("Embedder returned no vector — degraded retrieval "
                       "(lexical FTS5 ranking only)")
        self._set_retrieval_mode("lexical")
        return self._fallback_uuids(data_type, data_id,
                                    cross_profile=cross_profile, query=query,
                                    profile_name=profile_name)

    # Determine which collections to search
    if data_type:
        collections = [self._get_collection(data_type)]
    else:
        # Search all configured collections
        collections = list(set(self._collection_map.values()))

    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        all_results = []

        any_succeeded = False
        for coll in collections:
            conditions = []
            if data_type:
                conditions.append(FieldCondition(key="data_type", match=MatchValue(value=data_type)))
            if profile_name:
                conditions.append(FieldCondition(key="profile_name", match=MatchValue(value=profile_name)))
            search_filter = Filter(must=conditions) if conditions else None

            # Request more candidates for cross-profile to cover all profiles
            search_limit = self._layer0_top_k * 3 if cross_profile else self._layer0_top_k

            try:
                results = self._qdrant.query_points(
                    collection_name=coll,
                    query=query_vec,
                    limit=search_limit,
                    query_filter=search_filter,
                )
                # Convert Qdrant IDs back to SQLite hex UUIDs
                for hit in results.points:
                    uuid_hex = _from_qdrant_id(hit.id)
                    all_results.append((uuid_hex, 1.0 - hit.score))
                any_succeeded = True
            except Exception as e:
                logger.warning("Qdrant search failed for collection %s: %s", coll, e)
                self._record_qdrant_failure(e)
                continue

        if not any_succeeded and collections:
            # Every collection errored. Returning all_results here means
            # returning [] — retrieval then reports "no results" for a
            # backend failure, indistinguishable from an empty store. The
            # except clause below cannot catch this: the per-collection
            # try/except already swallowed each error, so the degraded
            # fallback built for exactly this case was unreachable.
            #
            # The common trigger is a dimension mismatch between the
            # collection and the embedding model — which _init_qdrant only
            # warns about — so the first four queries silently returned
            # nothing before the breaker even opened on the fifth.
            logger.warning(
                "Qdrant search failed for all %d collection(s) — falling back to "
                "brute-force semantic search over locally stored vectors", len(collections))
            return self._degraded_semantic_fallback(query, data_type, data_id, cross_profile,
                                                    profile_name=profile_name)
        if any_succeeded:
            self._record_qdrant_success()
            # Vector search worked — clear the "how is it ranking now?" state.
            # Deliberately NOT done inside _record_qdrant_success(): three of
            # that method's four call sites are writes (add/update/delete), so
            # resetting there would let a successful write mask degraded reads.
            self._set_retrieval_mode("vector")
        return all_results
    except Exception as e:
        self._record_qdrant_failure(e)
        logger.debug("Qdrant search failed — SQLite fallback")
        return self._degraded_semantic_fallback(query, data_type, data_id, cross_profile,
                                                profile_name=profile_name)


def _brute_force_search(self, query_vec: List[float], data_type: str = None,
                        data_id: str = None, limit: int = 30) -> List[tuple]:
    """Exact nearest-neighbour scan over the SQLite embedding column.

    At personal-memory scale this is faster than a Qdrant round trip: the
    vectors are already packed float32, so scoring a few thousand of them
    is one numpy matmul. It exists so the system has real semantic search
    without a running Qdrant — which is what let the test suite require a
    Docker container and an embedding endpoint just to run.

    Returns [(uuid, distance)] in the same shape as _layer0's Qdrant path.
    """
    try:
        import numpy as np
    except ImportError:
        logger.debug("numpy unavailable — cannot brute-force search")
        return []

    # **Active rows only, always.** This arm takes no `status`, so on the
    # degraded path `retrieve(status="deleted")` and `status="all"` come back
    # with nothing from the vector side while the healthy path honours both —
    # a silent difference in behaviour between the two, not just in latency.
    # Left as a documented limitation rather than threaded through: `status` is
    # an audit/debug filter, the combination needs Qdrant down *and* a
    # non-default status, and the alternative is adding a parameter to two more
    # signatures in the path T364 gates. Filed by the 2026-08-22 ox-alpha read
    # review (F1); revisit if anything starts relying on degraded audit reads.
    where = "status='active' AND superseded_by IS NULL AND embedding IS NOT NULL"
    params: List[Any] = []
    if data_type:
        where += " AND data_type = ?"
        params.append(data_type)
    if data_id:
        where += " AND data_id = ?"
        params.append(_norm_data_id(data_id))
    if self._profile_name and self._has_profile_column():
        where += " AND (profile_name IS NULL OR profile_name = ?)"
        params.append(self._profile_name)

    rows = self._get_conn().execute(
        f"SELECT uuid, embedding FROM memories WHERE {where}", params
    ).fetchall()
    if not rows:
        return []

    uuids, vectors = [], []
    for uuid, blob in rows:
        vec = _unpack_embedding(blob)
        if vec and len(vec) == len(query_vec):
            uuids.append(uuid)
            vectors.append(vec)
    if not vectors:
        return []

    matrix = np.asarray(vectors, dtype=np.float32)
    query = np.asarray(query_vec, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query)
    norms[norms == 0] = 1e-9
    sims = (matrix @ query) / norms

    top = np.argsort(-sims)[:limit]
    logger.debug("brute-force search: scored %d vectors, returning %d",
                 len(vectors), len(top))
    return [(uuids[i], float(1.0 - sims[i])) for i in top]


def _has_profile_column(self) -> bool:
    """True when the memories table carries a profile_name column."""
    if getattr(self, "_profile_col_checked", None) is None:
        cols = [r[1] for r in self._get_conn().execute("PRAGMA table_info(memories)")]
        self._profile_col_checked = "profile_name" in cols
    return self._profile_col_checked


def _fallback_uuids(self, data_type: str = None,
                    data_id: str = None, cross_profile: bool = False,
                    query: str = None, profile_name: str = None) -> List[tuple]:
    """SQLite fallback when Qdrant is unavailable.

    Ranks by FTS5/BM25 against the query. This used to be
    `SELECT uuid FROM memories WHERE status='active' LIMIT 30` — no
    ORDER BY and no query at all, so an outage returned the oldest 30
    records in insertion order, identically for every question, while the
    README advertised it as a working fallback. Measured during a circuit-
    breaker cooldown: recall@5 fell to 0.227 with byte-identical results
    across unrelated queries.

    The row scan remains as a last resort for queries with no usable
    lexical tokens, so an outage still returns *something*.

    When cross_profile=True, queries all discovered profile DBs.
    `profile_name`, naming a profile other than this backend's own, targets
    that one DB exclusively — same reasoning as `_fts5_fallback`, which
    every `query`-bearing call routes through: on the Qdrant-degraded path
    this function's result *is* the read, so an unpropagated `profile_name`
    here means a targeted read while Qdrant is down returns the caller's
    own text under the target's label, or nothing.
    """
    fallback_limit = self._layer0_top_k * 3 if cross_profile else self._layer0_top_k
    if query:
        try:
            ranked = self._fts5_fallback(query, data_type=data_type, data_id=data_id,
                                         limit=fallback_limit,
                                         cross_profile=cross_profile,
                                         profile_name=profile_name)
            if ranked:
                # Derive distance from FTS5 rank position: top result gets
                # 0.1 (near-perfect cosine equivalent), decreasing linearly
                # to 0.5 for the last result. This allows RRF fusion in
                # _layer2 to properly rank FTS5 matches against Qdrant results
                # and against each other, rather than all being identical 0.5.
                n = len(ranked)
                logger.debug("fallback: %d candidates ranked by FTS5", n)
                return [(ranked[i], 0.1 + 0.4 * (i / max(n - 1, 1))) for i in range(n)]
        except Exception as e:
            logger.debug("fallback: FTS5 ranking failed (%s), using row scan", e)
    # Determine which DBs to query — same three-way selection as
    # _fts5_fallback (cross_profile / targeted-other / own).
    own_profile = self._profile_name or "default"
    if cross_profile:
        all_dbs = self._discover_profile_dbs()
    elif profile_name and profile_name != own_profile:
        target_db = self._discover_profile_dbs().get(profile_name)
        all_dbs = {profile_name: target_db} if target_db else {}
    else:
        all_dbs = {own_profile: self._db_path}

    all_uuids = []
    for prof, db_path in all_dbs.items():
        if not db_path or not os.path.exists(db_path):
            continue
        try:
            import sqlite3 as _sqlite3
            conn = _sqlite3.connect(db_path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            where = "SELECT uuid FROM memories WHERE status='active'"
            # Check for superseded_by column (pre-v14 schemas don't have it)
            cols = [r[1] for r in conn.execute("PRAGMA table_info(memories)").fetchall()]
            if "superseded_by" in cols:
                where += " AND superseded_by IS NULL"
            params = []
            if data_type:
                where += " AND data_type = ?"
                params.append(data_type)
            if data_id:
                where += " AND data_id = ?"
                params.append(_norm_data_id(data_id))
            where += " ORDER BY trust_score DESC LIMIT ?"
            params.append(fallback_limit)
            rows = conn.execute(where, params).fetchall()
            # Derive distance from row position: top result = 0.1, last = 0.5.
            # Row-scan fallback returns UUIDs ordered by trust_score DESC;
            # use position as cosine-distance proxy for RRF fusion ranking.
            n = len(rows)
            if n:
                all_uuids.extend((rows[i][0], 0.1 + 0.4 * (i / max(n - 1, 1))) for i in range(n))
            conn.close()
        except Exception:
            continue
    # Not re-truncated to fallback_limit on the cross-profile path — same
    # per-DB-cap-is-the-real-bound reasoning as _fts5_fallback; each DB
    # already limited itself above, so a second global slice here would
    # again let the alphabetically-first profile crowd out the rest.
    return all_uuids if cross_profile else all_uuids[:fallback_limit]


#: Sentinel distinguishing "caller did not say" from "caller said None".
#: _detect_orphans must not suppress cleanup just because a direct caller
#: omitted the argument — only when the pipeline genuinely applied no
#: profile filter, which it signals by passing None explicitly.
_PROFILE_UNSET = object()


def _detect_orphans(self, candidates: List[tuple],
                    cross_profile: bool = False,
                    effective_profile=_PROFILE_UNSET) -> List[tuple]:
    """Filter candidates to those present in this profile's SQLite.

    Orphans: UUIDs in the candidate set but not active in *our* SQLite.

    Qdrant collections are shared across profiles, so a candidate set that
    was not narrowed by a `profile_name` filter legitimately contains other
    profiles' records — and this check only consults our own database.
    Deleting on that basis is destructive cleanup performed by a read,
    against data belonging to someone else: an observed cross-profile query
    saw 83 candidates, declared 78 orphaned, removed them from Qdrant, and
    returned 5.

    Three ways to reach a candidate set this backend's own SQLite cannot
    judge:
      - `cross_profile=True` (by request) — the pool is *meant* to span
        profiles. Nothing is filtered here; `_layer1` hydrates each
        candidate from its own profile's DB. Nothing is deleted.
      - `effective_profile` falsy, i.e. `_layer0` applied no profile filter
        at all. `peek()` never passed one, and a backend constructed
        without `profile_name` never has one, so single-profile reads
        could wipe other profiles' vectors just as thoroughly. The pool
        *is* filtered here, to our own active UUIDs — foreign points may
        be present and are not what this read asked for — but nothing is
        deleted.
      - `effective_profile` names a profile other than this backend's own
        — a single-profile read *targeting* another profile
        (`retrieve(profile_name=<other>)`). This candidate set was
        narrowed to the target's points, so our own SQLite is not merely
        incomplete for it, it is the wrong database entirely: querying it
        for validity made every one of the target's own points look like
        an orphan, and the previous code deleted them from the shared
        Qdrant collection on that basis — a read silently destroying
        another profile's vector index (measured: the target's specific
        point existed before the call and was gone after, with the
        caller's own query returning zero results). Nothing is filtered
        here either, for the same reason as the cross-profile case:
        `_layer1` already opens the *target* profile's DB directly to
        hydrate these candidates, so it is the check that can tell a
        genuine orphan from one of the target's real records. Nothing is
        deleted.

    Genuine orphans (a UUID no profile's SQLite owns) are reported by
    sync_check() and cleaned by rebuild(), not removed here.
    """
    if not candidates or not self._qdrant:
        return candidates
    if effective_profile is _PROFILE_UNSET:
        # Direct caller (or an older call site) — assume this backend's own
        # profile, which is what a single-profile read scopes to.
        effective_profile = self._profile_name
    own_profile = self._profile_name or "default"
    targets_foreign_profile = bool(effective_profile) and effective_profile != own_profile

    may_delete = True
    if cross_profile:
        logger.debug("orphan cleanup skipped: cross-profile candidates "
                     "belong to other profiles' databases")
        may_delete = False
    elif not effective_profile:
        logger.debug(
            "orphan cleanup skipped: candidates were not narrowed by a profile filter, "
            "so non-matching points may belong to other profiles")
        may_delete = False
    elif targets_foreign_profile:
        logger.debug(
            "orphan cleanup skipped: candidates were narrowed to profile %r, "
            "which is not this backend's own (%r) — our SQLite is not "
            "authoritative for them", effective_profile, own_profile)
        may_delete = False

    if not may_delete:
        if cross_profile or targets_foreign_profile:
            # The pool belongs to a database other than ours (every profile,
            # or specifically the targeted one). Filtering against our own
            # SQLite would drop it — or, for the targeted case, drop all of
            # it, reproducing the "read returns zero results" symptom the
            # deletion bug also caused. _layer1 is the check that consults
            # the right database.
            return candidates
        # effective_profile falsy: a same-profile read whose pool was not
        # narrowed. Drop what our own SQLite does not own — still don't
        # delete it, it may be someone else's real record.
        uuids = [c[0] for c in candidates]
        placeholders = ','.join(['?' for _ in uuids])
        rows = self._get_conn().execute(
            f"SELECT uuid FROM memories WHERE uuid IN ({placeholders}) AND status='active'",
            uuids
        ).fetchall()
        valid = {r[0] for r in rows}
        return [c for c in candidates if c[0] in valid]

    # Get all UUIDs from candidates
    uuids = [c[0] for c in candidates]
    placeholders = ','.join(['?' for _ in uuids])

    # Check which UUIDs exist in SQLite as active
    rows = self._get_conn().execute(
        f"SELECT uuid FROM memories WHERE uuid IN ({placeholders}) AND status='active'",
        uuids
    ).fetchall()
    valid_uuids = {r[0] for r in rows}

    # Find orphans
    orphans = [c for c in candidates if c[0] not in valid_uuids]
    # A total mismatch means we are probably not the authority — see
    # core.orphan_sweep_refused. Name guards above cannot catch this: a backend
    # over the wrong database still carries the right profile name.
    # `candidates` is a query-scoped batch, so "all of this batch is stale" is
    # ordinary and must still sweep (T195) — the batch cannot answer authority.
    # Ask the collection instead, and only when the sweep would take everything
    # we saw: do any of *our* active rows sit in it under our own claim? A
    # backend over another profile's live database answers no, which is the case
    # `own`-counting could never see (F13).
    _ovl = None
    if orphans and len(orphans) >= len(candidates):
        _rows = self._get_conn().execute(
            "SELECT uuid, data_type FROM memories WHERE status='active' "
            "LIMIT 64").fetchall()
        _by_coll = {}
        for _u, _dt in _rows:
            _by_coll.setdefault(self._get_collection(_dt or "CUSTOM"), []).append(_u)
        _ovl = 0
        for _c, _us in _by_coll.items():
            _ovl += _claimed_overlap(self._qdrant, _c, _us, self._profile_name)
            if _ovl:
                break
    if orphans and orphan_sweep_refused(
            self._get_conn(), self._profile_name, len(candidates), len(orphans),
            "_detect_orphans", getattr(self, "_db_path", None), overlap=_ovl):
        return [c for c in candidates if c[0] in valid_uuids]
    if orphans:
        logger.warning("orphan detection: %d orphaned records in Qdrant, cleaning", len(orphans))
        # Remove orphans from Qdrant
        try:
            from collections import defaultdict
            by_collection = defaultdict(list)
            # Batch the data_type lookup (IN clause) instead of one SELECT
            # per orphan — orphan counts can run into the dozens/hundreds.
            orphan_uuids = [o[0] for o in orphans]
            orphan_placeholders = ','.join(['?' for _ in orphan_uuids])
            dt_rows = self._get_conn().execute(
                f"SELECT uuid, data_type FROM memories WHERE uuid IN ({orphan_placeholders})",
                orphan_uuids
            ).fetchall()
            data_type_by_uuid = {u: dt for u, dt in dt_rows}
            # An orphan is by definition a uuid this SQLite has no row for, so
            # the data_type lookup above misses for exactly the records being
            # cleaned and every point was aimed at the *default* collection.
            # Points that actually live in `sessions`/`vault` were therefore
            # never removed, while the log still reported them as cleaned.
            # Fall back to deleting the id from every configured collection —
            # a delete for an id that isn't there is a no-op in Qdrant.
            all_collections = set(self._collection_map.values()) or {self._default_collection}
            for o in orphans:
                o_uuid = o[0]
                dt = data_type_by_uuid.get(o_uuid)
                qid = _to_qdrant_id(o_uuid)
                if dt:
                    by_collection[self._get_collection(dt)].append(qid)
                else:
                    for coll in all_collections:
                        by_collection[coll].append(qid)

            for coll, points in by_collection.items():
                # Per-collection try: get_collection() raises for a collection
                # that does not exist, which used to abort cleanup for every
                # remaining collection in the same pass.
                try:
                    if self._qdrant.get_collection(coll):
                        # Release first. This is the site that removed 30 of the
                        # profile-under-test's points in cycle 21, and a shared
                        # point deleted here takes the other owner's index with
                        # it — the 0.8.31 contract every other sweep already
                        # keeps (F16, class sweep).
                        keep = [pid for pid in points
                                if not _profile_release(self._qdrant, coll, pid,
                                                        self._profile_name)]
                        droppable = [pid for pid in points if pid not in set(keep)]
                        if droppable:
                            self._qdrant.delete(collection_name=coll,
                                                points_selector=droppable)
                        logger.info("orphan cleanup: removed %d point(s) from %s "
                                    "(%d kept for another profile)",
                                    len(droppable), coll, len(keep))
                except Exception as e:
                    logger.debug("orphan cleanup: skipped collection %s: %s", coll, e)
        except Exception as e:
            logger.warning("orphan cleanup failed: %s", e)

        return [c for c in candidates if c[0] in valid_uuids]

    return candidates


def _fts5_fallback(self, query: str, scope: str = None,
                   data_type: str = None, data_id: str = None,
                   session_name: str = None, limit: int = 30,
                   cross_profile: bool = False,
                   status: str = "active",
                   profile_name: str = None) -> List[str]:
    """Get UUIDs from FTS5 search for queries that vector search misses.

    When cross_profile=True, searches all discovered profile DBs.

    `profile_name`, when it names a profile other than this backend's own,
    targets that profile's DB exclusively — matching a single-profile
    `retrieve(profile_name=<other>)`. Before this parameter existed, a
    targeted read's lexical arm silently queried the *caller's* own FTS5
    index instead: `_layer0` already scopes the vector arm and `_layer1`
    already hydrates from the target's DB (both take `profile_name`), so a
    targeted read's lexical contribution was always empty — the query's
    keyword/BM25 recall was silently missing for exactly the read shape the
    lexical fallback exists to help, and on the Qdrant-degraded path (where
    this function's result *is* the read) it meant the wrong profile's
    text, or nothing.
    """
    # Build conditions explicitly qualified with m. (the memories table),
    # not via string concatenation with a single leading "m." prefix — that
    # pattern only disambiguated the *first* fragment and broke two ways:
    # (1) status="all" produced a bare "1=1" fragment, which becomes the
    #     invalid "m.1=1" once concatenated (SQLite syntax error, silently
    #     swallowed by the except below — status="all" always returned zero
    #     FTS5 results); (2) data_id exists on both `memories` and
    #     `memories_fts` (the FTS5 index columns), so the later, unprefixed
    #     "AND data_id = ?" fragment raised "ambiguous column name: data_id"
    #     — any data_id-filtered fallback query always returned zero results.
    conditions = []
    params = []
    # Held out of `conditions` and applied per-DB below, for the same reason
    # `_layer1` does it: superseded_by arrives with schema v14, a cross-profile
    # lexical search opens other profiles' files directly, and naming the column
    # on an older one raises "no such column" into the per-profile `except` —
    # which drops that entire profile out of the lexical results.
    supersede_clause = ""
    if status == "active":
        conditions.append("m.status='active'")
        # Superseding a record does not change its status: store.py sets
        # superseded_by/superseded_at and deliberately leaves status='active',
        # which is what makes supersession reversible. So the status filter
        # alone lets every stale version through, and this arm — unlike
        # `_layer1`, which has filtered them since the v14 work — never
        # excluded them. The two halves of the funnel disagreed about what
        # "active" means. Not visible in results, because hydration re-filters;
        # the cost is candidate-slot displacement. `fts_limit` is a fixed
        # budget (max(limit*3, 15)), so each stale version consumed a slot a
        # live record would have taken, and no downstream weight can rank a
        # record that never entered the candidate set.
        # 2026-08-26 ox-alpha round-2 review (bundle01 F4).
        supersede_clause = " AND m.superseded_by IS NULL"
    elif status == "deleted":
        conditions.append("m.status='deleted'")
    # else 'all': no status condition, and no supersession filter
    if data_type:
        conditions.append("m.data_type = ?")
        params.append(data_type)
    if data_id:
        conditions.append("m.data_id = ?")
        params.append(_norm_data_id(data_id))
    if session_name:
        conditions.append("m.session_name = ?")
        params.append(session_name)
    if scope:
        conditions.append("m.scope = ?")
        params.append(scope)
    where = " AND ".join(conditions) if conditions else "1=1"

    # Determine which DBs to query. A profile_name naming another profile
    # targets that one DB exclusively — the fix for the gap in this
    # docstring's own paragraph above. cross_profile still wins if both are
    # somehow set, matching _layer1's precedence (pipeline.py).
    own_profile = self._profile_name or "default"
    if cross_profile:
        all_dbs = self._discover_profile_dbs()
    elif profile_name and profile_name != own_profile:
        target_db = self._discover_profile_dbs().get(profile_name)
        all_dbs = {profile_name: target_db} if target_db else {}
    else:
        all_dbs = {own_profile: self._db_path}

    # Build FTS5 query. Every token is quoted, and a query with no word
    # tokens returns nothing rather than being handed to MATCH raw.
    #
    # The old form was `" OR ".join(tokens) if len(tokens) > 1 else query`,
    # which passed the *unescaped original string* to MATCH whenever it
    # yielded 0 or 1 tokens: MATCH 'C++' is `fts5: syntax error near "+"`,
    # MATCH '???' likewise. The error was caught and logged, so on the
    # degraded path (Qdrant down) any short or punctuated query silently
    # returned zero results. Quoting also stops a bare token being parsed
    # as an FTS5 operator or a `column:` filter.
    import re
    tokens = re.findall(r'\w+', query.lower())
    if not tokens:
        logger.debug(
            "_fts5_fallback: query %r has no searchable tokens", query[:40])
        return []
    fts_query = " OR ".join(f'"{t}"' for t in tokens)

    all_uuids = []
    for prof, db_path in all_dbs.items():
        if not db_path or not os.path.exists(db_path):
            continue
        conn = None
        try:
            import sqlite3 as _sqlite3
            conn = _sqlite3.connect(db_path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            db_where = where
            if supersede_clause:
                prof_cols = [r[1] for r in conn.execute("PRAGMA table_info(memories)")]
                if "superseded_by" in prof_cols:
                    db_where += supersede_clause
                else:
                    logger.debug("profile %s predates schema v14 — supersession "
                                 "filter not applied to the lexical arm", prof)
            rows = conn.execute(f"""
                SELECT m.uuid FROM memories_fts f
                JOIN memories m ON m.rowid = f.rowid
                WHERE {db_where} AND memories_fts MATCH ?
                ORDER BY rank LIMIT ?
            """, params + [fts_query, limit]).fetchall()
            all_uuids.extend(r[0] for r in rows)
        except Exception as e:
            # This silently masked the ambiguous-column-name and invalid-
            # SQL bugs fixed above for an unknown amount of time — a
            # malformed query here should be visible, not swallowed.
            logger.warning("_fts5_fallback query failed for profile=%s: %s", prof, e)
            continue
        finally:
            if conn:
                conn.close()

    # Deduplicate. Not re-truncated to `limit` here on the cross-profile
    # path: each DB already capped itself at `limit` rows via its own SQL
    # LIMIT above, so a single global `[:limit]` slice after concatenating
    # every profile's contribution took only the first DB's rows in
    # `_discover_profile_dbs()`'s alphabetical order — the alphabetically
    # first profile with `limit` matches silently zeroed out every later
    # profile's lexical contribution, including a profile whose only match
    # was more relevant than anything the first profile returned. Measured:
    # two profiles ("aaa" with 20 matches, "zzz" with one distinctive match)
    # returned exactly `limit` UUIDs, all "aaa"'s, "zzz"'s specific record
    # entirely absent. The natural bound is now `limit` per discovered
    # profile — for the single-profile path (one entry in `all_dbs`) that is
    # unchanged from before.
    return list(dict.fromkeys(all_uuids))

# ---- Layer 1: SQLite join + filters ----------------------------------


def _get_vector(self, record_uuid):
    """Extract embedding vector from SQLite by UUID.

    Stored as packed float32; legacy rows hold JSON text. Both are read.
    """
    row = self._get_conn().execute(
        "SELECT embedding FROM memories WHERE uuid = ?",
        (record_uuid,)
    ).fetchone()
    if row and row[0]:
        return _unpack_embedding(row[0])
    return None


def _embed_and_upsert(self, record_uuid, content, payload):
    """Compute the embedding, store it in SQLite, and upsert it to Qdrant.

    **SQLite first, because SQLite is the source of truth.** This used to
    upsert to Qdrant alone — its docstring said so — and `compact`'s merge is
    its only caller, whose `INSERT INTO memories` omits the `embedding`
    column. So every merged record was born with a NULL embedding in SQLite
    while its vector existed only in Qdrant, inverting the invariant the whole
    design rests on: "Qdrant holds vectors derived from SQLite and can always
    be rebuilt."

    Five readers take their embeddings from SQLite, and a merged record was
    invisible to all of them:

      * `_brute_force_search` — the degraded-mode fallback used when Qdrant is
        down. This is the serious one: compaction silently removed records
        from the index that exists *for* the Qdrant outage.
      * `check_surfaced_echo` — the auto-extraction self-loop shield
        (T497-T499), so a merged record could not be recognised as an echo.
      * `graph_health` — its scan is `embedding IS NOT NULL`, so the isolation
        report could never see a compacted record. Found exactly that way: a
        compaction on `hlm-hermes` left the one merged row as the only active
        record in the store with no embedding.
      * `_check_duplicate_sqlite` — dedup on the SQLite path.
      * `sync_check` — reports it as `null_embedding`, while `in_sync` stays
        true. That is correct and was the confusing part: `in_sync` compares
        *populations*, and Qdrant really did have the point.

    `rebuild()` backfills NULL embeddings, so the state was recoverable rather
    than permanent — but nothing prompts a rebuild, and the plugin's
    end-of-session line reads `in_sync=True`, which `AGENTS.md` tells the next
    reader to believe.

    Qdrant stays second and its failure stays non-fatal: sync_check and
    rebuild reconcile it, which is the documented recovery path. A failure to
    persist to SQLite is the one worth a warning, because nothing downstream
    detects it. 2026-09-16.
    """
    try:
        from qdrant_client.models import PointStruct
        fn = _get_embedding_fn(self._embedding_model)
        vectors = fn([content])
        if not vectors or not vectors[0]:
            logger.warning(
                "embed_and_upsert: embedder returned no vector for %s — record has no Qdrant point "
                "(will surface via sync_check until the next rebuild)", record_uuid)
            return
        try:
            self._get_conn().execute(
                "UPDATE memories SET embedding=? WHERE uuid=?",
                (_pack_embedding(vectors[0]), record_uuid))
            self._get_conn().commit()
        except Exception as e:
            logger.warning(
                "embed_and_upsert: could not store the embedding for %s in "
                "SQLite (%s) — the row stays vector-invisible to every "
                "SQLite-side reader, including the Qdrant-outage fallback, "
                "until a rebuild", record_uuid, e)
        self._qdrant.upsert(
            collection_name=self._get_collection(payload.get("data_type", "CUSTOM")),
            points=[PointStruct(
                id=_to_qdrant_id(record_uuid),
                vector=vectors[0],
                payload=payload
            )]
        )
    except Exception as e:
        logger.warning("embed_and_upsert failed for %s: %s", record_uuid, e)


def _cosine_similarity(self, vec_a, vec_b):
    """Compute cosine similarity between two vectors.

    Returns 0.0 — never raises — when the two vectors cannot be compared.
    Mixed dimensions are a real, reachable state: _unpack_embedding only
    *logs* a mismatch and hands the vector back, so a corpus that spans an
    embedding-model change holds both 384- and 1024-dim rows. np.dot on
    unequal shapes raises ValueError, and the caller that matters here —
    _detect_conflicts, via _run_pipeline — has no try/except, so a single
    stale row turned every max_layer>=2 retrieve() into an exception.
    """
    import numpy as np
    if not vec_a or not vec_b:
        return 0.0
    if not isinstance(vec_a, (list, np.ndarray)) or not isinstance(vec_b, (list, np.ndarray)):
        return 0.0
    if len(vec_a) != len(vec_b):
        logger.debug(
            "_cosine_similarity: dimension mismatch %d vs %d — treating as unrelated; "
            "run layered_maintenance(action='rebuild') to re-embed", len(vec_a), len(vec_b))
        return 0.0
    a = np.array(vec_a, dtype=np.float64)
    b = np.array(vec_b, dtype=np.float64)
    if a.ndim != 1 or b.ndim != 1:
        return 0.0
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _jaccard_similarity(self, set_a, set_b):
    """Jaccard similarity between two sets."""
    if not set_a or not set_b:
        return None
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    if union == 0:
        return 0.0
    return intersection / union


__all__ = ['_ensure_qdrant_collection', '_init_qdrant', '_physical_collection', '_record_qdrant_failure', '_record_qdrant_success', '_get_collection', '_check_duplicate', '_check_duplicate_sqlite', '_check_contradiction', '_embed_batch', '_layer0', '_set_retrieval_mode', '_degraded_semantic_fallback', '_brute_force_search', '_has_profile_column', '_fallback_uuids', '_detect_orphans', '_fts5_fallback', '_get_vector', '_embed_and_upsert', '_cosine_similarity', '_jaccard_similarity']
