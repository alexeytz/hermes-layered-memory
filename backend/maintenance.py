"""maintenance module — methods of LayeredBackend, kept in their own file.

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

from typing import List, Optional
from datetime import datetime, timedelta, timezone

import collections
import pwd
import re
import time
import os
import uuid as uuid_mod


from . import constants as _C
from .core import (
    orphan_sweep_refused,
    MAX_CONTENT_CHARS,
    MAX_METADATA_CHARS,
    _EMBED_NULL,
    _SELF_AUTHORED_SOURCES,
    _conflict_partners,
    _from_qdrant_id,
    _pack_embedding,
    _valid_timestamp,
    _retry_on_lock,
    _setting,
    _to_qdrant_id,
    profile_claim as _profile_claim,
    profile_release as _profile_release,
    _unpack_embedding,
    logger,
)


def _allowed_fs_roots(env_var: str) -> List[str]:
    """Allowed base directories for a caller-controlled filesystem read.

    Defaults to the real home directory — resolved via pwd, not
    os.path.expanduser("~"), since Hermes profile HOME overrides make
    expanduser unreliable (see docs/profile-isolation.md). Extend with
    `env_var` (colon-separated absolute paths) for locations outside home.
    Without a containment check, a caller-controlled tool argument would
    otherwise let a caller read any directory the process user can access.
    """
    real_home = pwd.getpwuid(os.getuid()).pw_dir
    roots = [os.path.realpath(real_home)]
    # Read via _setting(), not os.environ directly — under a multiplex
    # gateway a profile's own .env is loaded into an isolated scope, so a
    # direct read would silently ignore that profile's configured roots and
    # reject its legitimate vault.
    extra = _setting(env_var, "") or ""
    for p in extra.split(":"):
        p = p.strip()
        if p:
            roots.append(os.path.realpath(p))
    return roots


def _path_within_roots(path: str, roots: List[str]) -> bool:
    real = os.path.realpath(path)
    return any(real == root or real.startswith(root + os.sep) for root in roots)


def _obsidian_allowed_roots() -> List[str]:
    """Allowed base directories for Obsidian vault ingestion.

    Extend with HLM_OBSIDIAN_VAULT_ROOTS (colon-separated absolute paths)
    for vaults stored outside home.
    """
    return _allowed_fs_roots("HLM_OBSIDIAN_VAULT_ROOTS")


def rebuild(self, since: str = None) -> dict:
    if not self._qdrant:
        return {"status": "skipped", "reason": "qdrant not available"}
    # `since` reaches `WHERE updated_at > ?` verbatim, and `updated_at` holds
    # ISO *text*. SQLite's type ordering puts every INTEGER and REAL below
    # every TEXT value, so a non-string `since` — `12345`, or a provider that
    # serialised a timestamp as a number — matches **every** row: an
    # incremental rebuild silently becomes a full one, re-embedding and
    # re-upserting the entire store while reporting the incremental path's
    # own skip note. Not data loss (the result is a superset), which is what
    # kept it Minor, but it is wasted embedding spend and a multi-minute run
    # mistaken for a cheap one. Guarded at the backend chokepoint rather than
    # in one door, because both front ends forwarded it raw and neither
    # validated — this is the raw-forwarding shape `_coerce_int` exists for.
    # 2026-08-26 ox-alpha xhigh round, bundle02 F5.
    if since is not None and not isinstance(since, str):
        raise ValueError(
            f"since must be an ISO timestamp string, got {type(since).__name__} "
            f"({since!r}) — a non-string compares below every ISO text value in "
            f"SQLite, which silently turns an incremental rebuild into a full one")
    # A *string* that is not a timestamp is the other half, and it is worse,
    # because it compares unpredictably rather than always low. Driven against
    # SQLite with one row at 2026-09-01: `'12345'` and `'!'` sort below it and
    # match everything (the silent full rebuild the check above prevents for
    # non-strings), while `'not-a-date'` sorts ABOVE it and matches **nothing** —
    # so `rebuild(since="not-a-date")` returns {"status":"rebuilt","count":0}
    # having repaired none of the drift it was called to repair. rebuild is the
    # documented recovery path for index drift; a superset costs money, an empty
    # set costs the repair while reporting success.
    #
    # `_valid_timestamp` was already in this file, documented for exactly this
    # ("the fields this guards are compared as *strings* in SQL cutoffs"), and
    # this caller did not use it. 2026-09-14 review round 1, bundle01 F1 — which
    # named only the match-everything direction.
    if since and not _valid_timestamp(since):
        raise ValueError(
            f"since must be a valid ISO timestamp, got {since!r} — an "
            f"unparseable string compares unpredictably against ISO text in "
            f"SQLite: it can match every row (a silent full rebuild) or none "
            f"(a rebuild that reports success and repairs nothing)")
    where = "status = 'active'"
    params = []
    if since:
        where += " AND updated_at > ?"
        params.append(since)
    rows = self._get_conn().execute(f"SELECT uuid, embedding, data_type, data_id, session_name FROM memories WHERE {where}", params).fetchall()
    count = 0
    # Group by collection
    batches = {}
    # Cache collection dimensions (avoid repeated HTTP calls)
    dim_cache = {}
    repacked = 0

    # Phase 1: collect records that need re-embedding (dimension mismatch or NULL)
    reembed_queue = []  # (uuid, content, collection)
    for uuid, emb_json, dt, di, sn in rows:
        try:
            emb = _unpack_embedding(emb_json)
            coll = self._get_collection(dt or "CUSTOM")
            if coll not in dim_cache:
                coll_info = self._qdrant.get_collection(coll)
                dim_cache[coll] = coll_info.config.params.vectors.size
            expected_dim = dim_cache[coll]
            if emb is None or len(emb) != expected_dim:
                # Needs re-embedding — defer to batch phase
                reembed_queue.append((uuid, coll))
        except Exception as e:
            logger.warning("rebuild: dimension check failed for uuid=%s: %s", uuid[:8] if isinstance(uuid, str) else uuid, e)

    # Phase 2: batch re-embed, through `_embed_batch` rather than a second
    # implementation of it. This loop used to call the raw embedding function
    # inside one try/except, so **a single bad record abandoned all 64** — and
    # rebuild went on to report success, having silently skipped the records
    # it exists to repair. `_embed_batch` falls back to per-text embedding for
    # a failed batch, retries a null entry the embedder returned without
    # raising, and refuses a response whose length does not match the request
    # (this loop zipped uuids against whatever came back). 2026-09-24 external
    # review A5.3; `T699`.
    reembed_map = {}  # uuid → embedding
    for i in range(0, len(reembed_queue), _C.EMBED_BATCH_SIZE):
        batch = reembed_queue[i:i + _C.EMBED_BATCH_SIZE]
        contents = []
        batch_uuids = []
        for uuid, coll in batch:
            row = self._get_conn().execute(
                "SELECT content FROM memories WHERE uuid=?", (uuid,)
            ).fetchone()
            if row:
                contents.append(row[0])
                batch_uuids.append(uuid)
        if contents:
            try:
                embeddings = self._embed_batch(contents)
                for uuid, emb in zip(batch_uuids, embeddings):
                    if emb:
                        reembed_map[uuid] = emb
                        self._get_conn().execute(
                            "UPDATE memories SET embedding=? WHERE uuid=?",
                            (_pack_embedding(emb), uuid),
                        )
            except Exception as e:
                logger.warning("rebuild: batch re-embed failed (%d items): %s", len(contents), e)
    if reembed_map:
        self._get_conn().commit()
        logger.info("rebuild: re-embedded %d records (dimension mismatch or NULL)", len(reembed_map))

    # Phase 3: build Qdrant batches with resolved embeddings
    from qdrant_client.models import PointStruct
    for uuid, emb_json, dt, di, sn in rows:
        try:
            emb = _unpack_embedding(emb_json)
            # Use pre-computed re-embedding if available
            if uuid in reembed_map:
                emb = reembed_map[uuid]
            if emb is None:
                # Still NULL — embedding failed or was not needed
                logger.debug("rebuild: skipping NULL embedding for %s", uuid[:8])
                continue
            coll = self._get_collection(dt or "CUSTOM")
            if isinstance(emb_json, str) and emb_json != _EMBED_NULL:
                # Opportunistic migration: repack legacy JSON-text encoding
                self._get_conn().execute("UPDATE memories SET embedding=? WHERE uuid=?",
                                         (_pack_embedding(emb), uuid))
                repacked += 1
            payload = {"data_type": dt or "CUSTOM"}
            if di:
                payload["data_id"] = di
            if sn:
                payload["session_name"] = sn
            if self._profile_name:
                payload["profile_name"] = _profile_claim(
                    self._qdrant, coll, _to_qdrant_id(uuid), self._profile_name)
            batches.setdefault(coll, []).append(
                PointStruct(id=_to_qdrant_id(uuid), vector=emb, payload=payload)
            )
            if len(batches[coll]) >= 100:
                self._qdrant.upsert(collection_name=coll, points=batches[coll])
                count += len(batches[coll])
                batches[coll] = []
        except Exception as e:
            logger.warning("rebuild failed for %s: %s", uuid[:8], e)
            pass
    for coll, batch in batches.items():
        if batch:
            try:
                self._qdrant.upsert(collection_name=coll, points=batch)
                count += len(batch)
            except Exception as e:
                logger.warning("rebuild: final batch upsert failed for collection %s (%d points): %s", coll, len(batch), e)

    if repacked:
        self._get_conn().commit()
        logger.info("rebuild: repacked %d legacy JSON embeddings as float32", repacked)

    # Cleanup: delete orphaned Qdrant points (archived/deleted records).
    # When `since` is set, active_uuids is restricted to only the incremental
    # subset — using it as the "keep" set would wipe vectors for every active
    # record older than `since`. Skip orphan cleanup in incremental mode.
    if not since:
        from qdrant_client.models import PointIdsList, Filter, FieldCondition, MatchValue as MatchValueModel
        # Re-read, rather than reusing `rows` from the top of this function.
        # A rebuild over a large corpus takes minutes, and any record archived
        # or deleted during it is still in that snapshot — so its point was
        # upserted above and then *kept* here, because the snapshot says it is
        # active. A live vector for a retired record: it competes in `_layer0`
        # while `_layer1` excludes the row, which is the exact orphan class
        # 0.7.84's retirement work exists to remove.
        # 2026-08-24 audit, finding 7.
        active_uuids = {r[0] for r in self._get_conn().execute(
            "SELECT uuid FROM memories WHERE status = 'active'").fetchall()}
        # Iterate the live collection map, not self._config["collections"].
        # The config key is only populated when an operator sets it
        # explicitly or register_taxonomy() runs — LayeredBackend.__init__
        # builds the default map into _collection_map alone. So under the
        # documented default configuration this loop ran over an empty dict
        # and the entire orphan-cleanup phase silently did nothing, while
        # rebuild() still reported success and sync_check() kept reporting
        # the resulting drift with no way to fix it.
        for coll in set(self._collection_map.values()):
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

                # Delete points not in active UUIDs — but re-ask SQLite first.
                #
                # `active_uuids` was read at the top of rebuild(), and
                # everything between then and here takes real time: a
                # dimension check per collection, an embedding round trip per
                # 64 records, batched upserts, then a full scroll of the
                # collection. A record another session added inside that
                # window is in Qdrant (add() upserts immediately) and absent
                # from this snapshot, so it looked exactly like an orphan and
                # its vector was deleted — leaving a record that SQLite serves
                # and vector search cannot find until the next rebuild.
                #
                # Re-checking is cheap next to what it guards: one indexed
                # lookup per candidate, and the candidate list is normally
                # empty or tiny. Chunked because SQLite caps host parameters
                # per statement (999 on builds older than 3.32).
                candidates = [u for u in scroll_uuids if u not in active_uuids]
                revived = set()
                for i in range(0, len(candidates), 500):
                    chunk = candidates[i:i + 500]
                    ph = ",".join("?" for _ in chunk)
                    revived.update(r[0] for r in self._get_conn().execute(
                        f"SELECT uuid FROM memories WHERE uuid IN ({ph}) AND status='active'",
                        chunk).fetchall())
                if revived:
                    logger.info(
                        "Rebuild cleanup: %d point(s) matched a record that became "
                        "active during the rebuild — keeping them", len(revived))
                # Points sitting in a collection their record does not belong to.
                #
                # The sweep above asks "is this uuid still active?" — a record
                # filed under the *wrong* collection answers yes, so it can never
                # be seen. `_drop_points_everywhere` exists because a point can
                # sit under a data_type the record no longer has; that covers the
                # delete path, and this is the same class on the rebuild path.
                #
                # Found 2026-08-27: `profile-a` reported 117 points for 114 rows
                # for hours. Three SESSION-DATA records were indexed in both
                # `memories_*` and `sessions_*` — the copy in `memories_*` was
                # left behind by an older mapping, and every rebuild since had
                # scrolled straight past it. It was dismissed twice as Qdrant
                # count lag before anyone compared the collections.
                #
                # Released rather than deleted outright: another profile may map
                # the same data_type somewhere else entirely (`collections` is
                # per-profile config), so its copy here can be correctly filed
                # even while ours is not. `_profile_release` drops only our claim
                # and keeps the point if anyone else still holds it.
                misfiled = []
                present = [u for u in scroll_uuids if u in active_uuids]
                for i in range(0, len(present), 500):
                    chunk = present[i:i + 500]
                    ph = ",".join("?" for _ in chunk)
                    for _u, _dt in self._get_conn().execute(
                            f"SELECT uuid, data_type FROM memories WHERE uuid IN ({ph})",
                            chunk).fetchall():
                        if self._get_collection(_dt or "CUSTOM") != coll:
                            misfiled.append(_u)
                if misfiled:
                    _dropped = 0
                    for _u in misfiled:
                        _pid = _to_qdrant_id(_u)
                        if _profile_release(self._qdrant, coll, _pid, self._profile_name):
                            try:
                                self._qdrant.delete(collection_name=coll,
                                                    points_selector=[_pid])
                                _dropped += 1
                            except Exception as e:
                                logger.warning("could not remove misfiled point %s "
                                               "from %s: %s", _pid, coll, e)
                    logger.info("Rebuild cleanup: %d misfiled point(s) removed from %s "
                                "(record's data_type maps elsewhere); %d kept for "
                                "another profile", _dropped, coll, len(misfiled) - _dropped)

                orphan_ids = [_to_qdrant_id(u) for u in candidates if u not in revived]
                # Same guard as the read path — this is the site that actually
                # deleted 44 points on 2026-08-25 and ~90 on 2026-08-26.
                # Exact overlap is free here: both sets are already built.
                # A wrong-database backend claims none of them (F13).
                _ovl = len(set(scroll_uuids) & set(active_uuids))
                if orphan_ids and orphan_sweep_refused(
                        self._get_conn(), self._profile_name, len(scroll_uuids),
                        len(orphan_ids), "rebuild cleanup",
                        getattr(self, "_db_path", None), overlap=_ovl):
                    orphan_ids = []
                if orphan_ids:
                    # Release, then delete only what nobody else holds.
                    #
                    # A point's id is the record uuid, so two profiles that both
                    # hold a record share one point (0.8.31). "Orphaned" here
                    # means *this* profile has no active row for it — which says
                    # nothing about the other holder. Deleting outright destroyed
                    # their vector: measured 2026-08-27, profile A rebuilt after
                    # losing its copy of a shared record and B's point went with
                    # it, leaving B's row served only by the lexical path.
                    #
                    # `_drop_points_everywhere` was given exactly this treatment
                    # in 0.8.31 and this site was missed — one member of the
                    # class again, in the release that created the class.
                    # Releasing also fixes the stale-claim leak: a row that left
                    # SQLite without a `delete()` call (import cleanup, a manual
                    # DELETE) leaves this profile's name on the point, and this
                    # sweep is what removes it.
                    # 2026-08-27, after cycle 17.
                    droppable = [pid for pid in orphan_ids
                                 if _profile_release(self._qdrant, coll, pid,
                                                     self._profile_name)]
                    if droppable:
                        self._qdrant.delete(collection_name=coll,
                                            points_selector=PointIdsList(points=droppable))
                    logger.debug("Rebuild cleanup: %d orphaned point(s) in %s — "
                                 "%d deleted, %d released to another profile",
                                 len(orphan_ids), coll, len(droppable),
                                 len(orphan_ids) - len(droppable))
            except Exception as e:
                logger.warning("Rebuild cleanup failed for %s: %s", coll, e)

    result = {"status": "rebuilt", "count": count}
    if since:
        result["since"] = since
        # Say that orphan cleanup did not run. It is skipped deliberately in
        # incremental mode — `active_uuids` holds only the rows newer than
        # `since`, so using it as the keep-set would wipe the vector of every
        # active record older than that — but the result said `rebuilt` with no
        # hint, so an operator running incremental rebuilds saw success while
        # `sync_check` kept reporting drift that nothing here would ever clear.
        # The fix for the drift is a full `rebuild()`; this is what tells them.
        # 2026-08-25 bundle03 review (F1).
        result["orphan_cleanup"] = "skipped (incremental)"
        result["note"] = (
            "Incremental rebuild re-embeds changed records but does not remove "
            "orphaned points. If sync_check reports drift, run rebuild() with "
            "no `since`.")
    return result



def resolve_conflicts(self, execute: bool = False) -> dict:
    """Resolve conflict groups: keep the winner, soft-delete the rest.

    Args:
        execute: Apply the deletions. **Defaults to False** — without it this
            reports the proposed winner and losers and changes nothing.

    Strategy:
    1. Find all records flagged as conflict_candidate by _detect_conflicts
    2. Group by the detected pair (transitively, via union-find)
    3. Keep: highest trust_score, then NEWEST created_at
    4. Soft-delete the rest

    Why the default is dry-run: a "conflict" here is a heuristic verdict from
    cosine + Jaccard thresholds, and acting on it destroys one side of a pair
    of facts. Until the tiebreak was fixed, the side it destroyed was the
    *current* value. A heuristic that deletes should have to be asked twice.

    Grouping used to be by exact lowercased content match, which could
    almost never fire: a conflict candidate is by construction a record
    whose content is *similar but different* — exact duplicates are caught
    by dedup at write time. _detect_conflicts records the counterparts in
    layer3_flags.conflict_with, so the pairs it actually found are the
    grouping key.

    That field holds every counterpart, not the most recent one. It used to
    hold a single uuid and overwrite it, and the union-find below hid the
    consequence in the common case: A~B then A~C still reached one group
    through B's back-pointer to A. Only when *both* ends of a pair were later
    re-paired elsewhere did the edge disappear, splitting one conflict group
    into two and leaving the stale record alive — the exact outcome this
    action exists to prevent. Rows written before the change still carry a
    bare string, which _conflict_partners reads.
    """
    # Find all conflict-flagged records. `source` is carried through to the
    # dry-run preview below — without it the caller has no way to fence the
    # content snippet, and this action defaults to dry-run, so every first
    # invocation returned raw stored content with no way to tell it apart
    # from HLM's own text.
    flagged = self._get_conn().execute(
        "SELECT uuid, content, trust_score, created_at, data_type, layer3_flags, "
        "source FROM memories WHERE status='active' AND layer3_flags LIKE ?",
        ('%"conflict_candidate"%',)
    ).fetchall()

    if not flagged:
        return {"groups_resolved": 0, "records_deleted": 0, "details": [],
                "executed": bool(execute)}

    by_uuid = {}
    for row in flagged:
        partners = set()
        try:
            partners = _conflict_partners(
                (json.loads(row[5] or "{}") or {}).get("conflict_with"))
        except (json.JSONDecodeError, TypeError):
            pass
        by_uuid[row[0]] = {
            "uuid": row[0],
            "content": row[1],
            "trust_score": row[2] if row[2] is not None else 0.5,
            "created_at": row[3],
            "data_type": row[4],
            "partners": partners,
            "source": row[6],
        }

    # Union-find over the recorded pairs, so a chain A~B~C resolves as one
    # group instead of two overlapping ones.
    parent = {u: u for u in by_uuid}

    def _find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a, b):
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[rb] = ra

    for u, rec in by_uuid.items():
        for p in rec["partners"]:
            if p in parent:
                _union(u, p)

    groups = collections.defaultdict(list)
    for u, rec in by_uuid.items():
        groups[_find(u)].append(rec)

    deleted = 0
    details = []
    now = self._now()

    for group_key, members in groups.items():
        if len(members) < 2:
            continue  # no counterpart to resolve against

        # Sort: highest trust first, then NEWEST first.
        #
        # The tiebreak used to be oldest-first, and every record starts at
        # trust_score 0.5 — so in the ordinary case (nobody has given
        # feedback on either side) the *stale* value won and the current one
        # was soft-deleted. A conflict candidate is by construction "same
        # topic, different value", i.e. a fact that changed; both the L3
        # rerank prompt ("Prefer the most recent entry unless context
        # indicates otherwise") and the supersedes design treat recency as
        # the tiebreak. Match them. Two passes because the two keys sort in
        # opposite directions and sort() is stable.
        members.sort(key=lambda r: (r["created_at"] or ""), reverse=True)
        members.sort(key=lambda r: -r["trust_score"])
        winner = members[0]
        losers = members[1:]

        if not execute:
            details.append({
                "content": (winner["content"] or "")[:80],
                "source": winner["source"],
                "would_keep": winner["uuid"],
                "would_delete": [
                    {"uuid": r["uuid"], "content": (r["content"] or "")[:80],
                     "source": r["source"],
                     "trust_score": r["trust_score"], "created_at": r["created_at"]}
                    for r in losers],
            })
            continue

        for loser in losers:
            self._get_conn().execute(
                "UPDATE memories SET status='deleted', updated_at=? WHERE uuid=?",
                (now, loser["uuid"])
            )
            # Release anything the loser superseded — otherwise those
            # records stay active but permanently invisible (see delete()).
            self._get_conn().execute(
                "UPDATE memories SET superseded_by = NULL, superseded_at = NULL, updated_at = ? "
                "WHERE superseded_by = ?", (now, loser["uuid"]))
            # Remove the Qdrant point immediately, matching _compact_group's
            # pattern — otherwise it sits orphaned (drifting sync_check's
            # in_sync count) until the next purge() cycle picks it up.
            # Sweep every collection, like delete() and test_cleanup().
            # This deleted from the loser's *current* data_type only, so a
            # record whose type had moved kept its vector — and this is the
            # path that retires records a heuristic chose, which makes a
            # leftover here harder to notice than one from an explicit delete.
            # Fourth site of one class; the sweep now lives in one helper.
            # 2026-08-25 bundle03 review (F1).
            self._drop_points_everywhere(loser["uuid"])
            deleted += 1

        logger.info("resolve_conflicts: kept %s, deleted %d conflicting record(s) of '%s'",
                    winner["uuid"][:8], len(losers), (winner["content"] or "")[:60])
        details.append({
            "content": (winner["content"] or "")[:80],
            "source": winner["source"],
            "kept": winner["uuid"],
            "deleted": [r["uuid"] for r in losers],
        })

    if not execute:
        logger.info(
            "resolve_conflicts: dry run — %d group(s) would be resolved (execute=False)",
            len(details))
        return {
            "status": "dry_run",
            "executed": False,
            "groups_found": len(details),
            "records_affected": sum(len(d["would_delete"]) for d in details),
            "groups": details,
            "note": "Nothing was changed. Re-run with execute=True to apply. "
                    "Losing records are soft-deleted; the winner is the highest "
                    "trust_score, newest on a tie.",
        }

    # Clear only the conflict flags on the surviving records, not the whole
    # column.  resolve_conflicts is the writer; L3 writes rerank_score, gaps,
    # and confidence into the same column, and a later call to get_traces or
    # an annotated result still reads them back.  Wiping the lot would discard
    # every annotation the reranker placed.
    if details:
        try:
            for d in details:
                row = self._get_conn().execute(
                    "SELECT layer3_flags FROM memories WHERE uuid = ?",
                    (d["kept"],)).fetchone()
                flags = {}
                if row and row[0]:
                    try:
                        flags = json.loads(row[0]) or {}
                    except (json.JSONDecodeError, TypeError):
                        pass
                flags.pop("conflict_candidate", None)
                flags.pop("conflict_with", None)
                self._get_conn().execute(
                    "UPDATE memories SET layer3_flags = ? WHERE uuid = ?",
                    (json.dumps(flags), d["kept"]))
        except Exception as e:
            logger.debug("resolve_conflicts: flag clear failed: %s", e)

    self._get_conn().commit()
    logger.info("resolve_conflicts: %d groups resolved, %d records deleted", len(details), deleted)
    return {"groups_resolved": len(details), "records_deleted": deleted,
            "details": details, "executed": True}


def sync_check(self) -> dict:
    sqlite_count = self.count()
    # count() excludes superseded records; rebuild() indexes every
    # status='active' row, superseded ones included (so a status="all"
    # retrieval can still find them by vector). Comparing those two numbers
    # made in_sync permanently false after the very first supersede, and
    # on_session_end reacted by running a full rebuild at every single
    # session teardown, forever, for drift that could never be corrected.
    # Compare against the set that is actually indexed.
    try:
        indexed_count = self._get_conn().execute(
            "SELECT COUNT(*) FROM memories WHERE status = 'active'").fetchone()[0]
    except Exception:
        indexed_count = sqlite_count
    llm_ok = self.llm_configured()
    qdrant_count = 0
    profile_count = 0
    # Records whose stored vector no longer matches the collection it belongs
    # to. _unpack_embedding only logs a mismatch and _cosine_similarity now
    # returns 0.0 for one, so a corpus spanning an embedding-model change
    # degrades quietly: those records are unreachable by vector search and
    # invisible to conflict detection. sync_check is where an operator looks
    # for drift, so the count belongs here next to the others.
    dim_mismatch = 0
    # NULL embeddings: records written while the embedder was unreachable.
    # Equally invisible to vector search as a dimension mismatch, and equally
    # absent from the count above because typeof(NULL) != 'blob'.
    #
    # Counted here rather than inside the Qdrant block, which is where it first
    # landed: whether a row's embedding column is NULL is a fact about SQLite
    # and holds whether or not Qdrant answers. Nesting it under `if
    # self._qdrant` meant the one situation that produces these records — an
    # outage — could also be the situation that reports zero of them.
    try:
        # Both shapes of "no usable embedding": SQL NULL (legacy rows, explicit
        # clears) and the TEXT sentinel the write path stores when the embedder
        # fails. This counted only the first, so the diagnostic reported 0 for
        # exactly the rows it exists to find. See `SQL_EMBEDDING_MISSING`.
        null_embedding = self._get_conn().execute(
            "SELECT COUNT(*) FROM memories "
            "WHERE status='active' AND " + _C.SQL_EMBEDDING_MISSING).fetchone()[0]
    except Exception as e:
        logger.debug("sync_check: null-embedding audit skipped: %s", e)
        null_embedding = 0
    if self._qdrant:
        try:
            from qdrant_client.models import FieldCondition, MatchValue, Filter
            # Count points per profile in each collection
            for coll in set(self._collection_map.values()):
                try:
                    info = self._qdrant.get_collection(coll)
                    qdrant_count += info.points_count
                    try:
                        expected_dim = info.config.params.vectors.size
                        # Packed float32: 4 bytes per dimension. Legacy JSON-text
                        # rows are excluded by the typeof() guard rather than
                        # counted as mismatches.
                        dts = [dt for dt, c in self._collection_map.items() if c == coll]
                        if dts and expected_dim:
                            placeholders = ",".join("?" for _ in dts)
                            row = self._get_conn().execute(
                                f"SELECT COUNT(*) FROM memories "
                                f"WHERE status='active' AND embedding IS NOT NULL "
                                f"AND typeof(embedding)='blob' "
                                f"AND length(embedding) != ? "
                                f"AND data_type IN ({placeholders})",
                                [expected_dim * 4] + dts).fetchone()
                            dim_mismatch += row[0] if row else 0
                    except Exception as e:
                        logger.debug(
                            "sync_check: dimension audit skipped for %s: %s", coll, e)
                    # Filter by current profile
                    if self._profile_name:
                        filter_ = Filter(must=[FieldCondition(key="profile_name",
                           match=MatchValue(value=self._profile_name))])
                        count_result = self._qdrant.count(coll, count_filter=filter_)
                        profile_count += count_result.count
                except Exception as e:
                    logger.warning("sync_check: Qdrant count failed for collection %s: %s", coll, e)
        except Exception:
            logger.debug("sync_check: Qdrant unavailable, using SQLite count only")
    # Use profile-specific count for sync check
    effective_count = profile_count if self._profile_name else qdrant_count
    return {"sqlite_active": sqlite_count, "qdrant_total": qdrant_count,
            "qdrant_profile": effective_count,
            # What rebuild() would put in Qdrant — the number in_sync compares.
            "sqlite_indexed": indexed_count,
            # `HLM_QDRANT_ENABLED=false` is a supported mode, not a fault, and
            # in it there is no derived index for the source of truth to be out
            # of step with — so a count comparison against a structural 0 is not
            # a drift measurement, it is a category error. It reported
            # `in_sync=False` on a healthy SQLite-only profile every session,
            # which `initialize` then read as "Qdrant stale" and answered with a
            # rebuild that `rebuild()` immediately skipped ("qdrant not
            # available"). Pointless work, and worse than pointless in a repo
            # whose own guidance is to believe this line.
            # `qdrant_enabled` is reported alongside so a reader can tell the
            # two zeroes apart: nothing indexed, versus nothing to index.
            # 2026-08-26 external sweep, N5.
            "qdrant_enabled": bool(getattr(self, "_qdrant_enabled", True)),
            "in_sync": (indexed_count == effective_count
                        if getattr(self, "_qdrant_enabled", True) else True),
            "llm_configured": llm_ok,
            # >0 means those records were embedded with a different model and
            # are unreachable by vector search until a rebuild re-embeds them.
            "dimension_mismatch": dim_mismatch,
            # >0 means those records are unreachable by every SQLite-side
            # reader — the Qdrant-outage brute-force fallback, the extraction
            # echo shield, graph_health's scan, SQLite dedup — and invisible to
            # the dimension-mismatch audit because the column is NULL rather
            # than the wrong length.
            #
            # This comment used to say "stored during an embedder outage",
            # naming the only cause anyone had seen. `compact` produced the
            # same state with a perfectly healthy embedder: its merge INSERT
            # omitted the `embedding` column and `_embed_and_upsert` wrote
            # Qdrant alone (fixed 2026-09-16). A reader who trusted the old
            # wording would have gone looking for an outage that never
            # happened. Note `in_sync` can be true beside a non-zero count
            # here, and correctly so: it compares populations, and the record
            # does have its Qdrant point.
            "null_embedding": null_embedding,
            # Sticky ("ever degraded") vs current ("how did the last
            # retrieval rank?") — see retrieval_stats() for the distinction.
            "degraded_retrieval": self._degraded,
            "retrieval_mode": getattr(self, "_retrieval_mode", "vector"),
            "degraded_count": getattr(self, "_degraded_count", 0),
            "degraded_last": getattr(self, "_degraded_last_ts", None),
            "qdrant_circuit_open": bool(self._qdrant_broken_until
                                        and time.time() < self._qdrant_broken_until)}

# ---- Trust score feedback --------------------------------------------

@_retry_on_lock


def _retire_archived(self, uuids) -> None:
    """Release supersessions and drop Qdrant points for freshly archived rows.

    `sleep()` archives with three bare UPDATEs that never went near Qdrant, so
    every archived row left its vector behind. `_layer0` filters on
    data_type/profile_name and never on status, so those points kept competing
    for top_k slots while `_layer1` excluded the rows — noise that displaces
    live records. `sync_check()` compares the active-row count against the
    point count, so `in_sync` went permanently False after any sleep() run, and
    `rebuild()` selects `status='active'` and so never cleared them: a full
    re-embed on every teardown that fixed nothing.

    `update(status=...)` does this work for the same reason; this is the same
    retirement through the maintenance door. Found by the 2026-08-22 ox-alpha
    write review (F1).

    Failures are logged, never raised: archival already happened in SQLite,
    which is the source of truth, and sync_check plus rebuild are the documented
    recovery for a stale point.
    """
    uuids = [u for u in (uuids or []) if u]
    if not uuids:
        return
    now = self._now()
    try:
        placeholders = ",".join("?" * len(uuids))
        released = self._get_conn().execute(
            "UPDATE memories SET superseded_by = NULL, superseded_at = NULL, "
            "updated_at = ? WHERE superseded_by IN ({})".format(placeholders),
            [now] + uuids).rowcount
        if released:
            logger.info("sleep: archived records superseded %d row(s) — "
                        "restored them to active retrieval", released)
    except Exception as e:
        logger.warning("[S001] sleep: could not release supersessions: %s", e)

    if not self._qdrant:
        return
    # Sweep every collection, not the row's current data_type.
    #
    # Fifth member of the class `_drop_points_everywhere`'s docstring describes
    # — and that docstring said "now there is one site", which was premature.
    # A record's point may sit under a data_type it no longer has (a retype
    # whose resync did not complete leaves it under the old one), so deleting
    # from the current type alone strands the point, and only `sync_check` ever
    # notices. This is a retirement path, exactly like `delete`, `purge`,
    # `test_cleanup` and `resolve_conflicts`, all of which sweep everywhere.
    # Found by the 2026-08-26 ox-alpha round (bundle03 F11); T570 did not cover
    # it because it was written for `purge`, the member that was found first.
    try:
        self._drop_points_everywhere(uuids)
        self._record_qdrant_success()
    except Exception as e:
        self._record_qdrant_failure(e)
        logger.warning(
            "[S003] sleep: could not remove %d archived point(s): %s "
            "— sync_check will report the drift and rebuild clears it",
            len(uuids), e)


def _archived_of(self, uuids: list) -> list:
    """Of these uuids, the ones actually sitting at status='archived' now.

    A retirement list must be what the UPDATE *did*, not what the SELECT hoped
    it would do. `sleep`'s TTL and low-trust arms each SELECT candidates and
    then UPDATE with the same predicate as a separate statement, and extended
    the retire list from the SELECT — so anything that stopped qualifying in
    between (another connection setting `protected`, or moving the row out of
    'active') was still handed to `_retire_archived`, which deleted the Qdrant
    point of a row that is still live. Only `sync_check` would ever notice, and
    the record would be invisible to vector search until a rebuild.

    The duplicate arm in the same function already got this right — it gates
    `retired.append(uuid)` on `cur.rowcount` with a comment saying precisely
    why. Two of the three arms did not. 2026-08-26 ox-alpha round, bundle03 F3.

    Re-reading is the conservative direction: it can only shrink the list, and
    a point left behind is drift `sync_check` reports and `rebuild` clears,
    where a point deleted from under a live row is silent data loss.
    """
    if not uuids:
        return []
    out = []
    for i in range(0, len(uuids), 400):
        chunk = uuids[i:i + 400]
        rows = self._get_conn().execute(
            "SELECT uuid FROM memories WHERE status = 'archived' AND uuid IN ({})".format(
                ",".join("?" * len(chunk))), chunk).fetchall()
        out.extend(r[0] for r in rows)
    return out

def sleep(self, max_items: int = 100, min_age_hours: float = 24.0, archive_age_days: int = None) -> dict:
    """Archive stale/low-trust memories and enforce TTL.

    Lightweight pass — no LLM calls. Actual content merging is handled by compact().
    Steps:
    1. Archive TTL-expired records.
    2. Archive low-trust duplicates within the same data_type:data_id group.
    3. Archive very old (<0.3 trust, non-pinned) records.
    """
    now_ts = datetime.now(timezone.utc)
    archived = 0
    ttl_expired = 0
    # Every uuid this run archives, so their Qdrant points and supersession
    # back-pointers can be retired once at the end (see _retire_archived).
    retired = []

    # 0. TTL auto-expiry: archive records past their TTL
    #
    # `COALESCE(protected, 0) = 0` on BOTH statements. `protected` is an
    # explicit do-not-touch marker and every other archival path honours it —
    # the duplicate branch below, the low-trust branch below that, decay(),
    # and, since 0.8.113, compact()'s seed scan and both `review` doors. This
    # enumeration is the thing a reader checks a new sweep against, so it is
    # kept complete: compact() was missing for the whole life of this comment
    # and the comment is how that was eventually noticed (T709, T711).
    # This one did not, so a record carrying *both* `protected=true` and a
    # `ttl` (both first-class add()/update() parameters) was archived the
    # moment its ttl passed, vanished from every read path, and was then hard
    # deleted by purge(purge_archived=True) once past the 168h archived
    # threshold. That is precisely the two-step loss the comment on the
    # duplicate branch describes; the fix it documents was applied to the
    # branch where the loss was noticed and not to this one.
    #
    # The COUNT carries the guard too, not just the UPDATE: `ttl_expired` is
    # returned to the caller, and counting rows that were deliberately left
    # alone would report an archival that never happened.
    #
    # Narrower than the review suggested, deliberately: it flagged the absence
    # of a `priority` condition as well, but `priority` gates *decay*
    # exemption, and neither the duplicate branch nor decay() uses it to
    # qualify archival. Only `protected` is the established class guard here.
    #
    # The uuids are selected rather than counted so `_retire_archived` can drop
    # their Qdrant points; a bare COUNT left the vectors behind.
    # Validate every age BEFORE the first UPDATE, not between two of them.
    #
    # `min_age_hours` gates the duplicate arm below (a record that just
    # duplicated a higher-trust one gets a grace period rather than being
    # archived the instant it is written; the parameter was declared and never
    # read until 0.8.x). Its guard used to sit *after* the TTL arm's UPDATE,
    # so `sleep(min_age_hours=-1)` — the argument door this guard exists for,
    # per `_non_negative_age`'s own docstring — archived the TTL-expired rows,
    # then raised. Nothing commits and nothing rolls back on that path, and the
    # connection's implicit transaction is left open: the archival lands
    # whenever some later, unrelated write commits, and `_retire_archived`
    # never ran for those uuids, so SQLite says archived while Qdrant still
    # holds the points. A refusal that half-executes is worse than either
    # answer. 2026-08-26 external sweep, M-new-1.
    min_age_hours = _non_negative_age(min_age_hours, "min_age_hours")

    if archive_age_days is not None:
        archive_days = archive_age_days
    elif "low_trust_archive_days" in self._config:
        # Explicit env override (HLM_LOW_TRUST_ARCHIVE_DAYS) wins over the
        # documented cleanup.archive_age_days JSON key.
        archive_days = self._config["low_trust_archive_days"]
    else:
        # Previously fell straight to a hardcoded 365 here, silently
        # ignoring cleanup.archive_age_days whenever the env var wasn't set —
        # the documented config key had no effect.
        archive_days = self._get_cleanup_config()["archive_age_days"]
    # 2. Archive: very old low-trust (<0.3) non-pinned, non-protected
    #
    # Selected before the UPDATE for the same reason as the TTL branch: the
    # uuids are what `_retire_archived` needs to drop the Qdrant points.
    # Coerce before the sign check, because unparseable is a third shape.
    #
    # `_non_negative_age` documents "return value" for input it cannot parse —
    # written when every caller passed a number — so a malformed *config* value
    # sails through it and reaches `timedelta(days=...)`, which raises
    # `TypeError: unsupported type for timedelta days component: str`. 0.8.13
    # hardened the ARGUMENT door (`_do_sleep` refuses a malformed
    # archive_age_days) and this is the CONFIG door: `on_session_end` calls
    # `sleep()` with no arguments, so a typo in the profile's
    # `cleanup.archive_age_days` turned every session-end housekeeping pass into
    # an unhandled exception — after the TTL and duplicate arms had already run
    # their UPDATEs. One door guarded, the other not, again.
    #
    # Failing soft rather than raising, deliberately: this is unattended
    # teardown, `HLM_LOW_TRUST_ARCHIVE_DAYS` already fails soft through
    # `_apply_env_overrides`'s int() coercion, and refusing to run housekeeping
    # over a config typo is worse than running it on the documented default.
    # The argument door still refuses loudly, because there a human is asking.
    # 2026-08-26 ox-alpha round 2, bundle02 F2.
    try:
        archive_days = float(archive_days)
    except (TypeError, ValueError):
        # A hard default, NOT `_get_cleanup_config()["archive_age_days"]` — that
        # is where the malformed value came from, so falling back to it returns
        # the same string and crashes one line later. (First draft did exactly
        # that; the probe caught it.)
        _bad, archive_days = archive_days, 365.0
        logger.warning(
            "sleep: archive_age_days=%r is not a number — using %s. "
            "Check cleanup.archive_age_days in this profile's config.",
            _bad, archive_days)
    archive_days = _non_negative_age(archive_days, "archive_age_days")

    ttl_rows = self._get_conn().execute("""
        SELECT uuid FROM memories
        WHERE status = 'active' AND ttl IS NOT NULL AND ttl != '' AND ttl < ?
          AND COALESCE(protected, 0) = 0
    """, (now_ts.isoformat(),)).fetchall()
    ttl_uuids = [r[0] for r in ttl_rows]
    ttl_expired = len(ttl_uuids)
    if ttl_expired > 0:
        self._get_conn().execute("""
            UPDATE memories SET status = 'archived', updated_at = ?
            WHERE status = 'active' AND ttl IS NOT NULL AND ttl != '' AND ttl < ?
              AND COALESCE(protected, 0) = 0
        """, (self._now(), now_ts.isoformat()))
        # Count what the UPDATE did, not what the SELECT hoped for.
        #
        # The comment above already requires this — "the COUNT carries the
        # guard too ... counting rows that were deliberately left alone would
        # report an archival that never happened" — and 0.8.16 half-applied it:
        # it made the RETIRE list a post-UPDATE re-read and left `ttl_expired`
        # at the SELECT's length, so the two disagreed by construction in
        # exactly the race the re-read was added for. Found by the 2026-08-26
        # round-2 review (bundle01 F15), against a fix of mine from the day
        # before.
        _ttl_archived = _archived_of(self, ttl_uuids)
        retired.extend(_ttl_archived)
        ttl_expired = len(_ttl_archived)

    # 1. Archive low-trust duplicates within the same data_type:data_id group
    # Group by data_type + data_id, keep the highest trust_score record, archive rest
    groups = self._get_conn().execute("""
        SELECT data_type, data_id FROM memories
        WHERE status = 'active' AND data_type IS NOT NULL AND data_id IS NOT NULL
        GROUP BY data_type, data_id HAVING COUNT(*) > 1
        LIMIT ?
    """, (max_items,)).fetchall()

    min_age_cutoff = (now_ts - timedelta(hours=min_age_hours)).isoformat()
    for (dt, di) in groups:
        # CAST(priority AS INTEGER): every write path validates priority as
        # of 0.7.49, but a row written before that (or by any future path
        # that forgets to) can still carry a non-numeric value, and SQLite
        # sorts TEXT above every INTEGER — so an uncast `priority DESC` put
        # the corrupted row FIRST, making it this group's "winner" (kept)
        # rather than a candidate for archival. CAST('high' AS INTEGER) is 0
        # in SQLite, which sorts last here — the safe direction, since a
        # value this function cannot trust should lose the tiebreak, not win
        # it. Verified against a live SQLite table before this fix: a tied
        # trust_score group sorted the bad-priority row first.
        members = self._get_conn().execute("""
            SELECT uuid, trust_score, priority
            FROM memories WHERE data_type = ? AND data_id = ? AND status = 'active'
            ORDER BY trust_score DESC, CAST(priority AS INTEGER) DESC
        """, (dt, di)).fetchall()
        if len(members) < 2:
            continue
        # Keep best (first), archive the rest if they have low trust and are
        # old enough (min_age_hours grace period).
        archive_trust = self._get_cleanup_config()["duplicate_archive_trust"]
        for uuid, trust, prio in members[1:]:
            if trust < archive_trust:
                # protected = 0: `protected` is an explicit do-not-touch
                # marker, separate from priority (v10 migration). decay()
                # honours it; archival did not, so a record the user had
                # protected was archived here and then hard-deleted by
                # purge(purge_archived=True) a week later.
                # `AND status = 'active'` closes a cross-process race. The
                # `members` SELECT above ran in this transaction, but the
                # plugin and the MCP server are separate processes on one
                # SQLite file — so another writer can soft-delete or archive
                # the row between the read and this write, and without the
                # status test we would set `status='archived'` on a row that
                # is now `deleted`, resurrecting it as archived.
                #
                # The rest of this finding did not reproduce: `retired` is
                # already gated on `cur.rowcount` here (see below), and the
                # TTL arm's SELECT and UPDATE carry identical predicates in
                # one transaction, so its unconditional `extend` is exact.
                # 2026-08-24 audit, finding 4 (partly refuted).
                cur = self._get_conn().execute(
                    "UPDATE memories SET status = 'archived', updated_at = ? "
                    "WHERE uuid = ? AND status = 'active' AND updated_at < ? "
                    "AND COALESCE(protected, 0) = 0",
                    (self._now(), uuid, min_age_cutoff))
                archived += cur.rowcount
                if cur.rowcount:
                    # Only when the row actually moved: the guards above can
                    # decline it, and retiring a point whose row is still
                    # active would delete a live record's vector.
                    retired.append(uuid)

    old_cutoff = (now_ts - timedelta(days=archive_days)).isoformat()
    # `LIMIT ?` with `max_items`, matching the duplicate branch above it —
    # this arm had none, so a corpus with a large old/low-trust tail archived
    # in one unbounded UPDATE regardless of what the caller asked for
    # (2026-08-23 review round 7 maintenance F8). The UPDATE below
    # now targets exactly the uuids this SELECT found (`uuid IN (?)`) rather
    # than re-running the same unbounded filter — without that, bounding the
    # SELECT alone would still let the UPDATE archive every matching row.
    low_trust_uuids = [r[0] for r in self._get_conn().execute("""
        SELECT uuid FROM memories
        WHERE status = 'active'
          AND created_at < ?
          AND trust_score < 0.3
          AND priority < 2
          AND COALESCE(protected, 0) = 0
        LIMIT ?
    """, (old_cutoff, max_items)).fetchall()]
    if low_trust_uuids:
        _lt_placeholders = ",".join("?" for _ in low_trust_uuids)
        self._get_conn().execute(f"""
            UPDATE memories SET status = 'archived', updated_at = ?
            WHERE status = 'active'
              AND COALESCE(protected, 0) = 0
              AND uuid IN ({_lt_placeholders})
        """, (self._now(), *low_trust_uuids))
        # `archived` counted only the duplicate branch, so sleep() reported
        # fewer archived records than it archived — and 0.7.77 added the
        # retire step to this same block without noticing the counter was
        # missing (2026-08-22 ox-alpha maintenance review, F6).
        #
        # Count the re-read, not the SELECT. This arm took the `_archived_of`
        # re-read for its *retire* list and left the *counter* on the
        # pre-UPDATE list, so the two disagreed by construction in exactly the
        # race the re-read exists for — the same half-application 0.8.16 made
        # on the TTL arm and 0.8.21's round fixed there. Third member of the
        # set; the duplicate arm has gated on `cur.rowcount` all along.
        # 2026-08-26 external sweep, M-new-2.
        _low_archived = _archived_of(self, low_trust_uuids)
        archived += len(_low_archived)
        retired.extend(_low_archived)

    self._get_conn().commit()
    # After the commit: SQLite is the source of truth, so the rows are durably
    # archived before anything touches the derived index.
    _retire_archived(self, retired)
    self._get_conn().commit()
    return {"archived": archived, "ttl_expired": ttl_expired,
            "retired_points": len(retired)}

# ---- Compaction (content merge) --------------------------------------


def compact(self, similarity_threshold=_C.COMPACT_SIMILARITY_DEFAULT, topic=None, max_groups=50,
            execute=False):
    """Merge duplicate/overlapping memories via LLM.

    Args:
        similarity_threshold: Minimum vector similarity (0.8-0.99, default 0.90)
        topic: Optional topic filter
        max_groups: Maximum groups to process (default 50)
        execute: Apply the merges. **Defaults to False** — without it this
            reports what it would do and changes nothing.

    Returns:
        execute=False: dict with status='dry_run' and the proposed groups.
        execute=True:  dict with groups_merged, records_compacted, details.

    Why the default is dry-run: compaction soft-deletes every original record
    and replaces it with LLM-generated prose. It ran unattended from the MCP
    maintenance endpoint and, until the column-mapping fix, silently rewrote
    scope, dropped `sensitivity`/`protected` and mislabelled provenance on
    everything it touched. The flag lives here rather than in the provider
    handler because mcp_server.py calls this method directly — a default set
    one layer up would not cover it.
    """
    if not self._qdrant:
        logger.warning("compact: Qdrant not available")
        return {"error": "Qdrant not available"}

    groups = self.find_duplicate_groups(similarity_threshold, topic, max_groups)
    if not groups:
        return {"groups_merged": 0, "records_compacted": 0, "details": [],
                "executed": bool(execute)}

    if not execute:
        preview = []
        for group in groups:
            uuids = group["uuids"][:10]
            # source added so the caller can fence the content snippet below —
            # this is the default path (execute=False), so every first
            # invocation returned raw stored content with no way to tell it
            # from HLM's own text.
            # `content_chars` and `merge_sees_all` exist so the operator can
            # tell, *before* execute=true, whether the merge will be made from
            # a partial view. The preview shows 80 characters and the merge
            # prompt shows MERGE_CONTENT_CHARS; a record longer than that is
            # merged from text nobody in the loop has read, and the merge is
            # the record that survives.
            rows = self._get_conn().execute(
                "SELECT uuid, substr(content, 1, 80), sensitivity, protected, scope, "
                "source, LENGTH(content) FROM memories WHERE uuid IN ({})".format(
                    ",".join("?" for _ in uuids)), uuids).fetchall()
            preview.append({
                "similarity": round(group.get("similarity", 0.0), 3),
                "would_merge": [
                    {"uuid": r[0], "content": r[1], "sensitivity": r[2],
                     "protected": bool(r[3]), "scope": r[4], "source": r[5],
                     "content_chars": r[6],
                     "merge_sees_all": (r[6] or 0) <= _C.MERGE_CONTENT_CHARS} for r in rows],
            })
        logger.info(
            "compact: dry run — %d group(s) would be merged (execute=False)", len(groups))
        return {
            "status": "dry_run",
            "executed": False,
            "groups_found": len(groups),
            "records_affected": sum(len(g["would_merge"]) for g in preview),
            "groups": preview,
            "note": "Nothing was changed. Re-run with execute=True to apply. "
                    "Merging soft-deletes every original record.",
        }

    merged = 0
    details = []
    for group in groups:
        result = self._compact_group(group)
        if result:
            # Count what _compact_group actually merged, not the pre-cap group
            # size — it caps a group at 10 records, so the old arithmetic
            # over-reported every larger group.
            merged += max(len(result.get("original_uuids", [])) - 1, 0)
            details.append(result)

    return {"groups_merged": len(details), "records_compacted": merged,
            "details": details, "executed": True}


def find_duplicate_groups(self, similarity_threshold, topic=None, max_groups=50):
    """Find groups of similar records via Qdrant native search.

    Uses iterative Qdrant queries with score_threshold instead of O(N^2) pairwise.

    **Known blind spot, deliberate:** a Qdrant hit is only added to a group
    when its uuid is also in `active_uuids` — the same `scan_limit`-bounded,
    most-recently-updated SQLite set the seeds themselves come from. A genuine
    near-duplicate of a seed record is silently excluded from its group if
    that duplicate happens to be older than the scan window, even though
    Qdrant found it. Not fixed: widening the check would mean re-fetching an
    unbounded set of candidate records from SQLite per seed, defeating the
    scan_limit bound this function exists to enforce (see its comment above).
    On a corpus with duplicates spread across a long update history rather
    than clustered recently, some groups will go undetected until those older
    records are themselves touched and re-enter the window.
    2026-08-23 review round 7 maintenance F6.
    """
    if not self._qdrant:
        return []

    processed = set()
    groups = []

    # Bound the seed scan. max_groups caps how many groups are *collected*,
    # not how many records are examined — so in the healthy steady state
    # (few duplicates) the break was never reached and this issued one
    # Qdrant query plus one SQLite vector read for every active record:
    # 2,000+ sequential round trips per compact() call at the documented
    # vault scale. Seed from the most recently updated records, which is
    # where new duplicates appear.
    scan_limit = max(int(max_groups) * 20, 200)
    params = []
    # superseded_by IS NULL, like count(), list(), seed_overview(), graph_health()
    # and the dedup SQLite arm. Compaction was the one scan that omitted it, and
    # it is the worst place to: supersession exists because an old fact was
    # replaced by a near-identical one, which is precisely the shape compact()
    # merges. Merging a superseded record with its own successor resurrects the
    # value the chain was built to hide, then soft-deletes both originals and
    # destroys the chain.
    # `COALESCE(protected, 0) = 0`, for the reason sleep()'s TTL arm states at
    # the top of this file: `protected` is an explicit do-not-touch marker and
    # every *automatic* maintenance path honours it. That comment enumerates
    # the paths — sleep()'s three branches and decay() — and compaction was
    # not among them, so this was the last bulk-retirement scan without the
    # guard. Driven before the fix: two near-duplicate records, one written
    # with `protected=true`, seeded one group and `compact(execute=true)` set
    # `status='deleted'` on BOTH, folding the pinned record's content into a
    # new uuid that merely inherits the flag (`merged_protected`). The user's
    # record is gone and a model's paraphrase of it carries the marker. It is
    # the guarded side of the contract in docs/reference.md: compact names a
    # threshold and a group cap, never a uuid, so no caller "said so directly".
    # Guarding the seed scan covers membership as well as seeding — a Qdrant
    # hit only joins a group when its uuid is in this result set.
    # 0.8.113, T711; found one round after T709 fixed the same class in
    # `review`, by the reviewer that had just been shown that fix.
    sql = ("SELECT uuid, data_type, source FROM memories "
           "WHERE status='active' AND superseded_by IS NULL AND priority < 2 "
           "AND COALESCE(protected, 0) = 0")
    if topic:
        sql += " AND topic=?"
        params.append(topic)
    sql += " ORDER BY updated_at DESC LIMIT ?"
    params.append(scan_limit)
    records = self._get_conn().execute(sql, params).fetchall()
    if len(records) >= scan_limit:
        logger.info(
            "compact: seed scan capped at %d most-recently-updated records "
            "(raise max_groups to widen)", scan_limit)

    # Build lookup: uuid -> (data_type, source)
    record_map = {r[0]: (r[1], r[2]) for r in records}

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

            # Build group from results — only include records that are still active
            # and share the same source (preserves cross-source corroboration evidence)
            active_uuids = {r[0] for r in records}
            seed_source = record_map[uuid][1]
            group_uuids = []
            for hit in results.points:
                hit_uuid = _from_qdrant_id(hit.id)
                hit_source = record_map.get(hit_uuid, (None, None))[1]
                if (hit_uuid != uuid and hit_uuid not in processed
                        and hit_uuid in active_uuids and hit_source == seed_source):
                    group_uuids.append(hit_uuid)

            if group_uuids:
                group_uuids.insert(0, uuid)  # Include the seed record
                processed.add(uuid)
                processed.update(group_uuids)
                # `results.points[0]` is the seed itself — a vector search
                # seeded by a record's own vector returns that record first, at
                # ~1.0 — and the loop above deliberately skips it when building
                # the group (`hit_uuid != uuid`). So every group reported a
                # similarity of about 1.0 regardless of how alike its members
                # actually were, on the dry-run preview an operator reads
                # *before* setting execute=true. Report the best score among
                # the records that are actually in the group.
                # 2026-08-24 audit, minor 1.
                _member_scores = [h.score for h in results.points
                                  if _from_qdrant_id(h.id) in set(group_uuids[1:])]
                groups.append({
                    "uuids": group_uuids,
                    "similarity": max(_member_scores) if _member_scores else similarity_threshold
                })

                if len(groups) >= max_groups:
                    break

        except Exception as e:
            logger.warning("compact: search failed for %s: %s", uuid, e)
            continue

    return groups


def _compact_group(self, group):
    """Merge a group of similar records via LLM.

    Returns dict with merge details, or None on failure.
    """
    uuids = group["uuids"]

    # Cap to max 10 records per merge
    if len(uuids) > 10:
        uuids = uuids[:10]

    # Fetch full records (include source for compaction metadata inheritance)
    records = []
    placeholders = ",".join(["?" for _ in uuids])
    # scope/data_id/session_name/sensitivity/ttl/protected are selected because
    # the merged record inherits them below. The old SELECT omitted `scope`
    # while the INSERT read records[0].get("scope", "personal") — a key that
    # could never be present — so every merge silently rewrote scope to
    # 'personal'. sensitivity and protected are worse than cosmetic: dropping
    # them downgrades a redaction-marked record to sensitivity 0 (prefetch
    # then auto-injects it) and discards the user's explicit do-not-touch flag.
    rows = self._get_conn().execute(
        "SELECT uuid, content, summary, keywords, topic, data_type, trust_score, "
        "scope, created_at, source, data_id, session_name, sensitivity, ttl, "
        "protected, source_url, backlinks FROM memories WHERE uuid IN ({}) "
        # Deterministic order. `WHERE uuid IN (...)` has none of its own, so the
        # merge picked its inherited source_url and its backlink order out of
        # whatever sequence SQLite happened to return — stable within a run,
        # different between them. Oldest-first also makes the inherited
        # source_url the earliest attribution rather than an arbitrary one.
        "ORDER BY created_at".format(placeholders),
        uuids
    ).fetchall()

    for row in rows:
        records.append({
            "uuid": row[0], "content": row[1], "summary": row[2],
            "keywords": row[3], "topic": row[4], "data_type": row[5],
            "trust_score": row[6], "scope": row[7], "created_at": row[8],
            "source": row[9], "data_id": row[10], "session_name": row[11],
            "sensitivity": row[12], "ttl": row[13], "protected": row[14],
            "source_url": row[15], "backlinks": row[16],
        })

    if len(records) < 2:
        return None

    # Call LLM for merge
    try:
        merged = self._llm_merge(records)
    except Exception as e:
        logger.warning("compact: LLM merge failed: %s", e)
        return None

    if not merged or "content" not in merged:
        return None

    # The merge output bypasses every write-path length guard — it goes
    # straight into a raw INSERT below, not add()/update(), so
    # _check_fts_field_bounds never runs on it. Nothing bounds how much text
    # an LLM merge of several records can return, and the merged content is
    # written into the same FTS5-indexed column content's own guard exists to
    # protect. Failing this one group rather than raising, consistent with
    # the LLM-failure handling immediately above: a bad merge should not abort
    # the rest of compact()'s sweep. 2026-08-23 recovered finding F-NN.
    if len(merged["content"]) > MAX_CONTENT_CHARS:
        logger.warning(
            "compact: merged content for group %s exceeds max length (%d > %d chars) — "
            "skipping this group", uuids, len(merged["content"]), MAX_CONTENT_CHARS)
        return None

    # Create merged record
    new_uuid = uuid_mod.uuid4().hex
    max_trust = max(r["trust_score"] for r in records)
    n_records = len(records)
    # Cap total boost so low-trust noise can't be amplified into "trusted facts"
    # Each record adds at most 0.05, but total boost capped at 0.2 max
    max_boost = min(0.2, 0.05 * (n_records - 1))
    corroboration_boost = min(1.0, max_trust + max_boost)
    first_observed = min(r["created_at"] for r in records)

    # Most common data_type
    from collections import Counter
    data_types = [r["data_type"] for r in records if r["data_type"]]
    dominant_type = Counter(data_types).most_common(1)[0][0] if data_types else "CUSTOM"

    # Inherited classification/protection fields. Merging must never *lower*
    # a guarantee the originals carried, so sensitivity takes the maximum and
    # protected is a logical OR; a TTL is only inherited when every parent had
    # one (otherwise the merge would put an expiry on a permanent fact).
    _scopes = [r["scope"] for r in records if r["scope"]]
    merged_scope = Counter(_scopes).most_common(1)[0][0] if _scopes else "personal"
    _data_ids = [r["data_id"] for r in records if r["data_id"]]
    merged_data_id = Counter(_data_ids).most_common(1)[0][0] if _data_ids else None
    _sessions = [r["session_name"] for r in records if r["session_name"]]
    merged_session = Counter(_sessions).most_common(1)[0][0] if _sessions else None
    merged_sensitivity = max((int(r["sensitivity"] or 0) for r in records), default=0)
    merged_protected = 1 if any(r["protected"] for r in records) else 0
    # Provenance the merge used to drop on the floor. source_url is where the
    # content came from and backlinks are its references — losing them leaves a
    # merged record that cannot be traced back to anything, which matters most
    # for exactly the records worth merging (repeated notes about one source).
    # First non-empty wins for the url (records are ordered oldest-first, so
    # this is the earliest attribution); backlinks are unioned, order-preserved.
    merged_source_url = next((r["source_url"] for r in records if r.get("source_url")), None)
    _seen_links, merged_backlinks = set(), []
    for r in records:
        raw = r.get("backlinks")
        if not raw:
            continue
        try:
            links = json.loads(raw) if isinstance(raw, str) else raw
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(links, list):
            for link in links:
                if link not in _seen_links:
                    _seen_links.add(link)
                    merged_backlinks.append(link)
    _ttls = [r["ttl"] for r in records if r["ttl"]]
    merged_ttl = max(_ttls) if len(_ttls) == len(records) else None

    # Source inheritance: security-first OR (obsidian always wins)
    # When merging records from untrusted sources, use a distinct source
    # that is NOT in _SELF_AUTHORED_SOURCES so the merged content goes
    # through _wrap_untrusted on retrieval — the consolidation LLM blend
    # could contain injected content from any parent record.
    parent_sources = [r["source"] for r in records if r["source"]]
    # Use the canonical allowlist, not a copy. This was a hardcoded duplicate
    # and had already drifted: it omitted "hlm-consolidated", so re-compacting
    # an already-merged record treated it as untrusted and downgraded it to
    # "consolidated-untrusted". It also still listed "extraction" after that
    # was deliberately removed from the real set, which would have kept the
    # laundering path open through compaction alone.
    has_untrusted = any(s not in _SELF_AUTHORED_SOURCES for s in parent_sources)
    if "obsidian" in parent_sources:
        merged_source = "obsidian"
    elif has_untrusted:
        merged_source = "consolidated-untrusted"
    else:
        merged_source = "hlm-consolidated"

    # Add the merged record (include source for obsidian prompt injection defense)
    # priority=1 for ENV-DATA/SYSTEM (half-rate decay per decay()'s
    # priority_multipliers, NOT immunity — only priority>=3 or protected=True
    # are decay-exempt), 0 for others
    merged_priority = 1 if dominant_type in ("ENV-DATA", "SYSTEM") else 0
    try:
        # Wrap each group's multi-step merge in a savepoint — a failure
        # mid-merge should not leave a dangling INSERT committed by the
        # next successful group's commit.
        #
        # All SQLite work happens first and commits before any Qdrant call.
        # Previously the embedding HTTP call and N+1 Qdrant round trips ran
        # inside this transaction, holding the write lock across the lot —
        # times max_groups (50) — and blocking every other Hermes session
        # writing to the profile. Qdrant is reconciled by sync_check/rebuild
        # if a call fails, which is already the documented recovery path.
        self._get_conn().execute("SAVEPOINT compact_merge")
        self._get_conn().execute("""
            INSERT INTO memories (uuid, content, summary, keywords, topic, data_type,
                scope, data_id, session_name, sensitivity, ttl, protected,
                source, source_url, backlinks, created_at, updated_at,
                trust_score, first_observed_at, priority, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            new_uuid,
            merged["content"],
            merged.get("summary") or merged["content"][:100],
            json.dumps(merged.get("keywords", [])),
            merged.get("topic") or records[0]["topic"],
            dominant_type,
            merged_scope,
            merged_data_id,
            merged_session,
            merged_sensitivity,
            _safe_ttl(self, merged_ttl),
            merged_protected,
            merged_source,
            merged_source_url,
            json.dumps(merged_backlinks) if merged_backlinks else None,
            self._now(),
            self._now(),
            corroboration_boost,
            first_observed,
            merged_priority,
            json.dumps({"compacted_from": uuids, "first_observed_at": first_observed})
        ))

        # Soft-delete originals
        for r in records:
            self._get_conn().execute(
                "UPDATE memories SET status='deleted', compacted_into=?, updated_at=? WHERE uuid=?",
                (new_uuid, self._now(), r["uuid"]))
            # Release any supersession pointer aimed at a record we just
            # retired, so nothing is left active-but-invisible.
            self._get_conn().execute(
                "UPDATE memories SET superseded_by = NULL, superseded_at = NULL, updated_at = ? "
                "WHERE superseded_by = ?", (self._now(), r["uuid"]))

        self._get_conn().commit()  # releases savepoint + commits group
    except Exception as e:
        try:
            self._get_conn().execute("ROLLBACK TO compact_merge")
            self._get_conn().rollback()
        except Exception:
            pass  # rollback itself failed — already doomed
        logger.warning("compact: insert failed (rolled back): %s", e)
        return None

    # ---- Qdrant reconciliation, outside the write transaction ----
    _payload = {
        "data_type": dominant_type, "topic": merged.get("topic") or records[0]["topic"],
        "status": "active", "trust_score": corroboration_boost,
        "created_at": self._now(), "first_observed_at": first_observed,
    }
    # data_id/session_name are indexed payload fields that _layer0 and
    # _check_duplicate filter on — omitting them made the merged record
    # invisible to any data_id-scoped vector search.
    if merged_data_id:
        _payload["data_id"] = merged_data_id
    if merged_session:
        _payload["session_name"] = merged_session
    if self._profile_name:
        # `new_uuid` is freshly generated for the merged record, so no other
        # profile can already hold that point — a claim lookup would always
        # return just us. Plain assignment on purpose.
        _payload["profile_name"] = self._profile_name
    self._embed_and_upsert(new_uuid, merged["content"], _payload)

    if self._qdrant:
        # Sixth member of the same class — see `_drop_points_everywhere`.
        # Compaction retires the records it merged, so this is a retirement
        # path like `delete`, `purge`, `test_cleanup`, `resolve_conflicts` and
        # `_retire_archived`. Deleting from each record's *current* data_type
        # collection alone strands the point of any record whose type changed
        # since it was indexed, and the merged record then competes with its
        # own retired parents in layer 0. Found by the 2026-08-26 ox-alpha
        # round (bundle03 F12).
        try:
            self._drop_points_everywhere([r["uuid"] for r in records])
        except Exception as e:
            logger.warning("_compact_group: Qdrant delete failed for %d record(s): %s",
                           len(records), e)

    return {
        "merged_uuid": new_uuid,
        "original_uuids": uuids,
        "similarity": group.get("similarity", 0),
        "corroboration_boost": corroboration_boost
    }


def _is_future_timestamp(value, now_iso: str) -> bool:
    """Is `value` an instant later than `now_iso`? Compared as instants.

    Both sides were compared as **strings**. `_now()` always renders UTC as
    `+00:00`, so the comparison is only correct for values that happen to use
    that same offset — and `_valid_timestamp` accepts any offset
    `datetime.fromisoformat` parses, because an export/import round trip has to
    preserve real timestamps. The ordering of two ISO strings with different
    offsets is the ordering of their *local wall clocks*, not their instants,
    so it failed in both directions:

      * `2026-08-26T09:00:00+08:00` is 01:00Z — in the past at 02:15Z — but
        sorts after `2026-08-26T02:15:00+00:00`, so import "repaired" a
        perfectly good timestamp to `now()`, destroying the value the round
        trip exists to preserve;
      * `2026-08-27T20:00:00-08:00` is 2026-08-28T04:00Z — genuinely in the
        future — but sorts before `2026-08-27T21:00:00+00:00`, so the
        age-archival evasion this guard exists to close survived by writing
        the same instant with a negative offset.

    A naive value (no offset at all) is read as UTC, which is what every other
    reader of these columns already assumes.
    2026-08-26 external sweep, M-new-3.
    """
    left, right = _parse_timestamp(value), _parse_timestamp(now_iso)
    if left is None or right is None:
        # Not comparable: `_valid_timestamp` is the caller's gate for that, and
        # answering "future" here would rewrite a value on a parse failure.
        return False
    return left > right


def _parse_timestamp(value):
    """ISO-8601 -> aware UTC datetime, or None. Same rule as `_valid_timestamp`."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _non_negative_age(value, label):
    """Ages are durations, so a negative one inverts the guard it computes.

    Every cutoff here is `now - timedelta(hours=age)`. A negative age puts the
    cutoff in the *future*, so `created_at < cutoff` matches every row —
    `purge(min_age_hours=-1)` permanently deletes every soft-deleted and
    archived record inside the grace period the parameter exists to provide,
    and `sleep(min_age_hours=-1)` archives records written seconds ago.

    Neither front end guards it: the plugin's `_parse_age` float-coerces and
    forwards, MCP declares `min_age_hours: int | None` and forwards, and the
    `memory_write` maintenance door forwards too. A 0.7.3 changelog entry
    refuting this class is stale — it describes the old SQL-modifier shape
    (`'-%d hours'`), not the Python `timedelta` this became.

    **Raised, not clamped.** Clamping to 0 was written first and does not
    protect anything: 0 means "no grace period", so `purge(min_age_hours=-1)`
    would still delete the record it was about to delete — the cutoff stops
    being *inverted* but the grace period is gone either way. The test caught
    that, which is why it is worth saying here.

    Raising is safe on the automatic path because nothing on it passes a
    negative: teardown maintenance uses config values, and the plugin wraps
    that whole block in `try/except` (`__init__.py`'s `on_session_end`). What
    raising does reach is the caller that typed one — including the
    unauthenticated MCP door, where "delete everything inside its grace
    period" is the damage this guard exists to prevent.
    2026-08-24 audit, M2.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return value
    if v < 0:
        raise ValueError(
            f"{label}={value!r} is negative. Ages are durations: a negative one "
            f"puts the cutoff in the future, so every row matches and the grace "
            f"period this parameter exists to provide is inverted. Use 0 for no "
            f"grace period.")
    return value


def purge(self, purge_deleted: bool = True, purge_archived: bool = True,
          min_age_hours: Optional[float] = None,
          min_age_hours_deleted: Optional[float] = None,
          min_age_hours_archived: Optional[float] = None,
          vacuum: bool = True) -> dict:
    """Permanently delete soft-deleted and/or archived records.

    Args:
        purge_deleted: Delete records with status='deleted'
        purge_archived: Delete records with status='archived'
        min_age_hours: Minimum age in hours before purging both. Defaults:
            24h for deleted, 168h (7d) for archived.
        min_age_hours_deleted: Override just the deleted threshold.
        min_age_hours_archived: Override just the archived threshold.
        vacuum: Vacuum SQLite after purge to reclaim disk space.

    Returns:
        dict with deleted_count, archived_count, qdrant_removed,
        space_freed_bytes, vacuumed
    """
    now_ts = datetime.now(timezone.utc)
    cfg = self._get_cleanup_config()

    # Separate thresholds per category. `min_age_hours` is the legacy shared
    # knob kept for backward compat; it sets BOTH. The category-specific
    # overrides take priority. Defaults come from cleanup config
    # (purge_deleted_hours/purge_archived_hours) rather than hardcoded
    # literals, so the documented config keys actually take effect for a
    # direct purge() call, not just the automatic maintenance path.
    deleted_age = min_age_hours_deleted if min_age_hours_deleted is not None else (
        min_age_hours if min_age_hours is not None else cfg.get("purge_deleted_hours", 24.0))
    archived_age = min_age_hours_archived if min_age_hours_archived is not None else (
        min_age_hours if min_age_hours is not None else cfg.get("purge_archived_hours", 168.0))

    # **`purge` keys on `updated_at`, and only for rows already deleted or
    # archived.** `created_at` never reaches this function. Two independent
    # reviewers (2026-08-24 audit finding 5, 2026-08-25 bundle01 F1) have filed
    # "an imported far-past created_at makes a record instantly purge-eligible";
    # it does not, because an *active* row is not selected here at all. What a
    # far-past created_at does reach is `sleep()`'s low-trust archive arm, and
    # from there a record becomes purge-eligible only after the archive grace
    # period. Stated here so a third reviewer reads it before filing it again.
    deleted_age = _non_negative_age(deleted_age, "min_age_hours_deleted")
    archived_age = _non_negative_age(archived_age, "min_age_hours_archived")
    deleted_cutoff = (now_ts - timedelta(hours=deleted_age)).isoformat()
    archived_cutoff = (now_ts - timedelta(hours=archived_age)).isoformat()

    deleted_count = 0
    archived_count = 0
    qdrant_removed = 0

    # 1. Purge soft-deleted records
    if purge_deleted:
        to_delete = self._get_conn().execute("""
            SELECT uuid, data_type FROM memories
            WHERE status = 'deleted' AND updated_at < ?
        """, (deleted_cutoff,)).fetchall()

        for uuid, data_type in to_delete:
            # Remove from Qdrant. Guarded on self._qdrant like every other
            # call site — with HLM_QDRANT_ENABLED=false (a supported mode)
            # this raised AttributeError once per record and logged a
            # WARNING for each, burying real Qdrant errors.
            #
            # Sweeps **every** collection, via the shared helper. purge held
            # the last two single-collection deletes of that class — after
            # 0.7.93 (a no-op), 0.8.0 (delete, test_cleanup) and 0.8.4
            # (resolve_conflicts) — and it is the worst place to leave one:
            # this is the *hard* delete, so a point stranded under a data_type
            # the record no longer has outlives the SQLite row that could
            # explain it, and only a full rebuild's orphan pass ever clears it.
            # 2026-08-25 xhigh round, bundle03 (F1).
            if self._qdrant:
                try:
                    qdrant_removed += self._drop_points_everywhere(uuid)
                except Exception as e:
                    logger.warning("purge: Qdrant delete failed for uuid=%s: %s", uuid[:8] if isinstance(uuid, str) else uuid, e)
            deleted_count += 1

        # Release supersession back-pointers before the rows vanish —
        # otherwise the superseded record stays active but permanently
        # invisible to every read path (see delete()).
        #
        # SAVEPOINT around the release+delete pair: neither statement was
        # wrapped in try/except, so an exception between them (a malformed
        # SQL restriction, disk full — anything, not just a crash) propagated
        # uncaught with the release already staged and the DELETE never run,
        # leaving this connection holding an implicit open transaction —
        # neither committed nor rolled back. The same shape sleep()'s
        # min_age_hours guard was placed to prevent for that function's own
        # UPDATE ("nothing commits and nothing rolls back on that path... the
        # archival lands whenever some later, unrelated write commits").
        # Rollback-to-savepoint on failure and re-raise, matching
        # _compact_group's precedent for the same risk. 2026-08-23 review
        # round 6 maintenance F8.
        self._get_conn().execute("SAVEPOINT purge_deleted_supersession")
        try:
            self._get_conn().execute(
                "UPDATE memories SET superseded_by = NULL, superseded_at = NULL "
                "WHERE superseded_by IN (SELECT uuid FROM memories "
                "                        WHERE status = 'deleted' AND updated_at < ?)",
                (deleted_cutoff,))
            self._get_conn().execute("DELETE FROM memories WHERE status = 'deleted' AND updated_at < ?",
                               (deleted_cutoff,))
        except Exception:
            self._get_conn().execute("ROLLBACK TO purge_deleted_supersession")
            raise
        else:
            self._get_conn().execute("RELEASE purge_deleted_supersession")

    # 2. Purge archived records
    if purge_archived:
        to_archive = self._get_conn().execute("""
            SELECT uuid, data_type FROM memories
            WHERE status = 'archived' AND updated_at < ?
        """, (archived_cutoff,)).fetchall()

        for uuid, data_type in to_archive:
            # Remove from Qdrant (guarded — see the deleted branch above).
            if self._qdrant:
                try:
                    qdrant_removed += self._drop_points_everywhere(uuid)
                except Exception as e:
                    logger.warning("purge: Qdrant delete failed for archived uuid=%s: %s", uuid[:8] if isinstance(uuid, str) else uuid, e)
            archived_count += 1

        # Same SAVEPOINT reasoning as the deleted branch above.
        self._get_conn().execute("SAVEPOINT purge_archived_supersession")
        try:
            self._get_conn().execute(
                "UPDATE memories SET superseded_by = NULL, superseded_at = NULL "
                "WHERE superseded_by IN (SELECT uuid FROM memories "
                "                        WHERE status = 'archived' AND updated_at < ?)",
                (archived_cutoff,))
            self._get_conn().execute("DELETE FROM memories WHERE status = 'archived' AND updated_at < ?",
                               (archived_cutoff,))
        except Exception:
            self._get_conn().execute("ROLLBACK TO purge_archived_supersession")
            raise
        else:
            self._get_conn().execute("RELEASE purge_archived_supersession")

    # 3. Commit and get space freed estimate
    self._get_conn().commit()

    # Estimate space freed (page count before vs after)
    space_freed = 0
    vacuum_ran = False
    if vacuum and (deleted_count > 0 or archived_count > 0):
        # Only VACUUM if free pages exceed threshold
        cfg = self._get_cleanup_config()
        free_pct_threshold = cfg.get("vacuum_free_page_pct", 20)
        total_pages = self._get_conn().execute("PRAGMA page_count").fetchone()[0]
        free_pages = self._get_conn().execute("PRAGMA freelist_count").fetchone()[0]
        free_pct = (free_pages / total_pages * 100) if total_pages > 0 else 0

        if free_pct >= free_pct_threshold:
            page_size = self._get_conn().execute("PRAGMA page_size").fetchone()[0]
            before = total_pages
            try:
                self._get_conn().execute("VACUUM")
                vacuum_ran = True
            except Exception as e:
                logger.warning("Vacuum failed: %s", e)
            after = self._get_conn().execute("PRAGMA page_count").fetchone()[0]
            space_freed = (before - after) * page_size
            logger.debug("Vacuum: freed %d bytes (free_pages=%.1f%%, threshold=%.0f%%)",
                         space_freed, free_pct, free_pct_threshold)
        else:
            logger.debug("Vacuum skipped: free_pages=%.1f%% < threshold %.0f%%",
                         free_pct, free_pct_threshold)

    logger.info("Purge complete: deleted=%d, archived=%d, qdrant_removed=%d, space_freed=%d bytes",
                deleted_count, archived_count, qdrant_removed, space_freed)

    return {
        "deleted_count": deleted_count,
        "archived_count": archived_count,
        "qdrant_removed": qdrant_removed,
        "space_freed_bytes": max(0, space_freed),
        # Reported the *precondition* — "vacuum was requested and something was
        # purged" — so it read True when the free-page threshold was not met
        # and when the VACUUM itself raised and was swallowed by the warning
        # above. An operator reading `vacuumed: true` after a failed VACUUM has
        # been told the opposite of what happened. 2026-08-24 audit, minor 3.
        "vacuumed": vacuum_ran,
    }

# ---- State.db helpers (session metadata from Hermes) ----------------


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
    # protected/backlinks/layer3_flags were missing here — export's own
    # docstring calls this "restore" data, but a record's do-not-decay/
    # do-not-purge flag and its backlink graph silently vanished on any
    # export/import round trip (migration, restore after DB loss,
    # mode="overwrite"). reference_count is included below and imported via
    # _clamp_reference_count (0.7.55); before that it was hardcoded to 0 on
    # import regardless of what export produced.
    columns = (
        "uuid, content, summary, keywords, topic, scope, data_type, "
        "data_id, session_name, sensitivity, source, source_url, "
        "created_at, updated_at, ttl, status, trust_score, "
        "reference_count, priority, metadata, protected, backlinks, layer3_flags, "
        # The review verdict travels with the record. Without these two, an
        # export/import round trip returned a KEEP-reviewed record with
        # llm_review_status NULL: `review` then re-classified it, spending an
        # LLM call to re-derive a decision that had already been made, and a
        # later review(execute=true) could reach the opposite verdict and
        # delete it. A round trip is supposed to be lossless.
        "llm_review_status, llm_reviewed_at"
    )

    def _query_one(db_path, prof):
        conn = sqlite3.connect(db_path, check_same_thread=False)
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
                # keywords/backlinks are array-typed, metadata/layer3_flags
                # are objects — defaulting all four to {} exported
                # `"keywords": {}` for a record with none, which
                # import_memories then rejected via isinstance(kw, list).
                for jf in ("keywords", "metadata", "backlinks", "layer3_flags"):
                    default = [] if jf in ("keywords", "backlinks") else {}
                    try:
                        rec[jf] = json.loads(rec.get(jf, "null")) or default
                    except (json.JSONDecodeError, TypeError):
                        rec[jf] = default
                rec["protected"] = bool(rec.get("protected"))
                records.append(rec)
            return records
        finally:
            conn.close()

    # Determine which DBs to query
    # Naming a foreign profile used to cross the boundary on its own, which
    # made `cross_profile` decorative on this path: the whole point of the flag
    # is to be the single explicit opt-in for reading another profile's store,
    # and export returns raw record content — the least fenced output HLM
    # produces. Every other cross-profile reader (retrieve, discover, list)
    # requires the flag; export accepted a caller-supplied `profile_name`
    # instead. 2026-08-23 laguna-s-2.1 maintenance review (F1).
    #
    # The narrowing costs nothing functionally: `cross_profile=True` together
    # with `profile_name=X` still exports exactly X (the dict comprehension
    # below narrows to it), so a legitimate migration or restore call adds one
    # argument rather than losing a capability.
    if profile_name and profile_name != self._profile_name and not cross_profile:
        raise ValueError(
            f"export of profile {profile_name!r} from {self._profile_name!r} requires "
            f"cross_profile=True — naming another profile is not itself the opt-in")
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

    if len(all_records) > 50_000:
        # Not truncated — a silently partial export would be a worse
        # failure mode than high memory use for a rare very-large-corpus
        # export. Flagged so an operator sizing a many-profile cross-export
        # isn't surprised by it.
        logger.warning(
            "export_memories: %d records materialized in memory at once — "
            "large exports are not streamed, consider narrowing with data_type/topic/scope filters",
            len(all_records))

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
                # str() per element. `add()` has guarded keyword element types
                # since 0.7.93, but import does not — it stores whatever the
                # blob's list holds — and rows written before that guard are
                # still in every long-lived profile. `', '.join([1, 2])` raises
                # TypeError, and this is the only join in the export, so one
                # such row took down the entire markdown export of an entire
                # profile. 2026-08-24 audit, finding 6.
                lines.append(f"- **Keywords:** {', '.join(str(k) for k in kw)}")
            lines.append("")
            lines.append(f"_{rec.get('summary', '')}_")
            lines.append("")
            lines.append(rec.get("content", ""))
            lines.append("")
        return "\n".join(lines)

    return json.dumps(export_data, indent=2, ensure_ascii=False, default=str)


def _safe_ttl(be, value):
    """Normalise an imported TTL, dropping it rather than failing the import.

    add() and update() raise on an unparseable TTL because a caller is there to
    be told. The bulk paths — import over a file that may be years old or
    third-party, and compaction merging records written before 0.7.42 — should
    not abort a whole run over one bad row. A rejected TTL becomes
    None — the record simply does not expire on a schedule, which is the state
    it would have had without the field, rather than the string sorting above
    every ISO timestamp and meaning "never" by accident.
    """
    if value in (None, ""):
        return None
    from .store import _normalise_ttl
    try:
        return _normalise_ttl(value, be._now())
    except ValueError as e:
        logger.warning("ttl normalisation failed: dropping unusable ttl %r (%s)",
                       value, e)
        return None


def _clamp_trust(value) -> float:
    """Clamp an imported trust_score into 0.0-1.0.

    add() and update() both clamp; import_memories wrote whatever the file said.
    A trust_score of 99 from a hand-edited or hostile export defeats the
    invariant feedback(), reinforce() and decay() all assume, and the record
    then outranks everything forever. Import is the one write path that takes
    its values from a file rather than from a caller, so it is the one that most
    needs the check.

        NaN needs its own branch. Every comparison against NaN is False, so
    `min(1.0, nan)` returns 1.0 and `max(0.0, 1.0)` keeps it — a NaN
    trust_score does not fail the clamp, it lands at the **top** of the range
    and the record outranks everything, which is the exact outcome this
    function exists to prevent. It reaches here the same way 99 does: from a
    hand-edited or hostile export, where `"trust_score": NaN` is accepted by
    Python's json module. Falls back to the same neutral 0.5 the type-error
    branch uses. The int clamps beside this one are unaffected — `int(nan)`
    raises and their except branch catches it. 2026-08-23 ox-alpha write
    review (F2).
    """
    try:
        _v = float(value)
        if _v != _v:          # NaN
            return 0.5
        return max(0.0, min(1.0, _v))
    except (TypeError, ValueError):
        return 0.5


def _clamp_priority(value) -> int:
    """Clamp an imported priority into 0-3, matching update()'s validation."""
    try:
        return max(0, min(3, int(value)))
    except (TypeError, ValueError):
        return 0


def _clamp_reference_count(value) -> int:
    """Clamp an imported reference_count to >= 0.

    Not validated at all before 0.7.55 — the field wasn't imported, it was
    hardcoded to 0 in both write branches regardless of what export produced.
    """
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _clamp_sensitivity(value) -> int:
    """Clamp an imported sensitivity into 0-3, matching update()'s validation.

    import_memories clamped priority and trust_score on both its write paths
    and left sensitivity raw — the one field with no consumer that raises on
    a bad value, so a corrupted import silently stored an unenforceable
    sensitivity marker rather than failing anywhere visible.
    """
    try:
        return max(0, min(3, int(value)))
    except (TypeError, ValueError):
        return 0


def _import_json_fields(rec: dict) -> tuple:
    """`(keywords_json, metadata_json)` for an imported row, defaults on junk.

    Written out twice inside the import path — once on the UPDATE branch and
    once on the INSERT branch — with the same `isinstance` checks and the same
    `"[]"`/`"{}"` fallbacks. Two copies of a coercion rule on the door that
    `CHANGELOG.md` records four separate defects for, every one of them "the
    import path validates per-field what `add()` already enforces, and the fix
    patched one member". 2026-09-24 external review A5.2.
    """
    kw = rec.get("keywords", [])
    kw_json = json.dumps(kw) if isinstance(kw, list) else "[]"
    meta = rec.get("metadata", {})
    meta_json = json.dumps(meta) if isinstance(meta, dict) else "{}"
    return kw_json, meta_json


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

    # target_status lands in the INSERT verbatim. An unrecognised value is not
    # rejected by SQLite (there is no CHECK constraint), so importing with
    # target_status="activ" writes rows that every `status='active'` query
    # silently skips — records present in the table, absent from retrieval,
    # list, export and count, with nothing logged. Fail the call instead.
    if target_status not in ("active", "archived"):
        return {"imported": 0, "skipped": 0, "failed": 0,
                "errors": [f"invalid target_status {target_status!r} — "
                           f"must be 'active' or 'archived'"]}
    results = {"imported": 0, "skipped": 0, "failed": 0, "errors": []}

    # Reject unknown modes instead of silently falling through. Every
    # conflict branch below is keyed on an exact mode string, so an
    # unrecognised value skipped all of them and hit the plain
    # INSERT OR IGNORE with a colliding uuid — the record was silently
    # dropped while still being counted as imported.
    _valid_modes = ("skip_existing", "overwrite", "new_uuid")
    if mode not in _valid_modes:
        return {"imported": 0, "skipped": 0, "failed": 0,
                "errors": [f"Unknown import mode {mode!r}. Valid modes: {', '.join(_valid_modes)}."]}

    # Read from file if path provided. Gated only by string suffix, so this
    # is an unauthenticated file-existence/readability oracle unless
    # contained the same way ingest_obsidian's vault_path is — a caller
    # could otherwise probe for any file ending in .json/.md the process
    # user can read.
    if isinstance(data, str) and (
        data.endswith(".json") or data.endswith(".md")
    ):
        import_roots = _allowed_fs_roots("HLM_IMPORT_ALLOWED_ROOTS")
        if not _path_within_roots(data, import_roots):
            return {"imported": 0, "skipped": 0, "failed": 1,
                    "errors": [f"Import path {data!r} is outside allowed roots {import_roots}. "
                               f"Set HLM_IMPORT_ALLOWED_ROOTS (colon-separated) to allow additional locations."]}
        try:
            with open(data, "r", encoding="utf-8") as f:
                data = f.read()
        except Exception as e:
            # Uniform message, detail to the log. The errno text separated
            # "absent" from "present but unreadable" for any path inside the
            # allowed roots, and echoed the full path back. Both doors forward
            # this result verbatim; the MCP one needs HLM_MCP_ADMIN, but that
            # server has no authentication at all, and the plugin door has no
            # gate. The broad oracle the comment above describes closing
            # is genuinely closed by _path_within_roots; this is the narrowed
            # residual over the roots themselves.
            #
            # A uniform message with no log line would trade an oracle for an
            # undebuggable import, which is the wrong trade — the operator
            # still gets the errno, the caller does not. 2026-09-15 round 2,
            # bundle02 (F2), confirmed by probe. T640.
            logger.warning("import: cannot read %s: %s", data, e)
            return {"imported": 0, "skipped": 0, "failed": 1,
                    "errors": ["Failed to read file: not found or not readable"]}

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
            cursor = self._get_conn().execute("SELECT uuid FROM memories")
            existing_uuids = {row[0] for row in cursor.fetchall()}
        except Exception as e:
            logger.warning("import_memories: failed to fetch existing UUIDs: %s", e)

    # Trusted source allowlist — imported content must not be able to claim a
    # self-authored source, or it bypasses the prompt-injection fence. Read
    # from the single source of truth rather than a local copy, which would
    # drift silently the moment SELF_AUTHORED_SOURCES gains a member.
    _trusted_sources = _SELF_AUTHORED_SOURCES

    # Same write-payload bounds add() enforces. Import wrote straight to
    # SQLite with no check, so layered_io(action="import") was an unbounded
    # write path around the documented cap — and an oversized record bloats
    # the shared FTS5 index and Qdrant collection for the whole profile.
    max_content = MAX_CONTENT_CHARS
    max_metadata = MAX_METADATA_CHARS

    for rec in records:
        # A non-dict record — a bare string in a hand-edited export, a nested
        # list — used to reach `rec.get(...)` and raise AttributeError. That
        # was caught by the outer handler below, whose *own* message calls
        # `rec.get('uuid', '?')` and raised again, this time escaping the loop
        # entirely: the whole import aborted and the `commit()` after the loop
        # never ran, so records already imported were lost too. One bad row
        # discarded the batch. 2026-08-24 audit, minor 2.
        if not isinstance(rec, dict):
            results["failed"] += 1
            results["errors"].append(
                "Record is %s, not an object — skipped" % type(rec).__name__)
            continue
        try:
            # data_type through the same guard add()/update() use, at the top
            # of the loop so **every** branch passes it — the INSERT, the
            # overwrite UPDATE and the new_uuid path each write data_type
            # separately, and a guard placed in one of them is the
            # one-of-two mistake this review loop keeps finding (the first
            # attempt at this fix landed in a single branch and a `banana`
            # type imported cleanly through another).
            #
            # 0.7.77 gave add() and update() this check and skipped import,
            # which is the writer taking values from a *file* rather than a
            # caller — the argument `_clamp_trust`'s docstring makes for why
            # import needs it most. An unregistered type routes to the default
            # collection and partitions the record away from dedup and
            # contradiction detection, both type-scoped.
            from .store import _validate_data_type as _vdt
            try:
                _vdt(self, rec.get("data_type") if isinstance(rec, dict) else None)
            except ValueError as _e:
                results["failed"] += 1
                results["errors"].append(
                    "%s: %s" % (str(rec.get("uuid", "?"))[:8], _e))
                continue
            uuid = rec.get("uuid")
            if not uuid:
                results["failed"] += 1
                results["errors"].append("Record missing uuid")
                continue
            # Shape, not just presence. This field is interpolated into both
            # LLM review prompts **outside** the untrusted fence, so a uuid
            # carrying newlines and instruction text is attacker-controlled
            # prose in a prompt whose output can delete records. Driven
            # 2026-09-26: a record whose uuid was a fake prompt block imported
            # cleanly and was stored verbatim. `_llm_merge` already sanitised
            # this column for the same reason; the review renderers did not
            # (`T708`, which guards them).
            #
            # `is_storable_uuid`, **not** `is_record_uuid`. This door first
            # shipped with the strict form and that was a regression: import
            # is a portability path that accepts ids minted elsewhere, and
            # demanding canonical UUID form here failed every record whose id
            # merely looked foreign. Eight tests said so — an import fails the
            # bad row and commits the rest, and `t640-0000-...` is data. The
            # attack needs a prompt *block*: newlines and instructions. So
            # this rejects control characters and unbounded length, and the
            # renderers keep the strict check, which is where the
            # interpolation is. `T722`.
            if not _C.is_storable_uuid(uuid):
                results["failed"] += 1
                results["errors"].append(
                    "Record uuid is not storable (must be a single-line "
                    "string of at most 200 characters): %r"
                    % (str(uuid)[:40],))
                continue

            rec_content = rec.get("content") or ""
            if len(rec_content) > max_content:
                results["failed"] += 1
                results["errors"].append(
                    f"Record {uuid} content exceeds max length "
                    f"({len(rec_content)} > {max_content} chars)")
                continue
            _meta_len = len(json.dumps(rec.get("metadata") or {}, default=str))
            if _meta_len > max_metadata:
                results["failed"] += 1
                results["errors"].append(
                    f"Record {uuid} metadata exceeds max length "
                    f"({_meta_len} > {max_metadata} chars)")
                continue
            # add() and update() both call _check_fts_field_bounds on
            # summary/topic/data_id/keywords; import — the third write path —
            # checked only content and metadata, and the comment above claimed
            # parity with add(). summary/topic/keywords are FTS5-indexed, so an
            # unbounded value from an import blob bloats the index the bound
            # exists to protect, and data_id is a lookup key. Found by the
            # 2026-08-23 maintenance review (F4). Raised as a
            # per-record failure rather than aborting the import, matching how
            # the content/metadata guards above already behave.
            # created_at/updated_at were stored verbatim from the blob. Three
            # things follow, and the first is the one that matters:
            #
            #   * `_llm_merge` interpolates `created_at` into the compaction
            #     prompt (llm.py) beside `content` and `topic`, which are
            #     fenced. An imported record carrying
            #     `</untrusted_external_doc>` plus instructions in that field
            #     reaches the model outside the fence — and compaction writes
            #     the model's answer back with source="hlm-consolidated",
            #     which is self-authored and never fenced again. Injection
            #     with a persistence leg.
            #   * A `created_at` far in the past is immediately purge-eligible.
            #   * An unparseable one makes decay skip the record forever.
            #
            # Rejecting the record would fail an import on one bad row; these
            # are timestamps with a well-defined default, so an unusable value
            # is replaced with now() and counted. 2026-08-24 audit, M1.
            for _tsf in ("created_at", "updated_at"):
                _tsv = rec.get(_tsf)
                if _tsv is None:
                    continue
                # A *future* timestamp parses fine and is the residual half of
                # this finding. `sleep()`'s low-trust arm selects on
                # `created_at < cutoff`, so a record dated next year is immune
                # to age-based archival forever — an evasion primitive an
                # import blob can set for free. The far-past direction filed
                # alongside it is not fixed and should not be: preserving real
                # timestamps is what makes an export/import round trip a round
                # trip. 2026-08-24 audit, finding 5 (partly refuted).
                if _valid_timestamp(_tsv) and _is_future_timestamp(_tsv, self._now()):
                    results.setdefault("repaired", []).append(
                        f"Record {uuid} {_tsf}={_tsv!r} is in the future — using now()")
                    rec[_tsf] = self._now()
                    continue
                if not _valid_timestamp(_tsv):
                    results.setdefault("repaired", []).append(
                        f"Record {uuid} {_tsf}={_tsv!r} is not a timestamp — using now()")
                    rec[_tsf] = self._now()

            # add() rejects a keywords list whose elements are not strings
            # (`all(isinstance(k, str) ...)`); import checked only that the
            # container was a list and json.dumps'd whatever it held. That is
            # how `[1, 2, 3]` got into the store — and 0.8.0 fixed the
            # *symptom*, a markdown export dying on `', '.join(kw)`, by
            # coercing on the way out. This is the source. Third front of one
            # defect: add(), export, and the writer that let it in.
            # 2026-08-25 bundle01 review (F9).
            # add() and update() accept a JSON-array *string* here and parse
            # it; import's write branches test `isinstance(kw, list)` and fall
            # through to `kw_json = "[]"` otherwise, so a string arrived and the
            # keywords were **silently dropped** rather than rejected — data
            # loss on an export/import round trip, reported as success.
            #
            # 0.8.3's guard skipped strings entirely (`not isinstance(_kw, str)`)
            # which left that hole in place and made it look deliberate. Parse
            # it the way add() does instead. 2026-08-25 bundle01 review (F1).
            # The same silent-replacement shape as keywords, on three more
            # fields and in *both* write branches: a non-list `backlinks`
            # became `[]` and a non-dict `metadata`/`layer3_flags` became `{}`,
            # so a malformed export imported "successfully" having discarded
            # them. `add()` type-guards backlinks (0.7.93) and rejects; import
            # replaced. Reject here too, and accept the JSON-string forms the
            # other writers accept. 2026-08-25 bundle02 review (F2).
            # Honour the record's own `status` when the export carried one.
            #
            # `target_status` was applied to every row, so an export taken with
            # `status='all'` — the shape a backup uses, and the only one that
            # includes soft-deleted records — re-imported those records as
            # **active**. A round trip that is meant to be lossless resurrected
            # everything the store had deleted. The parameter stays as the
            # default for rows that carry no status of their own.
            # 2026-08-25 xhigh round, bundle02 (F4).
            _rec_status = rec.get("status")
            if _rec_status not in ("active", "archived", "deleted"):
                _rec_status = target_status

            _bad_container = None
            for _cf, _want, _label in (("backlinks", list, "a list"),
                                       ("metadata", dict, "an object"),
                                       ("layer3_flags", dict, "an object")):
                _cv = rec.get(_cf)
                if _cv is None:
                    continue
                if isinstance(_cv, str):
                    try:
                        _cv = json.loads(_cv)
                    except (json.JSONDecodeError, TypeError):
                        _cv = None
                if not isinstance(_cv, _want):
                    _bad_container = (
                        f"Record {uuid} {_cf} must be {_label} (or a JSON string "
                        f"of one), got {rec.get(_cf)!r}")
                    break
                rec[_cf] = _cv
            if _bad_container:
                results["failed"] += 1
                results["errors"].append(_bad_container)
                continue

            # `keywords` and `backlinks` through the same rules `add()` uses.
            #
            # **Only those two.** The numerics differ here *by design*: `T397`
            # pins that import **clamps** `sensitivity`, `priority` and
            # `trust_score` — writing 0, 3 and 1.0 for the record that test
            # imports — rather than dropping the row, because a bulk restore
            # that discards a record over one junk field loses data the
            # operator asked to get back. `add()` raises instead, for a single
            # interactive write whose caller can fix it. Two dispositions, one
            # rule set; passing the numerics through here made import *refuse*
            # them, and `T397` caught it.
            #
            # What was genuinely missing is the `backlinks` **elements** check.
            # The container guard above (2026-08-25 bundle02 F2) accepts any
            # list, so `backlinks=[1, 2]` was stored while `add()` refused it —
            # the identical container-only-check defect `add()` itself carried
            # on the same field, found by the laguna-s-2.1 write review.
            from .store import _coerce_record_fields, _UNSET
            try:
                _norm = _coerce_record_fields(
                    keywords=rec.get("keywords", _UNSET),
                    backlinks=rec.get("backlinks", _UNSET))
            except ValueError as _e:
                results["failed"] += 1
                results["errors"].append(f"Record {uuid} {_e}")
                continue
            rec.update(_norm)

            try:
                from .store import (_check_fts_field_bounds,
                                    _check_scalar_field_bounds)
                _check_scalar_field_bounds(
                    session_name=rec.get("session_name"), scope=rec.get("scope"),
                    source_url=rec.get("source_url"))
                _check_fts_field_bounds(
                    summary=rec.get("summary"), topic=rec.get("topic"),
                    data_id=rec.get("data_id"), keywords=rec.get("keywords"))
            except ValueError as _e:
                results["failed"] += 1
                results["errors"].append(f"Record {uuid} {_e}")
                continue

            # Skip existing
            if mode == "skip_existing" and uuid in existing_uuids:
                results["skipped"] += 1
                continue

            # Force untrusted source on import — an attacker-supplied source
            # of "agent" would permanently launder the content as fencing-exempt.
            import_source = rec.get("source", "import")
            if not import_source or import_source in _trusted_sources:
                import_source = "import"

            # Check if exists for overwrite
            if mode == "overwrite":
                if uuid in existing_uuids:
                    existing = self._get_conn().execute(
                        "SELECT content FROM memories WHERE uuid = ?",
                        (uuid,),
                    ).fetchone()
                    if existing:
                        # The new-record INSERT path below rejects empty
                        # content (add()'s own guard, mirrored there); this
                        # branch had no equivalent and would overwrite a live
                        # record's content and summary with blank on a
                        # hand-edited, corrupted, or partial export. The row
                        # stays status='active' — a phantom present in the
                        # table, unmatchable by any retrieval — the exact
                        # failure v0.7.54's update() fix exists to prevent,
                        # reached here through the sibling path that missed it.
                        _overwrite_content = rec.get("content")
                        if not _overwrite_content or not str(_overwrite_content).strip():
                            results["failed"] += 1
                            results["errors"].append(
                                f"Record {uuid} has empty content — cannot overwrite with blank")
                            continue
                        kw_json, meta_json = _import_json_fields(rec)
                        # backlinks/layer3_flags — see the matching comment
                        # on the INSERT branch below. Neither existed in this
                        # branch before 0.7.55; an overwrite import silently
                        # cleared both to their schema default (backlinks
                        # was untouched by the UPDATE at all, layer3_flags
                        # the same).
                        bl = rec.get("backlinks", [])
                        bl_json = json.dumps(bl) if isinstance(bl, list) else "[]"
                        l3 = rec.get("layer3_flags", {})
                        l3_json = json.dumps(l3) if isinstance(l3, dict) else "{}"
                        # Normalize data_id to lowercase for consistency (same as add())
                        import_data_id = rec.get("data_id")
                        if import_data_id:
                            import_data_id = import_data_id.lower()
                        # embedding = NULL: the content is being replaced, so
                        # the stored vector no longer describes it. rebuild()
                        # below only re-embeds rows whose embedding is NULL or
                        # the wrong dimension, and a stale-but-correctly-sized
                        # vector passed that check — leaving the record
                        # permanently indexed under its *old* content, found
                        # by semantic search for the old text and never the new.
                        self._get_conn().execute(
                            "UPDATE memories SET content = ?, summary = ?, keywords = ?, "
                            "topic = ?, scope = ?, data_type = ?, data_id = ?, "
                            "session_name = ?, sensitivity = ?, source = ?, source_url = ?, "
                            "updated_at = ?, ttl = ?, status = ?, trust_score = ?, "
                            "priority = ?, metadata = ?, protected = ?, "
                            "reference_count = ?, backlinks = ?, layer3_flags = ?, "
                            # The INSERT branch below carries both review
                            # columns and the export SELECT emits them, but
                            # this UPDATE dropped them — so a round trip run
                            # with mode="overwrite" silently cleared every
                            # imported record's verdict. 0.7.71 fixed exactly
                            # this loss on the *other* branch and this one was
                            # not checked; the comment above the export list
                            # states the invariant ("a round trip is supposed
                            # to be lossless") that both branches have to keep.
                            # Consequence of losing it: a KEEP-reviewed record
                            # comes back NULL, gets re-reviewed at LLM cost,
                            # and can reach the opposite verdict on a later
                            # review(execute=true).
                            "llm_review_status = ?, llm_reviewed_at = ?, "
                            "embedding = NULL "
                            "WHERE uuid = ?",
                            (rec.get("content", ""),
                             rec.get("summary", rec.get("content", "")[:100]),
                             kw_json,
                             rec.get("topic"),
                             rec.get("scope", "personal"),
                             rec.get("data_type", "CUSTOM"),
                             import_data_id,
                             rec.get("session_name"),
                             _clamp_sensitivity(rec.get("sensitivity", 0)),
                             import_source,
                             rec.get("source_url"),
                             self._now(),
                             _safe_ttl(self, rec.get("ttl")),
                             _rec_status,
                             _clamp_trust(rec.get("trust_score", 0.5)),
                             _clamp_priority(rec.get("priority", 0)),
                             meta_json,
                             bool(rec.get("protected", False)),
                             _clamp_reference_count(rec.get("reference_count", 0)),
                             bl_json,
                             l3_json,
                             rec.get("llm_review_status"),
                             rec.get("llm_reviewed_at"),
                             uuid),
                        )
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
            kw_json, meta_json = _import_json_fields(rec)

            # backlinks/layer3_flags/reference_count were absent from this
            # INSERT entirely (backlinks, layer3_flags) or bound to a literal
            # 0 regardless of what export produced (reference_count) — an
            # export/import round trip silently reset a record's backlink
            # graph, cleared any L3 rerank flags, and zeroed its usage
            # history. `protected` was bound correctly here but was never
            # exported (see the SELECT above), so it defaulted to False on
            # every import anyway: do-not-decay/do-not-purge records came
            # back decayable and archivable with no signal that it happened.
            bl = rec.get("backlinks", [])
            bl_json = json.dumps(bl) if isinstance(bl, list) else "[]"
            l3 = rec.get("layer3_flags", {})
            l3_json = json.dumps(l3) if isinstance(l3, dict) else "{}"

            # Normalize data_id to lowercase for consistency (same as add())
            import_data_id = rec.get("data_id")
            if import_data_id:
                import_data_id = import_data_id.lower()

            cur = self._get_conn().execute(
                """INSERT OR IGNORE INTO memories (
                    uuid, content, summary, keywords, topic, scope,
                    data_type, data_id, session_name, sensitivity,
                    source, source_url, created_at, updated_at, ttl,
                    status, trust_score, reference_count, priority,
                    metadata, embedding, protected, sequence, backlinks,
                    layer3_flags, llm_review_status, llm_reviewed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, 0, ?, ?, ?, ?)""",
                (uuid, content,
                 rec.get("summary", content[:100]), kw_json,
                 rec.get("topic"), rec.get("scope", "personal"),
                 rec.get("data_type", "CUSTOM"), import_data_id,
                 rec.get("session_name"), _clamp_sensitivity(rec.get("sensitivity", 0)),
                 import_source, rec.get("source_url"),
                 rec.get("created_at", self._now()),
                 rec.get("updated_at", self._now()),
                 _safe_ttl(self, rec.get("ttl")), _rec_status,
                 _clamp_trust(rec.get("trust_score", 0.5)),
                 _clamp_reference_count(rec.get("reference_count", 0)),
                 _clamp_priority(rec.get("priority", 0)), meta_json,
                 bool(rec.get("protected", False)),
                 bl_json, l3_json,
                 # The review verdict is part of the record. Exported since
                 # this release and dropped here until it was, so a
                 # KEEP-reviewed record came back NULL, got re-reviewed at the
                 # cost of an LLM call, and could reach the opposite verdict on
                 # a later review(execute=true). Fixing only the export half
                 # looks correct in a test of the export and still loses the
                 # data — measured before this line existed.
                 rec.get("llm_review_status"), rec.get("llm_reviewed_at"),
                ),
            )

            # OR IGNORE silently no-ops on a uuid collision — count what was
            # actually written, not what was attempted, or the caller gets a
            # success report for records that were never stored.
            if cur.rowcount:
                results["imported"] += 1
            else:
                results["skipped"] += 1

        except Exception as e:
            results["failed"] += 1
            # Belt and braces: an error handler that can itself raise turns a
            # single bad record into a lost batch, which is exactly what the
            # isinstance guard above now prevents at the other end.
            _uid = rec.get("uuid", "?") if isinstance(rec, dict) else "?"
            err_msg = f"Failed to import {_uid}: {e}"
            logger.debug(err_msg)
            results["errors"].append(err_msg)

    self._get_conn().commit()

    # Rebuild Qdrant for new records
    if results["imported"] > 0:
        try:
            self.rebuild()
        except Exception as e:
            logger.warning("Post-import rebuild failed: %s", e)

    logger.info("import: %d imported, %d skipped, %d failed",
                 results["imported"], results["skipped"], results["failed"])
    return results


def decay(self, min_age_days: int = None,
          max_age_days: int = None,
          decay_rate: float = None,
          min_score: float = None) -> dict:
    """Apply confidence decay to memories older than min_age_days.

    Decay formula: new_score = max(min_score,
        trust_score * (1 - effective_rate * min(age_days / max_age_days, 1)))

    Records with protected=True are never decayed.
    Records with trust_score > 0.9 are never decayed (high confidence).
    Priority controls decay rate multiplier:
      0: 1.00x (full decay)
      1: 0.50x (half decay)
      2: 0.20x (slow decay)
      3+: 0.00x (exempt — pinned)
    Records with reference_count > 10 decay 70% slower (reinforcement).

    Args are optional — falls back to JSON config `cleanup` section or defaults.
    Returns dict with decayed, unchanged counts.
    """
    cfg = self._get_cleanup_config()
    # Use `if x is not None` for 0-valued params — `or` treats 0 as falsy
    min_age_days = min_age_days if min_age_days is not None else cfg["decay_min_age_days"]
    max_age_days = max_age_days if max_age_days is not None else cfg["decay_max_age_days"]
    # The decay formula divides by this, so an explicit 0 (or a negative, which
    # would invert the age factor) raises ZeroDivisionError mid-sweep — after
    # some records have already been written. The default is 365 and every
    # in-repo caller relies on it, so this is only reachable from a caller
    # passing 0 deliberately; reject it up front rather than fail half-applied.
    try:
        max_age_days = int(max_age_days)
    except (TypeError, ValueError):
        raise ValueError(f"max_age_days must be a positive integer, got {max_age_days!r}")
    if max_age_days <= 0:
        raise ValueError(
            f"max_age_days must be > 0 (it is the denominator of the decay "
            f"age factor), got {max_age_days}")
    decay_rate = decay_rate if decay_rate is not None else cfg["decay_rate"]
    # The third parameter of three, and the one nobody validated.
    #
    # `max_age_days` is checked below because it is the denominator;
    # `min_age_days` gained `_non_negative_age` when 0.7.96's M2 found the same
    # inversion. `decay_rate` is the multiplier and got neither. It is bounded
    # by the formula's own meaning — `score * (1 - rate * age_factor)` is a
    # *fraction* of the score to erode — so a rate above 1 is not a faster
    # decay, it is a cliff: at age_factor 0.9 a rate of 2.0 takes 0.8 straight
    # to the floor in ONE pass, and trust_score at the floor is what sleep()'s
    # low-trust arm archives on. A negative rate computes growth, which the
    # `new_score < score` write guard already discards — so that half of the
    # report is refuted, and this guard exists for the over-decay half.
    # 2026-08-26 ox-alpha round, bundle03 F2.
    try:
        decay_rate = float(decay_rate)
    except (TypeError, ValueError):
        raise ValueError(f"decay_rate must be a number, got {decay_rate!r}")
    if not (0 < decay_rate <= 1):
        raise ValueError(
            f"decay_rate must be in (0, 1] — it is the fraction of trust_score "
            f"eroded at full age, so a value above 1 collapses every matched "
            f"record to min_score in one pass, got {decay_rate}")
    min_score = min_score if min_score is not None else cfg["decay_min_score"]

    # Priority → decay multiplier curve
    priority_multipliers = {0: 1.0, 1: 0.5, 2: 0.2}

    now = self._now()
    # `max_age_days` is validated above because it is the decay denominator;
    # `min_age_days` was not, and it is the same inversion 0.7.96's M2 fixed
    # for purge and sleep — a negative puts the cutoff in the *future*, so
    # every record reads as older than it and the whole store decays at once.
    # `_non_negative_age` reached those two functions and not this one.
    # 2026-08-25 xhigh round, bundle03 (F5).
    min_age_days = _non_negative_age(min_age_days, "min_age_days")
    cutoff = (datetime.fromisoformat(now) -
               timedelta(days=min_age_days)).isoformat()

    cursor = self._get_conn().execute(
        "SELECT uuid, trust_score, created_at, reference_count, priority "
        "FROM memories WHERE status = 'active' "
        "AND created_at < ? "
        "AND trust_score <= 0.9 "
        # COALESCE, matching sleep()'s guard two functions up — added there
        # because "protected is an explicit do-not-touch marker" and a raw
        # `= 0` treats NULL as neither 0 nor 1, so a future write path
        # producing NULL would make decay() skip the row (immortal) while
        # sleep() correctly archived it. No current writer produces NULL —
        # the v10 migration backfills the column — so this has no effect on
        # today's data; it closes the divergence before a future one opens it.
        "AND COALESCE(protected, 0) = 0",
        (cutoff,),
    )
    rows = cursor.fetchall()

    results = {"decayed": 0, "unchanged": 0, "details": []}

    for row in rows:
        uuid, score, created_at, ref_count, priority = row

        # Priority multiplier
        prio_mult = priority_multipliers.get(priority, 0.0)
        if prio_mult == 0.0:
            results["unchanged"] += 1
            continue

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

        # Effective decay rate — reinforce frequently retrieved + priority curve
        effective_rate = decay_rate * prio_mult
        if ref_count and ref_count > 10:
            effective_rate *= 0.3  # 70% slower decay

        # Apply decay
        new_score = max(min_score,
                        score * (1 - effective_rate * age_factor))
        new_score = round(new_score, 4)

        if new_score < score:
            self._get_conn().execute(
                "UPDATE memories SET trust_score = ?, updated_at = ? "
                "WHERE uuid = ?",
                (new_score, now, uuid),
            )
            results["decayed"] += 1
        else:
            results["unchanged"] += 1

    self._get_conn().commit()
    logger.debug("decay: %d decayed, %d unchanged (rate=%.3f, min_score=%.2f)",
                 results["decayed"], results["unchanged"], decay_rate, min_score)
    return results


def _parse_frontmatter(self, file_path: str) -> tuple:
    """Parse YAML frontmatter and return (content, frontmatter_dict)."""
    # Cap the read instead of buffering an arbitrarily large file — a vault
    # (or, pre-containment-fix, an attacker-chosen directory) can contain
    # very large .md-suffixed files, and ingest_obsidian's Phase 1 already
    # accumulates every parsed note in memory before any embedding call.
    max_bytes = MAX_CONTENT_CHARS * 4  # generous byte:char headroom for markdown/frontmatter
    try:
        if os.path.getsize(file_path) > max_bytes:
            logger.warning(
                "obsidian ingest: skipping oversized file %s (> %d bytes)", file_path, max_bytes)
            return "", {}
    except OSError:
        pass
    # Explicit UTF-8. Without it Python uses locale.getpreferredencoding(),
    # so under a non-UTF-8 locale a perfectly good note raised
    # UnicodeDecodeError, which ingest_obsidian counted as `skipped` behind
    # a debug log — notes vanished from the ingest with no visible reason.
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read(max_bytes)

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


def _embed_query_text(self, query: str) -> str:
    """Apply the configured query-side instruction prefix before embedding.

    Asymmetric embedding models (Qwen3-Embedding, E5, BGE) expect queries
    to carry a task instruction while documents are embedded raw. Without
    it, short generic queries land near the corpus centroid and score
    *higher* than specific ones, which makes any absolute score threshold
    useless. Empty by default — set `query_instruction` in config or
    HLM_QUERY_INSTRUCTION, and measure with
    `tests/eval_retrieval.py --compare` before changing it.
    """
    instruction = self._config.get("query_instruction") or ""
    return f"{instruction}{query}" if instruction else query


def ingest_obsidian(self, vault_path: str, exclude: List[str] = None) -> dict:
    """Ingest Obsidian vault notes into HLM.

    Parses frontmatter, extracts tags and wikilinks, embeds all notes.
    """
    # ".git" and ".DS_Store" are not Obsidian concepts, which is why they were
    # missing — but a vault under version control is the normal case, and
    # walking .git ingests commit messages, refs and hook scripts as if they
    # were notes. They are fenced (source="obsidian" is untrusted) but they are
    # still corpus noise that dilutes retrieval.
    # A caller's list **extends** these, it does not replace them.
    #
    # `exclude or [...]` meant `exclude=["Drafts"]` silently stopped excluding
    # `.obsidian` and `.git` — so asking to skip one folder started ingesting
    # the vault's own config, plugin data, commit messages, refs and hook
    # scripts as notes. The caller asked for *more* filtering and got less,
    # which is the opposite of what the argument reads as, and the comment
    # below explains why those defaults matter.
    # 2026-08-25 xhigh round, bundle03 (F7).
    _DEFAULT_EXCLUDE = [".obsidian", "_resources", "Templates", "Excalidraw",
                        ".git", ".DS_Store", "node_modules", ".trash"]
    if isinstance(exclude, str):
        exclude = [exclude]
    exclude = list(_DEFAULT_EXCLUDE) + [
        str(x) for x in (exclude or []) if str(x).strip()]

    # Resolve to realpath (follows symlinks) and validate it's a real directory
    # to prevent path traversal. Per-file symlink checks below guard against
    # a symlink *inside* an allowed vault escaping it.
    vault_path = os.path.realpath(vault_path)

    # Containment check: vault_path itself must live under an allowed root.
    # Without this, any caller of this tool action could read arbitrary
    # directories on the host (e.g. vault_path="/" or another user's home).
    #
    # This runs **before** the isdir() check, and the order is the whole
    # point. With isdir() first, the two errors answered different questions
    # about any path on the host: "/root" (exists, unreadable) returned the
    # containment message while "/root/nope" returned the existence message,
    # so an unauthenticated MCP caller could map directories anywhere the
    # process can stat — strictly broader than the import oracle beside it,
    # which refuses out-of-root paths before touching the filesystem at all.
    # Driven, not reasoned: /etc, /root and /var/log answered CONTAINMENT,
    # their nonexistent children answered EXISTENCE.
    #
    # Found by the class check on the 2026-09-15 round 2 bundle02 (F2)
    # import-oracle fix — the reviewer named this function as the sibling and
    # did not check it. T641.
    allowed_roots = _obsidian_allowed_roots()
    if not any(vault_path == root or vault_path.startswith(root + os.sep) for root in allowed_roots):
        raise ValueError(
            f"Vault path {vault_path!r} is outside allowed roots {allowed_roots}. "
            "Set HLM_OBSIDIAN_VAULT_ROOTS (colon-separated) to allow additional vault locations."
        )

    if not os.path.isdir(vault_path):
        raise ValueError(f"Vault path does not exist or is not a directory: {vault_path}")

    ingested = 0
    skipped = 0

    # Phase 1: parse every note. No embedding calls yet — this used to
    # embed one note at a time, which on a large vault means one HTTP
    # round trip per file and is what tripped the Qdrant circuit breaker.
    pending = []
    for root, dirs, files in os.walk(vault_path):
        dirs[:] = [d for d in dirs if d not in exclude]
        for f in files:
            if not f.endswith('.md'):
                continue
            file_path = os.path.join(root, f)
            # Validate realpath stays inside the vault (symlink protection).
            # Plain startswith(vault_path) is a string-prefix bug: a sibling
            # path like "/home/t/vault-evil/x.md" starts with "/home/t/vault"
            # without being inside it — require an exact match or a path
            # separator boundary, same as _path_within_roots.
            real_file = os.path.realpath(file_path)
            if not (real_file == vault_path or real_file.startswith(vault_path + os.sep)):
                skipped += 1
                logger.debug("symlink escape: %s -> %s", file_path, real_file)
                continue
            relative = os.path.relpath(file_path, vault_path)
            try:
                content, frontmatter = self._parse_frontmatter(file_path)
                if not content.strip():
                    skipped += 1
                    continue
                folder = os.path.dirname(relative).split('/')[0] if '/' in relative else ""
                filename = os.path.splitext(f)[0]
                tags = frontmatter.get('tags', [])
                if isinstance(tags, str):
                    tags = [tags]
                pending.append({
                    "content": content,
                    "summary": frontmatter.get('title', filename),
                    "data_id": folder,
                    "keywords": tags,
                    "topic": filename,
                    "source_url": relative,
                    "backlinks": self._extract_wikilinks(content),
                    "metadata": {"file_path": file_path, "tags": tags,
                                 "frontmatter": frontmatter},
                })
            except Exception as e:
                skipped += 1
                logger.debug("Failed to parse %s: %s", relative, e)

    if not pending:
        logger.info("obsidian ingest: %d ingested, %d skipped", ingested, skipped)
        return {"ingested": ingested, "skipped": skipped, "vault": vault_path}

    # Phase 2: one embedding request per 64 notes, order preserved.
    contents = [n["content"] for n in pending]
    vectors = self._embed_batch(contents)

    # Guard against vector misalignment: a partial failure (fewer vectors
    # than inputs) would silently zip the wrong vectors with the wrong
    # contents, corrupting retrieval for those records. If mismatched, fall
    # back to individual embedding with per-record error tracking.
    if len(vectors) != len(contents):
        logger.warning(
            "embed_batch returned %d vectors for %d inputs — "
            "falling back to individual embedding",
            len(vectors), len(contents))
        vectors = []
        for note in pending:
            try:
                v = self._embed_batch([note["content"]])
                vectors.append(v[0] if v else None)
            except Exception as e:
                logger.debug(
                    "Failed to embed %s: %s", note.get("source_url"), e)
                vectors.append(None)

    # Phase 3: store, handing each record its own pre-computed vector.
    for note, vector in zip(pending, vectors):
        try:
            self.add(
                content=note["content"],
                summary=note["summary"],
                data_type="OBSIDIAN",
                data_id=note["data_id"],
                keywords=note["keywords"],
                topic=note["topic"],
                source="obsidian",
                source_url=note["source_url"],
                metadata=note["metadata"],
                backlinks=note["backlinks"],
                embedding=vector,
            )
            ingested += 1
            logger.debug("Ingested: %s (data_id=%s, topic=%s)",
                         note["source_url"], note["data_id"], note["topic"])
        except Exception as e:
            skipped += 1
            logger.warning("Failed to ingest %s: %s", note["source_url"], e)

    logger.info("Obsidian ingestion complete: %d ingested, %d skipped", ingested, skipped)
    return {"ingested": ingested, "skipped": skipped}


def graph_health(self, threshold: float = 0.5, limit: int = 200,
                 data_type: str = None) -> dict:
    """Report records that cluster with nothing — semantic orphans.

    A record whose nearest neighbour is further away than `threshold` is
    isolated: nothing else in the store is about the same thing. That is
    usually one of two problems worth seeing — a fact captured once and never
    reinforced (so decay will quietly retire it), or one whose wording shares
    no ground with how it would ever be asked for, which is a retrieval miss
    that no query will ever reveal.

    Read-only, and exact rather than approximate: the vectors are already in
    SQLite as packed float32, so this is one numpy matmul over the whole set
    rather than N Qdrant round trips. `limit` caps the scan because the cost
    is quadratic in record count; the result says so when it truncates rather
    than reporting a clean bill of health over a sample.

    Content is returned unfenced — the caller fences it, because only the
    tool boundary knows what a source column means (see docs/security.md).
    """
    # A non-string `data_type` reached `data_type = ?` and raised
    # `sqlite3.ProgrammingError: type 'list' is not supported` straight out of
    # the plugin door, or — for an int — reported `orphan_count: 0` over a
    # corpus it never scanned, which reads as a healthy graph. One of six
    # instances of the same class; see `constants.str_filter_error`.
    # 2026-09-16 argument-validation enumeration.
    _C.require_str_filters(data_type=data_type)
    try:
        import numpy as np
    except ImportError:
        return {"error": "numpy unavailable — graph_health needs it for the scan"}

    try:
        threshold = float(threshold)
    except (TypeError, ValueError):
        return {"error": f"threshold must be a number, got {threshold!r}"}
    if not 0.0 <= threshold <= 1.0:
        return {"error": f"threshold must be in [0.0, 1.0], got {threshold}"}
    try:
        limit = max(2, min(int(limit), 5000))
    except (TypeError, ValueError):
        limit = 200

    where = "status='active' AND superseded_by IS NULL AND embedding IS NOT NULL"
    params = []
    if data_type:
        where += " AND data_type = ?"
        params.append(data_type)
    if self._profile_name and self._has_profile_column():
        # `IS NULL OR = ?`, not exact match — deliberately, and not the same
        # shape as summaries.py's fixed NULL-ownership bug despite looking
        # identical. `digests.db` is one file *shared by every profile*, so a
        # NULL row there could be anyone's and treating it as "mine" was a
        # real cross-profile leak. `memories.db` is one file *per profile* —
        # this connection can only ever see rows this profile wrote, so a
        # NULL `profile_name` here means "written before the column existed,
        # in this very database," not "some other profile's." Excluding it
        # would drop this profile's own pre-migration records from its own
        # health report. 2026-08-22 review round 3, maintenance F15.
        where += " AND (profile_name IS NULL OR profile_name = ?)"
        params.append(self._profile_name)

    total = self._get_conn().execute(
        f"SELECT COUNT(*) FROM memories WHERE {where}", params).fetchone()[0]
    # Newest first: a truncated scan should cover what was written recently
    # rather than an arbitrary slice.
    rows = self._get_conn().execute(
        f"SELECT uuid, embedding, topic, data_type, source, created_at, "
        f"trust_score, COALESCE(summary, substr(content, 1, 120)) "
        f"FROM memories WHERE {where} ORDER BY created_at DESC LIMIT ?",
        params + [limit]).fetchall()

    kept, vectors = [], []
    dim = None
    for row in rows:
        vec = _unpack_embedding(row[1])
        if not vec:
            continue
        if dim is None:
            dim = len(vec)
        elif len(vec) != dim:
            # Written by a different embedding model — not comparable, and
            # scoring it against this set would invent a distance.
            continue
        kept.append(row)
        vectors.append(vec)

    if len(kept) < 2:
        return {"checked": len(kept), "total_eligible": total, "threshold": threshold,
                "orphans": [], "orphan_count": 0, "truncated": total > len(kept),
                "note": "need at least 2 comparable records to find orphans"}

    m = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    m = m / norms
    sims = m @ m.T
    np.fill_diagonal(sims, -1.0)  # a record is not its own neighbour
    best = sims.max(axis=1)

    orphans = []
    for i, row in enumerate(kept):
        if float(best[i]) >= threshold:
            continue
        orphans.append({
            "uuid": row[0], "topic": row[2], "data_type": row[3],
            "source": row[4], "created_at": row[5], "trust_score": row[6],
            "excerpt": row[7], "nearest_similarity": round(float(best[i]), 4),
        })
    orphans.sort(key=lambda o: o["nearest_similarity"])

    return {"checked": len(kept), "total_eligible": total, "threshold": threshold,
            "orphans": orphans, "orphan_count": len(orphans),
            "truncated": total > len(kept),
            "mean_nearest": round(float(best.mean()), 4)}


__all__ = ['graph_health', 'rebuild', 'resolve_conflicts', 'sync_check', 'sleep', 'compact', 'find_duplicate_groups', '_compact_group', 'purge', 'export_memories', 'import_memories', 'decay', '_parse_frontmatter', '_extract_wikilinks', '_embed_query_text', 'ingest_obsidian']
