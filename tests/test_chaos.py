"""T180-T196, T215: Degraded-mode and cross-profile safety tests.

Most of this system is fallback logic — Qdrant offline, embeddings failing,
FTS5 unusable, LLM unconfigured. Until now exactly one test covered a degraded
path, while the last three real defects all lived in that code:

  * the Qdrant-outage fallback returned the oldest 30 records in insertion
    order, ignoring the query entirely (fixed, T171);
  * the circuit breaker logged "OPEN" on the first failure and never re-opened
    after a failed half-open probe (fixed, T172/T173);
  * `_call_llm` returning "" silently disabled six features (fixed, T161).

Every one of those looked healthy from the happy path. These tests break
things on purpose and assert that the system degrades usefully rather than
quietly.

T215 is the operational preflight: it validates that the embedding endpoint
and model are configured and reachable before the rest of the suite runs,
catching dead endpoints early instead of letting 200+ tests degrade silently.

No Docker manipulation: failures are injected by pointing the backend at a
dead endpoint or forcing breaker state, so the suite stays fast and does not
disturb a running Qdrant.
"""

import json
import os
import time

from conftest import _make_backend, _cleanup_db, _cleanup_qdrant_coll, _get_uuid

CORPUS = [
    ("Redis uses an allkeys-lru eviction policy with 512MB maxmemory.", "ENV-DATA", "sw"),
    ("The workstation has an NVIDIA RTX 4090 with 24GB of VRAM.", "ENV-DATA", "hw"),
    ("PostgreSQL 16 listens on port 5432 for the analytics warehouse.", "ENV-DATA", "sw"),
    ("The user prefers Neovim over VS Code for all editing work.", "USER-DATA", "preferences"),
    ("Never push to the remote repository without explicit approval.", "SYSTEM", "rules"),
]


def _seed(be):
    return {c[:20]: _get_uuid(be.add(c, data_type=dt, data_id=di, force=True))
            for c, dt, di in CORPUS}


def test_t180():
    """Qdrant offline: retrieval still answers, and answers relevantly."""
    be = _make_backend("t180")
    try:
        _seed(be)
        be._qdrant_broken_until = time.time() + 120  # circuit forced open

        results = be.retrieve("redis eviction policy", max_layer=2, limit=5)
        assert results, "an outage must not produce an empty result set"
        assert "Redis" in results[0]["content"], (
            f"degraded retrieval returned an irrelevant record first: "
            f"{results[0]['content'][:60]!r}"
        )
    finally:
        be._qdrant_broken_until = None
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t180")


def test_t181():
    """Qdrant offline: the degradation is reported, not silent."""
    be = _make_backend("t181")
    try:
        _seed(be)
        assert be.sync_check()["degraded_retrieval"] is False, "should start healthy"

        be._qdrant_broken_until = time.time() + 120
        be.retrieve("redis eviction policy", max_layer=1, limit=5)

        assert be._degraded is True, "degraded flag not set"
        status = be.sync_check()
        assert status["degraded_retrieval"] is True, "sync_check hides the degradation"
        assert status["qdrant_circuit_open"] is True, "circuit state not reported"
        assert be.retrieval_stats()["degraded_retrieval"] is True, "stats hide it"
    finally:
        be._qdrant_broken_until = None
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t181")


def test_t182():
    """Writes still succeed while Qdrant is unavailable.

    SQLite is the source of truth; an index outage must not block durability.
    """
    be = _make_backend("t182")
    try:
        be._qdrant_broken_until = time.time() + 120
        uuid = _get_uuid(be.add("Grafana dashboards are served on port 3000.",
                                data_type="ENV-DATA", data_id="sw", force=True))
        assert uuid, "add() must succeed during a Qdrant outage"

        row = be._get_conn().execute(
            "SELECT content, status FROM memories WHERE uuid = ?", (uuid,)).fetchone()
        assert row and row[1] == "active", "record should be durable in SQLite"

        found = be.retrieve("grafana dashboards port", max_layer=1, limit=5)
        assert any(r["uuid"] == uuid for r in found), (
            "a record written during an outage should still be findable"
        )
    finally:
        be._qdrant_broken_until = None
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t182")


def test_t183():
    """A dead embedding endpoint degrades to lexical search, not a crash."""
    import backend.core as _b

    be = _make_backend("t183")
    saved = _b._embedding_fn
    try:
        _seed(be)

        def dead_embedder(texts):
            raise ConnectionError("embedding endpoint refused connection")

        _b._embedding_fn = dead_embedder
        try:
            results = be.retrieve("redis eviction policy", max_layer=1, limit=5)
        except Exception as e:
            raise AssertionError(
                f"a dead embedding endpoint should degrade, not raise: {e!r}"
            )
        # Whatever comes back must not be an exception or a stack trace; an
        # empty list is acceptable, wrong-but-confident results are not.
        if results:
            assert "Redis" in results[0]["content"], (
                f"lexical fallback returned an irrelevant record: "
                f"{results[0]['content'][:60]!r}"
            )
    finally:
        _b._embedding_fn = saved
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t183")


def test_t184():
    """A failing embedder during add() does not lose the record."""
    import backend.core as _b

    be = _make_backend("t184")
    saved = _b._embedding_fn
    try:
        def dead_embedder(texts):
            return [None] * len(texts)

        _b._embedding_fn = dead_embedder
        uuid = _get_uuid(be.add("The backup window runs from 02:00 to 04:00 UTC.",
                                data_type="ENV-DATA", data_id="sw", force=True))
        assert uuid, "add() must not fail when embedding is unavailable"

        row = be._get_conn().execute(
            "SELECT content FROM memories WHERE uuid = ?", (uuid,)).fetchone()
        assert row, "record must be durable even without a vector"
    finally:
        _b._embedding_fn = saved
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t184")


def test_t185():
    """Batch embedding survives a failing batch by falling back per text."""
    be = _make_backend("t185")
    try:
        import backend.core as _b
        saved = _b._embedding_fn  # None unless a fault is already injected — restored in finally
        real_fn = _b._get_embedding_fn(be._embedding_model)  # the actual working embedder to delegate to
        state = {"batch_calls": 0}

        def flaky(texts):
            # Fail on multi-text calls, succeed one at a time
            if len(texts) > 1:
                state["batch_calls"] += 1
                raise RuntimeError("batch too large")
            return real_fn(texts)

        _b._embedding_fn = flaky
        try:
            vectors = be._embed_batch(["alpha text", "beta text", "gamma text"])
        finally:
            _b._embedding_fn = saved

        assert state["batch_calls"] >= 1, "the batch path should have been attempted"
        assert len(vectors) == 3, f"every text must get a slot: {len(vectors)}"
        assert all(v is not None for v in vectors), (
            "per-text fallback should recover all three vectors"
        )
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t185")


def test_t186():
    """Recovery: the circuit closes again and vector search resumes."""
    be = _make_backend("t186")
    try:
        _seed(be)
        be._qdrant_broken_until = time.time() + 120
        be.retrieve("redis eviction policy", max_layer=1, limit=5)
        assert be._degraded is True, "should be degraded while open"

        # Cooldown elapses, the half-open trial succeeds
        be._qdrant_broken_until = None
        be._degraded = False
        results = be.retrieve("redis eviction policy", max_layer=2, limit=5)

        assert results, "retrieval should work after recovery"
        assert be._degraded is False, "recovered retrieval should not be flagged degraded"
        assert be.sync_check()["qdrant_circuit_open"] is False, "circuit should be closed"
    finally:
        be._qdrant_broken_until = None
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t186")


def test_t187():
    """Semantic search works with Qdrant switched off entirely.

    HLM_QDRANT_ENABLED=false previously meant lexical-only retrieval. Vectors
    are already stored as packed float32, so an exact scan is one numpy matmul
    at this scale — which is what lets the suite run without a Docker
    container.
    """
    be = _make_backend("t187")
    try:
        _seed(be)
        be._qdrant_enabled = False   # SQLite-only mode

        # A query with no lexical overlap with the target — only semantics can
        # find it, so this fails if the path falls back to FTS5.
        results = be.retrieve("what graphics card is installed", max_layer=1, limit=5)
        assert results, "SQLite-only mode returned nothing"
        assert any("RTX 4090" in r["content"] for r in results), (
            "brute-force vector search did not find the semantically matching "
            f"record: {[r['content'][:40] for r in results]}"
        )
    finally:
        be._qdrant_enabled = True
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t187")


def test_t188():
    """Brute-force search ranks by similarity and respects filters."""
    be = _make_backend("t188")
    try:
        _seed(be)
        import backend.core as _b
        fn = _b._get_embedding_fn()
        qv = fn(["which editor is preferred for writing code"])[0]

        hits = be._brute_force_search(qv, limit=5)
        assert hits, "brute-force search returned nothing"
        assert len(hits[0]) == 2, "should return (uuid, distance) tuples"

        distances = [d for _, d in hits]
        assert distances == sorted(distances), f"results not sorted by distance: {distances}"
        assert all(0.0 <= d <= 2.0 for d in distances), f"distances out of range: {distances}"

        top_uuid = hits[0][0]
        content = be._get_conn().execute(
            "SELECT content FROM memories WHERE uuid = ?", (top_uuid,)).fetchone()[0]
        assert "Neovim" in content, f"expected the editor preference first, got {content[:50]!r}"

        # data_type filter must be honoured
        filtered = be._brute_force_search(qv, data_type="SYSTEM", limit=5)
        for uuid, _ in filtered:
            dt = be._get_conn().execute(
                "SELECT data_type FROM memories WHERE uuid = ?", (uuid,)).fetchone()[0]
            assert dt == "SYSTEM", f"filter leaked a {dt} record"
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t188")


def test_t194():
    """Cross-profile search must not delete other profiles' Qdrant points.

    Qdrant collections are shared across profiles, but _detect_orphans checked
    candidates against the *current* profile's SQLite only. During a
    cross-profile read every other profile's record therefore looked orphaned
    and was deleted. Observed in a real E2E run: 83 candidates in, 78 declared
    orphaned and removed from Qdrant, 5 returned — destructive cleanup
    performed by a read, against other profiles' data.
    """
    be = _make_backend("t194")
    try:
        _seed(be)
        # Candidates that do not exist in this profile's SQLite at all
        foreign = [("ffffffffffffffffffffffffffffff01", 0.2),
                   ("ffffffffffffffffffffffffffffff02", 0.3)]

        deleted = []
        real_delete = be._qdrant.delete
        be._qdrant.delete = lambda **kw: deleted.append(kw)
        try:
            kept = be._detect_orphans(list(foreign), cross_profile=True)
        finally:
            be._qdrant.delete = real_delete

        assert not deleted, (
            f"cross-profile read deleted from Qdrant: {deleted}"
        )
        assert len(kept) == len(foreign), (
            "other profiles' candidates must survive to be hydrated by Layer 1, "
            f"got {len(kept)} of {len(foreign)}"
        )
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t194")


def test_t195():
    """Single-profile reads still clean genuine orphans."""
    be = _make_backend("t195")
    try:
        _seed(be)
        orphan = [("ffffffffffffffffffffffffffffff03", 0.2)]

        deleted = []
        real_delete = be._qdrant.delete
        be._qdrant.delete = lambda **kw: deleted.append(kw)
        try:
            kept = be._detect_orphans(list(orphan), cross_profile=False)
        finally:
            be._qdrant.delete = real_delete

        assert deleted, "a true orphan should still be cleaned on a single-profile read"
        assert kept == [], f"orphan should be filtered from candidates: {kept}"
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t195")


def test_t196():
    """A profile on an older schema still contributes to cross-profile results.

    The supersession filter references superseded_by, a v14 column added when
    this code opens a database. A cross-profile read opens other profiles'
    files directly, so an un-migrated peer raised "no such column" — swallowed
    by a broad except into a debug line, silently contributing nothing.
    """
    import shutil
    import sqlite3

    be = _make_backend("t196")
    try:
        be.add("The legacy profile records nginx on port 8443.",
               data_type="ENV-DATA", data_id="net", force=True)
        # Materialise the WAL so the copy is complete
        be._get_conn().execute("PRAGMA wal_checkpoint(TRUNCATE)")
        be._get_conn().commit()

        legacy = os.path.join(os.path.dirname(be._db_path), "legacy-peer.db")
        shutil.copy(be._db_path, legacy)
        lc = sqlite3.connect(legacy)
        # The v14 index references the column, so it must go first
        lc.execute("DROP INDEX IF EXISTS idx_mem_superseded")
        lc.execute("ALTER TABLE memories DROP COLUMN superseded_by")
        lc.execute("ALTER TABLE memories DROP COLUMN superseded_at")
        lc.commit()
        cols = [r[1] for r in lc.execute("PRAGMA table_info(memories)")]
        lc.close()
        assert "superseded_by" not in cols, "peer should be on the older schema"

        be._discover_profile_dbs = lambda: {"legacy": legacy,
                                            be._profile_name: be._db_path}
        results = be.retrieve("nginx port 8443", max_layer=1, limit=10,
                              cross_profile=True)
        profiles = {r.get("profile_name") for r in results}
        assert "legacy" in profiles, (
            f"older-schema profile silently dropped from cross-profile results: "
            f"{sorted(p for p in profiles if p)}"
        )
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t196")


def test_t215():
    """Preflight — embedding endpoint and model are configured and reachable.

    The embedding model is configured per-profile in <profile>/.env. When
    the endpoint is down or the model name is unset, every vector operation
    silently degrades to lexical-only — which most tests still pass but
    never exercise the paths they claim to. This test fails loudly so the
    whole suite doesn't waste its time on results from a broken embedder.
    """
    embed_url = os.environ.get("HLM_EMBED_URL")
    embed_model = os.environ.get("HLM_EMBED_MODEL")

    assert embed_url, (
        "HLM_EMBED_URL is not set — the embedding endpoint is "
        "configured per-profile in <profile>/.env, not in shared config. "
        "Run tests under `hermes -p <profile>` or set the var explicitly."
    )
    assert embed_model, (
        "HLM_EMBED_MODEL is not set — the embedding model name "
        "is configured per-profile in <profile>/.env. Run tests under "
        "`hermes -p <profile>` or set the var explicitly."
    )

    # Probe the endpoint with a minimal embedding request — the Ollama
    # /api/embed path only accepts POST, so a GET probe would return 405
    # even when the server and model are perfectly healthy.
    import urllib.request
    try:
        payload = json.dumps({"model": embed_model, "input": ["preflight"]}).encode()
        req = urllib.request.Request(embed_url, data=payload,
                                     headers={"Content-Type": "application/json"})
        resp = urllib.request.urlopen(req, timeout=5)
        body = json.loads(resp.read())
        assert body.get("embeddings"), (
            f"embedding endpoint {embed_url} returned no embeddings — "
            f"the model '{embed_model}' may not be loaded"
        )
    except Exception as e:
        raise AssertionError(
            f"embedding endpoint {embed_url} is unreachable: {e}"
        ) from e


def test_t216():
    """Retrieval mode distinguishes brute-force from lexical degradation.

    `_degraded` alone conflates two very different outcomes: a brute-force
    cosine scan is exact and semantic (only slower), while an FTS5/BM25
    fallback loses most of the recall (measured 0.227). An operator reading
    `degraded_retrieval: true` could not tell which had happened.
    """
    be = _make_backend("t216")
    try:
        _seed(be)
        # Healthy: Qdrant ANN, nothing degraded yet.
        be.retrieve("redis eviction policy", max_layer=1, limit=5)
        s = be.sync_check()
        assert s["retrieval_mode"] == "vector", f"expected vector, got {s['retrieval_mode']}"
        assert s["degraded_retrieval"] is False
        assert s["degraded_count"] == 0

        # Circuit open: brute-force over locally stored vectors — degraded,
        # but NOT lexical, and that distinction is the whole point.
        be._qdrant_broken_until = time.time() + 120
        be.retrieve("redis eviction policy", max_layer=1, limit=5)
        s = be.sync_check()
        assert s["retrieval_mode"] == "brute_force", f"expected brute_force, got {s['retrieval_mode']}"
        assert s["degraded_retrieval"] is True, "sticky flag should latch"
        assert s["degraded_count"] == 1, f"expected 1 degraded retrieval, got {s['degraded_count']}"
        assert s["degraded_last"], "degraded_last timestamp should be set"

        # Recovered: mode returns to vector; the sticky flag stays True
        # because it answers "did this ever degrade?", not "is it now?".
        be._qdrant_broken_until = None
        be.retrieve("redis eviction policy", max_layer=1, limit=5)
        s = be.sync_check()
        assert s["retrieval_mode"] == "vector", f"expected vector after recovery, got {s['retrieval_mode']}"
        assert s["degraded_retrieval"] is True, "sticky flag must not be cleared by recovery"
        assert s["degraded_count"] == 1, "a healthy retrieval must not increment the counter"
    finally:
        be._qdrant_broken_until = None
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t216")


def test_t217():
    """A successful write must not clear read-degradation state.

    _record_qdrant_success() is called from add/update/delete as well as
    _layer0 — three of its four call sites are writes. Resetting retrieval
    health there would let ordinary write traffic mask degraded reads, which
    is exactly when an operator most needs to see it.
    """
    be = _make_backend("t217")
    try:
        _seed(be)
        be._set_retrieval_mode("lexical")
        assert be.sync_check()["retrieval_mode"] == "lexical"

        # A write that reaches Qdrant successfully closes the circuit...
        be.add("an unrelated healthy write", force=True)

        # ...but must leave the retrieval-quality state alone.
        s = be.sync_check()
        assert s["retrieval_mode"] == "lexical", (
            f"a successful write cleared read-degradation state: {s['retrieval_mode']}")
        assert s["degraded_retrieval"] is True
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t217")
