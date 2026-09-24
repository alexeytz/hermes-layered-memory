"""T71-T104: Compaction source, multi-collection, history, SQLite-only, entities, tracing."""

import os
import json
import tempfile
import shutil
import time

from conftest import (
    _make_backend, _make_summaries, _cleanup_db, _cleanup_qdrant_coll,
    _get_uuid,
)


# ── Compaction source preservation ──────────────────────────────────────────

def test_t71():
    """Compaction source preservation — obsidian precedence."""
    be = _make_backend("t71")
    try:
        be.add("obsidian note", data_type="OBSIDIAN", data_id="test", source="obsidian")
        be.add("agent note", data_type="OBSIDIAN", data_id="test", source="agent")
        assert isinstance(be.compact(similarity_threshold=0.85), dict)
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t71")


def test_t72():
    """Compaction source preservation — agent-only cluster."""
    be = _make_backend("t72")
    try:
        be.add("agent note A", data_type="ENV-DATA", data_id="hw", source="agent")
        be.add("agent note B", data_type="ENV-DATA", data_id="hw", source="agent")
        assert isinstance(be.compact(similarity_threshold=0.85), dict)
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t72")


def test_t73():
    """Compaction source preservation — empty source fallback."""
    be = _make_backend("t73")
    try:
        be.add("no source A", data_type="CUSTOM", source=None)
        be.add("no source B", data_type="CUSTOM", source=None)
        assert isinstance(be.compact(similarity_threshold=0.85), dict)
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t73")


def test_t74():
    """Dedup with default data_type — checks correct collection."""
    be = _make_backend("t74")
    try:
        be.add("My GPU is an RTX 3090 with 24GB VRAM")
        result = be.add("My GPU is an RTX 3090 with 24GB VRAM")
        assert isinstance(result, dict)
        assert result.get("status") in ("duplicate", "possible_duplicate")
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t74")


def test_t75():
    """Profile-scoped summary delete — missing profile_name refused."""
    sb = _make_summaries("t75")
    try:
        r = sb.add("https://example.com/t75", "web", "T75", highlights=["h"], full_text="text")
        assert sb.delete(r["uuid"], profile_name=None, profile="own") is False
    finally:
        sb.close(); _cleanup_db("t75"); shutil.rmtree(sb._summaries_dir, ignore_errors=True)


def test_t76():
    """L2 RRF fusion score range."""
    be = _make_backend("t76")
    try:
        for i in range(5):
            be.add(f"GPU item {i} with different topic", data_type="ENV-DATA", data_id="hw")
        results = be.retrieve("GPU", limit=5, max_layer=2)
        assert len(results) >= 1 and results[0].get("fusion_score") is not None
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t76")


def test_t76b():
    """L2 keyword/topic boost strips punctuation (BUG-016).

    query.lower().split() splits on whitespace only, so "setup?" != "setup".
    The keyword/topic boost should match words with trailing punctuation.
    """
    be = _make_backend("t76b")
    try:
        # Add record with keyword "setup" and topic "config"
        be.add("The system setup requires config.",
               data_type="USER-DATA", data_id="prefs",
               keywords=["setup", "deploy"],
               topic="config", force=True)

        # Query with punctuation — should still match keyword/topic
        results = be.retrieve("What's the setup?", max_layer=2, limit=5)
        assert len(results) >= 1, f"expected results, got {results}"

        # The keyword/topic boost should have been applied
        top = results[0]
        assert "fusion_score" in top, f"L2 should set fusion_score: {top}"

        # The top result should be our record (not a punctuation miss)
        assert "setup" in (top.get("content") or "").lower() or \
               "setup" in (top.get("keywords") or []), \
               f"top result should match 'setup' keyword: {top}"
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t76b")


def test_t77():
    """Conflict temporal guard — 1 day default."""
    be = _make_backend("t77")
    try:
        assert be._conflict_thresholds().get("temporal_guard_days") == 1
    finally:
        be.close(); _cleanup_db("t77")


def test_t78():
    """Compaction trust boost — capped at +0.2."""
    be = _make_backend("t78")
    try:
        for i in range(3):
            be.add(f"similar low trust {i}", data_type="ENV-DATA", data_id="hw", trust_score=0.2)
        assert isinstance(be.compact(similarity_threshold=0.85), dict)
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t78")


def test_t79():
    """Heuristic gate — only upgrades, never downgrades."""
    be = _make_backend("t79")
    try:
        be.add("simple word content", data_type="CUSTOM")
        results = be.retrieve("simple word", max_layer=3)
        assert len(results) >= 1
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t79")


def test_t80():
    """LLM enrichment fallback — uses heuristic on failure."""
    be = _make_backend("t80")
    try:
        uuid = be.add("RTX 4090 GPU", data_type="CUSTOM")
        record = be._get_record(uuid)
        assert record["data_type"] in ("ENV-DATA", "CUSTOM")
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t80")


# ── Multi-collection ────────────────────────────────────────────────────────

def test_t81():
    """YouTube /shorts canonicalization."""
    sb = _make_summaries("t81")
    try:
        url = sb._canonicalize_url("https://youtube.com/shorts/abc123")
        assert url == "https://www.youtube.com/watch?v=abc123"
    finally:
        sb.close(); _cleanup_db("t81"); shutil.rmtree(sb._summaries_dir, ignore_errors=True)


def test_t82():
    """Multi-collection — OBSIDIAN routes to vault."""
    be = _make_backend("t82")
    try:
        tmp = tempfile.mkdtemp()
        with open(os.path.join(tmp, "note.md"), "w") as f:
            f.write("Obsidian test note")
        be.ingest_obsidian(tmp)
        shutil.rmtree(tmp, ignore_errors=True)
        sync = be.sync_check()
        assert sync.get("qdrant_total", 0) >= 1
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t82")


def test_t83():
    """Multi-collection — SESSION-DATA routes to sessions."""
    be = _make_backend("t83")
    try:
        be.add("session note", data_type="SESSION-DATA", data_id="sess_abc")
        results = be.retrieve("session note", data_type="SESSION-DATA", max_layer=1)
        assert len(results) >= 1
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t83")


def test_t84():
    """Multi-collection — cross-collection search."""
    be = _make_backend("t84")
    try:
        be.add("hw spec", data_type="ENV-DATA", data_id="hw")
        be.add("session note", data_type="SESSION-DATA", data_id="sess84")
        results = be.retrieve("anything", max_layer=1)
        types = {r["data_type"] for r in results}
        assert len(types) >= 2
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t84")


# ── History archive ────────────────────────────────────────────────────────

def test_t85():
    """History archive — add logs JSONL entry."""
    be = _make_backend("t85")
    try:
        be.add("history test add", data_type="CUSTOM")
        hist_path = be._get_history_path()
        assert os.path.exists(hist_path)
        with open(hist_path) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        assert any(l["action"] == "add" for l in lines)
    finally:
        be.close(); _cleanup_db("t85")


def test_t86():
    """History archive — update logs old + new fields."""
    be = _make_backend("t86")
    try:
        uuid = be.add("history update test", data_type="CUSTOM")
        be.update(uuid, content="updated content")
        with open(be._get_history_path()) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        update_entries = [l for l in lines if l.get("action") == "update"]
        assert len(update_entries) >= 1
        entry = update_entries[0]
        assert "fields" in entry and "new_fields" in entry
    finally:
        be.close(); _cleanup_db("t86")


def test_t87():
    """History archive — delete logs old fields."""
    be = _make_backend("t87")
    try:
        uuid = be.add("history delete test", data_type="CUSTOM")
        be.delete(uuid)
        with open(be._get_history_path()) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        delete_entries = [l for l in lines if l.get("action") == "delete"]
        assert len(delete_entries) >= 1 and "fields" in delete_entries[0]
    finally:
        be.close(); _cleanup_db("t87")


def test_t88():
    """History archive — rotation at 10MB."""
    be = _make_backend("t88")
    try:
        hist_path = be._get_history_path()
        with open(hist_path, "w") as f:
            entry = json.dumps({"ver": 1, "ts": "2025-01-01", "uuid": "x" * 32,
                                "action": "add", "new_fields": {"content": "x" * 1000}})
            for _ in range(11000):
                f.write(entry + "\n")
        assert os.path.getsize(hist_path) > 10 * 1024 * 1024
        be._rotate_history(hist_path)
        assert os.path.exists(hist_path + ".1")
    finally:
        be.close(); _cleanup_db("t88")


# ── SQLite-only mode ───────────────────────────────────────────────────────

def test_t89():
    """Enrichment llm=False skips LLM."""
    be = _make_backend("t89")
    try:
        result = be._enrich_metadata("RTX 4090 GPU", llm=False)
        assert result.get("data_type") in ("ENV-DATA", "CUSTOM")
    finally:
        be.close(); _cleanup_db("t89")


def test_t90():
    """Enrichment llm=True forces LLM."""
    be = _make_backend("t90")
    try:
        result = be._enrich_metadata("RTX 4090 GPU", llm=True)
        assert isinstance(result, dict)
    finally:
        be.close(); _cleanup_db("t90")


def test_t91():
    """SQLite-only mode — HLM_QDRANT_ENABLED=false."""
    old = os.environ.get("HLM_QDRANT_ENABLED")
    os.environ["HLM_QDRANT_ENABLED"] = "false"
    try:
        be = _make_backend("t91")
        try:
            assert be._qdrant is None
            uuid = be.add("sqlite only test", data_type="CUSTOM")
            assert len(be.retrieve("sqlite only", max_layer=1)) >= 1
        finally:
            be.close(); _cleanup_db("t91")
    finally:
        if old is not None:
            os.environ["HLM_QDRANT_ENABLED"] = old
        else:
            os.environ.pop("HLM_QDRANT_ENABLED", None)


def test_t92():
    """SQLite-only mode — circuit breaker not triggered on retrieves."""
    old = os.environ.get("HLM_QDRANT_ENABLED")
    os.environ["HLM_QDRANT_ENABLED"] = "false"
    try:
        be = _make_backend("t92")
        try:
            be.add("no cb test", data_type="CUSTOM")
            initial_failures = be._qdrant_failures
            for _ in range(10):
                results = be.retrieve("no cb", max_layer=1)
                assert len(results) >= 1
            assert be._qdrant_failures == initial_failures, \
                f"Circuit breaker triggered during retrieves: {initial_failures} → {be._qdrant_failures}"
        finally:
            be.close(); _cleanup_db("t92")
    finally:
        if old is not None:
            os.environ["HLM_QDRANT_ENABLED"] = old
        else:
            os.environ.pop("HLM_QDRANT_ENABLED", None)


# ── Entity patterns ────────────────────────────────────────────────────────

def test_t97():
    """Entity patterns — loaded from default."""
    be = _make_backend("t97")
    try:
        assert len(be._load_entity_patterns()) >= 1
    finally:
        be.close(); _cleanup_db("t97")


def test_t98():
    """Entity patterns — extraction from text."""
    be = _make_backend("t98")
    try:
        entities = be._extract_entities("Using RTX 4090 with qwen3-embedding:8b on embed-host.example.net")
        assert len(entities) >= 1
    finally:
        be.close(); _cleanup_db("t98")


def test_t99():
    """Query expansion."""
    be = _make_backend("t99", config={"query_expand": True})
    try:
        be.add("RTX GPU with CUDA cores", data_type="ENV-DATA", data_id="hw")
        assert len(be.retrieve("GPU", max_layer=1)) >= 1
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t99")


def test_t100():
    """Entity patterns — custom config override."""
    be = _make_backend("t100", config={})
    try:
        patterns = be._load_entity_patterns()
        assert len(patterns) >= 1
    finally:
        be.close(); _cleanup_db("t100")


# ── Tracing ────────────────────────────────────────────────────────────────

def test_t101():
    """Query-path tracing — write and read."""
    be = _make_backend("t101", config={"tracing": True})
    try:
        be.add("trace test", data_type="CUSTOM")
        be.retrieve("trace", max_layer=1)
        assert len(be.get_traces()) >= 1
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t101")


def test_t102():
    """Query-path tracing — disabled by default."""
    be = _make_backend("t102", config={})
    try:
        be.add("no trace", data_type="CUSTOM")
        be.retrieve("no trace", max_layer=1)
        traces = be.get_traces()
        if traces:
            our_traces = [t for t in traces if "no trace" in t.get("query", "").lower()]
            assert len(our_traces) == 0
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t102")


def test_t103():
    """Query-path tracing — search by query."""
    be = _make_backend("t103", config={"tracing": True})
    try:
        be.add("trace A", data_type="CUSTOM")
        be.add("trace B", data_type="CUSTOM")
        be.retrieve("trace A", max_layer=1)
        be.retrieve("trace B", max_layer=1)
        assert len(be.get_traces(query="trace A")) >= 1
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t103")


def test_t104():
    """Query-path tracing — limit results."""
    be = _make_backend("t104", config={"tracing": True})
    try:
        for i in range(10):
            be.add(f"trace item {i}", data_type="CUSTOM")
            be.retrieve(f"trace item {i}", max_layer=1)
        assert len(be.get_traces(limit=3)) <= 3
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t104")

def test_t633():
    """`get_traces` returns the NEWEST traces, not the oldest.

    The files were sorted by mtime descending — correct — and then each was
    read *line-forward* with a `break` at `limit`. A `.jsonl` trace file is
    appended to, so its lines run oldest-to-newest: the newest file's *oldest*
    lines came back, and everything after the first `limit` entries was
    unreachable by any argument, because rotation prunes whole files and never
    lines inside one.

    Found on the 2026-09-13 e2e, not by an assertion but by reading what came
    back: `.hlm-traces/unknown.jsonl` held 6,067 entries spanning 2026-07-22 to
    2026-09-13T03:47 — the newest written three minutes earlier by the run that
    was reading them — and `layered_advanced(action="traces")` returned the ten
    from **July 22**. 6,057 of 6,067 were unreachable.

    `test_t104` already covered this surface and passed throughout: it asserts
    `len(get_traces(limit=3)) <= 3` and never asks *which* three. That is the
    house rule — *a count is not a membership test* — costing a real defect, so
    this one asserts identity and order and nothing about length alone.

    Worst case for a debugging tool: live write path, confident read path, and
    an answer describing a different week. The standing advice is to suspect
    the measurement before the system; here the measurement was the system.
    """
    be = _make_backend("t633", config={"tracing": True})
    try:
        issued = []
        for i in range(12):
            q = f"trace seq {i:02d}"
            be.add(f"record for {q}", data_type="CUSTOM")
            be.retrieve(q, max_layer=1)
            issued.append(q)

        got = be.get_traces(limit=3)
        assert len(got) == 3, f"expected 3 traces, got {len(got)}"
        queries = [t.get("query") for t in got]

        assert queries == issued[-1:-4:-1], (
            f"get_traces(limit=3) returned {queries}, expected the three most "
            f"recent queries newest-first {issued[-1:-4:-1]}. Returning "
            f"{issued[:3]} instead means the read still walks the file forward "
            f"from its start, so every trace after the first `limit` entries is "
            f"unreachable — which is what shipped until 0.8.61")

        # The query filter must also return recent matches, not the oldest
        # ones: the deque is per-file and filters before it fills.
        for i in range(4):
            be.retrieve("trace seq 11", max_layer=1)
        filtered = be.get_traces(query="trace seq 11", limit=2)
        assert filtered and all("trace seq 11" in t.get("query", "") for t in filtered), (
            f"query-filtered traces came back as {[t.get('query') for t in filtered]}")
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t633")
