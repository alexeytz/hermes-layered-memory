"""T59-T70, T167-T168: Conflict detection, thresholds, L3 offline fallback."""

from conftest import _make_backend, _cleanup_db, _cleanup_qdrant_coll, _get_uuid


def test_t59():
    """Conflict detection — proxy port mutation."""
    be = _make_backend("t59")
    try:
        be.add("Proxy is on port 8080", data_type="ENV-DATA", data_id="net")
        be.add("Proxy moved to port 9090", data_type="ENV-DATA", data_id="net")
        results = be.retrieve("proxy port", max_layer=2)
        assert len(results) >= 1
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t59")


def test_t60():
    """Conflict detection — daily notes skipped."""
    be = _make_backend("t60")
    try:
        for i in range(5):
            be.add(f"Project status update {i}", data_type="SESSION-DATA", data_id="sess60")
        results = be.retrieve("project status", max_layer=2)
        assert len(results) >= 1
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t60")


def test_t61():
    """Conflict detection — different folders skipped."""
    be = _make_backend("t61")
    try:
        be.add("API endpoint /v1", data_type="ENV-DATA", data_id="sw")
        be.add("API endpoint /v2", data_type="ENV-DATA", data_id="net")
        results = be.retrieve("API endpoint", max_layer=2)
        assert len(results) >= 1
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t61")


def test_t62():
    """Conflict detection — edge cases (no crash)."""
    be = _make_backend("t62")
    try:
        results = be.retrieve("nonexistent xyzzy", max_layer=2)
        assert isinstance(results, list)
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t62")


def test_t63():
    """Conflict detection — null embeddings."""
    be = _make_backend("t63")
    try:
        uuid = be.add("null embed conflict", data_type="CUSTOM")
        be._get_conn().execute("UPDATE memories SET embedding = 'null' WHERE uuid=?", (uuid,))
        be._get_conn().commit()
        results = be.retrieve("null embed", max_layer=2)
        assert isinstance(results, list)
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t63")


def test_t64():
    """Conflict detection — empty keywords."""
    be = _make_backend("t64")
    try:
        be.add("similar content A", data_type="CUSTOM")
        be.add("similar content B", data_type="CUSTOM")
        results = be.retrieve("similar content", max_layer=2)
        assert isinstance(results, list)
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t64")


def test_t65():
    """Compaction with contradictions."""
    be = _make_backend("t65")
    try:
        be.add("proxy port 8080", data_type="ENV-DATA", data_id="net")
        be.add("proxy port 9090", data_type="ENV-DATA", data_id="net")
        result = be.compact(similarity_threshold=0.85)
        assert isinstance(result, dict)
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t65")


def test_t66():
    """No regression — existing tests still work."""
    be = _make_backend("t66")
    try:
        be.add("regression test", data_type="CUSTOM")
        results = be.retrieve("regression", max_layer=1)
        assert len(results) >= 1
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t66")


def test_t67():
    """Config conflict thresholds — defaults."""
    be = _make_backend("t67", config={})
    try:
        t = be._conflict_thresholds()
        assert t.get("cosine_min") == 0.85
        assert t.get("jaccard_min") == 0.7
        assert t.get("temporal_guard_days") == 1
    finally:
        be.close(); _cleanup_db("t67")


def test_t68():
    """Config conflict thresholds — custom config."""
    be = _make_backend("t68", config={
        "conflict_thresholds": {"cosine_min": 0.75, "jaccard_min": 0.6, "temporal_guard_days": 14}})
    try:
        t = be._conflict_thresholds()
        assert t.get("cosine_min") == 0.75
        assert t.get("jaccard_min") == 0.6
        assert t.get("temporal_guard_days") == 14
    finally:
        be.close(); _cleanup_db("t68")


def test_t69():
    """Config conflict thresholds — env var override."""
    import os
    old = os.environ.get("HLM_CONFLICT_THRESHOLDS")
    os.environ["HLM_CONFLICT_THRESHOLDS"] = '{"cosine_min": 0.7, "jaccard_min": 0.5}'
    try:
        be = _make_backend("t69", config={})
        try:
            t = be._conflict_thresholds()
            assert t.get("cosine_min") == 0.7
            assert t.get("jaccard_min") == 0.5
        finally:
            be.close(); _cleanup_db("t69")
    finally:
        if old is not None:
            os.environ["HLM_CONFLICT_THRESHOLDS"] = old
        else:
            os.environ.pop("HLM_CONFLICT_THRESHOLDS", None)


def test_t70():
    """L3 offline conflict fallback warning."""
    be = _make_backend("t70", config={
        "layer3_provider_config": {"base_url": "http://localhost:99999/v1"}})
    try:
        be.add("conflict A", data_type="ENV-DATA", data_id="net")
        be.add("conflict B", data_type="ENV-DATA", data_id="net")
        results = be.retrieve("conflict", max_layer=3)
        assert len(results) >= 1
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t70")

def test_t167():
    """Contradiction guards use the caller's real data_id, not a fabricated one.

    _check_contradiction built its comparison record with
    _heuristic_classify("dummy"), which returns data_id=None — so guard 1
    ("different data_id -> skip") could never fire. The old comment claimed it
    "will match existing"; it disabled the guard by short-circuiting.
    """
    import inspect
    from backend import LayeredBackend

    raw = inspect.getsource(LayeredBackend._check_contradiction)
    # Strip comments — the fix is described in one, so match on code only.
    code = "\n".join(line.split("#")[0] for line in raw.splitlines())

    assert '_heuristic_classify("dummy")' not in code, "fabricated data_id still in the code path"
    assert "data_id" in inspect.signature(LayeredBackend._check_contradiction).parameters, \
        "check should accept the caller's data_id"

    # And the Jaccard step is no longer dead code
    assert "_jaccard_similarity" in code, "Jaccard comparison never runs"
    assert code.count("jaccard = None") <= 1, (
        "both branches still assign None — the documented Jaccard step is dead"
    )


def test_t168():
    """Records in different data_id partitions are not flagged as contradictions."""
    be = _make_backend("t167")
    try:
        first = be.add("The cache server holds 512MB with an LRU policy.",
                       data_type="ENV-DATA", data_id="sw", force=True)
        uuid = _get_uuid(first)
        be._get_conn().execute("UPDATE memories SET created_at = ? WHERE uuid = ?",
                               ("2020-01-01T00:00:00+00:00", uuid))
        be._get_conn().commit()

        # Same topic, deliberately different partition — guard 1 should skip it
        result = be.add("The cache server holds 512MB with an LRU policy today.",
                        data_type="ENV-DATA", data_id="hw")
        status = result.get("status") if isinstance(result, dict) else None
        assert status != "contradiction", (
            f"cross-partition records should not be flagged as contradictions: {result}"
        )
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t167")


def test_t189():
    """_call_llm disables reasoning by default, or reasoning models return null.

    Contract updated 2026-08-08: suppression is now ON by default and the
    spelling is chosen per host. See T319-T321 for the portability rules.

    _llm_merge defaulted layer3_reasoning_effort to "none" (commit eb17dbc);
    _call_llm read the same key with no default. With the key absent the field
    was never sent, Qwen3-class models reasoned, `content` came back null, and
    _call_llm returned "" — silently disabling L3 rerank, L4 gap detection,
    fact extraction, LLM review and enrichment classification. Verified against
    a live endpoint: 'LLM returned null content' before, 'OK' after.
    """
    import json as _json
    import urllib.request

    be = _make_backend("t189", config={
        "layer3_model": "some-reasoning-model",
        "layer3_provider_config": {"base_url": "http://localhost:9/v1", "api_key": "k"},
    })
    captured = {}
    real_urlopen = urllib.request.urlopen

    class _Resp:
        def __init__(self, payload): self._p = payload
        def read(self): return _json.dumps(self._p).encode()
        def __enter__(self): return self
        def __exit__(self, *a): return False

    try:
        def fake_urlopen(req, timeout=None):
            captured["body"] = _json.loads(req.data.decode())
            return _Resp({"choices": [{"message": {"content": "hello"}}]})

        urllib.request.urlopen = fake_urlopen
        out = be._call_llm("ping")
        assert out == "hello", f"expected the content through: {out!r}"

        # Reasoning IS suppressed by default — which is what this test's own
        # docstring describes and what the old assertion contradicted.
        #
        # The previous contract was "send nothing unless the operator sets
        # layer3_reasoning_effort". That is the exact failure written up above:
        # with the key absent, a Qwen3-class model reasons, `content` comes back
        # null, and every LLM-backed feature silently no-ops. It only appeared
        # fixed because both HLM profiles happen to set HLM_REASONING_EFFORT=none
        # in their .env; a fresh profile got the broken default.
        #
        # The reason the old contract existed at all was portability — sending
        # an unknown parameter 400s on a strict provider. That is now handled by
        # choosing the spelling per host (T319) and repairing on rejection
        # (T321), rather than by sending nothing to everyone.
        body = captured["body"]
        assert body.get("reasoning_effort") == "none", (
            f"reasoning must be suppressed by default: {body!r}")
        # base_url here is localhost, i.e. a permissive endpoint, so the
        # Qwen3-on-vLLM chat-template switch must be present too —
        # reasoning_effort alone does not stop that template thinking.
        assert body.get("chat_template_kwargs") == {"enable_thinking": False}, body

        # An explicit config value still wins
        be._config["layer3_reasoning_effort"] = "low"
        be._call_llm("ping")
        assert captured["body"].get("reasoning_effort") == "low", "explicit config ignored"
        assert "chat_template_kwargs" not in captured["body"], (
            "asking for a reasoning budget must not also send a disable switch")

        # ...and an operator who wants the model's own default can say so
        be._config["layer3_reasoning_effort"] = "provider_default"
        be._call_llm("ping")
        assert "reasoning_effort" not in captured["body"], captured["body"]
        assert "chat_template_kwargs" not in captured["body"], captured["body"]
        be._config.pop("layer3_reasoning_effort")

        # Servers that answer with reasoning_content only are still usable
        def reasoning_only(req, timeout=None):
            captured["body"] = _json.loads(req.data.decode())
            return _Resp({"choices": [{"message": {"content": None,
                                                   "reasoning_content": "fallback text"}}]})

        urllib.request.urlopen = reasoning_only
        assert be._call_llm("ping") == "fallback text", (
            "should fall back to reasoning_content when content is null"
        )
    finally:
        urllib.request.urlopen = real_urlopen
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t189")


def test_t206():
    """L3 must not rank unseen records above the ones it scored.

    The reranker sees limit*3 candidates and scores them 0.0-1.0, but records
    outside that slice kept their fusion_score — unbounded RRF, typically
    1.0-1.3. Both went into one sort key, so every record the LLM never saw
    outranked everything it did, and L3 returned its worst candidates.
    Measured against a live endpoint: recall@5 0.273 vs 1.000 at L2.
    """
    import json as _json

    be = _make_backend("t206", config={
        "layer3_model": "stub",
        "layer3_provider_config": {"base_url": "http://localhost:9/v1", "api_key": "k"},
    })
    try:
        # 8 records: L2 order puts the answer first, with realistic RRF scores
        records = []
        for i in range(8):
            records.append({
                "uuid": f"{i:032x}",
                "topic": "answer" if i == 0 else f"filler{i}",
                "summary": "the answer" if i == 0 else f"unrelated {i}",
                "content": "the answer" if i == 0 else f"unrelated {i}",
                "fusion_score": 1.30 - i * 0.01,   # unbounded, all > 1.0
                "layer3_flags": {},
            })

        # limit=2 -> the LLM sees only the first 6; it ranks the answer top
        ranking = [[0, 0.95], [1, 0.40], [2, 0.30], [3, 0.20], [4, 0.10], [5, 0.05]]
        be._call_llm = lambda prompt: _json.dumps(
            {"rerank_needed": True, "ranking": ranking})

        out = be._layer3(records, "what is the answer", limit=2)

        assert out, "layer3 returned nothing"
        assert out[0]["uuid"] == f"{0:032x}", (
            "the record the LLM scored highest must come first; got "
            f"{out[0]['topic']!r} (fusion={out[0].get('fusion_score')}) — "
            "unseen records are outranking scored ones again"
        )
        # Nothing beyond the LLM's slice may appear above a scored record
        returned_topics = [r["topic"] for r in out]
        assert "filler6" not in returned_topics and "filler7" not in returned_topics, (
            f"records the reranker never saw were promoted: {returned_topics}"
        )
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t206")


def test_t206b():
    """Conflict alert indices filtered to visible results (BUG-FMT).

    Conflict indices from LLM reference positions in `candidates`
    (up to limit*3 records). If indices point to records not in the
    returned results (top-N), they should be filtered out, not cause
    index errors or reference invisible records.
    """
    be = _make_backend("t206b")
    try:
        # Create results with different summaries
        results = [
            {"uuid": f"{i:032x}", "summary": f"Result {i}"}
            for i in range(5)
        ]

        # Conflict with indices beyond the returned results
        # LLM sees candidates[:15] but only 5 are returned
        conflicts = [
            {"indices": [1, 12], "preferred": 1, "description": "out of bounds",
             "reason": "recency"},  # index 12 out of bounds
            {"indices": [0, 3], "preferred": 0, "description": "in bounds",
             "reason": "recency"},  # both in bounds
            {"indices": [10, 14], "preferred": 10, "description": "both out",
             "reason": "recency"},  # preferred out of bounds, skip entire conflict
        ]

        alert = be._format_conflict_alert(results, conflicts)

        # Should include the in-bounds conflict
        assert "in bounds" in alert, f"in-bounds conflict missing: {alert}"
        assert "Result 0" in alert, f"primary summary missing: {alert}"
        assert "Result 3" in alert, f"contradicting summary missing: {alert}"

        # Should NOT include the out-of-bounds conflict (index 12 > 4)
        assert "out of bounds" not in alert, f"out-of-bounds conflict should be skipped: {alert}"

        # Should NOT include the all-out-of-bounds conflict
        assert "both out" not in alert, f"all-out-of-bounds conflict should be skipped: {alert}"
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t206b")
