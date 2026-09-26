"""T1-T30: Core CRUD, filters, layers, maintenance, export/import."""

import os
import json
import tempfile
import shutil
import time

from conftest import (
    _make_backend, _cleanup_db, _cleanup_qdrant_coll,
    _get_uuid, _docker, _check_qdrant_alive, QDRANT_URL,
)


# ── CRUD ────────────────────────────────────────────────────────────────────

def test_t01():
    """Smoke — add + retrieve."""
    be = _make_backend("t01")
    try:
        uuid = be.add("RTX 6000 Blackwell 96GB", data_type="ENV-DATA", data_id="hw")
        results = be.retrieve("what GPU", max_layer=1)
        assert len(results) >= 1, f"Expected ≥1 result, got {len(results)}"
        assert any("RTX" in r.get("content", "") for r in results)
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t01")


def test_t02():
    """Filter by data_type."""
    be = _make_backend("t02")
    try:
        be.add("user pref", data_type="USER-DATA")
        be.add("hw spec", data_type="ENV-DATA")
        be.add("session note", data_type="SESSION-DATA", data_id="sess1")
        results = be.retrieve("anything", data_type="ENV-DATA", max_layer=1)
        assert all(r["data_type"] == "ENV-DATA" for r in results)
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t02")


def test_t03():
    """Filter by data_id."""
    be = _make_backend("t03")
    try:
        be.add("RTX 6000", data_type="ENV-DATA", data_id="hw")
        be.add("Ubuntu 26.04", data_type="ENV-DATA", data_id="sw")
        results = be.retrieve("system", data_type="ENV-DATA", data_id="hw", max_layer=1)
        assert all(r["data_id"] == "hw" for r in results)
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t03")


def test_t04():
    """Filter by session_name."""
    be = _make_backend("t04")
    try:
        be.add("max_num_seqs=50", data_type="SESSION-DATA",
               data_id="sess_abc", session_name="GPU tuning")
        be.add("layered plugin 1090 lines", data_type="SESSION-DATA",
               data_id="sess_def", session_name="plugin review")
        results = be.retrieve("config", data_type="SESSION-DATA",
                              session_name="GPU tuning", max_layer=1)
        assert all(r["session_name"] == "GPU tuning" for r in results)
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t04")


def test_t05():
    """Combined filters."""
    be = _make_backend("t05")
    try:
        be.add("max_num_seqs=50", data_type="SESSION-DATA",
               data_id="sess_abc", session_name="GPU tuning")
        be.add("other note", data_type="SESSION-DATA",
               data_id="sess_def", session_name="plugin review")
        results = be.retrieve("anything", data_type="SESSION-DATA",
                              data_id="sess_abc", session_name="GPU tuning", max_layer=1)
        assert len(results) == 1
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t05")


# ── Layers ──────────────────────────────────────────────────────────────────

def test_t06():
    """Layer depth comparison."""
    be = _make_backend("t06")
    try:
        be.add("RTX 6000 GPU", data_type="ENV-DATA", data_id="hw")
        r2 = be.retrieve("GPU", max_layer=2)
        assert len(r2) >= 1 and "fusion_score" in r2[0]
        r3 = be.retrieve("GPU", max_layer=3)
        r4 = be.retrieve("GPU", max_layer=4)
        assert len(r3) >= 1 and len(r4) >= 1
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t06")


def test_t07():
    """Peek at layers."""
    be = _make_backend("t07")
    try:
        be.add("RTX 6000 GPU", data_type="ENV-DATA", data_id="hw")
        l1 = be.peek("GPU", layer=1)
        l2 = be.peek("GPU", layer=2)
        assert isinstance(l1, list) and isinstance(l2, list)
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t07")


def test_t08():
    """List recent."""
    be = _make_backend("t08")
    try:
        for i in range(3):
            be.add(f"memory {i}", data_type="CUSTOM")
        results = be.list(limit=5, sort="created_at")
        assert len(results) >= 3
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t08")


def test_t09():
    """Sync check."""
    be = _make_backend("t09")
    try:
        be.add("sync test", data_type="CUSTOM")
        sync = be.sync_check()
        assert "sqlite_active" in sync and "in_sync" in sync
        assert sync["sqlite_active"] >= 1
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t09")


def test_t10():
    """Rebuild."""
    be = _make_backend("t10")
    try:
        be.add("rebuild test", data_type="CUSTOM")
        result = be.rebuild()
        assert isinstance(result, dict)
        assert result.get("status") == "rebuilt" or "count" in result
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t10")


def test_t11():
    """Circuit breaker — Qdrant offline."""
    be = _make_backend("t11")
    be.add("cb test", data_type="CUSTOM")
    try:
        _docker("stop")
        for attempt in range(10):
            if not _check_qdrant_alive():
                break
            time.sleep(1)
        else:
            _docker("start"); be.close(); _cleanup_db("t11")
            raise AssertionError("Qdrant did not shut down, test skipped")

        for _ in range(10):
            be.retrieve("cb test", max_layer=1)

        if be._qdrant_failures >= 5:
            assert be._qdrant_broken_until is not None
        elif be._qdrant_failures >= 3:
            pass
        else:
            results = be.retrieve("cb test", max_layer=1)
            assert len(results) >= 1, "Fallback should return results"

        _docker("start"); time.sleep(3)
    except Exception as e:
        try: _docker("start"); time.sleep(2)
        except Exception: pass
        raise
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t11")


# ── Update / Delete / Maintenance ──────────────────────────────────────────

def test_t12():
    """Update."""
    be = _make_backend("t12")
    try:
        uuid = _get_uuid(be.add("RTX 6000 Blackwell 96GB test12",
                                data_type="ENV-DATA", data_id="hw"))
        be.update(uuid, content="RTX 6000 Blackwell 96GB — tested")
        record = be._get_record(uuid)
        assert record["content"] == "RTX 6000 Blackwell 96GB — tested"
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t12")


def test_t13():
    """Delete (soft)."""
    be = _make_backend("t13")
    try:
        uuid = be.add("delete me", data_type="CUSTOM")
        be.delete(uuid)
        record = be._get_record(uuid)
        assert record is None
        row = be._get_conn().execute(
            "SELECT status FROM memories WHERE uuid=?", (uuid,)
        ).fetchone()
        assert row and row[0] == "deleted"
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t13")


def test_t14():
    """Sleep / consolidation."""
    be = _make_backend("t14")
    try:
        be.add("high trust", data_type="ENV-DATA", data_id="hw", trust_score=0.8)
        be.add("low trust", data_type="ENV-DATA", data_id="hw", trust_score=0.1)
        be.add("low trust 2", data_type="ENV-DATA", data_id="hw", trust_score=0.1)
        result = be.sleep(max_items=50, min_age_hours=0)
        assert isinstance(result, dict)
        # sleep() returns archived + ttl_expired — no "merged", no "cutoff"
        assert "archived" in result and "ttl_expired" in result
        assert "merged" not in result, "sleep() does not do content merging"
        assert "cutoff" not in result, "cutoff was removed (was unused internally)"
        # Two of three records are low-trust duplicates — should be archived
        assert result["archived"] >= 1, f"expected ≥1 archived, got {result}"
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t14")


def test_t15():
    """Feedback."""
    be = _make_backend("t15")
    try:
        uuid = be.add("feedback test", data_type="CUSTOM")
        be.feedback(uuid, helpful=True)
        r1 = be._get_record(uuid)
        assert r1["trust_score"] > 0.5
        result = be.feedback("nonexistent", helpful=True)
        assert "not_found" in result or result.get("status") == "not_found"
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t15")


def test_t16():
    """Enrich (standalone)."""
    be = _make_backend("t16")
    try:
        be.add("RTX 4090 GPU with 24GB VRAM on my workstation", data_type="CUSTOM")
        result = be.enrich_existing(max_items=10)
        assert isinstance(result, dict)
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t16")


def test_t17():
    """Review (no crash)."""
    be = _make_backend("t17")
    try:
        be.add("review test", data_type="CUSTOM")
        assert hasattr(be, "rebuild")
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t17")


def test_t18():
    """Compact."""
    be = _make_backend("t18")
    try:
        be.add("RTX 4090 GPU 24GB", data_type="ENV-DATA", data_id="hw")
        be.add("RTX 4090 graphics card 24GB VRAM", data_type="ENV-DATA", data_id="hw")
        be.add("NVIDIA RTX 4090 24GB GPU card", data_type="ENV-DATA", data_id="hw")
        result = be.compact(similarity_threshold=0.85)
        assert isinstance(result, dict)
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t18")


def test_t19():
    """Cross-profile retrieve."""
    be = _make_backend("t19")
    try:
        be.add("cross profile test", data_type="CUSTOM")
        results = be.retrieve("cross profile", cross_profile=True, max_layer=1)
        assert isinstance(results, list)
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t19")


def test_t20():
    """Export / Import round-trip."""
    be = _make_backend("t20")
    try:
        be.add("export test data", data_type="CUSTOM")
        exported = be.export_memories(fmt="json")
        assert exported is not None and len(exported) > 0
        be2 = _make_backend("t20b")
        try:
            data = json.loads(exported)
            result = be2.import_memories(json.dumps(data), mode="skip_existing")
            assert isinstance(result, dict)
        finally:
            _cleanup_qdrant_coll(be2); be2.close(); _cleanup_db("t20b")
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t20")


def test_t20b():
    """Import overwrite mode updates all fields, not just content.

    BUG-005: overwrite mode only updated content, updated_at, and status.
    All other fields (summary, keywords, topic, scope, data_type, data_id,
    session_name, sensitivity, source, source_url, ttl, trust_score,
    priority, metadata, protected) were silently dropped on overwrite.
    """
    be = _make_backend("t20b")
    try:
        # Add original record
        uuid = be.add("Original content.",
                      data_type="USER-DATA", data_id="prefs",
                      summary="Original summary",
                      keywords=["original", "old"],
                      topic="settings", scope="personal",
                      trust_score=0.5, priority=0,
                      sensitivity=0, source="agent",
                      session_name="test-session", force=True)

        # Verify original
        rec = be._get_record(uuid)
        assert rec["content"] == "Original content."
        assert rec["topic"] == "settings"
        assert rec["trust_score"] == 0.5
        assert rec["keywords"] == ["original", "old"]

        # Export and modify
        exported = be.export_memories(fmt="json")
        data = json.loads(exported)
        for r in data["records"]:
            if r["uuid"] == uuid:
                r["content"] = "Updated content."
                r["summary"] = "Updated summary"
                r["keywords"] = ["updated", "new"]
                r["topic"] = "new_settings"
                r["trust_score"] = 0.9
                r["priority"] = 3
                r["sensitivity"] = 2
                r["session_name"] = "updated-session"
                break

        # Import with overwrite mode
        result = be.import_memories(json.dumps(data), mode="overwrite")
        assert result["imported"] == 1, f"Expected 1 imported, got {result}"

        # Verify all fields were updated
        rec = be._get_record(uuid)
        assert rec["content"] == "Updated content.", f"content not updated: {rec['content']}"
        assert rec["topic"] == "new_settings", f"topic not updated: {rec['topic']}"
        assert rec["trust_score"] == 0.9, f"trust_score not updated: {rec['trust_score']}"
        assert rec["keywords"] == ["updated", "new"], f"keywords not updated: {rec['keywords']}"
        assert rec["priority"] == 3, f"priority not updated: {rec['priority']}"
        assert rec["sensitivity"] == 2, f"sensitivity not updated: {rec['sensitivity']}"
        assert rec["session_name"] == "updated-session", f"session_name not updated: {rec['session_name']}"
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t20b")


def test_t20c():
    """Import normalizes data_id to lowercase (BUG-027).

    import_memories did not normalize data_id to lowercase, inconsistent
    with add() which does. This broke case-insensitive lookups for imported
    records (e.g. data_id="SW" vs data_id="sw").
    """
    be = _make_backend("t20c")
    try:
        # Add record with lowercase data_id
        uuid1 = be.add("Lowercase data_id record.",
                       data_type="USER-DATA", data_id="sw", force=True)

        # Export record with modified (uppercase) data_id
        exported = be.export_memories(fmt="json")
        data = json.loads(exported)
        for r in data["records"]:
            if r["uuid"] == uuid1:
                r["data_id"] = "SW"  # uppercase
                break

        # Import with overwrite mode — data_id should be normalized to lowercase
        result = be.import_memories(json.dumps(data), mode="overwrite")
        assert result["imported"] == 1, f"Expected 1 imported, got {result}"

        rec = be._get_record(uuid1)
        assert rec["data_id"] == "sw", f"data_id should be lowercase: {rec['data_id']}"

        # Test INSERT path with mixed case data_id
        be2 = _make_backend("t20c2")
        try:
            # Create a record with mixed-case data_id
            import uuid as uuid_mod
            new_uuid = uuid_mod.uuid4().hex
            import_data = {
                "records": [{
                    "uuid": new_uuid,
                    "content": "Mixed case data_id test.",
                    "data_type": "ENV-DATA",
                    "data_id": "MixedCase",
                }]
            }
            result2 = be2.import_memories(json.dumps(import_data), mode="skip_existing")
            assert result2["imported"] == 1, f"Expected 1 imported, got {result2}"

            rec2 = be2._get_record(new_uuid)
            assert rec2["data_id"] == "mixedcase", f"data_id should be lowercase: {rec2['data_id']}"
        finally:
            _cleanup_qdrant_coll(be2); be2.close(); _cleanup_db("t20c2")
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t20c")


# ── Decay / Rebuild / Purge ────────────────────────────────────────────────

def test_t21():
    """Decay (continuous curve)."""
    from datetime import datetime, timedelta, timezone
    be = _make_backend("t21")
    try:
        u0 = be.add("p0", data_type="CUSTOM", priority=0, trust_score=0.5)
        u1 = be.add("p1", data_type="CUSTOM", priority=1, trust_score=0.5)
        u3 = be.add("p3", data_type="CUSTOM", priority=3, trust_score=0.5)
        up = be.add("protected", data_type="CUSTOM", priority=0,
                    trust_score=0.5, protected=True)
        past = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
        be._get_conn().execute(
            "UPDATE memories SET created_at = ? WHERE uuid IN (?, ?, ?)",
            (past, u0, u1, u3))
        be._get_conn().commit()
        be.decay(min_age_days=0, max_age_days=365, decay_rate=0.05, min_score=0.1)
        assert be._get_record(u3)["trust_score"] == 0.5
        assert be._get_record(up)["trust_score"] == 0.5
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t21")


def test_t22():
    """Rebuild with NULL embedding recovery."""
    be = _make_backend("t22")
    try:
        uuid = be.add("null embed test", data_type="CUSTOM")
        be._get_conn().execute("UPDATE memories SET embedding = 'null' WHERE uuid=?", (uuid,))
        be._get_conn().commit()
        be.rebuild()
        assert be._get_record(uuid) is not None
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t22")


def test_t23():
    """Rebuild with dimension mismatch."""
    be = _make_backend("t23")
    try:
        uuid = be.add("dim mismatch test", data_type="CUSTOM")
        be._get_conn().execute("UPDATE memories SET embedding = ? WHERE uuid=?",
                         (json.dumps([0.1] * 999), uuid))
        be._get_conn().commit()
        # Rebuild should handle dimension mismatch gracefully
        result = be.rebuild()
        assert isinstance(result, dict), f"Expected dict from rebuild, got {type(result).__name__}"
        # Record should still exist after rebuild
        record = be._get_record(uuid)
        assert record is not None, "Record should exist after rebuild with dimension mismatch"
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t23")


def test_t24():
    """Purge."""
    be = _make_backend("t24")
    try:
        uuid = be.add("purge me", data_type="CUSTOM")
        be.delete(uuid)
        be.purge(purge_deleted=True, purge_archived=True, min_age_hours=0, vacuum=True)
        row = be._get_conn().execute("SELECT * FROM memories WHERE uuid=?", (uuid,)).fetchone()
        assert row is None
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t24")


def test_t25():
    """List sort variants."""
    be = _make_backend("t25")
    try:
        be.add("low pri", data_type="CUSTOM", priority=0)
        be.add("high pri", data_type="CUSTOM", priority=3)
        assert len(be.list(limit=5, sort="priority")) >= 2
        assert len(be.list(limit=5, sort="trust_score")) >= 2
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t25")


# ── L3 / L4 ────────────────────────────────────────────────────────────────

def test_t26():
    """L3 smart-skip."""
    be = _make_backend("t26")
    try:
        be.add("RTX GPU with CUDA cores", data_type="ENV-DATA", data_id="hw")
        results = be.retrieve("hardware GPU memory configuration settings comparison",
                              max_layer=3)
        assert len(results) >= 1
        assert isinstance(results[0].get("layer3_flags", {}), dict)
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t26")


def test_t27():
    """L3 explicit rerank flag."""
    be = _make_backend("t27")
    try:
        be.add("RTX GPU config", data_type="ENV-DATA", data_id="hw")
        results = be.retrieve("GPU", max_layer=2, rerank=True)
        assert len(results) >= 1
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t27")


def test_t28():
    """L3 fallback when LLM unavailable."""
    be = _make_backend("t28", config={
        "layer3_provider_config": {"base_url": "http://localhost:99999/v1"}})
    try:
        be.add("fallback test", data_type="CUSTOM")
        results = be.retrieve("anything", max_layer=3)
        assert len(results) >= 1
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t28")


def test_t29():
    """L4 gap detection partial coverage."""
    be = _make_backend("t29")
    try:
        be.add("RTX GPU hardware", data_type="ENV-DATA", data_id="hw")
        results = be.retrieve("hardware GPU configuration memory settings comparison",
                              max_layer=4)
        assert len(results) >= 1
        assert results[0].get("layer3_flags", {}).get("gap_checked") is True
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t29")


def test_t30():
    """L4 gap detection missing info."""
    be = _make_backend("t30")
    try:
        be.add("one small fact", data_type="CUSTOM")
        results = be.retrieve(
            "quantum computing neural network machine learning architecture",
            max_layer=4)
        assert results[0].get("layer3_flags", {}).get("gap_checked") is True
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t30")


def test_t129():
    """Retrieval stats — explicit retrieve increments counter."""
    be = _make_backend("t129")
    try:
        be.add("stats test", data_type="CUSTOM")
        stats = be.retrieval_stats()
        assert "prefetch" in stats, f"Stats missing 'prefetch': {stats}"
        assert "explicit" in stats, f"Stats missing 'explicit': {stats}"
        assert stats["prefetch"] == 0
        assert stats["explicit"] == 0
        be.retrieve("stats", source="explicit")
        stats = be.retrieval_stats()
        assert stats["explicit"] >= 1, f"Expected ≥1 explicit: {stats}"
        assert stats["prefetch"] == 0
        assert stats["total"] == stats["prefetch"] + stats["explicit"]
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t129")


def test_t130():
    """Retrieval stats — prefetch increments separate counter."""
    be = _make_backend("t130")
    try:
        be.add("prefetch stats test", data_type="CUSTOM")
        # Simulate prefetch calls
        be.retrieve("prefetch", source="prefetch")
        be.retrieve("prefetch 2", source="prefetch")
        be.retrieve("explicit", source="explicit")
        stats = be.retrieval_stats()
        assert stats["prefetch"] == 2, f"Expected 2 prefetch: {stats}"
        assert stats["explicit"] == 1, f"Expected 1 explicit: {stats}"
        assert stats["total"] == 3
        assert 0 < stats["prefetch_pct"] < 100
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t130")


def test_t130b():
    """Retrieval stats — maintenance_anchors reads from maintenance_state table.

    Regression: retrieval_stats() queried 'maintenance_status' but the table
    is 'maintenance_state', so maintenance_anchors was always {} silently.
    """
    be = _make_backend("t130b")
    try:
        # Write a timestamp into the maintenance_state table
        be._set_maintenance_state("last_purge", "1700000000")
        be._set_maintenance_state("last_decay", "1700001000")
        stats = be.retrieval_stats()
        assert "maintenance_anchors" in stats, f"Stats missing 'maintenance_anchors': {stats}"
        anchors = stats["maintenance_anchors"]
        assert anchors.get("last_purge") == 1700000000, f"last_purge not read: {anchors}"
        assert anchors.get("last_decay") == 1700001000, f"last_decay not read: {anchors}"
        assert anchors.get("last_enrich") is None  # not set
        assert anchors.get("last_sleep") is None   # not set
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t130b")


# ── M0: L3 hardcode and regex fixes ─────────────────────────────────────────

def test_t131():
    """M0-2: L3 is skipped gracefully when layer3_provider_config has no base_url."""
    be = _make_backend("l3_skip", config={
        "layer3_model": "test-model",
        "layer3_provider_config": {}  # no base_url, no api_key
    })
    try:
        for i in range(3):
            be.add(f"l3 skip test content {i}", data_type="CUSTOM")
        results = be.retrieve("l3 skip test", max_layer=3)
        # Should complete without trying to reach embed-host.example.net
        assert len(results) >= 1, "Should get results even when L3 is skipped"
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("l3_skip")


def test_t132():
    """M0-3: L3 regex parses responses with nested JSON (conflicts array)."""
    sample = '{"rerank_needed": true, "ranking": [[0, 0.9]], "conflicts": [{"indices": [0,1], "reason": "recency"}]}'

    # Old non-greedy regex fails on nested JSON
    import re
    match_old = re.search(r'\{.*?\}', sample, re.DOTALL)
    # Should stop at first } — truncated output
    assert match_old.group() != sample, "Non-greedy regex should truncate nested JSON"

    # New raw_decode should parse the full response
    decoder = json.JSONDecoder()
    result, _ = decoder.raw_decode(sample)
    assert result["rerank_needed"] is True
    assert result["ranking"] == [[0, 0.9]]
    assert result["conflicts"][0]["reason"] == "recency"


def test_t133():
    """M0-3: L3 regex also handles simple responses (rerank_needed: false)."""
    sample = '{"rerank_needed": false}'
    decoder = json.JSONDecoder()
    result, _ = decoder.raw_decode(sample)
    assert result["rerank_needed"] is False


# ── M1: force parameter on write path ──────────────────────────────────────

def test_t136():
    """M1: force=True bypasses duplicate check and stores the memory."""
    be = _make_backend("force_dup")
    try:
        uuid1 = be.add("exact same content", data_type="CUSTOM")
        # Without force, this would be blocked as duplicate
        result = be.add("exact same content", data_type="CUSTOM")
        assert isinstance(result, dict) and result.get("status") == "duplicate"

        # With force, it stores a new record
        uuid2 = be.add("exact same content", data_type="CUSTOM", force=True)
        assert isinstance(uuid2, str), f"force=True should return uuid string, got {type(uuid2).__name__}: {uuid2}"
        assert uuid2 != uuid1, "force=True should create a new record"

        # Verify both exist
        count = be._get_conn().execute(
            "SELECT COUNT(*) FROM memories WHERE status='active'"
        ).fetchone()[0]
        assert count == 2, f"Expected 2 active records, got {count}"
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("force_dup")


def test_t137():
    """M1: force=True bypasses contradiction check."""
    import datetime
    be = _make_backend("force_contra", config={
        "conflict_thresholds": {"temporal_guard_days": 0}
    })
    try:
        # Store first fact, age it past the temporal guard
        uuid1 = be.add("port is 8080", data_type="ENV-DATA", data_id="sw")
        old_time = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=8)).isoformat()
        be._get_conn().execute("UPDATE memories SET created_at = ? WHERE uuid = ?", (old_time, uuid1))
        be._get_conn().commit()

        # With force, it stores regardless of contradiction
        uuid2 = be.add("port is 9090", data_type="ENV-DATA", data_id="sw", force=True)
        assert isinstance(uuid2, str), f"force=True should return uuid string, got {type(uuid2).__name__}"
        assert uuid2 != uuid1

        # Verify both exist
        count = be._get_conn().execute(
            "SELECT COUNT(*) FROM memories WHERE status='active'"
        ).fetchone()[0]
        assert count == 2, f"Expected 2 active records, got {count}"
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("force_contra")


def test_t138():
    """M1: Without force, dedup still blocks (sanity check)."""
    be = _make_backend("no_force")
    try:
        uuid1 = be.add("dedup test content", data_type="CUSTOM")
        result = be.add("dedup test content", data_type="CUSTOM")
        assert isinstance(result, dict), f"Without force, duplicate should return dict, got {type(result).__name__}"
        assert result.get("status") in ("duplicate", "possible_duplicate"), (
            f"Expected duplicate/possible_duplicate, got {result.get('status')}"
        )
        # Only one record should exist
        count = be._get_conn().execute(
            "SELECT COUNT(*) FROM memories WHERE status='active'"
        ).fetchone()[0]
        assert count == 1, f"Expected 1 active record, got {count}"
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("no_force")


# ── M2: Fact extraction refactoring ────────────────────────────────────────

def test_t139():
    """M2: _extract_facts_background does not use subprocess."""
    import inspect
    import sys as _sys
    import os as _os
    sys_path = _sys.path.copy()
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    _init_path = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "__init__.py")
    with open(_init_path) as f:
        src = f.read()
    # Find _extract_facts_background method and verify no subprocess.Popen
    # The method should call _extract_facts_direct instead
    assert "_extract_facts_direct" in src, "Should have _extract_facts_direct method"
    # The _extract_facts_background method should not contain subprocess.Popen
    # (initialize compaction extraction also replaced)
    in_extract_bg = False
    for line in src.split('\n'):
        if 'def _extract_facts_background' in line:
            in_extract_bg = True
        elif in_extract_bg and line.startswith('    def ') and '_extract_facts_background' not in line:
            in_extract_bg = False
        if in_extract_bg and 'subprocess.Popen' in line:
            raise AssertionError("_extract_facts_background should not contain subprocess.Popen")
    _sys.path[:] = sys_path


def test_t140():
    """M2: _extract_facts_direct parsing handles JSON arrays correctly."""
    be = _make_backend("extract_direct")
    try:
        # Test the JSON array parsing used by _extract_facts_direct
        import re
        sample = json.dumps([
            {"content": "fact A", "topic": "t", "keywords": ["a"]},
            {"content": "fact B", "topic": "t", "keywords": ["b"]},
        ])
        # Use the same parsing approach as _extract_facts_direct
        # Need to handle nested brackets — find the matching ] for the outer [
        bracket_start = sample.index("[")
        depth = 0
        bracket_end = bracket_start
        for i in range(bracket_start, len(sample)):
            if sample[i] == "[":
                depth += 1
            elif sample[i] == "]":
                depth -= 1
                if depth == 0:
                    bracket_end = i + 1
                    break
        matched = sample[bracket_start:bracket_end]
        facts = json.loads(matched)
        assert len(facts) == 2
        assert facts[0]["content"] == "fact A"
        assert facts[1]["keywords"] == ["b"]
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("extract_direct")


# ── M3: Connection-per-thread ──────────────────────────────────────────────

def test_t141():
    """M3: Background enrichment doesn't race with main thread writes."""
    be = _make_backend("enrich_race", config={
        "enrich_llm": "test-model",
        "dedup_threshold": 0  # disable dedup for this test
    })
    try:
        # Add records with distinct content that trigger background enrichment
        distinct_contents = [
            "RTX 4090 GPU has 24GB GDDR6X memory",
            "Ubuntu 24.04 LTS released April 2024",
            "Python 3.12 adds tomllib to standard library",
            "Docker Compose V2 is written in Go",
            "Qdrant uses HNSW for vector search indexing",
        ]
        for content in distinct_contents:
            be.add(content, data_type="CUSTOM")
        # Give enrichment threads time to run
        time.sleep(2)
        # Verify no data corruption
        rows = be._get_conn().execute(
            "SELECT COUNT(*) FROM memories WHERE status='active'"
        ).fetchone()[0]
        assert rows == 5, f"Expected 5 active records, got {rows}"
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("enrich_race")


def test_t142():
    """M3: close() joins background threads before closing DB."""
    be = _make_backend("close_wait", config={"enrich_llm": "test-model"})
    try:
        be.add("close wait test content with details", data_type="CUSTOM")
        # close() should not raise even with active enrichment thread
    finally:
        be.close()  # should not throw
        _cleanup_db("close_wait")

def test_t147():
    """Embeddings round-trip as packed float32, and legacy JSON rows still read.

    Packing cut a 4096-dim vector from ~56KB of JSON text to 16KB. Rows written
    before the change hold JSON and must stay readable without a migration.
    """
    from backend import _pack_embedding, _unpack_embedding, _EMBED_NULL

    vec = [0.125, -0.5, 3.25, 0.0]
    packed = _pack_embedding(vec)
    assert not isinstance(packed, str), "embedding should be packed binary, not text"
    assert [round(x, 4) for x in _unpack_embedding(packed)] == vec, "packed round-trip lost data"

    # Legacy encodings stay readable in both the str and bytes forms a driver may return
    assert _unpack_embedding('[1.0, 2.0]') == [1.0, 2.0], "legacy JSON string not readable"
    assert _unpack_embedding(b'[1.0, 2.0]') == [1.0, 2.0], "legacy JSON bytes not readable"

    # Null forms all collapse to None
    for null_form in (None, "null", _EMBED_NULL, _pack_embedding(None), _pack_embedding([])):
        assert _unpack_embedding(null_form) is None, f"{null_form!r} should read as None"

    # Size claim is load-bearing — assert it rather than trusting the comment.
    # Measured on a real 4096-dim row: 56,242 bytes of JSON vs 16,384 packed (3.4x).
    big = [0.1234567] * 4096
    packed_size = len(bytes(_pack_embedding(big)))
    assert packed_size == 4096 * 4, "expected 4 bytes per dimension"
    assert packed_size < len(json.dumps(big)) / 2.5, (
        f"packed form should be >2.5x smaller than JSON text "
        f"({packed_size} vs {len(json.dumps(big))})"
    )


def test_t148():
    """A record written with a packed embedding survives a full add/retrieve cycle."""
    be = _make_backend("t148")
    try:
        uuid = _get_uuid(be.add("Grafana dashboards live on port 3000 behind the reverse proxy.",
                                data_type="ENV-DATA", data_id="sw", force=True))
        # Vector must come back as a usable list of floats, not bytes or text
        vec = be._get_vector(uuid)
        assert isinstance(vec, list) and vec, f"_get_vector returned {type(vec).__name__}"
        assert all(isinstance(x, float) for x in vec[:8]), "embedding elements should be floats"

        results = be.retrieve("where do grafana dashboards run", max_layer=1, limit=5)
        assert any(r["uuid"] == uuid for r in results), "packed-embedding record not retrievable"
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t148")


def test_t149():
    """Supersession — the replacement is returned, the old value is retained.

    Facts change. Before this, the only options were overwrite (losing the old
    value) or store both (leaving the reader to guess which is current).
    """
    be = _make_backend("t149")
    try:
        old = _get_uuid(be.add("The API gateway listens on port 8080.",
                               data_type="ENV-DATA", data_id="net", force=True))
        result = be.add("The API gateway listens on port 9090.",
                        data_type="ENV-DATA", data_id="net", supersedes=old)

        assert isinstance(result, dict), f"superseding add should report status, got {result!r}"
        assert result["status"] == "superseded", f"unexpected status: {result}"
        assert result["superseded_uuid"] == old, f"wrong record superseded: {result}"
        new = result["uuid"]

        # The old record still exists and is linked to its replacement
        row = be._get_conn().execute(
            "SELECT superseded_by, superseded_at, status FROM memories WHERE uuid = ?", (old,)
        ).fetchone()
        assert row is not None, "superseded record was deleted — it must be retained"
        assert row[0] == new, f"superseded_by not set correctly: {row[0]}"
        assert row[1], "superseded_at not stamped"
        assert row[2] == "active", "superseding must not soft-delete the old record"

        # Retrieval returns the new value and not the stale one
        results = be.retrieve("what port does the API gateway use", max_layer=1, limit=5)
        uuids = [r["uuid"] for r in results]
        assert new in uuids, "replacement record should be retrievable"
        assert old not in uuids, "superseded record must not appear in normal retrieval"

        # ...but it is still reachable when explicitly asking for everything
        all_results = be.retrieve("API gateway port", max_layer=1, limit=10, status="all")
        assert old in [r["uuid"] for r in all_results], (
            "superseded record should remain queryable with status='all'"
        )
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t149")


def test_t150():
    """Supersession bypasses the dedup/contradiction block by design.

    An updated value is necessarily similar to the value it replaces — that is
    exactly the case the blocking paths used to reject.
    """
    be = _make_backend("t150")
    try:
        old = _get_uuid(be.add("The user prefers dark mode in every application.",
                               data_type="USER-DATA", data_id="preferences", force=True))
        # Near-identical content: without supersedes this returns a duplicate dict
        result = be.add("The user prefers dark mode in every application.",
                        data_type="USER-DATA", data_id="preferences", supersedes=old)
        assert isinstance(result, dict) and result["status"] == "superseded", (
            f"supersedes must bypass dedup blocking, got {result!r}"
        )
        assert result["uuid"] != old, "a new record should have been created"
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t150")


def test_t151():
    """A supersedes pointer to an unknown uuid is ignored, not fatal."""
    be = _make_backend("t151")
    try:
        result = be.add("Standalone fact with a bogus supersedes pointer.",
                        data_type="CUSTOM", supersedes="ffffffffffffffffffffffffffffffff")
        uuid = _get_uuid(result)
        assert uuid, "add should still succeed when supersedes target is missing"
        row = be._get_conn().execute(
            "SELECT status FROM memories WHERE uuid = ?", (uuid,)).fetchone()
        assert row and row[0] == "active", "record should be stored normally"
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t151")


def test_t151b():
    """count() excludes superseded records.

    Superseded records remain status='active' but have superseded_by set,
    so they should not count toward active total. Affects maintenance gating
    and sync_check SQLite vs Qdrant comparison.
    """
    be = _make_backend("t151b")
    try:
        old = _get_uuid(be.add("Original fact.",
                               data_type="ENV-DATA", data_id="net", force=True))
        assert be.count() == 1, f"Expected 1, got {be.count()}"

        # Supersede the original
        be.add("Updated fact.",
               data_type="ENV-DATA", data_id="net", supersedes=old, force=True)

        # count() should not count the superseded record
        assert be.count() == 1, f"Expected 1 (old superseded), got {be.count()}"

        # Add another independent record
        be.add("Another fact.", data_type="ENV-DATA", data_id="other", force=True)
        assert be.count() == 2, f"Expected 2, got {be.count()}"

        # Direct DB check: 3 rows with status='active', only 2 without superseded_by
        raw = be._get_conn().execute(
            "SELECT COUNT(*) FROM memories WHERE status = 'active'"
        ).fetchone()[0]
        assert raw == 3, f"Expected 3 raw active, got {raw}"
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t151b")


def test_t151c():
    """seed_overview() excludes superseded records.

    BUG-031: seed_overview queries did not filter superseded_by IS NULL,
    so a superseded record could be the top trusted record injected into
    the system prompt, presenting stale content as current knowledge.
    """
    be = _make_backend("t151c")
    try:
        # Add a high-trust record and supersede it
        old = _get_uuid(be.add("Stale but trusted fact.",
                               data_type="USER-DATA", data_id="prefs",
                               trust_score=0.95, force=True))
        new = _get_uuid(be.add("Current fact.",
                               data_type="USER-DATA", data_id="other",
                               trust_score=0.5, supersedes=old, force=True))

        # The old record has higher trust_score but is superseded
        old_rec = be._get_record(old)
        assert old_rec["superseded_by"] is not None, "old should be superseded"
        assert old_rec["trust_score"] == 0.95, "old should have high trust"

        # seed_overview should not mention the superseded record
        overview = be.seed_overview()
        assert isinstance(overview, str), f"expected str, got {overview!r}"

        # The overview should mention the non-superseded record, not the stale one
        assert "Current fact" in overview, f"overview should include current fact: {overview[:200]}"
        assert "Stale but trusted" not in overview, f"overview must not include stale fact: {overview[:200]}"
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t151c")


def test_t152():
    """Extraction ledger records what the write path learned and discarded.

    Auto-extraction decides what memory contains, and used to report nothing —
    no count, no reason, only a logger.debug that went to DEVNULL in practice.
    """
    be = _make_backend("t152")
    try:
        be.write_extraction_ledger({"hook": "session_end", "candidates": 5,
                                    "stored": 3, "rejected": 2, "session_id": "abc"})
        be.write_extraction_ledger({"hook": "sync_turn", "candidates": 0,
                                    "stored": 0, "rejected": 0, "reason": "llm_empty_response"})

        stats = be.extraction_stats()
        assert stats["runs"] == 2, f"expected 2 runs, got {stats}"
        assert stats["candidates"] == 5, f"candidate total wrong: {stats}"
        assert stats["stored"] == 3, f"stored total wrong: {stats}"
        assert stats["rejected"] == 2, f"rejected total wrong: {stats}"
        assert stats["failed"] == 1, f"failed run not counted: {stats}"
        assert stats["by_hook"]["session_end"] == 1, f"hook breakdown wrong: {stats}"
        assert stats["by_hook"]["sync_turn"] == 1, f"hook breakdown wrong: {stats}"
        assert stats["last_run"], "last_run timestamp missing"

        # The ledger shares the history sidecar and must not corrupt it
        with open(be._get_history_path(), encoding="utf-8") as fh:
            for line in fh:
                json.loads(line)
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t152")


def test_t153():
    """extraction_stats is empty, not broken, before any extraction has run."""
    be = _make_backend("t153")
    try:
        stats = be.extraction_stats()
        assert stats["runs"] == 0 and stats["last_run"] is None, f"unexpected: {stats}"
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t153")


def test_t163():
    """Extraction supersedes a changed fact instead of rejecting it forever.

    A fact that changes trips contradiction against its stale predecessor. If
    that is treated as a rejection, the new value can never be learned: every
    later extraction is refused against the record it was supposed to replace.
    """
    be = _make_backend("t163")
    try:
        # Age the original past the temporal guard so contradiction can fire
        old = _get_uuid(be.add("The API gateway listens on port 8080.",
                               data_type="ENV-DATA", data_id="net", force=True))
        old_ts = "2020-01-01T00:00:00+00:00"
        be._get_conn().execute("UPDATE memories SET created_at = ? WHERE uuid = ?",
                               (old_ts, old))
        be._get_conn().commit()

        # Simulate the producer path: contradiction -> retry with supersedes
        first = be.add("The API gateway listens on port 9090.",
                       data_type="ENV-DATA", data_id="net")
        if isinstance(first, dict) and first.get("status") == "contradiction":
            retry = be.add("The API gateway listens on port 9090.",
                           data_type="ENV-DATA", data_id="net",
                           supersedes=first["uuid"])
            assert retry["status"] == "superseded", f"retry should supersede: {retry}"
            assert retry["superseded_uuid"] == old, f"wrong target: {retry}"
        else:
            # Contradiction did not fire (thresholds/guards); supersession must
            # still work when driven explicitly.
            retry = be.add("The API gateway listens on port 9090 now.",
                           data_type="ENV-DATA", data_id="net", supersedes=old)
            assert retry["status"] == "superseded", f"explicit supersede failed: {retry}"

        # Whichever path ran, the stale value must be out of normal retrieval
        results = be.retrieve("what port does the API gateway use", max_layer=1, limit=5)
        assert old not in [r["uuid"] for r in results], "stale value still retrievable"
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t163")


def test_t164():
    """The ledger distinguishes superseded writes from rejected ones."""
    be = _make_backend("t164")
    try:
        be.write_extraction_ledger({"hook": "session_end", "candidates": 4,
                                    "stored": 1, "superseded": 2, "rejected": 1})
        stats = be.extraction_stats()
        assert stats["superseded"] == 2, f"superseded not tracked: {stats}"
        assert stats["stored"] == 1 and stats["rejected"] == 1, f"counts wrong: {stats}"
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t164")


def test_t165():
    """Prefetch impressions do not inflate reference_count.

    retrieve() incremented reference_count for everything it returned, and
    prefetch calls retrieve() on every turn — so whatever prefetch surfaced
    crossed the >10 reinforcement threshold within ten turns and then decayed
    70% slower, entrenching records that were never actually used.
    """
    be = _make_backend("t165")
    try:
        uuid = _get_uuid(be.add("Grafana runs on port 3000 behind the proxy.",
                                data_type="ENV-DATA", data_id="sw", force=True))

        def refs():
            return be._get_conn().execute(
                "SELECT reference_count FROM memories WHERE uuid = ?", (uuid,)
            ).fetchone()[0]

        start = refs()
        for _ in range(3):
            be.retrieve("where does grafana run", max_layer=1, source="prefetch")
        assert refs() == start, f"prefetch inflated reference_count: {start} -> {refs()}"

        be.retrieve("where does grafana run", max_layer=1, source="explicit")
        assert refs() == start + 1, f"explicit retrieve should count: {start} -> {refs()}"
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t165")


def test_t166():
    """reinforce() nudges trust upward for records the agent actually used."""
    be = _make_backend("t166")
    try:
        uuid = _get_uuid(be.add("The deploy pipeline requires manual approval.",
                                data_type="SYSTEM", data_id="rules", force=True))

        def trust():
            return be._get_conn().execute(
                "SELECT trust_score FROM memories WHERE uuid = ?", (uuid,)
            ).fetchone()[0]

        before = trust()
        assert be.reinforce(uuid) is True, "reinforce should succeed on an active record"
        after = trust()
        assert after > before, f"trust did not increase: {before} -> {after}"
        assert after - before < 0.1, "implicit signal must be smaller than explicit feedback"

        # Never exceeds the cap, and does not resurrect deleted records
        for _ in range(80):
            be.reinforce(uuid)
        assert trust() <= 1.0, f"trust exceeded cap: {trust()}"

        be.delete(uuid)
        assert be.reinforce(uuid) is False, "should not reinforce a deleted record"
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t166")


def test_t170():
    """max_layer=0 must not silently run the Layer 2 fusion pass.

    The early return tested `max_layer == 1`, so max_layer=0 — documented as
    the cheapest setting — fell through to L2. It measured identically to
    max_layer=2 in the eval harness because it was max_layer=2.
    """
    be = _make_backend("t170")
    try:
        for i in range(5):
            be.add(f"Node {i} runs service {i} on port {9000 + i}.",
                   data_type="ENV-DATA", data_id="sw", force=True)

        l0 = be.retrieve("which node runs a service", max_layer=0, limit=5)
        l2 = be.retrieve("which node runs a service", max_layer=2, limit=5)
        assert l0, "layer 0 should still return results"

        # L2 annotates records with a fusion score; L0 must not.
        assert all("fusion_score" not in r for r in l0), (
            "max_layer=0 ran the L2 fusion pass — it should stop after Layer 1"
        )
        assert any("fusion_score" in r for r in l2), "max_layer=2 should score"
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t170")


def test_t171():
    """Qdrant-outage fallback ranks by relevance, not by insertion order.

    _fallback_uuids() was `SELECT uuid FROM memories WHERE status='active'
    LIMIT 30` — no ORDER BY, and the query was never passed in. During a
    circuit-breaker cooldown every question returned the same oldest-30
    records. Measured at 2,024 records: recall@5 0.227, byte-identical results
    across unrelated queries, while the README called it a working fallback.
    """
    be = _make_backend("t171")
    try:
        # Insert the answer LAST so insertion order cannot accidentally pass
        for i in range(8):
            be.add(f"Filler record {i} about unrelated scheduling matters.",
                   data_type="CUSTOM", force=True)
        target = _get_uuid(be.add("Redis uses an allkeys-lru eviction policy with 512MB maxmemory.",
                                  data_type="ENV-DATA", data_id="sw", force=True))

        # Force the circuit open — no vector search available
        be._qdrant_broken_until = time.time() + 120
        results = be.retrieve("redis eviction policy", max_layer=1, limit=5)

        assert results, "fallback must still return something during an outage"
        uuids = [r["uuid"] for r in results]
        assert target in uuids, (
            "relevant record missing during degraded retrieval — fallback is "
            f"returning arbitrary rows: {[r['content'][:40] for r in results]}"
        )
        assert uuids[0] == target, (
            f"fallback should rank the lexical match first, got "
            f"{results[0]['content'][:50]!r}"
        )
        assert be._degraded is True, "degraded retrieval should be flagged"
        assert be.sync_check()["degraded_retrieval"] is True, "sync_check must report it"
    finally:
        be._qdrant_broken_until = None
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t171")


def test_t172():
    """Circuit breaker: a single transient failure must not claim the circuit is open.

    The rewritten breaker logged "Qdrant circuit OPEN, 120s cooldown" on the
    FIRST failure, because the log-once flag started False and nothing had
    opened yet. It also deleted the accurate "(n/5)" progress line, so the only
    signal was a false one.
    """
    be = _make_backend("t172")
    try:
        be._record_qdrant_failure(Exception("transient"))
        assert be._qdrant_failures == 1, f"should count the failure: {be._qdrant_failures}"
        assert be._qdrant_broken_until is None, (
            "one failure must not open the circuit"
        )

        for _ in range(4):
            be._record_qdrant_failure(Exception("boom"))
        assert be._qdrant_broken_until is not None, "5 failures should open the circuit"
        assert time.time() < be._qdrant_broken_until, "cooldown should be in the future"

        # Further failures while open are no-ops (no cooldown extension, no spam)
        opened_at = be._qdrant_broken_until
        for _ in range(10):
            be._record_qdrant_failure(Exception("still down"))
        assert be._qdrant_broken_until == opened_at, (
            "failures during cooldown must not extend it — a burst kept the "
            "circuit open far longer than the advertised 120s"
        )
    finally:
        be._qdrant_broken_until = None
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t172")


def test_t173():
    """Circuit breaker: a failed half-open probe re-opens immediately.

    Opening resets the failure counter, so a failed probe used to count as 1 of
    5 — sending four more requests into a backend known to be down, which is
    exactly what a half-open state exists to prevent.
    """
    be = _make_backend("t173")
    try:
        for _ in range(5):
            be._record_qdrant_failure(Exception("down"))
        assert be._qdrant_broken_until is not None, "circuit should be open"

        be._qdrant_broken_until = time.time() - 1  # cooldown elapsed
        be._record_qdrant_failure(Exception("probe failed"))

        assert be._qdrant_broken_until is not None, "probe failure must re-open"
        assert time.time() < be._qdrant_broken_until, (
            "half-open probe failure should arm a fresh cooldown, not fall through"
        )

        # A success closes it completely
        be._record_qdrant_success()
        assert be._qdrant_broken_until is None, "success should close the circuit"
        assert be._qdrant_failures == 0, "success should clear the failure count"
    finally:
        be._qdrant_broken_until = None
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t173")


def test_t174():
    """Batched re-embed must not cross-wire vectors between records.

    rebuild() re-embeds in batches of 64. Nothing exercised the batch loop —
    T10 and T23 rebuild a handful of records — so the boundary, the partial
    final batch, and the mapping of batch results back to their source rows
    were untested. Misalignment there gives every record someone else's
    vector: retrieval still returns plausible-looking results and no existing
    test would notice.
    """
    be = _make_backend("t174")
    try:
        # Straddle the batch size (64) so there is a full batch and a partial one
        markers = {}
        for i in range(70):
            content = f"Distinct fact number {i}: the widget codenamed zeta{i} weighs {i} kilograms."
            markers[i] = _get_uuid(be.add(content, data_type="CUSTOM", force=True))

        # Force the NULL-embedding re-embed path for every record
        be._get_conn().execute("UPDATE memories SET embedding = 'null'")
        be._get_conn().commit()

        result = be.rebuild()
        assert isinstance(result, dict), f"rebuild should report status: {result}"

        # Two questions, kept apart — `T690`'s rule, learned here the hard way.
        #
        # *Is a record's vector its own?* is a question about this code and is
        # asserted. *Did the embedder answer at all?* is a question about this
        # machine and is reported. They used to share one assertion, `assert
        # vec, "record N lost its embedding"`, which fires first — so a
        # transient embedder failure reported itself as a cross-wiring
        # regression and the cross-wiring check never ran.
        #
        # Measured 2026-09-26 on a second host: this test failed inside the
        # suite at 52.2s, immediately after `Qdrant initialization failed:
        # [Errno 104] Connection reset by peer` (T11 cycles the container),
        # and passed 3/3 in isolation on the same host at 21.2s. Nothing was
        # cross-wired; the embedder had returned nothing for one record of 70
        # during a degraded window. `_embed_batch` guarantees
        # `len(out) == len(texts)` with order preserved and appends `None` on
        # failure, and `rebuild()` guards its UPDATE with `if emb:`, so a
        # missing vector cannot misalign the rest or clear an existing one.
        #
        # When no probe has a vector the comparison is vacuous, and a green
        # result from a check that did not run is the `T353` failure. So this
        # does not simply skip: it asks the embedder directly, and only
        # excuses the run when the embedder is the thing that is broken.
        probes = (0, 63, 64, 69)
        missing, compared = [], 0
        for i in probes:
            uuid = markers[i]
            vec = be._get_vector(uuid)
            if not vec:
                missing.append(i)
                continue
            hits = be.retrieve(f"widget codenamed zeta{i} weighs {i} kilograms",
                               max_layer=1, limit=3)
            assert hits, (
                f"record {i} has a vector but is not retrievable after "
                f"rebuild — that is a retrieval defect, not an embedder one")
            assert hits[0]["uuid"] == uuid, (
                f"record {i} ranked behind another record after batched rebuild — "
                f"vectors are cross-wired. got {hits[0]['content'][:60]!r}"
            )
            compared += 1

        if missing:
            print(f"  T174: the embedder returned no vector for record(s) "
                  f"{missing} of {len(probes)} probed; those are reported, not "
                  f"asserted — see this test's docstring")

        if not compared:
            # Decide the cause instead of assuming it.
            probe_vec = None
            try:
                probe_vec = (be._embed_batch(["T174 embedder liveness probe"]) or [None])[0]
            except Exception as e:                      # noqa: BLE001
                print(f"  T174: embedder probe raised {type(e).__name__}: {e}")
            assert not probe_vec, (
                "no probe record got a vector, yet the embedder answers a "
                "direct call — rebuild()'s batch loop is dropping every "
                "result, which is this test's subject and a real defect")
            print("  T174: embedder is not answering on this host, so the "
                  "cross-wiring comparison could not run. This run proves "
                  "nothing about batching — re-run when the embedder is "
                  "healthy (T215's preflight covers startup, not mid-run "
                  "degradation)")
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t174")


def test_t175():
    """Bulk ingest embeds in batches, and each note keeps its own vector.

    ingest_obsidian() embedded one note per HTTP round trip. On a large vault
    that is one request per file — the pattern that tripped the Qdrant circuit
    breaker while indexing 2,024 records. Batching is only safe if the vectors
    come back aligned with their notes, so this compares each stored vector
    against a freshly computed embedding of that note's own text rather than
    relying on retrieval ranking (which measures the embedder, not alignment).
    """
    import tempfile as _tf
    from pathlib import Path
    from backend import _get_embedding_fn

    # Deliberately unrelated subjects — alignment errors must not be masked by
    # the notes being near-identical.
    SUBJECTS = [
        "the migration of arctic terns across the Atlantic",
        "sourdough fermentation temperatures and hydration ratios",
        "the tuning of a harpsichord in meantone temperament",
        "sediment layers in the Burgess Shale",
        "routing protocols used by undersea telegraph cables",
        "the chemistry of indigo dye extraction",
    ]

    be = _make_backend("t175")
    calls = {"n": 0, "sizes": []}
    real_batch = be._embed_batch

    def counting_batch(texts, batch_size=64):
        calls["n"] += 1
        calls["sizes"].append(len(texts))
        return real_batch(texts, batch_size)

    be._embed_batch = counting_batch
    try:
        with _tf.TemporaryDirectory() as vault:
            for i, subject in enumerate(SUBJECTS):
                Path(vault, f"note{i}.md").write_text(
                    f"---\ntitle: Note {i}\n---\n\nA detailed account of {subject}.\n",
                    encoding="utf-8")

            result = be.ingest_obsidian(vault)
            assert result["ingested"] == len(SUBJECTS), f"expected all ingested: {result}"

            # One batched call for the whole vault, not one per note
            assert calls["n"] == 1, f"expected a single batch pass, got {calls['n']}"
            assert calls["sizes"] == [len(SUBJECTS)], (
                f"batch should cover every note: {calls['sizes']}"
            )

            # Each stored vector must match its OWN content, not a neighbour's.
            fn = _get_embedding_fn()
            rows = be._get_conn().execute(
                "SELECT content, embedding FROM memories WHERE data_type='OBSIDIAN'"
            ).fetchall()
            assert len(rows) == len(SUBJECTS), f"expected {len(SUBJECTS)} rows, got {len(rows)}"

            from backend import _unpack_embedding
            for content, blob in rows:
                stored = _unpack_embedding(blob)
                assert stored, f"no embedding stored for {content[:40]!r}"
                fresh = fn([content])[0]
                sim = be._cosine_similarity(stored, fresh)
                assert sim > 0.99, (
                    f"stored vector does not match its own content (cos={sim:.4f}) — "
                    f"batched embeddings are misaligned for {content[:50]!r}"
                )
    finally:
        be._embed_batch = real_batch
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t175")


def test_t197():
    """Updating an indexed column must not raise, and must refresh the index.

    memories_fts is an external-content table, but fts5_au used a plain
    `UPDATE memories_fts SET ...`. FTS5 must re-read the old row to remove its
    terms, and an AFTER trigger runs once the content table already holds the
    new values — so every update raised "database disk image is malformed".
    Observed in production: 7 of 7 background enrichments failed that way, so
    LLM-derived topics and keywords were computed and then thrown away.
    """
    be = _make_backend("t197")
    try:
        uuid = _get_uuid(be.add("Nginx serves the staging site on port 8443.",
                                summary="staging proxy", data_type="ENV-DATA",
                                data_id="net", force=True))
        conn = be._get_conn()

        # This is exactly what background enrichment does
        conn.execute("UPDATE memories SET topic = ?, updated_at = ? WHERE uuid = ?",
                     ("networking", be._now(), uuid))
        conn.commit()

        row = conn.execute("SELECT topic FROM memories WHERE uuid = ?", (uuid,)).fetchone()
        assert row[0] == "networking", "enrichment update did not persist"

        # The new term is searchable and the stale one is gone
        be.update(uuid, content="Nginx serves staging on port 9443.",
                  summary="staging proxy 9443")
        assert conn.execute("SELECT COUNT(*) FROM memories_fts WHERE memories_fts MATCH '9443'"
                            ).fetchone()[0] == 1, "new text not indexed"
        assert conn.execute("SELECT COUNT(*) FROM memories_fts WHERE memories_fts MATCH '8443'"
                            ).fetchone()[0] == 0, "stale text still searchable after update"
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t197")


def test_t198():
    """Deleting a row removes its terms instead of leaving them behind.

    fts5_ad used `DELETE FROM memories_fts WHERE rowid = old.rowid`, which on
    an external-content table cannot resolve the old values — it reported
    success and left the terms in the index, so deleted content stayed
    searchable and inflated BM25 for everything else.
    """
    be = _make_backend("t198")
    try:
        uuid = _get_uuid(be.add("The decommissioned relay used codename thistledown.",
                                summary="decommissioned relay thistledown",
                                data_type="CUSTOM", force=True))
        conn = be._get_conn()
        assert conn.execute("SELECT COUNT(*) FROM memories_fts WHERE memories_fts MATCH 'thistledown'"
                            ).fetchone()[0] == 1, "term should be indexed after add"

        conn.execute("DELETE FROM memories WHERE uuid = ?", (uuid,))
        conn.commit()

        assert conn.execute("SELECT COUNT(*) FROM memories_fts WHERE memories_fts MATCH 'thistledown'"
                            ).fetchone()[0] == 0, (
            "deleted content is still searchable — the FTS index retained its terms"
        )
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t198")


def test_t199():
    """The v15 migration repairs databases created with the broken triggers."""
    import sqlite3

    be = _make_backend("t199")
    db_path = be._db_path
    try:
        be.add("Legacy row indexed under the old triggers.", data_type="CUSTOM", force=True)
        be._get_conn().execute("PRAGMA wal_checkpoint(TRUNCATE)")
        be._get_conn().commit()
        be.close()

        # Reinstate the broken trigger form, as an older database would have.
        # Use busy_timeout so we wait for WAL cleanup rather than failing
        # immediately with "database is locked".
        import time as _time
        _time.sleep(0.3)  # let WAL checkpoint flush settle
        raw = sqlite3.connect(db_path, timeout=30)
        raw.execute("PRAGMA busy_timeout=10000")
        raw.execute("DROP TRIGGER IF EXISTS fts5_au")
        raw.execute("""CREATE TRIGGER fts5_au AFTER UPDATE ON memories BEGIN
                           UPDATE memories_fts SET content = new.content
                           WHERE rowid = new.rowid;
                       END""")
        raw.commit()
        sql = raw.execute("SELECT sql FROM sqlite_master WHERE name='fts5_au'").fetchone()[0]
        raw.close()
        assert "'delete'" not in sql, "precondition: trigger should be the broken form"

        # Reopening runs the migration
        from backend import LayeredBackend
        be2 = LayeredBackend(db_path=db_path, qdrant_url=QDRANT_URL,
                             profile_name="test-t199", config={"enrich_llm": False})
        try:
            sql2 = be2._get_conn().execute(
                "SELECT sql FROM sqlite_master WHERE name='fts5_au'").fetchone()[0]
            assert "'delete'" in sql2, "migration did not replace the broken trigger"
            # And the repaired database accepts an update
            uuid = be2._get_conn().execute(
                "SELECT uuid FROM memories LIMIT 1").fetchone()[0]
            be2._get_conn().execute("UPDATE memories SET topic=? WHERE uuid=?",
                                    ("repaired", uuid))
            be2._get_conn().commit()
        finally:
            be2.close()
    finally:
        _cleanup_db("t199")


def test_t201():
    """Enrichment tolerates the record being deleted while the LLM is in flight.

    add-then-delete is a normal sequence, and _get_record() filters on
    status='active'. Calling .get() on the resulting None raised
    "'NoneType' object has no attribute 'get'" — swallowed into a debug line,
    killing the enrichment thread. Seen in a live E2E run, inside a step the
    agent marked PASS.
    """
    be = _make_backend("t201")
    try:
        uuid = _get_uuid(be.add("Ephemeral note that will be deleted at once.",
                                data_type="CUSTOM", force=True))
        be.delete(uuid)
        assert be._get_record(uuid) is None, "precondition: record is gone"

        # Drive the enrichment body directly against the deleted record
        captured = {}
        real_debug = be.__class__._enrich_metadata

        def fake_enrich(self, content, summary=None, llm=None):
            return {"topic": "some topic", "keywords": ["a", "b"]}

        be.__class__._enrich_metadata = fake_enrich
        try:
            be._enrich_background(uuid, "Ephemeral note", "summary")
            if be._enrich_thread:
                be._enrich_thread.join(timeout=10)
        finally:
            be.__class__._enrich_metadata = real_debug

        # It must not have crashed, and must not have resurrected the record
        row = be._get_conn().execute(
            "SELECT status, topic FROM memories WHERE uuid = ?", (uuid,)).fetchone()
        assert row and row[0] == "deleted", f"record should stay deleted: {row}"
        assert not captured, "no update should have been attempted"
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t201")


def test_t202():
    """test_cleanup() soft-deletes all [HLM-TEST] records without crashing.

    Regression: test_cleanup referenced self.HLM_TEST_MARKER which was never
    set on the instance (module-level constant in maintenance.py).
    """
    be = _make_backend("t202")
    try:
        be.add("[HLM-TEST] first test record", data_type="CUSTOM", force=True)
        be.add("[HLM-TEST] second test record", data_type="CUSTOM", force=True)
        be.add("Normal record that should survive", data_type="ENV-DATA", force=True)

        count_before = be.count()
        assert count_before == 3, f"expected 3 active, got {count_before}"

        result = be.test_cleanup()
        assert result["test_deleted"] == 2, f"expected 2 deleted, got {result}"
        assert result["marker"] == "[HLM-TEST] ", f"wrong marker: {result}"

        count_after = be.count()
        assert count_after == 1, f"expected 1 active, got {count_after}"

        # The normal record should be untouched
        row = be._get_conn().execute(
            "SELECT content FROM memories WHERE status='active'"
        ).fetchone()
        assert row[0] == "Normal record that should survive", (
            f"normal record was affected: {row}"
        )
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t202")
