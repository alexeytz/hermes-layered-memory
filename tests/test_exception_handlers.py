#!/usr/bin/env python3
"""Exception handler audit tests (T207-T212).

Verify that broad `except Exception` blocks wrapping *local logic* (parsing,
scoring, DB queries) emit at WARNING level so that coding mistakes
(NameError, AttributeError, KeyError) are visible in production logs rather
than silently degrading features.

Corresponds to the audit described in docs/handover-2026-07-28.md item 5.
"""

from __future__ import annotations

import logging
import os
import sys
import sqlite3
import tempfile
import unittest
from unittest.mock import patch, MagicMock

# Ensure project root is on path (must happen before any backend import)
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from conftest import _make_backend, _cleanup_db


class _LogCapture(logging.Handler):
    """Capture log records for assertion."""
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


def _get_logger():
    """Get the backend logger instance."""
    from backend import logger
    return logger


class _BrokenConn:
    """Mock SQLite connection that fails on FTS5 queries."""
    def __init__(self, real_conn):
        self._real = real_conn

    def execute(self, query, *args, **kwargs):
        if "memories_fts" in str(query):
            raise RuntimeError("FTS5 query failed")
        return self._real.execute(query, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._real, name)


class _BrokenConnAll:
    """Mock SQLite connection that fails on ALL queries."""
    def execute(self, query, *args, **kwargs):
        raise RuntimeError("DB locked")

    def __getattr__(self, name):
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")


def test_t207_bm25_failure_logs_warning():
    """T207 BM25 failure logs WARNING (not silent)."""
    be = _make_backend("t207")
    try:
        uuid = be.add("BM25 test content with keywords for scoring")
        if isinstance(uuid, dict):
            uuid = uuid["uuid"]
        record = be._get_record(uuid)
        assert record is not None
        # _get_record() omits rowid; add it from the DB for BM25 scoring
        row = be._get_conn().execute(
            "SELECT rowid FROM memories WHERE uuid = ?", (uuid,)
        ).fetchone()
        record["rowid"] = row[0] if row else 1

        cap = _LogCapture()
        _get_logger().addHandler(cap)

        try:
            broken_conn = _BrokenConn(be._get_conn())
            be._add_bm25_conn([record], "test query", broken_conn)
        finally:
            _get_logger().removeHandler(cap)

        assert record.get("bm25_score") == 0.0, "Fallback should set bm25_score=0.0"

        warnings = [r for r in cap.records if r.levelno == logging.WARNING]
        assert len(warnings) > 0, "Expected WARNING log for BM25 failure"
        assert any("BM25" in r.getMessage() for r in warnings), \
            "Expected 'BM25' in warning message"
    finally:
        _cleanup_db("t207")


def test_t208_layer3_parse_failure_logs_warning():
    """T208 L3 parse failure logs WARNING (not DEBUG)."""
    be = _make_backend("t208")
    try:
        uuids = []
        for i in range(3):
            uuid = be.add(f"L3 test record number {i} with different content")
            if isinstance(uuid, dict):
                uuid = uuid["uuid"]
            uuids.append(uuid)

        records = []
        for uid in uuids:
            rec = be._get_record(uid)
            if rec:
                rec["fusion_score"] = 0.5 + len(records) * 0.1
                records.append(rec)
        assert len(records) > 0

        cap = _LogCapture()
        _get_logger().addHandler(cap)

        try:
            with patch.object(be, '_call_llm', return_value='{"some": "json"}'):
                with patch.object(be, '_parse_llm_json', side_effect=AttributeError("bad key")):
                    result = be._layer3(records, "test query", limit=5)
        finally:
            _get_logger().removeHandler(cap)

        assert len(result) > 0, "Fallback should return records"

        warnings = [r for r in cap.records if r.levelno == logging.WARNING]
        assert len(warnings) > 0, "Expected WARNING log for L3 parse failure"
        assert any("Layer 3 parse" in r.getMessage() for r in warnings), \
            "Expected 'Layer 3 parse' in warning message"
    finally:
        _cleanup_db("t208")


def test_t209_layer4_parse_failure_logs_warning():
    """T209 L4 parse failure logs WARNING (not DEBUG)."""
    be = _make_backend("t209")
    try:
        uuid = be.add("L4 gap detection test content")
        if isinstance(uuid, dict):
            uuid = uuid["uuid"]
        record = be._get_record(uuid)
        assert record is not None
        records = [record]

        cap = _LogCapture()
        _get_logger().addHandler(cap)

        try:
            with patch.object(be, '_call_llm', return_value='{"gaps": []}'):
                with patch.object(be, '_parse_llm_json', side_effect=ValueError("parse error")):
                    result = be._layer4(records, "test query")
        finally:
            _get_logger().removeHandler(cap)

        assert len(result) == 1, "Records still returned with gaps skipped"

        warnings = [r for r in cap.records if r.levelno == logging.WARNING]
        assert len(warnings) > 0, "Expected WARNING log for L4 parse failure"
        assert any("Layer 4 parse" in r.getMessage() for r in warnings), \
            "Expected 'Layer 4 parse' in warning message"
    finally:
        _cleanup_db("t209")


def test_t210_layer1_profile_db_failure_logs_warning():
    """T210 L1 profile DB failure logs WARNING (not DEBUG)."""
    be = _make_backend("t210")
    try:
        uuid = be.add("L1 profile test content")
        if isinstance(uuid, dict):
            uuid = uuid["uuid"]

        # Create a corrupt DB file (exists but fails on query)
        corrupt_db = tempfile.NamedTemporaryFile(
            suffix=".db", dir=os.path.dirname(be._db_path), delete=False
        )
        corrupt_db.write(b"not a sqlite database")
        corrupt_db.close()

        cap = _LogCapture()
        _get_logger().addHandler(cap)

        try:
            with patch.object(be, '_discover_profile_dbs',
                    return_value={"corrupt-profile": corrupt_db.name}):
                candidates = [(uuid, 0.1)]
                result = be._layer1(candidates, "test query", cross_profile=True)
        finally:
            _get_logger().removeHandler(cap)
            os.unlink(corrupt_db.name)

        # Result should be empty since the corrupt profile contributed nothing
        # (and the current profile's own query uses the real conn, not cross-profile)
        warnings = [r for r in cap.records if r.levelno == logging.WARNING]
        assert len(warnings) > 0, "Expected WARNING log for profile DB failure, got " + str([r.getMessage() for r in cap.records])
        assert any("profile DB" in r.getMessage() for r in warnings), \
            "Expected 'profile DB' in warning message"
    finally:
        _cleanup_db("t210")


def test_t211_reference_count_bump_logs_debug():
    """T211 reference_count bump failure logs DEBUG (best-effort)."""
    be = _make_backend("t211")
    try:
        uuid = be.add("Reference count test content")
        if isinstance(uuid, dict):
            uuid = uuid["uuid"]

        cap = _LogCapture()
        log = _get_logger()
        old_level = log.level
        log.setLevel(logging.DEBUG)  # DEBUG filtered at default INFO level
        log.addHandler(cap)

        try:
            def broken_get_conn():
                return _BrokenConnAll()

            with patch.object(be, '_run_pipeline', return_value=[{"uuid": uuid, "score": 0.9}]):
                with patch.object(be, '_get_conn', side_effect=broken_get_conn):
                    result = be.retrieve("test query", source="explicit")
        finally:
            log.removeHandler(cap)
            log.setLevel(old_level)

        assert len(result) == 1, "Result returned despite reference_count failure"

        debugs = [r for r in cap.records if r.levelno == logging.DEBUG]
        assert len(debugs) > 0, "Expected DEBUG log for reference_count bump failure"
        assert any("reference_count" in r.getMessage() for r in debugs), \
            "Expected 'reference_count' in debug message"

        # Verify NO WARNING for reference_count (best-effort)
        warnings = [r for r in cap.records
                    if r.levelno == logging.WARNING and "reference_count" in r.getMessage()]
        assert len(warnings) == 0, "reference_count bump should log DEBUG, not WARNING"
    finally:
        _cleanup_db("t211")


def test_t212_run_pipeline_l3_outer_logs_error():
    """T212 _run_pipeline L3/L4 outer handler logs ERROR."""
    be = _make_backend("t212")
    try:
        uuid = be.add("L3 outer handler test content")
        if isinstance(uuid, dict):
            uuid = uuid["uuid"]

        cap = _LogCapture()
        _get_logger().addHandler(cap)

        try:
            with patch.object(be, '_layer0', return_value=[(uuid, 0.1)]):
                with patch.object(be, '_detect_orphans', side_effect=lambda c, **kw: c):
                    with patch.object(be, '_fts5_fallback', return_value=[]):
                        with patch.object(be, '_layer1', return_value=[
                            {"uuid": uuid, "distance": 0.1, "score": 0.9,
                             "bm25_score": 0.5, "fusion_score": 0.7,
                             "profile_name": "test-t212"}
                        ]):
                            with patch.object(be, '_layer3', side_effect=RuntimeError("LLM crash")):
                                with patch.object(be, '_write_trace'):
                                    result = be._run_pipeline("test query", max_layer=3)
        finally:
            _get_logger().removeHandler(cap)

        assert len(result) > 0, "Fallback to L2 results"

        errors = [r for r in cap.records if r.levelno == logging.ERROR]
        assert len(errors) > 0, "Expected ERROR log for L3 outer handler failure"
        assert any("Layer 3 reranker failed" in r.getMessage() for r in errors), \
            "Expected 'Layer 3 reranker failed' in error message"
    finally:
        _cleanup_db("t212")