#!/usr/bin/env python3
"""HLM Regression Test Runner — discovers and runs all test_*.py files.

Usage:
    python3 tests/run-regression.py

Requires: Qdrant running on localhost:6333, Docker available.
Results written to tests/regression-results.json.
"""

from __future__ import annotations

import glob
import json
import os
import sys
import importlib.util
import time
from typing import Any, Dict, List

RESULTS: List[dict] = []
TEST_START = time.time()

# ── Project root ────────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from backend import logger
from conftest import QDRANT_URL, _check_qdrant_alive


# ── Test discovery ──────────────────────────────────────────────────────────
def discover_tests() -> List[Dict[str, Any]]:
    """Discover all test functions from test_*.py files."""
    test_files = sorted(glob.glob(os.path.join(TESTS_DIR, "test_*.py")))
    tests = []
    for fpath in test_files:
        fname = os.path.basename(fpath)
        spec = importlib.util.spec_from_file_location(fname.replace(".py", ""), fpath)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for name in sorted(dir(mod)):
            if name.startswith("test_t"):
                fn = getattr(mod, name)
                if callable(fn):
                    # Extract test ID and title from function name/docstring
                    parts = name.replace("test_t", "T").split("_", 1)
                    test_id = parts[0].upper()
                    title = fn.__doc__.strip().split("\n")[0] if fn.__doc__ else parts[1].replace("_", " ").title() if len(parts) > 1 else test_id
                    tests.append({"test_id": test_id, "title": title, "fn": fn, "file": fname})
    return tests


# ── Test runner ─────────────────────────────────────────────────────────────
def run_test(test_id: str, title: str, fn) -> dict:
    """Run a single test function, capture result."""
    result = {"test": test_id, "title": title, "status": "PASS", "detail": ""}
    t0 = time.time()
    try:
        fn()
    except Exception as e:
        result["status"] = "FAIL"
        result["detail"] = f"{type(e).__name__}: {e}"
    elapsed = time.time() - t0
    result["elapsed_s"] = round(elapsed, 2)
    # 120s catches a hang. A test that legitimately takes longer says so with
    # `fn.max_seconds`, rather than forcing the ceiling up for everything: the
    # retrieval gate embeds 540 records and runs 51 queries against a live
    # endpoint and lands around 117s, close enough to the flat limit that a
    # slow afternoon would have reported a passing gate as a timeout.
    limit = getattr(fn, "max_seconds", 120)
    if elapsed > limit:
        result["status"] = "TIMEOUT"
        result["detail"] = f"exceeded {limit}s ({elapsed:.0f}s)"
    RESULTS.append(result)
    status_icon = "✓" if result["status"] == "PASS" else "✗"
    print(f"  {status_icon} {test_id} {title:60s} [{result['elapsed_s']:5.1f}s] {result['status']}", flush=True)
    return result


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 80)
    print("HLM Regression Test Suite")
    print("=" * 80)

    if not _check_qdrant_alive():
        print(f"ERROR: Qdrant not reachable at {QDRANT_URL}", file=sys.stderr)
        sys.exit(1)
    print(f"Qdrant OK at {QDRANT_URL}\n")

    tests = discover_tests()
    print(f"Discovered {len(tests)} tests\n")

    for test in tests:
        run_test(test["test_id"], test["title"], test["fn"])

    # ── Summary ──────────────────────────────────────────────────────
    total = len(RESULTS)
    passed = sum(1 for r in RESULTS if r["status"] == "PASS")
    failed = sum(1 for r in RESULTS if r["status"] == "FAIL")
    timed_out = sum(1 for r in RESULTS if r["status"] == "TIMEOUT")
    elapsed = time.time() - TEST_START

    print(f"\n{'=' * 80}")
    print(f"RESULTS: {passed} PASS, {failed} FAIL, {timed_out} TIMEOUT ({total} total, {elapsed:.0f}s)")
    print(f"{'=' * 80}")

    if failed:
        print("\nFailed tests:")
        for r in RESULTS:
            if r["status"] in ("FAIL", "TIMEOUT"):
                print(f"  {r['test']} {r['title']}: {r['detail']}")

    # Write results JSON
    results_path = os.path.join(TESTS_DIR, "regression-results.json")
    with open(results_path, "w") as f:
        json.dump(RESULTS, f, indent=2, default=str)
    print(f"\nResults written to {results_path}")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()