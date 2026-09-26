#!/usr/bin/env python3
"""Shared test fixtures for HLM regression tests.

Imported by all test_*.py files. Provides:
  - _make_backend()     isolated LayeredBackend per test
  - _make_summaries()   isolated SummariesBackend per test
  - _cleanup_db()       remove test DB artifacts
  - _cleanup_qdrant_coll()  delete Qdrant points for a profile
  - _reset_qdrant_for_test()  pre-test Qdrant cleanup
  - _docker()           stop/start Qdrant container
  - _check_qdrant_alive()  health check
  - _get_uuid()         handle add() returning dict or str
  - fake_secret_scope() simulate a multiplex gateway's profile secret scope
"""

from __future__ import annotations

import atexit
import contextlib
import glob
import json
import os
import pwd
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional

# ── Project root ────────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

# Redirect logging BEFORE importing backend — it resolves the log file at
# module import. Without this the suite writes into whichever profile log the
# environment points at, so running the tests during an E2E dumps 194 tests'
# worth of output — including chaos tests that fail on purpose — into the
# production log. The E2E log gate then reports failures that never happened
# in production, which is exactly how a gate earns a reputation for crying
# wolf and gets rationalised away.
_TEST_LOG_DIR = tempfile.mkdtemp(prefix="hlm-test-logs-")
os.environ["HLM_LOG_FILE"] = os.path.join(_TEST_LOG_DIR, "tests.log")
atexit.register(lambda: shutil.rmtree(_TEST_LOG_DIR, ignore_errors=True))

# And redirect $HERMES_HOME for the same reason, one file over. Every
# `LayeredBackend` construction ends up in `_ensure_db_config_defaults()` ->
# `_save_db_config()` -> `_sync_config_to_file()`, which writes
# `$HERMES_HOME/hermes-layered-memory.json` — layer 2 of the three-layer
# config, the one *real* profiles read at startup. So the suite was writing
# its own config into the operator's live file, and what it wrote included
# the routing table:
#
#     ~/.hermes/hermes-layered-memory.json.bak.20260926T053648
#     {"collections": {"SYSTEM": "hlmtest_memories", "USER-DATA":
#      "hlmtest_memories", ..., "OBSIDIAN": "hlmtest_vault"}, ...}
#
# `hlmtest_*` is this suite's Qdrant namespace and exists for the reason the
# AGENTS.md note gives — the suite must not write into live profile data. The
# leak put the test namespace where every profile would read it as its own
# routing table, which is that same mistake arriving by the other door. It
# also rewrote the file (keeping 3 timestamped backups) on any machine that
# ran the suite, whether or not a test touched config.
#
# One test already did this for itself (the `T353`-era HERMES_HOME dance in
# test_pass2_regressions.py) — applied where it was noticed, not where it was
# needed, which is the same shape as `T642` and `T651`. It belongs here, with
# the log redirect it is a sibling of.
# 0.8.113, T716. Found 2026-09-26 while driving bundle03 F2.
_TEST_HERMES_HOME = tempfile.mkdtemp(prefix="hlm-test-hermes-home-")
os.environ["HERMES_HOME"] = _TEST_HERMES_HOME
atexit.register(lambda: shutil.rmtree(_TEST_HERMES_HOME, ignore_errors=True))

# ingest_obsidian() only accepts vault_path under an allowed root (real home
# by default) to prevent a caller reading arbitrary directories — see
# backend/maintenance.py's _obsidian_allowed_roots(). Obsidian-ingest tests
# build fixture vaults under the system temp dir via tempfile.mkdtemp(),
# which sits outside home on most systems; extend the allowlist for the test
# session the same way an operator would for a real vault stored elsewhere.
os.environ.setdefault("HLM_OBSIDIAN_VAULT_ROOTS", tempfile.gettempdir())

# backup(dest_dir=...) is contained the same way — dest_dir reaches
# os.makedirs() and the retention sweep's os.remove(), so it must resolve
# under an allowed root (see backend/store.py's backup() and
# _do_backup's pre-check). Backup tests write to tempfile.mkdtemp()
# directories outside home, so extend the allowlist for the test session.
os.environ.setdefault("HLM_BACKUP_ALLOWED_ROOTS", tempfile.gettempdir())

# ── Profile env var isolation ─────────────────────────────────────────────────
# When tests run under `hermes -p hlm-test`, the active profile's .env is
# loaded into os.environ BEFORE the test process starts. Without stripping
# these, tests that assert on defaults (T102 tracing off, T161 LLM unconfigured,
# T190 no profile vars in os.environ) see the profile's values instead.
# We save and remove them at module load so every test starts clean; tests
# that need a specific value (T122, T192, T69) set it explicitly themselves.
_PROFILE_ENV_VARS = [
    "HLM_DB_PATH",
    "HLM_ENRICH_LLM",
    "HLM_MAX_LAYER",
    "HLM_LAYER3_MODEL",
    "HLM_LAYER3_BASE_URL",
    "HLM_LAYER3_API_KEY",
    "HLM_TRACING",
    "HLM_LAYER0_TOP_K",
    "HLM_LAYER3_MODE",
    # "HLM_LAYER3_PROVIDER" was here and is gone: nothing in this repo reads
    # it. It arrived in the mass HERMES_LAYERED_* -> HLM_* rename (5502e8b)
    # and never corresponded to a setting — the config key is
    # `layer3_provider_config`, which has no env form. Clearing a name nothing
    # reads protects nothing, and leaving it in this list is how a reader
    # concludes the variable exists. Found by T657 in the 2026-09-15 drift
    # pass. If an env form is ever added, add the name back deliberately.
    "HLM_CONFLICT_THRESHOLDS",
    "HLM_SEED_OVERVIEW",
]
for _var in _PROFILE_ENV_VARS:
    os.environ.pop(_var, None)

# ── Preflight: verify required packages before importing backend ───────────
# Fails fast with a readable message rather than 18+ assertion failures deep
# inside individual tests. Runs at module level so it fires on the first
# conftest load (before any test collects).

def _check_dependencies():
    """Check that required packages are importable. Exit with readable error if not."""
    required = {
        "qdrant_client": "pip install qdrant-client",
        "numpy": "pip install numpy",
        "sentence_transformers": "pip install sentence-transformers",
    }
    missing = []
    for module_name, install_hint in required.items():
        try:
            __import__(module_name)
        except ImportError:
            missing.append((module_name, install_hint))

    if missing:
        print("\n" + "=" * 72)
        print("  HLM test preflight FAILED — missing dependencies")
        print("=" * 72)
        print(f"\n  checked {len(required)} required package(s): {', '.join(required.keys())}")
        print(f"  missing {len(missing)} package(s): {', '.join(m[0] for m in missing)}")
        print()
        for pkg, hint in missing:
            print(f"  fix: {hint}")
        print()
        print("  Quick restore after Hermes upgrade:")
        print("  ~/.hermes/hermes-agent/venv/bin/pip install qdrant-client numpy sentence-transformers")
        print("=" * 72 + "\n")
        sys.exit(1)


_check_dependencies()

from backend import LayeredBackend, logger
from summaries import SummariesBackend

# ── Globals ─────────────────────────────────────────────────────────────────
QDRANT_URL = os.environ.get("HLM_QDRANT_URL", "http://localhost:6333")

# ── Preflight: verify the embedding endpoint is reachable ───────────────────
# The embedding model is configured per-profile in <profile>/.env (see
# docs/profile-isolation.md). When the endpoint is down, retrieval degrades
# to FTS5-only — which looks healthy to most tests but silently loses all
# vector signals. T183/T184 cover the degraded path; this catches the
# *operational* failure (wrong URL, dead server) before 200+ tests waste
# their time on empty vector results.
_EMBED_URL = os.environ.get("HLM_EMBED_URL")
_EMBED_MODEL = os.environ.get("HLM_EMBED_MODEL")
if _EMBED_URL and _EMBED_MODEL:
    try:
        import urllib.request
        _payload = json.dumps({"model": _EMBED_MODEL, "input": ["preflight"]}).encode()
        _req = urllib.request.Request(_EMBED_URL, data=_payload,
                                      headers={"Content-Type": "application/json"})
        _resp = urllib.request.urlopen(_req, timeout=5)
        _body = json.loads(_resp.read())
        if not _body.get("embeddings"):
            logger.warning("preflight: embedding endpoint %s returned no embeddings "
                           "(model %s may not be loaded)", _EMBED_URL, _EMBED_MODEL)
    except Exception as e:
        logger.warning("preflight: embedding endpoint %s unreachable: %s",
                       _EMBED_URL, e)
    else:
        logger.debug("preflight: embedding endpoint %s OK (%s)",
                     _EMBED_URL, _EMBED_MODEL)

# Test databases live in a throwaway directory, never in the production
# DB directory (~/.hermes/hermes-layered-memory-dbs). A crashed run must not
# be able to leave artifacts next to real memories. Override with
# HLM_TEST_DB_DIR to inspect artifacts after a run.
TEST_DB_DIR = os.environ.get("HLM_TEST_DB_DIR") or tempfile.mkdtemp(
    prefix="hlm-tests-"
)
os.makedirs(TEST_DB_DIR, exist_ok=True)

if not os.environ.get("HLM_TEST_DB_DIR"):
    atexit.register(lambda: shutil.rmtree(TEST_DB_DIR, ignore_errors=True))


# ── Helpers ─────────────────────────────────────────────────────────────────
def _real_home() -> str:
    """Real user home (not Hermes profile HOME)."""
    return pwd.getpwuid(os.getuid()).pw_dir


def _make_db_path(test_id: str) -> str:
    """Unique temp DB path for a test (isolated from the production DB dir)."""
    return os.path.join(TEST_DB_DIR, f"hlm-test-{test_id}.db")


# ── Qdrant collection isolation ───────────────────────────────────────────────
# The suite used to write into the *operator's* live collections — "memories",
# "sessions", "vault", "code" — because _make_backend passed
# qdrant_collection="memories" and let LayeredBackend build its default map.
# Two problems with that:
#   1. Tests mutate real profile data (they only delete by profile_name filter,
#      which is a thin guarantee).
#   2. Those collections are sized for the operator's embedding model
#      (e.g. 4096-dim qwen3-embedding), while the suite defaults to 384-dim
#      all-MiniLM-L6-v2. Every Qdrant call then fails with a dimension error,
#      so vector search never actually runs and tests silently exercise the
#      fallback paths instead — which is what let a real retrieval regression
#      (T01/T05/T84) and a false "recovery" pass (T186) hide for so long.
# Give the suite its own namespace. Override with HLM_TEST_QDRANT_PREFIX.
TEST_COLL_PREFIX = os.environ.get("HLM_TEST_QDRANT_PREFIX", "hlmtest_")
TEST_MEMORIES_COLL = f"{TEST_COLL_PREFIX}memories"
TEST_SESSIONS_COLL = f"{TEST_COLL_PREFIX}sessions"
TEST_VAULT_COLL = f"{TEST_COLL_PREFIX}vault"
# Mirrors LayeredBackend's default map, but namespaced. Passed explicitly
# because the default map hardcodes "sessions"/"vault" for SESSION-DATA and
# OBSIDIAN regardless of qdrant_collection — so setting only the latter would
# still route those two data types into the operator's collections.
TEST_COLLECTIONS = {
    "SYSTEM": TEST_MEMORIES_COLL,
    "USER-DATA": TEST_MEMORIES_COLL,
    "ENV-DATA": TEST_MEMORIES_COLL,
    "CUSTOM": TEST_MEMORIES_COLL,
    "SESSION-DATA": TEST_SESSIONS_COLL,
    "OBSIDIAN": TEST_VAULT_COLL,
}
TEST_COLL_NAMES = [TEST_MEMORIES_COLL, TEST_SESSIONS_COLL, TEST_VAULT_COLL]


def _live_test_collections() -> list:
    """Every collection in this Qdrant that belongs to the suite.

    Physical names are keyed to the embedding model since 0.6.0
    (`hlmtest_memories_qwen3-embedding_8b_4096`), so a static list no longer
    names what is actually there — and cleanup that misses leaves fixtures
    behind for the next run to trip over. Match on the prefix instead, which
    is the one part that stays put.
    """
    try:
        from qdrant_client import QdrantClient
        qc = QdrantClient(url=QDRANT_URL)
        try:
            return [c.name for c in qc.get_collections().collections
                    if c.name.startswith(TEST_COLL_PREFIX)]
        finally:
            qc.close()
    except Exception:
        return list(TEST_COLL_NAMES)


def _resolved_test_collections(dim: Optional[int]) -> list:
    """The suite's collections as *this* embedder would name them.

    `_live_test_collections()` answers a different question — everything with
    the `hlmtest_` prefix, which is right for cleanup and wrong for anything
    destructive, because it names collections belonging to other embedders and
    to any profile configured against the same prefix.
    """
    try:
        from backend.core import _collection_suffix, _suffix_enabled
    except Exception:
        return list(TEST_COLL_NAMES)
    if not _suffix_enabled():
        return list(TEST_COLL_NAMES)
    try:
        suffix = _collection_suffix(None, dim)
    except Exception:
        return list(TEST_COLL_NAMES)
    if not suffix:
        return list(TEST_COLL_NAMES)
    return [c if c.endswith(f"_{suffix}") else f"{c}_{suffix}"
            for c in TEST_COLL_NAMES]


def _foreign_profiles_in(qc, coll: str) -> set:
    """Profile names in `coll` that this suite did not create.

    Returns a non-empty set on any doubt, including a scroll that fails: the
    caller uses this to decide whether dropping a collection is safe, and
    "could not tell" must not read as "safe".
    """
    try:
        points, _ = qc.scroll(collection_name=coll, limit=256,
                              with_payload=["profile_name"], with_vectors=False)
    except Exception as e:
        return {f"<unreadable: {e}>"}
    foreign = set()
    for point in points:
        name = (point.payload or {}).get("profile_name") or ""
        if not str(name).startswith("test-"):
            foreign.add(str(name) or "<unnamed>")
    return foreign


def _collections_to_realign(qc, dim: Optional[int]) -> list:
    """Which collections `_align_test_collection_dims` may delete, and why.

    Split out from the caller so the decision can be tested without deleting
    anything: a test that reproduces the 2026-08-13 incident by dropping a real
    collection *is* the incident. Returns (name, existing_dim) pairs.
    """
    doomed = []
    for coll in _resolved_test_collections(dim):
        try:
            existing = qc.get_collection(coll).config.params.vectors.size
        except Exception:
            continue  # absent — _init_qdrant will create it at the right size
        if existing == dim:
            continue
        foreign = _foreign_profiles_in(qc, coll)
        if foreign:
            print(f"  test collection {coll}: NOT dropped (dim {existing} "
                  f"!= {dim}) — holds points from {sorted(foreign)[:3]}")
            continue
        doomed.append((coll, existing))
    return doomed


def _align_test_collection_dims() -> Optional[int]:
    """Drop test collections whose vector size no longer matches the embedder.

    Switching embedders — which is exactly what `eval_retrieval.py --profile`
    does, moving from the 384-dim local fallback to a 4096-dim endpoint —
    leaves the hlmtest_* collections sized for the previous model. Every upsert
    then fails with "Vector dimension error", the circuit breaker opens, and
    the run aborts with a message about Qdrant being down rather than about the
    dimension change that caused it. _init_qdrant only *warns* on a mismatch,
    by design, because dropping a collection in production would destroy data.
    Here the collections are disposable by construction (TEST_COLL_PREFIX), so
    recreate them instead of failing.

    Only ever touches the collections *this run would use*, resolved through the
    same model-key suffix the backend applies. It used to iterate
    `_live_test_collections()`, which matches on the `hlmtest_` prefix alone —
    and that made the drop reachable for a collection this run has no business
    touching. On 2026-08-13 a `run-regression.py` invocation without the profile
    environment probed 384-dim MiniLM, saw `hlmtest_memories_qwen3-embedding_8b_4096`
    at 4096, and dropped it. Those are the collections the **hlm-test profile**
    is configured to use, so a stray suite import destroyed a live profile's
    vectors (`initialize: sync check — SQLite=24, Qdrant=0, in_sync=False`).
    SQLite is the source of truth so the next session rebuilt them, but nothing
    should be able to reach that state by accident.

    A second guard backs it up: a collection holding points from any profile
    that is not this suite's is never dropped, whatever its dimension. The
    suite names its profiles `test-<id>` (see `_make_backend`), so anything
    else in there belongs to someone.

    Returns the detected dimension, or None if it could not be determined.
    """
    try:
        from backend.core import _get_embedding_fn
        vectors = _get_embedding_fn()(["dimension probe"])
        if not vectors or not vectors[0]:
            return None
        dim = len(vectors[0])
    except Exception as e:
        logger.debug("_align_test_collection_dims: cannot probe embedder: %s", e)
        return None
    try:
        from qdrant_client import QdrantClient
        qc = QdrantClient(url=QDRANT_URL)
    except Exception:
        return dim
    try:
        # With model-keyed names (0.6.0) a mismatch is unreachable for the
        # collections this run resolves to: a different embedder means a
        # different name. That is why the list below is the *resolved* set and
        # not every `hlmtest_` collection present — see the docstring. The drop
        # remains a safety net for HLM_QDRANT_COLLECTION_SUFFIX=false, where the
        # names stay flat and a real collision can happen.
        for coll, existing in _collections_to_realign(qc, dim):
            qc.delete_collection(coll)
            print(f"  test collection {coll}: dropped (dim {existing} != {dim})")
    finally:
        try:
            qc.close()
        except Exception:
            pass
    return dim


def _autoalign_test_collections_once() -> None:
    """Realign the test collections with this run's embedder, at import.

    The suite runs on the 384-dim local fallback; `eval_retrieval.py --profile
    <name>` runs on the profile's endpoint (4096-dim here). Whichever ran last
    leaves the shared hlmtest_* collections sized for its model, and the next
    run of the other one fails every upsert with a dimension error, opens the
    circuit breaker, and reports itself as "Qdrant down" — observed as T82,
    T186 and T216 failing for a reason that has nothing to do with them.

    Cheap: one embedding probe per process, and the model is loaded anyway.
    Only ever touches TEST_COLL_NAMES.
    """
    try:
        _align_test_collection_dims()
    except Exception as e:  # never block collection on a housekeeping step
        logger.debug("test collection auto-align skipped: %s", e)


_autoalign_test_collections_once()


def _reset_qdrant_for_test(test_id: str):
    """Reset Qdrant collections for a test (idempotent, handles stale data)."""
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        qc = QdrantClient(url=QDRANT_URL)
        profile = f"test-{test_id}"
        for coll in _live_test_collections():
            try:
                f = Filter(must=[FieldCondition(key="profile_name",
                           match=MatchValue(value=profile))])
                qc.delete(collection_name=coll, points_selector=f)
            except Exception:
                pass
        qc.close()
    except Exception:
        pass


#: Cached under a fixed, importable name so every caller gets one instance.
_PLUGIN_MODULE_NAME = "hlm_plugin_under_test"


def plugin_module():
    """Load the plugin package by **file path**, not by directory name.

    Seven call sites across three test files did
    `importlib.import_module(os.path.basename(root))`, which imports the plugin
    by whatever the checkout directory happens to be called. That works only
    while the directory name is also a valid Python module name, and nothing
    guarantees it: a clone into `hermes-layered-memory.git`, or the staging
    directory `hermes-layered-memory.github` used to build the public snapshot,
    both fail with

        ModuleNotFoundError: No module named 'hermes-layered-memory.github'

    because the dot makes Python read it as a package path. Found when the
    published tree's own suite was run before publishing it — T665 failed in a
    correctly-built tree for a reason that had nothing to do with the code.

    A suite whose result depends on the name of the directory it was checked
    out into is the same class as a suite whose result depends on the host
    (`T353`, `T349`, `T582`): it answers a question about this machine while
    reading as a question about the repo. Loading by path answers the second
    one. T675.
    """
    import importlib.util
    if _PLUGIN_MODULE_NAME in sys.modules:
        return sys.modules[_PLUGIN_MODULE_NAME]
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    spec = importlib.util.spec_from_file_location(
        _PLUGIN_MODULE_NAME, os.path.join(root, "__init__.py"),
        submodule_search_locations=[root])
    mod = importlib.util.module_from_spec(spec)
    # Registered before exec so any self-referential import inside the package
    # resolves to this module rather than starting a second load.
    sys.modules[_PLUGIN_MODULE_NAME] = mod
    spec.loader.exec_module(mod)
    return mod


def _make_backend(test_id: str, **kwargs) -> LayeredBackend:
    """Create a LayeredBackend with an isolated DB for a test."""
    profile = kwargs.pop("profile_name", f"test-{test_id}")
    config = kwargs.pop("config", {})
    config["enrich_llm"] = False  # avoid background thread races
    # Route every data_type into the test namespace unless the test asked for
    # its own map (see TEST_COLLECTIONS for why this must be explicit).
    config.setdefault("collections", dict(TEST_COLLECTIONS))
    _reset_qdrant_for_test(test_id)
    # Save & unset env vars that _apply_env_overrides() would use to
    # silently override the explicit config values (db_path & enrich_llm).
    saved = {
        "HLM_DB_PATH": os.environ.pop("HLM_DB_PATH", None),
        "HLM_ENRICH_LLM": os.environ.pop("HLM_ENRICH_LLM", None),
    }
    try:
        return LayeredBackend(
            db_path=_make_db_path(test_id),
            qdrant_url=QDRANT_URL,
            qdrant_collection=TEST_MEMORIES_COLL,
            profile_name=profile,
            config=config,
            **kwargs,
        )
    finally:
        for key, val in saved.items():
            if val is not None:
                os.environ[key] = val


def _make_summaries(test_id: str) -> SummariesBackend:
    """Create a SummariesBackend with isolated DB."""
    md_dir = tempfile.mkdtemp(prefix=f"hlm-sum-{test_id}-", dir=TEST_DB_DIR)
    return SummariesBackend(
        db_path=os.path.join(TEST_DB_DIR, f"hlm-sum-{test_id}.db"),
        summaries_dir=md_dir,
    )


def _cleanup_qdrant_coll(be: LayeredBackend):
    """Delete test points from Qdrant collections."""
    if not be or not be._qdrant:
        return
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        for coll in set(be._collection_map.values()):
            try:
                profile_filter = Filter(
                    must=[FieldCondition(key="profile_name",
                          match=MatchValue(value=be._profile_name))]
                )
                be._qdrant.delete(
                    collection_name=coll,
                    points_selector=profile_filter,
                )
            except Exception:
                pass
    except Exception:
        pass


def _cleanup_db(test_id: str):
    """Remove test DB files."""
    import time as _time
    _time.sleep(0.2)
    db_path = _make_db_path(test_id)
    if os.path.exists(db_path):
        try:
            import sqlite3 as _sqlite3
            _conn = _sqlite3.connect(db_path, timeout=5)
            _conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            _conn.close()
        except Exception:
            pass
        for ext in ["-wal", "-shm", ""]:
            try:
                os.remove(db_path + ext)
            except OSError:
                pass
    for pattern in [f"hlm-sum-{test_id}.db*"]:
        for f in glob.glob(os.path.join(TEST_DB_DIR, pattern)):
            try:
                os.remove(f)
            except OSError:
                pass


def _get_uuid(result) -> str:
    """Handle add() returning either str (uuid) or dict (duplicate detected)."""
    if isinstance(result, dict):
        return result["uuid"]
    return result


def _docker(action: str, container: str = None):
    """Run docker stop/start."""
    if container is None:
        try:
            out = subprocess.run(
                ["docker", "ps", "--format", "{{.Names}}"],
                capture_output=True, text=True, timeout=10,
            ).stdout
            containers = [n for n in out.split() if "qdrant" in n.lower()]
            container = containers[0] if containers else "qdrant"
        except Exception:
            container = "qdrant"
    subprocess.run(["docker", action, container], capture_output=True, timeout=30)


def _check_qdrant_alive() -> bool:
    """Quick health check."""
    try:
        import urllib.request
        resp = urllib.request.urlopen(QDRANT_URL + "/", timeout=5)
        return resp.status == 200
    except Exception:
        return False


# ── Multiplex secret-scope simulation ───────────────────────────────────────
class UnscopedSecretError(RuntimeError):
    """Stands in for agent.secret_scope.UnscopedSecretError."""


@contextlib.contextmanager
def fake_secret_scope(mapping: Dict[str, str], multiplex: bool = False):
    """Install a fake profile secret scope for the duration of the block.

    Under a multiplex gateway the active profile's .env is loaded into an
    isolated mapping rather than os.environ (docs/profile-isolation.md), so
    code that reads os.environ directly silently gets root-level values. That
    path cannot be reached from a normal test run, which is precisely why it
    went unnoticed — this fixture makes it reachable.

    Args:
        mapping: the profile's scoped variables.
        multiplex: when True, a lookup for an unscoped name raises rather than
            falling through, matching multiplex-on behaviour.
    """
    # The seam lives in backend.core, where _setting() reads it — the shared
    # primitives were split out of backend.backend so the method modules could
    # import them directly instead of going through sys.modules.
    import backend.core as _backend

    def _get_secret(name):
        if name in mapping:
            return mapping[name]
        if multiplex:
            raise UnscopedSecretError(f"{name} is not scoped to this profile")
        return os.environ.get(name)

    saved = _backend._secret_getter
    _backend._secret_getter = _get_secret
    try:
        yield
    finally:
        _backend._secret_getter = saved
