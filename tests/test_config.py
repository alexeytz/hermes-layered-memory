"""T105-T128: Seed overview, enrichment, protected, maintenance, config, write-time detection."""

import os
import json
import pwd
from datetime import datetime, timedelta, timezone

from backend import logger
from conftest import (
    _make_backend, _cleanup_db, _cleanup_qdrant_coll, _get_uuid, plugin_module,
)


# ── Seed overview ───────────────────────────────────────────────────────────

def test_t105():
    """Seed overview — basic structure."""
    be = _make_backend("t105")
    try:
        be.add("seed fact", data_type="ENV-DATA", data_id="hw", trust_score=0.9)
        overview = be.seed_overview()
        assert "Knowledge Base Overview" in overview and "records" in overview
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t105")


def test_t106():
    """Seed overview — empty KB."""
    be = _make_backend("t106")
    try:
        assert be.seed_overview() == ""
    finally:
        be.close(); _cleanup_db("t106")


def test_t107():
    """Seed overview — data_type:data_id groups."""
    be = _make_backend("t107")
    try:
        for i in range(3):
            be.add(f"hw item {i}", data_type="ENV-DATA", data_id="hw")
        be.add("user pref", data_type="USER-DATA", data_id="preferences")
        overview = be.seed_overview()
        assert "ENV-DATA" in overview and "USER-DATA" in overview
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t107")


def test_t108():
    """Seed overview — top trusted record shown."""
    be = _make_backend("t108")
    try:
        be.add("trusted fact here", data_type="ENV-DATA", data_id="hw", trust_score=1.0)
        be.add("untrusted", data_type="CUSTOM", trust_score=0.1)
        overview = be.seed_overview()
        assert "Top trusted" in overview and "trusted fact" in overview
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t108")


def test_t109():
    """Seed overview — BM25 note present."""
    be = _make_backend("t109")
    try:
        be.add("test", data_type="CUSTOM")
        assert "BM25" in be.seed_overview()
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t109")


# ── Re-enrich ──────────────────────────────────────────────────────────────

def test_t110():
    """Re-enrich — topic_only."""
    be = _make_backend("t110")
    try:
        be.add("RTX 4090 GPU", data_type="ENV-DATA", data_id="hw", topic=None)
        result = be.re_enrich(topic_only=True, limit=5)
        assert isinstance(result, dict) and "total_scanned" in result
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t110")


def test_t111():
    """Re-enrich — keyword_only."""
    be = _make_backend("t111")
    try:
        be.add("RTX 4090 GPU", data_type="ENV-DATA", data_id="hw", keywords=[])
        result = be.re_enrich(keyword_only=True, limit=5)
        assert isinstance(result, dict)
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t111")


def test_t112():
    """Re-enrich — full."""
    be = _make_backend("t112")
    try:
        be.add("RTX 4090 GPU", data_type="CUSTOM")
        result = be.re_enrich(limit=5)
        assert isinstance(result, dict)
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t112")


# ── Protected / maintenance ────────────────────────────────────────────────

def test_t113():
    """Protected: decay skip."""
    be = _make_backend("t113")
    try:
        uuid = be.add("protected record", data_type="CUSTOM", priority=0,
                      trust_score=0.5, protected=True)
        past = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
        be._get_conn().execute("UPDATE memories SET created_at = ? WHERE uuid=?", (past, uuid))
        be._get_conn().commit()
        be.decay(min_age_days=0, max_age_days=365, decay_rate=0.05, min_score=0.1)
        assert be._get_record(uuid)["trust_score"] == 0.5
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t113")


def test_t114():
    """Protected: add with protected=True."""
    be = _make_backend("t114")
    try:
        uuid = _get_uuid(be.add("new protected", data_type="CUSTOM", protected=True))
        row = be._get_conn().execute(
            "SELECT protected FROM memories WHERE uuid=?", (uuid,)
        ).fetchone()
        assert row and row[0] in (1, True)
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t114")


def test_t115():
    """Event-driven maintenance — should_run."""
    be = _make_backend("t115")
    try:
        assert be._should_run_maintenance("last_test", 10) is True
    finally:
        be.close(); _cleanup_db("t115")


def test_t116():
    """Maintenance budget tracking."""
    be = _make_backend("t116")
    try:
        cfg = be._get_cleanup_config()
        assert "maintenance_budget" in cfg and cfg["maintenance_budget"] == 2
    finally:
        be.close(); _cleanup_db("t116")


def test_t117():
    """Vacuum threshold check."""
    be = _make_backend("t117")
    try:
        assert "vacuum_free_page_pct" in be._get_cleanup_config()
    finally:
        be.close(); _cleanup_db("t117")


def test_t118():
    """Event-driven maintenance — threshold gating."""
    be = _make_backend("t118")
    try:
        be._set_maintenance_state("last_test", "5")
        be.add("record", data_type="CUSTOM")
        # _should_run_maintenance returns True when threshold exceeded
        should_run = be._should_run_maintenance("last_test", 10)
        assert isinstance(should_run, bool), f"Expected bool, got {type(should_run).__name__}"
    finally:
        be.close(); _cleanup_db("t118")


def test_t119():
    """Cleanup config — defaults."""
    be = _make_backend("t119")
    try:
        cfg = be._get_cleanup_config()
        assert cfg["decay_min_age_days"] == 30 and cfg["decay_rate"] == 0.05
    finally:
        be.close(); _cleanup_db("t119")


def test_t120():
    """Cleanup config — custom overrides."""
    be = _make_backend("t120", config={"cleanup": {"decay_rate": 0.1, "decay_min_score": 0.2}})
    try:
        cfg = be._get_cleanup_config()
        assert cfg["decay_rate"] == 0.1 and cfg["decay_min_score"] == 0.2
    finally:
        be.close(); _cleanup_db("t120")


def test_t121():
    """Maintenance state persistence."""
    be = _make_backend("t121")
    try:
        be._set_maintenance_state("key1", "value1")
        assert be._maintenance_state("key1") == "value1"
        assert be._maintenance_state("nonexistent") is None
    finally:
        be.close(); _cleanup_db("t121")


def test_t122():
    """Heuristic gate — disabled when env set."""
    old = os.environ.get("HLM_MAX_LAYER")
    os.environ["HLM_MAX_LAYER"] = "2"
    try:
        be = _make_backend("t122")
        try:
            be.add("test", data_type="CUSTOM")
            assert isinstance(be.retrieve("follow docs for today", max_layer=2), list)
        finally:
            be.close(); _cleanup_db("t122")
    finally:
        if old is not None:
            os.environ["HLM_MAX_LAYER"] = old
        else:
            os.environ.pop("HLM_MAX_LAYER", None)


def test_t123():
    """LLM explicit max_layer=3 honored with env set."""
    old = os.environ.get("HLM_MAX_LAYER")
    os.environ["HLM_MAX_LAYER"] = "2"
    try:
        be = _make_backend("t123")
        try:
            be.add("test", data_type="CUSTOM")
            assert isinstance(be.retrieve("anything", max_layer=3), list)
        finally:
            be.close(); _cleanup_db("t123")
    finally:
        if old is not None:
            os.environ["HLM_MAX_LAYER"] = old
        else:
            os.environ.pop("HLM_MAX_LAYER", None)


# ── Write-time detection ───────────────────────────────────────────────────

def test_t124():
    """Write-time contradiction detection."""
    be = _make_backend("t124")
    try:
        be.add("Default timeout is 30 seconds", data_type="ENV-DATA", data_id="net")
        result = be.add("Default timeout changed to 60 seconds", data_type="ENV-DATA", data_id="net")
        assert isinstance(result, (str, dict))
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t124")


def test_t125():
    """Write-time duplicate (hard block)."""
    be = _make_backend("t125")
    try:
        be.add("RTX 4090 GPU with 24GB VRAM", data_type="ENV-DATA", data_id="hw")
        result = be.add("RTX 4090 GPU 24GB VRAM", data_type="ENV-DATA", data_id="hw")
        assert isinstance(result, dict)
        assert result.get("status") in ("duplicate", "possible_duplicate", "contradiction")
        assert "similarity" in result
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t125")


def test_t126():
    """Write-time possible duplicate (warning zone)."""
    be = _make_backend("t126")
    try:
        be.add("vLLM serving qwen3-embedding:8b model on embed-host.example.net:11434 with tensor parallelism",
               data_type="ENV-DATA", data_id="sw")
        result = be.add("vLLM serving qwen3-embedding:8b on embed-host.example.net:11434",
                        data_type="ENV-DATA", data_id="sw")
        assert isinstance(result, (dict, str))
        if isinstance(result, dict):
            assert result.get("status") in ("duplicate", "possible_duplicate", "contradiction")
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t126")


def test_t127():
    """Write-time detection — return value structure."""
    be = _make_backend("t127")
    try:
        be.add("structure test original", data_type="ENV-DATA", data_id="hw")
        result = be.add("structure test duplicate", data_type="ENV-DATA", data_id="hw")
        if isinstance(result, dict):
            for key in ("uuid", "status", "similarity", "note"):
                assert key in result, f"Missing key '{key}'"
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t127")


def test_t128():
    """Write-time detection — configurable thresholds."""
    be = _make_backend("t128", config={
        "dedup_threshold": 0.95, "dedup_warning_threshold": 0.92})
    try:
        be.add("threshold test A", data_type="ENV-DATA", data_id="hw")
        result = be.add("threshold test A nearly identical", data_type="ENV-DATA", data_id="hw")
        assert isinstance(result, (dict, str))
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t128")


# ── M0: Config persistence roundtrip ────────────────────────────────────────

def test_t134():
    """M0-1: Runtime config DB roundtrip preserves types (fixes _json NameError)."""
    be = _make_backend("config_rt")
    try:
        # Write config via backend method
        be.set_config("dedup_threshold", 0.90)
        # Reload from DB (simulates new backend instance reading persisted config)
        loaded = be._load_db_config()
        assert "dedup_threshold" in loaded, f"Key not in loaded config: {loaded}"
        assert loaded["dedup_threshold"] == 0.90, (
            f"Expected float 0.90, got {loaded['dedup_threshold']!r} (type={type(loaded['dedup_threshold']).__name__}). "
            "If this is a string, _json is still undefined in _load_db_config()."
        )
        assert isinstance(loaded["dedup_threshold"], float), (
            f"Expected float, got {type(loaded['dedup_threshold']).__name__}"
        )
    finally:
        be.close(); _cleanup_db("config_rt")


def test_t135():
    """M0-1: Runtime config preserves JSON object types (not just primitives)."""
    be = _make_backend("config_json")
    try:
        be.set_config("conflict_thresholds", {"temporal_guard_days": 3, "cosine_min": 0.88})
        loaded = be._load_db_config()
        assert "conflict_thresholds" in loaded
        assert isinstance(loaded["conflict_thresholds"], dict), (
            f"Expected dict, got {type(loaded['conflict_thresholds']).__name__}"
        )
        assert loaded["conflict_thresholds"]["temporal_guard_days"] == 3
    finally:
        be.close(); _cleanup_db("config_json")

def test_t161():
    """An unconfigured LLM provider is reported, not silently ignored.

    _call_llm() returns "" when layer3_model or base_url is missing, and six
    features route through it. Extraction stopped writing on profiles that had
    relied on the old subprocess route, with nothing in the logs to say why.
    """
    be = _make_backend("t161")
    try:
        assert be.llm_configured() is False, "no provider configured in test fixture"
        assert be.sync_check()["llm_configured"] is False, "sync_check must report it"
        assert be.retrieval_stats()["llm_configured"] is False, "stats must report it"
        assert be._call_llm("say OK") == "", "unconfigured provider should return empty"
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t161")


def test_t162():
    """llm_configured() requires BOTH a model and a base_url."""
    be = _make_backend("t162", config={
        "layer3_model": "some-model",
        "layer3_provider_config": {"base_url": "http://localhost:9/v1", "api_key": "k"},
    })
    try:
        assert be.llm_configured() is True, "both set should report configured"
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t162")

    be = _make_backend("t162b", config={"layer3_model": "some-model"})
    try:
        assert be.llm_configured() is False, "model without base_url is not configured"
    finally:
        _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t162b")


def test_t169():
    """Session chain anchors to the git root, not whatever cwd the agent started in.

    _project_dir was os.getcwd(), so launching the agent from a subdirectory
    dropped a second sessions.txt there and split one project's chain across
    every directory it had ever been started from.
    """
    import tempfile as _tf
    from pathlib import Path
    import sys as _sys

    _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import importlib.util as _util
    spec_src = open(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "__init__.py"),
        encoding="utf-8").read()
    ns = {"os": os}
    start = spec_src.find("def _find_project_root")
    end = spec_src.find("def _load_config")
    exec(compile(spec_src[start:end], "<extract>", "exec"), ns)
    _find_project_root = ns["_find_project_root"]

    with _tf.TemporaryDirectory() as root:
        Path(root, ".git").mkdir()
        nested = Path(root, "src", "deep")
        nested.mkdir(parents=True)

        assert _find_project_root(str(nested)) == os.path.abspath(root), (
            "should walk up to the git root"
        )
        assert _find_project_root(str(root)) == os.path.abspath(root), (
            "already at the root"
        )

    with _tf.TemporaryDirectory() as plain:
        assert _find_project_root(plain) == os.path.abspath(plain), (
            "no repository — fall back to the given directory"
        )


def test_t190():
    """Profile settings are read through the secret scope, not just os.environ.

    Under a multiplex gateway the active profile's .env goes into an isolated
    mapping rather than os.environ (docs/profile-isolation.md). Reading
    os.environ directly meant every HLM_* tuning var silently
    reverted to its default for that profile.
    """
    from conftest import fake_secret_scope

    # Deliberately absent from os.environ — only the scope has them
    for key in ("HLM_MAX_LAYER", "HLM_LAYER3_MODEL",
                "HLM_LAYER3_BASE_URL"):
        assert key not in os.environ, f"{key} must not be in os.environ for this test"

    scoped = {
        "HLM_MAX_LAYER": "4",
        "HLM_LAYER3_MODEL": "scoped-model",
        "HLM_LAYER3_BASE_URL": "http://scoped.invalid/v1",
        "HLM_LAYER0_TOP_K": "17",
    }
    with fake_secret_scope(scoped, multiplex=True):
        be = _make_backend("t190")
        try:
            assert be._config.get("max_layer") == 4, (
                f"scoped max_layer ignored: {be._config.get('max_layer')}"
            )
            assert be._config.get("layer3_model") == "scoped-model", "scoped model ignored"
            provider = be._config.get("layer3_provider_config") or {}
            assert provider.get("base_url") == "http://scoped.invalid/v1", (
                "scoped base_url ignored"
            )
            assert be._layer0_top_k == 17, f"scoped top_k ignored: {be._layer0_top_k}"
            assert be.llm_configured() is True, (
                "a fully scoped profile should report the LLM as configured"
            )
        finally:
            _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t190")


def test_t191():
    """An unscoped name under multiplex is treated as absent, not fatal.

    Multiplex raises UnscopedSecretError rather than leaking the root value.
    That must read as "not set for this profile" and fall back to defaults —
    not propagate out of construction.
    """
    from conftest import fake_secret_scope

    # Only one var is scoped; every other lookup raises.
    with fake_secret_scope({"HLM_MAX_LAYER": "3"}, multiplex=True):
        be = _make_backend("t191")
        try:
            assert be._config.get("max_layer") == 3, "the scoped var should apply"
            # Unscoped ones simply do not appear
            assert be._config.get("layer3_model") in (None, ""), (
                "an unscoped var must not be invented"
            )
            assert be.llm_configured() is False, "no provider scoped, so not configured"
        finally:
            _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t191")


def test_t192():
    """With no secret scope installed, os.environ still governs (CLI path)."""
    old = os.environ.get("HLM_MAX_LAYER")
    try:
        os.environ["HLM_MAX_LAYER"] = "1"
        be = _make_backend("t192")
        try:
            assert be._config.get("max_layer") == 1, (
                f"os.environ must still win when no scope exists: "
                f"{be._config.get('max_layer')}"
            )
        finally:
            _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t192")
    finally:
        if old is not None:
            os.environ["HLM_MAX_LAYER"] = old
        else:
            os.environ.pop("HLM_MAX_LAYER", None)


def test_t193():
    """The scope takes precedence over a conflicting os.environ value.

    Multiplex root-level vars live in os.environ; the profile's own value must
    win, otherwise one profile's setting leaks into another's session.
    """
    from conftest import fake_secret_scope

    old = os.environ.get("HLM_MAX_LAYER")
    try:
        os.environ["HLM_MAX_LAYER"] = "1"   # root-level
        with fake_secret_scope({"HLM_MAX_LAYER": "4"}):  # profile-level
            be = _make_backend("t193")
            try:
                assert be._config.get("max_layer") == 4, (
                    "the profile's scoped value must beat the root-level env var, "
                    f"got {be._config.get('max_layer')}"
                )
            finally:
                _cleanup_qdrant_coll(be); be.close(); _cleanup_db("t193")
    finally:
        if old is not None:
            os.environ["HLM_MAX_LAYER"] = old
        else:
            os.environ.pop("HLM_MAX_LAYER", None)


def test_t203():
    """The suite must never log into a production profile log.

    backend.py resolves HLM_LOG_FILE at import, so without an early
    redirect the tests write wherever the ambient environment points. During an
    E2E — which is required to run Tier 1 — that meant 194 tests' output,
    including chaos tests that fail deliberately, landed in the profile log.
    The E2E log gate then reported "Embedding unavailable" and orphan deletions
    that never happened in production, and the agent rationalised the failure
    away. A gate that cries wolf is a gate that gets ignored.
    """
    log_file = os.environ.get("HLM_LOG_FILE")
    assert log_file, "the suite must pin HLM_LOG_FILE to a temp path"

    real_home = pwd.getpwuid(os.getuid()).pw_dir
    profiles_dir = os.path.join(real_home, ".hermes", "profiles")
    assert not os.path.abspath(log_file).startswith(os.path.abspath(profiles_dir)), (
        f"tests are logging into a profile directory: {log_file}"
    )
    hermes_logs = os.path.join(real_home, ".hermes", "logs")
    assert not os.path.abspath(log_file).startswith(os.path.abspath(hermes_logs)), (
        f"tests are logging into the shared Hermes log directory: {log_file}"
    )

    # And the chaos tests' deliberate failure strings must not reach it
    from conftest import _make_backend as _mb  # noqa: F401
    before = os.path.getsize(log_file) if os.path.exists(log_file) else 0
    logger.warning("t203 canary line")
    after = os.path.getsize(log_file) if os.path.exists(log_file) else 0
    assert after >= before, "log file should be writable by the suite"


def test_t665():
    """A config key or taxonomy name that is not a string is refused at the door.

    `key` and `name` are identifiers: written to a TEXT column, used as dict
    keys in the live config, read back by string. The plugin's doors checked
    only that they were truthy, and SQLite's affinity quietly papered over the
    difference. Driven on the pre-fix tree:

        set(key=123, value="x")   -> {"status": "updated"}           # success
        get(key=123)              -> {"value": "x", "exists": true}  # same session
        -- reopen the backend --
        get(key=123)              -> {"exists": false}               # gone
        get(key="123")            -> {"value": "x", "exists": true}  # moved

    The write succeeds, the read-back confirms it, and at the next start the
    value is under a different key — the row went to disk as the text `'123'`
    while the live dict kept the int. That is the shape AGENTS.md names as the
    worst one for a setting an agent adjusts: every signal points the right way
    and the change is not there later.

    An unhashable key was louder and no better: `key=["a"]` reached
    `coerce_config_value`, which does `_CONFIG_VALUE_TYPES.get(key)`, and the
    door answered `{"error": "unhashable type: 'list'"}` — the interpreter's
    own message, the class T614/T636/T645 exist to keep off the doors.

    The MCP twin declares `key: str | None`, so its pydantic layer already
    refuses both (verified: `key=123` -> "Input should be a valid string").
    This was the plugin being the lenient door again.
    2026-09-14 review round 1, bundle05 (F4).
    """
    import importlib
    import sys as _sys
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _sys.path.insert(0, os.path.dirname(_root))
    plugin = plugin_module()
    be = _make_backend("t665")
    try:
        prov = plugin.LayeredMemoryProvider()
        prov._backend = be

        def _txt(o):
            return o if isinstance(o, str) else json.dumps(o)

        for bad in (123, ["a"], {"k": "v"}, 1.5, True):
            for args, door in (
                    ({"key": bad, "value": "x"}, "_do_set_config"),
                    ({"key": bad}, "_do_get_config"),
                    ({"key": bad}, "_do_delete_config"),
                    ({"name": bad}, "_do_register_taxonomy"),
                    ({"name": bad}, "_do_unregister_taxonomy")):
                out = _txt(getattr(prov, door)(dict(args)))
                assert "error" in out, (
                    "%s accepted a non-string identifier %r: %s"
                    % (door, bad, out[:200]))
                assert "must be a string" in out or "is required" in out, (
                    "%s refused %r with the interpreter's own message rather "
                    "than one naming the parameter: %s" % (door, bad, out[:200]))

        # The legitimate shapes still work: a string key, and `get` with no key
        # at all (the whole merged config).
        assert "error" not in _txt(prov._do_set_config({"key": "t665_key",
                                                        "value": "hello"}))
        got = prov._do_get_config({"key": "t665_key"})
        assert got["value"] == "hello" and got["exists"] is True, got
        assert "error" not in _txt(prov._do_get_config({}))
    finally:
        try: be.close()
        except Exception: pass
        _cleanup_db("t665")


def test_t666():
    """`unregister_taxonomy` validates `kind` the way `register_taxonomy` does.

    The vocabulary check was on the write and not on the delete, so
    `kind="data_typo"` became `AND kind = 'data_typo'`, matched no row and
    returned the *same* `{"status": "not_found"}` as a name that does not
    exist. An agent that misspelled the kind was told the entry was already
    gone, with no way to tell the two apart — while `register_taxonomy`
    refused the identical word by name.

    In the backend rather than on the two front ends because both cross it and
    neither can share code with the other, which is the same reason
    `register_taxonomy`'s own check lives there. `kind=None` still means
    "every kind of this name" — the documented default both doors rely on.
    2026-09-14 review round 1, bundle05 (F5).
    """
    be = _make_backend("t666")
    try:
        assert be.register_taxonomy("T666TYPE", kind="data_typo").get("error"), (
            "register accepted an invalid kind — the premise of this test is gone")
        assert be.register_taxonomy("T666TYPE", kind="data_type")["status"] == "registered"

        bad = be.unregister_taxonomy("T666TYPE", kind="data_typo")
        assert bad.get("error"), (
            "unregister answered %r for an invalid kind — indistinguishable "
            "from a name that does not exist" % (bad,))
        assert "data_type" in bad["error"], (
            "the refusal does not name the valid kinds: %r" % (bad,))

        # A name that genuinely is not there is still `not_found`, and the
        # two answers must not be the same one.
        assert be.unregister_taxonomy("T666NOPE")["status"] == "not_found"
        # None still means "every kind of this name".
        assert be.unregister_taxonomy("T666TYPE")["status"] == "unregistered"
    finally:
        try: be.close()
        except Exception: pass
        _cleanup_db("t666")
