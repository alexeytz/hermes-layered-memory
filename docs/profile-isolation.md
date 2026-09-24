# Hermes Profile Isolation and .env Loading

## The short version

Hermes profiles are **isolated** — each profile has its own `HERMES_HOME`, its own `.env`, its own SQLite DB, and its own set of environment variables. HLM must respect this isolation: it must never read another profile's secrets, and it must always use the **active profile's** config.

## How .env gets loaded (three paths)

### HLM does not require a gateway

HLM works as a memory provider through the Hermes agent interface. A gateway is only needed for persistent chat platforms (Telegram, Discord, Slack) or multiplex mode. The most common usage — `hermes -p <profile> -z "..."` or `hermes -p <profile> chat` — is a CLI one-shot that never touches a gateway.

### Path 1: CLI (`hermes -p <profile> ...`)

1. `hermes_cli/main.py::_apply_profile_override()` pre-parses `-p <profile>` and sets:
   ```python
   os.environ["HERMES_HOME"] = "~/.hermes/profiles/<profile>"
   ```
2. `load_hermes_dotenv(hermes_home=HERMES_HOME)` loads `<profile>/.env` into `os.environ` with `override=True`.
3. The Python process now has the profile's vars in `os.environ`.
4. `LayeredBackend.__init__()` (in `backend/backend.py`) calls `_apply_env_overrides()` which reads `HLM_*` from `os.environ`.

**Result:** `os.environ` has the profile's vars. `os.environ.get("HLM_MAX_LAYER")` works.

### Path 2: Per-profile gateway (`<profile> gateway start`)

Per the [Hermes docs](https://hermes-agent.nousresearch.com/docs/user-guide/profiles): "Each profile runs its own gateway as a separate process." The alias `coder gateway start` is `hermes -p coder gateway start` under the hood, so it goes through Path 1: `HERMES_HOME` is set to the profile dir, and `<profile>/.env` is loaded into `os.environ`.

**Result:** `os.environ` has the profile's vars. Same as Path 1.

### Path 3: Multiplex gateway (single process, many profiles)

When `gateway.multiplex_profiles` is enabled in config.yaml, ONE gateway process serves MANY profiles. This is the only case where profile isolation is tricky:

1. Gateway starts with `HERMES_HOME=~/.hermes` (root, not a profile).
2. `load_hermes_dotenv(hermes_home=~/.hermes)` loads `~/.hermes/.env` (root-level only).
3. Per-session, `_profile_runtime_scope(profile_home)` wraps the handler:
   - `set_hermes_home_override(profile_home)` — redirects `get_hermes_home()` to the profile dir
   - `set_secret_scope(build_profile_secret_scope(profile_home))` — loads `<profile>/.env` into an **isolated mapping** (NOT `os.environ`)
4. Agent subprocesses inherit `env=dict(os.environ)` **merged with** the secret scope mapping.

**Result:** `os.environ` has root-level vars only. The profile's vars are in the **secret scope** (accessible via `secret_scope.get_secret("HLM_MAX_LAYER")`), **not** in `os.environ`. The agent subprocess sees both.

### Which path does your deployment use?

- `hermes -p hlm-test -z "..."` → Path 1 (CLI, `.env` in `os.environ`)
- `hlm-test gateway start` → Path 2 (per-profile gateway, `.env` in `os.environ`)
- Multiplex mode with `gateway.multiplex_profiles: true` → Path 3 (secret scope)

**For HLM:** Paths 1 and 2 are the common cases. `os.environ.get("HLM_...")` works. Path 3 is rare and would require `secret_scope.get_secret()` for profile-specific vars.

## How the secret scope works

`agent/secret_scope.py` provides `get_secret(name)` with this resolution order:

1. **Global env vars** (PATH, HOME, HERMES_MAX_ITERATIONS, etc.) — always read `os.environ`
2. **Secret scope** (if installed) — reads the profile's isolated mapping
3. **Fallback** — if multiplex is OFF, falls through to `os.environ`; if ON, **raises** `UnscopedSecretError`

`HLM_*` vars are **not** in the global env whitelist, so they go through the secret scope path.

## How to test .env-dependent code

### In unit tests (`tests/test_*.py`)

Set the env var directly in the test, scoped to the test's lifetime:

```python
def test_something():
    old = os.environ.get("HLM_MAX_LAYER")
    try:
        os.environ["HLM_MAX_LAYER"] = "4"
        be = _make_backend("test_something")
        # ... assertions ...
    finally:
        if old is not None:
            os.environ["HLM_MAX_LAYER"] = old
        else:
            os.environ.pop("HLM_MAX_LAYER", None)
```

`_make_backend()` already saves/restores `HLM_DB_PATH` and `HLM_ENRICH_LLM` to prevent config leaking between tests.

### In dispatch tests (`tests/test_dispatch.py`)

Same pattern — set env var before `_get_provider()` which calls `provider.initialize()`:

```python
def test_d21():
    old = os.environ.get("HLM_MAX_LAYER")
    try:
        os.environ["HLM_MAX_LAYER"] = "4"
        provider, db = _get_provider("d21")
        assert provider._max_layer == 4
    finally:
        # restore ...
```

### In E2E tests (`hlm-test chat -q "..."`, or any `hermes -p <profile>` CLI form)

Set the var in the **profile's `.env`** file (`~/.hermes/profiles/<profile>/.env`). The CLI path loads this into `os.environ` before the Python process starts.

**Do NOT set vars in `~/.hermes/.env`** unless they're shared across all profiles (e.g., `HLM_LOG`). Profile-specific vars belong in the profile's `.env`.

## Common mistakes

| Mistake | Why it fails | Fix |
|---------|-------------|-----|
| Reading `os.environ` in multiplex mode | Profile's `.env` is in secret scope, not `os.environ` | Use `secret_scope.get_secret()` or rely on per-profile gateway |
| Setting vars in `~/.hermes/.env` | Leaks across profiles, defeats isolation | Put in `<profile>/.env` |
| Forgetting to refresh cached values after `_apply_env_overrides()` | Cache was set at module import (default), env override applies later | Refresh after `LayeredBackend.__init__()` |
| Testing with `os.environ` but not restoring | Pollutes subsequent tests | Always save/restore in `try/finally` |
| Assuming `HERMES_HOME` is `~/.hermes/` | Hermes sets it to the profile dir | Use `get_hermes_home()` or `pwd.getpwuid()` for real home |

## Rule of thumb

- **CLI or per-profile gateway:** `os.environ` has profile vars → `os.environ.get()` works (the common case)
- **Multiplex gateway:** `os.environ` has root vars → use `secret_scope.get_secret()` for profile vars
- **Tests:** always set env var directly, always save/restore
- **Profile-specific config:** always in `<profile>/.env`, never in `~/.hermes/.env`

## max_layer is a default, not a ceiling

`HLM_MAX_LAYER` controls the **default** retrieval depth (used by prefetch and when the LLM omits `max_layer`). The LLM can always escalate by passing `max_layer=3` or `max_layer=4` explicitly, or `rerank=true` to auto-escalate to L3. There is no hard cap — `__init__.py` `_do_retrieve()` uses `args.get("max_layer", self._max_layer)` which gives the LLM's value precedence. The env var is the floor, not the ceiling.