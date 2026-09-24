"""`memory_config` — runtime configuration and taxonomy.

Split out of mcp_server.py alongside io_tools; see that module's docstring for
the registration contract. Reads are open, writes need HLM_MCP_ADMIN: they
persist to the config file and change behaviour for the Hermes plugin sharing
the same profile.
"""

from __future__ import annotations

import asyncio
import json
import textwrap
from typing import Any

from backend.constants import coerce_config_value

TOOL_NAME = "memory_config"

ACTIONS = ("get", "set", "delete", "register_taxonomy", "get_taxonomy",
           "unregister_taxonomy")

POLICY = {
    "get": frozenset(),
    "get_taxonomy": frozenset(),
    "set": frozenset({"write", "admin"}),
    "delete": frozenset({"write", "admin"}),
    "register_taxonomy": frozenset({"write", "admin"}),
    "unregister_taxonomy": frozenset({"write", "admin"}),
}


def register(mcp, ctx):
    """Attach the tool to `mcp` and return it.

    The return value matters: FastMCP registration does not create a name
    in the importing module, and in-process callers — the tests, and
    anything else importing mcp_server — address tools by attribute.
    """

    @mcp.tool(
        name="memory_config",
        description=textwrap.dedent("""\
            Runtime configuration. Reads are open; writes need HLM_MCP_ADMIN=true
            because they persist to the config file and change behaviour for the
            Hermes plugin reading the same profile. Pass action=

            get                 view config (all, or one `key`)
            get_taxonomy        list registered data_type/data_id entries
            set                 change a key at runtime (key, value)
            delete              remove a custom key
            register_taxonomy   add a data_type or data_id (name, kind, collection)
            unregister_taxonomy remove one
        """),
    )
    async def memory_config(
        action: str,
        profile: str | None = None,
        key: str | None = None,
        value: str | dict | list | bool | int | float | None = None,
        name: str | None = None,
        # `None`, not "data_type". The backend reads kind=None as "every kind
        # of this name" and the plugin passes `args.get("kind")`, so omitting
        # it there removes both rows — while this door's default silently
        # narrowed the same call to the data_type row alone. That started
        # mattering in 0.7.96, which made `(name, kind)` the primary key so a
        # name can hold a data_type row *and* a data_id row at once: the two
        # doors now leave the store in different states for the same request.
        # register_taxonomy keeps a real default because creating something
        # needs a kind; removing does not. 2026-08-25 bundle04 review (F1).
        kind: str | None = None,
        collection: str | None = None,
        description: str | None = None,
        filter: str | None = None,
    ) -> str:
        """Config operations mirroring the plugin's layered_config."""
        async with ctx.semaphore:
            try:
                ctx.guard("memory_config", action, profile)
            except (ValueError, PermissionError) as e:
                return json.dumps({"error": str(e)}, default=str)
            try:
                be = await ctx.registry.get(profile)
            except (ValueError, PermissionError) as e:
                # A deliberate refusal is not an internal error. `_get_db_path`
                # raises ValueError when HLM_DB_PATH is set and the caller asks
                # for another profile (0.8.31), and the allowlist raises
                # PermissionError — both name the variable the caller must
                # change. `_safe_error` reduces an exception to its class,
                # which is right for an unexpected failure on an
                # unauthenticated server and useless here: the caller sees
                # "get backend failed: ValueError" and cannot act on it, while
                # the guard path one block up returns the full text. Same
                # disclosure either way (a variable name and a profile name,
                # no paths, no SQL) — so make the two paths agree.
                # 2026-08-27 external MCP re-validation, E-5.
                return json.dumps({"error": str(e)}, default=str)
            except Exception as e:
                return json.dumps({"error": ctx.safe_error("get backend", e)}, default=str)

            try:
                if action == "get":
                    return json.dumps(await asyncio.to_thread(be.get_config, key),
                                      default=str, indent=2)
                if action == "get_taxonomy":
                    return json.dumps(await asyncio.to_thread(be.get_taxonomy, filter),
                                      default=str, indent=2)
            except Exception as e:
                return json.dumps({"error": ctx.safe_error(action, e)}, default=str)

            async with await ctx.registry.lock_for(profile):
                try:
                    if action == "set":
                        if not key:
                            return json.dumps({"error": "key required"}, default=str)
                        if value is None:
                            return json.dumps({"error": "value required"}, default=str)
                        # The plugin's coercion itself, not a mirror of it.
                        # This was a hand-written copy that recognised only
                        # "true"/"false" and otherwise fell back to json.loads,
                        # so `auto_extract="yes"` failed validation here while
                        # succeeding through Hermes, and an unknown key set to
                        # "3" stored an int here and a string there (M-004).
                        # A JSON object/array may also arrive already parsed —
                        # see _as_text — and passes through untouched.
                        coerced: Any = coerce_config_value(key, value)
                        return json.dumps(await asyncio.to_thread(be.set_config, key, coerced),
                                          default=str, indent=2)
                    if action == "delete":
                        if not key:
                            return json.dumps({"error": "key required"}, default=str)
                        return json.dumps(await asyncio.to_thread(be.delete_config, key),
                                          default=str, indent=2)
                    if action == "register_taxonomy":
                        if not name:
                            return json.dumps({"error": "name required"}, default=str)
                        # `kind or "data_type"` turned a *present but empty*
                        # kind into a silent data_type registration, where the
                        # plugin passes `""` through to the backend and gets a
                        # clean `invalid kind` error. Absent means default;
                        # present-but-invalid means error — the same
                        # distinction T636 and T637 draw elsewhere, and the
                        # reason it matters here is that the caller typed
                        # something and was told it worked.
                        # 2026-09-15 round 2, bundle05 (F2). T653.
                        if kind is not None and not kind:
                            return json.dumps(
                                {"error": "invalid kind: %r — omit it for the "
                                          "default, or name one" % (kind,)},
                                default=str)
                        return json.dumps(await asyncio.to_thread(
                            be.register_taxonomy, name, kind=(kind or "data_type"), collection=collection,
                            description=description), default=str, indent=2)
                    if action == "unregister_taxonomy":
                        if not name:
                            return json.dumps({"error": "name required"}, default=str)
                        return json.dumps(await asyncio.to_thread(
                            be.unregister_taxonomy, name, kind=kind), default=str, indent=2)
                except Exception as e:
                    return json.dumps({"error": ctx.safe_error(action, e)}, default=str)
        return json.dumps({"error": f"unhandled action: {action}"}, default=str)

    return memory_config
