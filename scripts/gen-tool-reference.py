#!/usr/bin/env python3
"""Generate docs/tools.md from the live tool schemas.

The tool table was hand-maintained in four places and disagreed with the code
in all of them: README and AGENTS.md both claimed "5 meta-tools, 29 actions"
against an actual 6 and 40, and layered_config was absent from the README
entirely. Numbers that are copied by hand drift; numbers that are generated
cannot.

Usage:
    python3 scripts/gen-tool-reference.py           # write docs/tools.md
    python3 scripts/gen-tool-reference.py --check   # exit 1 if out of date
"""

from __future__ import annotations

import argparse
import ast
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INIT_PATH = os.path.join(ROOT, "__init__.py")
OUT_PATH = os.path.join(ROOT, "docs", "tools.md")

HEADER = """# Tool Reference

<!-- GENERATED FILE — do not edit by hand.
     Regenerate with: python3 scripts/gen-tool-reference.py
     Source of truth: META_DISPATCH and META_*_SCHEMA in __init__.py -->

Every HLM operation is dispatched through a meta-tool plus an `action`
parameter. This file is generated from the schemas the agent actually sees.

"""


def _literal(node):
    """Best-effort literal eval that tolerates non-literal nodes."""
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return None


def extract() -> tuple[dict, dict]:
    """Return (dispatch_map, {tool: {param: description}}) parsed from source."""
    tree = ast.parse(open(INIT_PATH, encoding="utf-8").read())
    dispatch, schemas = {}, {}

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            name = getattr(target, "id", None)
            if name == "META_DISPATCH":
                value = _literal(node.value)
                if value:
                    dispatch = value
            elif name and name.startswith("META_") and name.endswith("_SCHEMA"):
                value = _literal(node.value)
                if isinstance(value, dict):
                    schemas[value.get("name", name)] = value

    return dispatch, schemas


def render(dispatch: dict, schemas: dict) -> str:
    total_actions = sum(len(v) for v in dispatch.values())
    out = [HEADER]
    out.append(f"**{len(dispatch)} meta-tools, {total_actions} actions.**\n")

    out.append("| Meta-tool | Actions |")
    out.append("|---|---|")
    for tool in sorted(dispatch):
        actions = ", ".join(f"`{a}`" for a in dispatch[tool])
        out.append(f"| `{tool}` | {actions} |")
    out.append("")

    for tool in sorted(dispatch):
        schema = schemas.get(tool, {})
        out.append(f"## `{tool}`")
        out.append("")
        if schema.get("description"):
            out.append(schema["description"].strip())
            out.append("")
        out.append(f"**Actions:** {', '.join('`' + a + '`' for a in dispatch[tool])}")
        out.append("")

        props = (schema.get("parameters") or {}).get("properties") or {}
        if props:
            out.append("| Parameter | Type | Description |")
            out.append("|---|---|---|")
            for param, spec in props.items():
                if param == "action":
                    continue
                desc = (spec.get("description") or "").replace("|", "\\|").replace("\n", " ")
                out.append(f"| `{param}` | {spec.get('type', 'any')} | {desc} |")
            out.append("")

    return "\n".join(out).rstrip() + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero if docs/tools.md is stale")
    args = ap.parse_args()

    dispatch, schemas = extract()
    if not dispatch:
        print("ERROR: could not parse META_DISPATCH from __init__.py", file=sys.stderr)
        return 2

    content = render(dispatch, schemas)

    if args.check:
        existing = open(OUT_PATH, encoding="utf-8").read() if os.path.exists(OUT_PATH) else ""
        if existing != content:
            print("docs/tools.md is out of date — run scripts/gen-tool-reference.py",
                  file=sys.stderr)
            return 1
        print("docs/tools.md is up to date")
        return 0

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        fh.write(content)
    total = sum(len(v) for v in dispatch.values())
    print(f"Wrote {OUT_PATH} ({len(dispatch)} meta-tools, {total} actions)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
