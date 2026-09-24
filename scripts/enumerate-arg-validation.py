#!/usr/bin/env python3
"""Where is each action argument validated — at the chokepoint, or on one door?

`docs/review-backlog.md` has asked for this table twice. Its "class behind the
thirteen" section names the shape that produced most of two review rounds'
findings: **the two front ends validate arguments differently, and the backend
function they share catches some cases and not others.** Fixing those one door
at a time patches instances while the class survives; the question worth
answering mechanically is which arguments have a net under them and which
depend on whichever door the caller happened to use.

The output is one row per (action, argument) with three columns — plugin, MCP,
backend — and a verdict. The row that matters is **`one-door`**: guarded on
exactly one front end and not at the chokepoint. Every such row is a place
where the same call is accepted through one entry point and refused through the
other, which is precisely what `T400` cannot see (it proves reachability, not
agreement).

**What counts as a guard, and why it is a construct and not a name.** A
parameter is "guarded" in a function when its name appears in the *test* of an
`if` whose body returns or raises, in an `isinstance` call, or inside a
`try/except` around a coercion. Searching for the bare name instead would
count the sentence *describing* a missing guard as the guard — `T652`, `T657`
and `T654` were each written that way first and each found its own prose.
Comments and docstrings are invisible to `ast`, which is the point of using it.

**This is a map, not a verdict.** A `one-door` row is a question ("should this
be at the chokepoint?"), not a defect: `sleep`/`decay` are gated on MCP and
immediate on the plugin *deliberately* (`docs/mcp.md`), and a door may
legitimately refuse earlier than the net below it. Read the row, then read the
code. The value is that the list is short and complete rather than rediscovered
one finding at a time.

Known limits, stated because a reader will otherwise assume they are covered:
it does not follow a parameter renamed between door and backend, does not see
validation inside a helper the guard calls (`_coerce_int`, `_bounded_limit`,
`coerce_config_value`), and treats the MCP door's pydantic annotations as a
guard only where the annotation is not `Any`/untyped — that layer refused four
of round 1's findings and is easy to forget.

`internal` is not a gap: it is a parameter the plugin fills from its own state
rather than from the caller (`profile_name=self._profile_name`), so there is no
value to refuse. Reporting those as unguarded said the opposite of the truth.
"""
from __future__ import annotations

import argparse
import ast
import os
import sys
from collections import OrderedDict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── locating the pieces ──────────────────────────────────────────────────

def _parse(rel: str):
    path = os.path.join(ROOT, rel)
    return ast.parse(open(path, encoding="utf-8").read()), path


def dispatch_map(plugin_tree) -> dict:
    """action -> handler name, read from META_DISPATCH itself."""
    for node in ast.walk(plugin_tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "META_DISPATCH":
                    table = ast.literal_eval(node.value)
                    out = {}
                    for tool, actions in table.items():
                        for action, handler in actions.items():
                            out[(tool, action)] = handler
                    return out
    raise SystemExit("could not read META_DISPATCH from __init__.py")


def functions(tree) -> dict:
    return {n.name: n for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def mcp_branches(trees: list) -> dict:
    """action -> list of AST bodies for `if action == "<name>"` branches."""
    out = {}
    for tree in trees:
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            test = node.test
            if (isinstance(test, ast.Compare) and isinstance(test.left, ast.Name)
                    and test.left.id == "action"
                    and len(test.comparators) == 1
                    and isinstance(test.comparators[0], ast.Constant)
                    and isinstance(test.comparators[0].value, str)):
                out.setdefault(test.comparators[0].value, []).append(node)
    return out


def mcp_annotations(trees: list) -> dict:
    """argument -> set of annotations the MCP tool functions declare.

    The pydantic layer refuses a wrong type before any branch body runs, which
    is why `bundle05 F2` and `F6` were refuted. An untyped or `Any` parameter
    has no such net.
    """
    out = {}
    for tree in trees:
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not fn.name.startswith(("memory_", "mcp_")):
                continue
            for a in list(fn.args.args) + list(fn.args.kwonlyargs):
                if a.annotation is not None:
                    out.setdefault(a.arg, set()).add(
                        ast.unparse(a.annotation))
    return out


# ── what counts as a guard ───────────────────────────────────────────────

_TYPED = {"int", "float", "bool", "str"}


def _names_in(node) -> set:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)} | {
        n.value for n in ast.walk(node)
        if isinstance(n, ast.Constant) and isinstance(n.value, str)}


def guarded_args(body) -> set:
    """Arguments this code refuses on, by construct.

    Four constructs, each of which is a refusal and not a mention:
      * `if <test involving X>: return/raise ...`
      * `isinstance(X, ...)` anywhere
      * a `try:` whose body coerces X and whose handler returns or raises
      * `for _, v in ((..., X), ...): if bad(v): raise` — a loop over a tuple
        of (label, value) pairs, which is how this codebase prefers to guard a
        class of arguments at once (`PREVIEW_FENCED_FIELDS`,
        `UPDATE_ALLOWED_FIELDS`, `delete_many`'s filter types). Missing it
        reported four freshly-guarded `delete_many` filters as gaps — the
        analyser blind to exactly the shape the house style asks for.
    """
    out = set()
    nodes = body if isinstance(body, list) else [body]

    for root in nodes:
        for node in ast.walk(root):
            if isinstance(node, ast.If):
                exits = any(isinstance(s, (ast.Return, ast.Raise))
                            for s in ast.walk(node))
                if exits:
                    out |= _names_in(node.test)
            elif isinstance(node, ast.Call):
                fname = (getattr(node.func, "id", None)
                         or getattr(node.func, "attr", None))
                if fname == "isinstance" and node.args:
                    out |= _names_in(node.args[0])
                elif fname in _REFUSERS:
                    # Both shapes: require_str_filters(tag=tag) names the
                    # argument in the keyword, str_filter_error("tag", tag)
                    # in the positionals.
                    for kw in node.keywords:
                        if kw.arg:
                            out.add(kw.arg)
                    for a in node.args:
                        out |= _names_in(a)
            elif isinstance(node, ast.For):
                # The parameter is named in what the loop iterates over, and
                # the body refuses. The loop variable carries the value, so
                # the name only appears in the iterable.
                body_exits = any(isinstance(s, (ast.Return, ast.Raise))
                                 for s in ast.walk(node))
                if body_exits:
                    out |= _names_in(node.iter)
            elif isinstance(node, ast.Try):
                handler_exits = any(
                    isinstance(s, (ast.Return, ast.Raise))
                    for h in node.handlers for s in ast.walk(h))
                if handler_exits:
                    for s in node.body:
                        for c in ast.walk(s):
                            if isinstance(c, ast.Call) and getattr(
                                    c.func, "id", None) in _TYPED:
                                out |= _names_in(c)
    return out


#: Helpers whose whole job is to refuse. A call naming an argument counts as a
#: guard on it, because that is what the house style asks a guard to look like:
#: one shared definition, called from every chokepoint, rather than a copy of
#: the same `if` in each.
#:
#: **This set is the analyser's blind spot, and it has already bitten three
#: times.** The detector understands four constructs (`if`, `isinstance`,
#: `try`, `for`), and each time a fix was written in an idiom it did not know,
#: the count went *down* and read as new gaps: the `for`-over-tuple guard in
#: `delete_many`, then this — `require_str_filters(source_type=...)` — which
#: took the freshly-guarded chokepoint back to `ONE-DOOR`. An analyser that
#: reads code for guards needs the project's vocabulary of guards, and that
#: vocabulary is a maintenance surface. `T669` pins that every name here still
#: exists, so a rename shows up as a failure rather than as a false gap.
_REFUSERS = {"require_str_filters", "str_filter_error", "_require_str_arg",
             "_validate_config_value", "_validate_data_type",
             "_valid_profile_scope", "_valid_timestamp"}

#: Helpers that clamp-or-default rather than refuse. A parameter routed
#: through one of these is neither guarded nor naked: a bad value becomes the
#: documented default, which `_bounded_limit`'s own docstring argues is the
#: right contract for a pagination bound. Counted as `coerced` so the
#: `ONE-DOOR` list means "nothing between the caller and the SQL".
#: `_bounded_offset` was in this set on the first draft and does not exist —
#: it is the helper `bundle05 F2` *suggested*, never written. `T669` caught it
#: on its first run, which is the argument for that test: a vocabulary entry
#: naming nothing is invisible until an argument it was meant to cover reads
#: as a gap.
_COERCERS = {"_coerce_int", "_coerce_bool", "_bounded_limit",
             "coerce_config_value", "coerce_tool_bool", "coerce_tool_json"}


def coerced_args(handler) -> set:
    """Backend keyword arguments whose value passes through a coercion helper.

    Read off the keyword binding itself rather than by scanning the function
    for helper calls, so `limit=_coerce_int(args.get("limit"), 200)` counts for
    `limit` and not for every other parameter in the same call.
    """
    out = set()
    for node in ast.walk(handler):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if kw.arg is None:
                continue
            for c in ast.walk(kw.value):
                if isinstance(c, ast.Call):
                    fname = (getattr(c.func, "id", None)
                             or getattr(c.func, "attr", None))
                    if fname in _COERCERS:
                        out.add(kw.arg)
    return out


def caller_sourced(handler) -> set:
    """Backend keyword arguments the plugin fills from the *caller's* args.

    A parameter the door supplies itself — `profile_name=self._profile_name`,
    `profile="own"` — is not a caller argument, and reporting it as unguarded
    says the opposite of the truth: it cannot be attacked because it cannot be
    set. Detected by whether the keyword's value reads from `args` at all.
    """
    out = set()
    for node in ast.walk(handler):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if kw.arg is None:
                continue
            reads_args = any(
                isinstance(n, ast.Name) and n.id == "args"
                for n in ast.walk(kw.value))
            if reads_args:
                out.add(kw.arg)
    return out


def _helpers_called(fn, backend_fns: dict) -> list:
    """Backend helpers `fn` calls, one hop, excluding itself."""
    out = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            name = (getattr(node.func, "attr", None)
                    or getattr(node.func, "id", None))
            if name and name in backend_fns and name != fn.name and name not in out:
                out.append(name)
    return out


def backend_call_targets(handler, backend_fns: dict) -> list:
    """Backend functions a plugin handler calls, by attribute name."""
    hits = []
    for node in ast.walk(handler):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            name = node.func.attr
            if name in backend_fns and name not in hits:
                hits.append(name)
    return hits


# ── the table ────────────────────────────────────────────────────────────

def build() -> list:
    plugin_tree, _ = _parse("__init__.py")
    mcp_trees = [_parse("mcp_server.py")[0]]
    for rel in ("mcp_tools/io_tools.py", "mcp_tools/config_tools.py"):
        if os.path.exists(os.path.join(ROOT, rel)):
            mcp_trees.append(_parse(rel)[0])

    backend_fns = {}
    for rel in ("backend/store.py", "backend/maintenance.py", "backend/pipeline.py",
                "backend/index.py", "backend/llm.py", "summaries.py"):
        if os.path.exists(os.path.join(ROOT, rel)):
            backend_fns.update(functions(_parse(rel)[0]))

    plugin_fns = functions(plugin_tree)
    dispatch = dispatch_map(plugin_tree)
    branches = mcp_branches(mcp_trees)
    annotations = mcp_annotations(mcp_trees)

    rows = []
    for (tool, action), handler_name in sorted(dispatch.items()):
        handler = plugin_fns.get(handler_name)
        if handler is None:
            continue
        targets = backend_call_targets(handler, backend_fns)
        if not targets:
            continue
        be_fn = backend_fns[targets[0]]
        params = [a.arg for a in be_fn.args.args if a.arg != "self"]

        plugin_guards = guarded_args(handler.body)
        from_caller = caller_sourced(handler)
        plugin_coerced = coerced_args(handler)
        mcp_guards = set()
        for node in branches.get(action, []):
            mcp_guards |= guarded_args(node.body)
        # One hop into the helpers the backend function calls. `list_summaries`
        # does not guard `profile` itself — `_profile_sql` does, and every
        # summaries read crosses it (0.8.74, T663). Reading only the named
        # function reported three such arguments as one-door gaps, which is the
        # inverse of the truth: the net is one level down, where the house
        # style says to put it.
        be_guards = guarded_args(be_fn.body)
        for helper in _helpers_called(be_fn, backend_fns):
            be_guards |= guarded_args(backend_fns[helper].body)

        for p in params:
            ann = annotations.get(p, set())
            typed = any(a.split("|")[0].strip() in _TYPED for a in ann)
            in_plugin = p in plugin_guards
            in_mcp = (p in mcp_guards) or typed
            in_be = p in be_guards
            if p not in from_caller and not in_plugin:
                # The plugin fills this itself; there is no caller value to
                # refuse. Counted separately rather than as a gap.
                verdict = "internal"
            elif in_be:
                verdict = "chokepoint"
            elif in_plugin and in_mcp:
                verdict = "both-doors"
            elif p in plugin_coerced and in_mcp:
                verdict = "coerced"
            elif in_plugin or in_mcp:
                verdict = "ONE-DOOR"
            else:
                verdict = "unguarded"
            rows.append(OrderedDict(
                tool=tool, action=action, backend=targets[0], arg=p,
                plugin=("y" if in_plugin else
                        "c" if p in plugin_coerced else "-"),
                mcp=("T" if typed and p not in mcp_guards
                     else "y" if in_mcp else "-"),
                chokepoint="y" if in_be else "-",
                verdict=verdict))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", help="show one verdict class (e.g. ONE-DOOR)")
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()

    rows = build()
    # The same floor every scanner in this repo needs: a run that collected
    # nothing must not read as a clean result (T353, T652).
    if len(rows) < 50:
        print(f"ERROR: enumerated only {len(rows)} (action, arg) pairs — the "
              f"analyser is broken, not the code", file=sys.stderr)
        return 2

    from collections import Counter
    counts = Counter(r["verdict"] for r in rows)
    if args.summary or not args.only:
        print("(action, argument) pairs: %d over %d actions\n"
              % (len(rows), len({(r["tool"], r["action"]) for r in rows})))
        for v in ("chokepoint", "both-doors", "coerced", "ONE-DOOR",
                  "unguarded", "internal"):
            print("  %-12s %4d" % (v, counts.get(v, 0)))
        print("\n  plugin column: y = refuses, c = clamps/defaults via a helper")
        print("  mcp column:    y = an explicit guard in the branch, "
              "T = typed by the pydantic annotation only\n")
    if args.summary:
        return 0

    show = [r for r in rows if not args.only or r["verdict"] == args.only]
    print("%-20s %-18s %-22s %-24s %s %s %s  %s" % (
        "TOOL", "ACTION", "BACKEND", "ARG", "P", "M", "C", "VERDICT"))
    for r in show:
        print("%-20s %-18s %-22s %-24s %s %s %s  %s" % (
            r["tool"], r["action"], r["backend"], r["arg"],
            r["plugin"], r["mcp"], r["chokepoint"], r["verdict"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
