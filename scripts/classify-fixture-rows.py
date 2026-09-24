#!/usr/bin/env python3
"""Classify a profile's memory rows as test-suite fixtures or genuine records.

Written for the residue the E2E suite left in `hlm-hermes`: 335 active rows of
which the 0.8.73 investigation found ~19 genuine. That investigation matched
deleted rows against "a literal test-corpus string" by hand, in a session whose
transcript is the only record of how. A bulk delete wants better evidence than
that, and the next one will want it again — the suite wrote into a real store
four times before anyone noticed.

**Read-only.** It classifies and prints; it deletes nothing. Feeding its output
to `layered_memory(action="delete_many")` is a separate, deliberate step.

The method, and its limits:

- **Harvest** every string literal the test suite could write as `content` —
  `add(...)`/`store(...)` call arguments and `content=` keywords across
  `tests/*.py`, plus the seeded corpora under `tests/` and the fixture strings
  in `docs/e2e-prompt.md`. Harvesting from the *source* rather than from a
  hand-written list is the point: a list drifts, the suite does not.
- **Normalise** for loop counters. The suite writes `f"item {i}"` and
  `f"t448 clamp probe {n}"`, so a literal with a `{...}` placeholder becomes a
  regex with `.+` in its place. Without this the largest fixture batches read
  as genuine, which is the failure direction that matters.
- **Classify** each row: `fixture` if its content matches a harvested literal
  exactly, case-insensitively, or through a counter pattern; `unmatched`
  otherwise.

`unmatched` means *this script could not prove it is a fixture* — not that it
is genuine. Every unmatched row is printed in full, because the only safe use
of this output is a human reading that list before anything is deleted. A
fixture the harvester misses lands in `unmatched` and survives, which is the
right way for it to fail.

**Content alone is not enough, and `hlm-hermes` proves it.** Four rows reading
`My GPU is an RTX 3090 with 24GB VRAM` are genuine — they are in the set the
2026-09-14 incident agent deliberately kept — and that exact string is also a
test-suite literal, so content matching classifies them as fixtures. `--protect-from`
takes a database whose uuids must never be classified as fixtures: an earlier
snapshot, a known-good export, the survivors of a previous purge. Without it
this script would have deleted a record with `reference_count=10`.

**`reference_count` was the obvious second signal and it does not work.**
Measured on `hlm-hermes`: 202 rows that are unambiguously fixtures have been
retrieved at least once, `hw spec` 48 times, because the suite retrieves what
it writes. The distribution of fixture and genuine rows over that column
overlaps almost completely. Recorded so the next person does not re-derive it —
the same reason `docs/handover.md` keeps a table of measured-and-rejected ideas.
"""
from __future__ import annotations

import argparse
import ast
import os
import re
import sqlite3
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Call names whose arguments are candidate memory content.
_WRITERS = {"add", "store", "add_memory", "_add", "write"}


def _literals_from_python(path: str) -> set:
    """Every string literal that could reach a record's `content`.

    Deliberately over-collects: any string argument to a writer call, any
    `content=` keyword anywhere, and any f-string template among them. A
    false positive here can only classify a *genuine* row as a fixture, so
    the printed `unmatched` list is the control — see the module docstring.
    """
    out = set()
    try:
        tree = ast.parse(open(path, encoding="utf-8").read())
    except (SyntaxError, OSError):
        return out

    def _text(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.JoinedStr):
            # f"item {i}" -> "item {}" so the counter normaliser can see it.
            parts = []
            for v in node.values:
                if isinstance(v, ast.Constant) and isinstance(v.value, str):
                    parts.append(v.value)
                else:
                    parts.append("{}")
            return "".join(parts)
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
            if name in _WRITERS:
                for a in node.args:
                    t = _text(a)
                    if t:
                        out.add(t)
            for kw in node.keywords:
                if kw.arg in ("content", "text", "summary"):
                    t = _text(kw.value)
                    if t:
                        out.add(t)
    return out


def _literals_from_markdown(path: str) -> set:
    """Fixture strings the e2e prompt tells a driver to write verbatim."""
    out = set()
    try:
        text = open(path, encoding="utf-8").read()
    except OSError:
        return out
    # Backticked strings long enough to be content rather than a field name.
    for m in re.finditer(r"`([^`\n]{12,200})`", text):
        out.add(m.group(1))
    return out


def harvest() -> set:
    lits = set()
    tests_dir = os.path.join(ROOT, "tests")
    for fname in sorted(os.listdir(tests_dir)):
        if fname.endswith(".py"):
            lits |= _literals_from_python(os.path.join(tests_dir, fname))
    for rel in ("docs/e2e-prompt.md",):
        lits |= _literals_from_markdown(os.path.join(ROOT, rel))
    # Strings too short or too generic to identify anything.
    return {s.strip() for s in lits if len(s.strip()) >= 3}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def build_matchers(literals: set):
    """(exact set, [compiled counter patterns]) from harvested literals."""
    exact, patterns = set(), []
    for lit in literals:
        n = _norm(lit)
        if "{}" in n:
            # A template: "item {}" -> ^item .+$
            rx = "^" + ".+".join(re.escape(p) for p in n.split("{}")) + "$"
            try:
                patterns.append(re.compile(rx))
            except re.error:
                pass
        else:
            exact.add(n)
    return exact, patterns


def protected_uuids(paths: list) -> set:
    """Every uuid in the given databases — rows that must never be deleted.

    See the module docstring: a genuine record whose content happens to be a
    test literal is invisible to content matching, and the only evidence that
    it is genuine may be that some earlier, trusted pass kept it.
    """
    out = set()
    for path in paths or []:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            out |= {r[0] for r in conn.execute("SELECT uuid FROM memories")}
        finally:
            conn.close()
    return out


def classify(db_path: str, exact: set, patterns: list, status: str = "active",
             protect: set = frozenset()):
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT uuid, content, data_type, data_id, source, created_at, status "
        "FROM memories WHERE status = ? ORDER BY created_at", (status,)
    ).fetchall()
    conn.close()

    fixture, unmatched, prot = [], [], []
    for r in rows:
        if r["uuid"] in protect:
            prot.append(r)
            continue
        n = _norm(r["content"])
        hit = n in exact or any(p.match(n) for p in patterns)
        (fixture if hit else unmatched).append(r)
    return fixture, unmatched, prot


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", required=True, help="path to the profile's .db")
    ap.add_argument("--status", default="active")
    ap.add_argument("--show-fixtures", action="store_true",
                    help="also print the matched rows, grouped by content")
    ap.add_argument("--protect-from", action="append", metavar="DB", default=[],
                    help="a database whose uuids may never be called fixtures; "
                         "repeatable. Use a snapshot you trust — see the module "
                         "docstring for the four rows this saved.")
    ap.add_argument("--print-delete-contents", action="store_true",
                    help="print the distinct contents of the fixture set, one "
                         "per line, for feeding to delete_many(content_like=)")
    args = ap.parse_args()

    if not os.path.exists(args.db):
        print(f"ERROR: no such database: {args.db}", file=sys.stderr)
        return 2

    literals = harvest()
    exact, patterns = build_matchers(literals)
    # A harvester that collected nothing would classify every row as genuine
    # and read as a clean bill of health. Same failure as T353's: a green
    # result from a check that did not run.
    if len(exact) + len(patterns) < 50:
        print(f"ERROR: harvested only {len(exact)} literals and {len(patterns)} "
              f"patterns from tests/ — the harvester is broken, not the store",
              file=sys.stderr)
        return 2

    protect = protected_uuids(args.protect_from)
    fixture, unmatched, prot = classify(args.db, exact, patterns, args.status, protect)
    total = len(fixture) + len(unmatched) + len(prot)

    # A content-keyed delete is only safe if no content string appears in both
    # the delete set and the keep set — otherwise one `content_like` call takes
    # a kept row with it. Asserted, not assumed.
    keep = unmatched + prot
    del_contents = {_norm(r["content"]) for r in fixture}
    keep_contents = {_norm(r["content"]) for r in keep}
    overlap = del_contents & keep_contents

    if args.print_delete_contents:
        if overlap:
            print("REFUSED: %d content string(s) appear in both the fixture and "
                  "keep sets, so a content-keyed delete cannot separate them:"
                  % len(overlap), file=sys.stderr)
            for c in sorted(overlap):
                print("   " + c[:120], file=sys.stderr)
            return 2
        for c in sorted(del_contents):
            print(c)
        return 0

    print(f"corpus:    {len(exact)} exact literals, {len(patterns)} counter patterns")
    print(f"database:  {args.db}")
    print(f"rows:      {total} {args.status}")
    print(f"  fixture:   {len(fixture)}  ({len(del_contents)} distinct contents)")
    print(f"  unmatched: {len(unmatched)}")
    print(f"  protected: {len(prot)}  (from {len(protect)} uuids in "
          f"{len(args.protect_from)} snapshot(s))")
    print(f"  content overlap between delete and keep sets: {len(overlap)}"
          + ("  <-- a content-keyed delete is NOT safe" if overlap else "  (safe)"))
    print()

    if args.show_fixtures:
        from collections import Counter
        print("── matched as fixtures, by content ─────────────────────────")
        for content, n in Counter(_norm(r["content"]) for r in fixture).most_common():
            print(f"  {n:4d}  {content[:100]}")
        print()

    if prot:
        print("── protected by snapshot: never classified as fixtures ─────")
        for r in prot:
            print(f"  {r['uuid'][:8]}  {r['created_at'][:10]}  {r['data_type']:<13}"
                  f"  {_norm(r['content'])[:100]}")
        print()

    print("── unmatched: read every one before deleting anything ──────")
    for r in unmatched:
        print(f"  {r['uuid'][:8]}  {r['created_at'][:10]}  {r['data_type']:<13}"
              f"  {r['source']:<10}  {_norm(r['content'])[:110]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
