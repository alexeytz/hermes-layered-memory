#!/usr/bin/env python3
"""Repair records damaged by the pre-fix `_compact_group` column mis-read.

Background
----------
`_compact_group()` unpacked its SELECT one column short (see
05c5b4b^:archives/CODE_REVIEW_PASS2_2026-08-07.md, P2-16). For every merge it performed:

  * `created_at` held the record's **scope**, so the merged row was written with
    `first_observed_at = "personal"` (or whatever the scope was) instead of the
    oldest parent timestamp — in the row, in `metadata.first_observed_at`, and
    in the Qdrant payload.
  * `source` held a **timestamp**, which is never in `SELF_AUTHORED_SOURCES`, so
    `has_untrusted` was always true and every merged record was stamped
    `source = "consolidated-untrusted"`. That is not cosmetic: it means the
    system's own consolidated text is wrapped in `<untrusted_external_doc>`
    fences on every retrieval, telling the model to treat its own memory as
    hostile input.

Fixing the code does not fix rows already written. This does.

What it changes
---------------
  1. `source = 'consolidated-untrusted'` → `'hlm-consolidated'`, but **only**
     when every parent listed in `metadata.compacted_from` was itself
     self-authored. A merge that really did absorb external content keeps its
     untrusted marker — that fence is load-bearing.
  2. `first_observed_at` that is not a timestamp → the oldest `created_at`
     among the surviving parents, or the record's own `created_at` if none
     survive. Same for `metadata.first_observed_at`.

Dry-run by default. Nothing is written without `--apply`.

Usage
-----
    python3 scripts/repair-compaction-damage.py --db PATH        # report only
    python3 scripts/repair-compaction-damage.py --db PATH --apply
    python3 scripts/repair-compaction-damage.py --all-profiles   # scan every DB
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pwd
import sqlite3
import sys

# Sources the system authored itself — must mirror SELF_AUTHORED_SOURCES in
# backend/constants.py. Kept as a literal rather than imported so this script
# stays runnable against an old checkout or a copied database; T358 asserts the
# two agree, because a copy that drifts here is not cosmetic.
#
# It had drifted, in the direction that matters: it listed "tool-call",
# "import" and "memory-write" — all *untrusted* — alongside "extraction"
# (untrusted since 0.4.1) and None/"" (untrusted since 0.6.1). Below, a merged
# record whose parents are all "self-authored" has its source rewritten to
# "hlm-consolidated", which is trusted and never fenced. With those values in
# the set, a record merged from imported rows — an attacker-supplied export —
# came out of this repair permanently exempt from the prompt-injection fence.
#
# "compaction" left the real set in 0.7.53 — same laundering shape as
# "extraction" (LLM-mediated write from fenced input, stored self-authored),
# left behind when "extraction" was fixed. A merge that absorbed a
# compaction-extracted parent must now keep its untrusted marker too.
#
# "prefetch" left the real set in 0.8.45 — a dead entry, not a trust
# correction: nothing ever wrote a memory row with that source. Removing it
# here changes no runtime behavior (no compaction parent has ever carried it),
# it just keeps T358's mirror check honest.
SELF_AUTHORED = {"agent", "hlm-consolidated"}


def _real_home() -> str:
    return pwd.getpwuid(os.getuid()).pw_dir


def _default_db_dir() -> str:
    return os.path.join(_real_home(), ".hermes", "hermes-layered-memory-dbs")


def _looks_like_timestamp(value) -> bool:
    return isinstance(value, str) and value[:2] == "20" and "-" in value


def scan(db_path: str, apply: bool) -> dict:
    """Report (and optionally repair) damaged rows in one database."""
    out = {"db": db_path, "source_fixed": 0, "first_observed_fixed": 0,
           "source_kept_untrusted": 0, "examined": 0, "errors": []}
    if not os.path.exists(db_path):
        out["errors"].append("no such file")
        return out

    conn = sqlite3.connect(db_path)
    try:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(memories)").fetchall()]
        if "first_observed_at" not in cols:
            out["errors"].append("pre-v8 schema (no first_observed_at) — nothing to repair")
            return out

        rows = conn.execute(
            "SELECT uuid, source, first_observed_at, created_at, metadata "
            "FROM memories "
            "WHERE source = 'consolidated-untrusted' "
            "   OR (first_observed_at IS NOT NULL AND first_observed_at NOT LIKE '20%')"
        ).fetchall()
        out["examined"] = len(rows)

        for uuid, source, first_obs, created_at, metadata_json in rows:
            try:
                meta = json.loads(metadata_json) if metadata_json else {}
                if not isinstance(meta, dict):
                    meta = {}
            except (json.JSONDecodeError, TypeError):
                meta = {}
            parents = meta.get("compacted_from") or []
            updates, params = [], []

            # 1. Provenance. Only downgrade the fence when every parent was
            #    self-authored; a genuinely external parent must keep it.
            if source == "consolidated-untrusted":
                parent_sources = []
                if parents:
                    placeholders = ",".join("?" for _ in parents)
                    parent_sources = [
                        r[0] for r in conn.execute(
                            f"SELECT source FROM memories WHERE uuid IN ({placeholders})",
                            parents).fetchall()]
                # No surviving parents (purged) means no evidence either way.
                # Leave the fence in place — failing safe here costs a little
                # prompt noise; failing open removes an injection boundary.
                if parent_sources and all(s in SELF_AUTHORED for s in parent_sources):
                    updates.append("source = ?")
                    params.append("hlm-consolidated")
                    out["source_fixed"] += 1
                else:
                    out["source_kept_untrusted"] += 1

            # 2. first_observed_at that is not a timestamp.
            if first_obs is not None and not _looks_like_timestamp(first_obs):
                replacement = created_at
                if parents:
                    placeholders = ",".join("?" for _ in parents)
                    row = conn.execute(
                        f"SELECT MIN(created_at) FROM memories WHERE uuid IN ({placeholders})",
                        parents).fetchone()
                    if row and row[0]:
                        replacement = row[0]
                updates.append("first_observed_at = ?")
                params.append(replacement)
                if meta.get("first_observed_at") is not None and \
                        not _looks_like_timestamp(meta.get("first_observed_at")):
                    meta["first_observed_at"] = replacement
                    updates.append("metadata = ?")
                    params.append(json.dumps(meta))
                out["first_observed_fixed"] += 1

            if updates and apply:
                conn.execute(
                    f"UPDATE memories SET {', '.join(updates)} WHERE uuid = ?",
                    params + [uuid])

        if apply:
            conn.commit()
    finally:
        conn.close()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", metavar="PATH", help="database to repair")
    ap.add_argument("--all-profiles", action="store_true",
                    help=f"scan every *.db under {_default_db_dir()}")
    ap.add_argument("--apply", action="store_true",
                    help="write the repairs (default: report only)")
    args = ap.parse_args()

    if args.all_profiles:
        targets = [p for p in sorted(glob.glob(os.path.join(_default_db_dir(), "*.db")))
                   if not os.path.basename(p).startswith("digests")]
    elif args.db:
        targets = [args.db]
    else:
        ap.error("pass --db PATH or --all-profiles")
        return 2

    if not targets:
        print("no databases found")
        return 0

    mode = "APPLY" if args.apply else "DRY RUN (nothing written)"
    print(f"repair-compaction-damage — {mode}\n")
    total = 0
    for db in targets:
        r = scan(db, args.apply)
        total += r["source_fixed"] + r["first_observed_fixed"]
        name = os.path.basename(db)
        if r["errors"]:
            print(f"  {name}: {'; '.join(r['errors'])}")
            continue
        if not r["examined"]:
            print(f"  {name}: clean")
            continue
        print(f"  {name}: {r['examined']} damaged row(s) — "
              f"source→hlm-consolidated: {r['source_fixed']}, "
              f"first_observed_at→timestamp: {r['first_observed_fixed']}"
              + (f", fence kept (external parent or parents purged): "
                 f"{r['source_kept_untrusted']}" if r["source_kept_untrusted"] else ""))

    if not args.apply and total:
        print(f"\n{total} field(s) would be repaired. Re-run with --apply.")
        print("Back up first: layered_io(action=\"backup\") or scripts against a copy.")
    elif args.apply:
        print(f"\nRepaired {total} field(s).")
        print("Run layered_maintenance(action='rebuild') to refresh Qdrant payloads.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
