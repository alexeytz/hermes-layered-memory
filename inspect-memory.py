#!/usr/bin/env python3
"""Inspect layered memory — Qdrant + SQLite side by side."""
import json
import os
import pwd
import sys
import argparse
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend import LayeredBackend, _from_qdrant_id

def inspect(db_path, qdrant_url, collection, limit=20, data_type=None):
    backend = LayeredBackend(
        db_path=db_path,
        qdrant_url=qdrant_url,
        qdrant_collection=collection or "memories",
    )

    # --- SQLite ---
    where = "status = 'active'"
    params = []
    if data_type:
        where += " AND data_type = ?"
        params.append(data_type)

    cursor = backend._get_conn().execute(
        f"SELECT uuid, content, summary, data_type, data_id, session_name, topic, scope, priority, trust_score, created_at FROM memories WHERE {where} ORDER BY created_at DESC LIMIT ?",
        params + [limit]
    )
    rows = cursor.fetchall()
    cols = [d[0] for d in cursor.description]

    print(f"\n{'='*80}")
    print(f"SQLite — {len(rows)} records (data_type={data_type or 'all'})")
    print(f"{'='*80}")
    for row in rows:
        rec = dict(zip(cols, row))
        print(f"\n  UUID:       {rec['uuid']}")
        print(f"  Content:    {(rec['content'] or '')[:100]}")
        # `or ''` rather than a .get() default — the key is always present,
        # it is the stored value that is NULL, and slicing None raises.
        print(f"  Summary:    {(rec.get('summary') or '')[:80]}")
        print(f"  data_type:  {rec['data_type']} | data_id: {rec.get('data_id','—')} | session_name: {rec.get('session_name','—')}")
        print(f"  topic:      {rec.get('topic','—')} | scope: {rec.get('scope','—')}")
        trust = rec.get('trust_score')
        print(f"  priority:   {rec['priority']} | trust: {trust:.2f}" if trust is not None
              else f"  priority:   {rec['priority']} | trust: —")
        print(f"  created:    {rec['created_at']}")

    # --- Qdrant ---
    print(f"\n{'='*80}")
    # Print the *physical* name, not the configured one. Collections are keyed
    # by embedding model (`memories_qwen3-embedding_8b_4096`), so the label read
    # `memories` while the query below correctly used the resolved name — an
    # operator comparing this output against `curl .../collections` saw a name
    # that is not there and concluded the collection was missing. The query was
    # never wrong; the label was. 2026-08-24 audit, minor 29 (partly refuted).
    print(f"Qdrant — {backend._default_collection} collection")
    print(f"{'='*80}")

    if backend._qdrant:
        try:
            info = backend._qdrant.get_collection(backend._default_collection)
            print(f"\n  Points:     {info.points_count}")
            print(f"  Vectors:    {info.vectors_config}")
            print(f"  Status:     {info.status}")

            # Get all points
            all_points = backend._qdrant.scroll(
                collection_name=backend._default_collection,
                limit=limit,
                with_payload=True,
            )
            points, _ = all_points

            if data_type:
                points = [p for p in points if p.payload.get("data_type") == data_type]

            print(f"\n  Points returned: {len(points)}")
            for pt in points:
                uid = _from_qdrant_id(pt.id)
                print(f"\n  ID:         {uid}")
                print(f"  Payload:    {json.dumps(pt.payload, indent=4)}")
        except Exception as e:
            print(f"\n  Qdrant error: {e}")
    else:
        print("\n  Qdrant client not available")

    # --- Sync check ---
    sync = backend.sync_check()
    print(f"\n{'='*80}")
    print(f"Sync: sqlite_active={sync['sqlite_active']} "
          f"sqlite_indexed={sync.get('sqlite_indexed', 'N/A')} "
          f"qdrant={sync.get('qdrant_total', 'N/A')} in_sync={sync['in_sync']}")
    print(f"{'='*80}\n")

    backend.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect layered memory")
    parser.add_argument("--db", default=None, help="SQLite DB path (default: ~/.hermes/hermes-layered-memory-dbs/<profile>.db)")
    parser.add_argument("--profile", default=None, help="Profile name (used for default DB path)")
    parser.add_argument("--qdrant", default="http://localhost:6333", help="Qdrant URL")
    parser.add_argument("--collection", default="memories", help="Qdrant collection")
    parser.add_argument("--limit", type=int, default=20, help="Max records")
    parser.add_argument("--data-type", default=None, help="Filter by data_type")
    args = parser.parse_args()

    if args.db:
        # Resolve ~ to real home (Hermes remaps $HOME)
        real_home = pwd.getpwuid(os.getuid()).pw_dir
        db = args.db
        if db.startswith("~"):
            db = db.replace("~", real_home, 1)
        if "$" in db:
            db = os.path.expandvars(db)
    elif args.profile:
        real_home = pwd.getpwuid(os.getuid()).pw_dir
        db = f"{real_home}/.hermes/hermes-layered-memory-dbs/{args.profile}.db"
    else:
        real_home = pwd.getpwuid(os.getuid()).pw_dir
        db = f"{real_home}/.hermes/hermes-layered-memory-dbs/default.db"
    inspect(db, args.qdrant, args.collection, args.limit, args.data_type)
