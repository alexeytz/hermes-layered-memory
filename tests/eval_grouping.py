#!/usr/bin/env python3
"""Dedup and compaction quality, measured against labelled near-duplicates.

`eval_retrieval.py` proves the pipeline *finds* things. This proves it can tell
when two records say the same thing — the half of the system that decides what
the store looks like after a year of writes, and the half nothing measures.

Today `compact` is tested for its dry-run contract and its merge mechanics, and
`find_duplicate_groups` has no precision or recall number attached to it
anywhere in the suite. Two external evaluations independently reported "the DB
accumulates near-duplicates", both diagnosed it wrongly (one blamed the 0.97
threshold, one blamed `force=true`), and the actual defect — dedup searching a
`data_type` partition its own records were never stored in — went unnoticed
through both. A labelled corpus is what turns that from an argument into a
measurement.

## The corpus, and what labels a duplicate

`~/Documents/REDIS/redis_docs_qa.jsonl`: 173,150 QA pairs over 34,462 chunks of
the Redis documentation, each carrying `source_chunk_key` =
`file:::heading:::first-50-chars`.

The obvious label is wrong, and the smoke run said so before any number was
published. Sharing a chunk does *not* make two answers duplicates: five
questions about one chunk produce five answers about different aspects of it —
"you adjust CPU and memory allocation per node" and "the persistentSpec.volumeSize
field controls storage size" share a chunk and state different facts. Scoring
against that label reports recall 0.011 and means nothing.

The real label is the **question**. 8,949 questions appear against more than one
chunk, and their answers are paraphrases of one fact written from different
source text — "Redis 8.6 is tested on Ubuntu 22.04 and 24.04, Rocky Linux…"
against "Redis 8.6 is tested on Ubuntu 22.04 (Jammy Jellyfish) and 24.04 (Noble
Numbat)…". That is exactly the population that accumulates in a real store.

So the corpus is built from two labelled classes:

  positives   answers to the *same question* — should be caught
  hard negs   answers to *different questions from the same chunk* — must not be

The hard negatives are the point. They share vocabulary, headings and source,
so they are what a threshold-only dedup gets wrong, and no synthetic corpus
produces them convincingly.

The chunk bodies live in the sibling `redis-docs-in/` checkout; the 50-char
prefix locates them, so `--mode retrieval` can use real prose rather than the
answers.

## What is measured

    --mode dedup      (default) ingest without force and score what add() said
    --mode compaction ingest with force, then score find_duplicate_groups()
    --mode retrieval  ingest chunk bodies, query with questions, recall@k

Dedup is scored per write against the `fact` label: flagging a record whose
match answers the same question is a true positive, flagging one that answers a
different question is a false positive, and storing silently when a same-fact
record already exists is a false negative. Compaction is scored per group and
per pair the same way.

Usage:
    python3 tests/eval_grouping.py --groups 200
    python3 tests/eval_grouping.py --groups 200 --mode compaction
    python3 tests/eval_grouping.py --groups 800 --mode dedup --json out.json

Requires Qdrant and an embedding endpoint, like the other harnesses. Runs in a
throwaway profile (`grpNNNN`) with its own Qdrant namespace — never point it at
a real profile, because it writes thousands of records and the collections are
shared across profiles by payload filter alone.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import statistics
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conftest import (  # noqa: E402
    _align_test_collection_dims,
    _check_qdrant_alive,
    _cleanup_db,
    _cleanup_qdrant_coll,
    _get_uuid,
    _make_backend,
)

DEFAULT_QA = os.path.expanduser("~/Documents/REDIS/redis_docs_qa.jsonl")
DEFAULT_DOCS = os.path.expanduser("~/Documents/REDIS")
EVAL_ID = "grp%d" % os.getpid()


# ── Corpus loading ──────────────────────────────────────────────────────────
def load_corpus(path: str, n_groups: int, seed: int) -> List[dict]:
    """Build a labelled record list: positives by question, hard negatives by chunk.

    Every record carries `fact` (the question it answers — the positive label)
    and `chunk` (its source chunk — the hard-negative label). Two records with
    the same `fact` state the same thing and should be caught; two with the same
    `chunk` but different `fact` do not and must not be.

    Sampling is seeded, because a benchmark whose corpus changes per run cannot
    show a regression.
    """
    by_q: Dict[str, List[dict]] = collections.defaultdict(list)
    by_chunk: Dict[str, List[dict]] = collections.defaultdict(list)
    with open(path) as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("answer") and d.get("question") and d.get("source_chunk_key"):
                by_q[d["question"]].append(d)
                by_chunk[d["source_chunk_key"]].append(d)

    rng = random.Random(seed)
    positives = sorted(q for q, v in by_q.items() if len(v) >= 2)
    rng.shuffle(positives)
    records: List[dict] = []
    used_chunks: set = set()
    for q in positives[:n_groups]:
        for d in by_q[q][:3]:
            records.append({"content": d["answer"], "fact": q,
                            "chunk": d["source_chunk_key"]})
            used_chunks.add(d["source_chunk_key"])

    # Hard negatives: same chunk, different question. Drawn from chunks the
    # positives did not touch, so a hard negative can never accidentally be a
    # positive's partner.
    negatives = sorted(k for k, v in by_chunk.items()
                       if len(v) >= 3 and k not in used_chunks)
    rng.shuffle(negatives)
    for k in negatives[:n_groups]:
        seen_q: set = set()
        for d in by_chunk[k]:
            if d["question"] in seen_q:
                continue
            seen_q.add(d["question"])
            records.append({"content": d["answer"], "fact": d["question"],
                            "chunk": k})
            if len(seen_q) >= 3:
                break
    return records


def recover_body(chunk_key: str, docs_root: str) -> Optional[str]:
    """Recover a chunk's real text from the docs checkout.

    The key truncates the body at 50 characters, so the file has to be read to
    get anything usable as a record. Returns None when the checkout is absent
    or the prefix no longer matches — the docs move, the QA file does not.
    """
    parts = chunk_key.split(":::")
    if len(parts) != 3:
        return None
    rel, _heading, prefix = parts
    path = os.path.join(docs_root, rel)
    if not os.path.isfile(path):
        return None
    try:
        body = open(path, encoding="utf-8", errors="replace").read()
    except OSError:
        return None
    i = body.find(prefix)
    if i < 0:
        return None
    return body[i:i + 1500].strip()


# ── Modes ───────────────────────────────────────────────────────────────────
def run_dedup(be, corpus: List[dict], verbose: bool) -> Dict[str, Any]:
    """Ingest without force; score every add() against the fact label.

    The interesting number is not "how many were caught" but *what* was caught.
    A flag pointing at a record that answers a different question is a false
    merge suggestion, and the hard negatives — same chunk, same vocabulary,
    different fact — are where that happens. Nothing synthetic produces them.
    """
    uuid_to_fact: Dict[str, str] = {}
    seen_facts: set = set()
    tp = fp = fn = tn = 0
    fp_examples, fn_examples = [], []
    statuses = collections.Counter()
    t0 = time.time()

    items = list(corpus)
    random.Random(0).shuffle(items)   # interleave; sequential is easy mode

    for rec in items:
        result = be.add(rec["content"], source="redis-docs-qa")
        uuid = _get_uuid(result)
        status = result.get("status", "stored") if isinstance(result, dict) else "stored"
        statuses[status] += 1
        flagged = isinstance(result, dict) and status in (
            "duplicate", "possible_duplicate", "contradiction")

        if flagged:
            matched_fact = uuid_to_fact.get(result.get("uuid"))
            if matched_fact == rec["fact"]:
                tp += 1
            else:
                fp += 1
                if len(fp_examples) < 5:
                    fp_examples.append({
                        "status": status,
                        "similarity": result.get("similarity"),
                        "wrote": rec["content"][:88],
                        "matched": (result.get("existing_content") or "")[:88],
                    })
        else:
            if rec["fact"] in seen_facts:
                fn += 1
                if len(fn_examples) < 5:
                    fn_examples.append({"fact": rec["fact"][:70],
                                        "wrote": rec["content"][:88]})
            else:
                tn += 1
            if uuid:
                uuid_to_fact[uuid] = rec["fact"]
        seen_facts.add(rec["fact"])

    prec = tp / (tp + fp) if (tp + fp) else None
    rec_ = tp / (tp + fn) if (tp + fn) else None
    return {
        "mode": "dedup", "writes": len(items), "seconds": round(time.time() - t0, 1),
        "true_positive": tp, "false_positive": fp,
        "false_negative": fn, "true_negative": tn,
        "precision": round(prec, 3) if prec is not None else None,
        "recall": round(rec_, 3) if rec_ is not None else None,
        "statuses": dict(statuses),
        "false_positive_examples": fp_examples,
        "false_negative_examples": fn_examples,
    }


def run_compaction(be, corpus: List[dict], threshold: float,
                   verbose: bool) -> Dict[str, Any]:
    """Force-ingest everything, then score find_duplicate_groups() by fact.

    force=True is deliberate: dedup would otherwise block the very records the
    grouping is meant to find, and the two mechanisms are measured separately
    on purpose. A group is pure when every member answers the same question.
    """
    uuid_to_fact: Dict[str, str] = {}
    t0 = time.time()
    for rec in corpus:
        uuid = _get_uuid(be.add(rec["content"], source="redis-docs-qa", force=True))
        if uuid:
            uuid_to_fact[uuid] = rec["fact"]
    ingest_s = time.time() - t0

    t1 = time.time()
    groups = be.find_duplicate_groups(threshold, max_groups=len(corpus))
    group_s = time.time() - t1

    pure = impure = 0
    grouped: set = set()
    pair_tp = pair_fp = 0
    impure_examples = []
    for g in groups:
        uuids = [u for u in g.get("uuids", []) if u in uuid_to_fact]
        grouped.update(uuids)
        labels = collections.Counter(uuid_to_fact[u] for u in uuids)
        if len(labels) <= 1:
            pure += 1
        else:
            impure += 1
            if len(impure_examples) < 5:
                impure_examples.append({
                    "size": len(uuids), "distinct_facts": len(labels),
                    "facts": [f[:52] for f in list(labels)[:2]],
                })
        for i in range(len(uuids)):
            for j in range(i + 1, len(uuids)):
                if uuid_to_fact[uuids[i]] == uuid_to_fact[uuids[j]]:
                    pair_tp += 1
                else:
                    pair_fp += 1

    by_fact = collections.Counter(uuid_to_fact.values())
    total_pairs = sum(n * (n - 1) // 2 for n in by_fact.values())
    pair_prec = pair_tp / (pair_tp + pair_fp) if (pair_tp + pair_fp) else None
    pair_rec = pair_tp / total_pairs if total_pairs else None
    return {
        "mode": "compaction", "threshold": threshold,
        "records": len(uuid_to_fact), "facts": len(by_fact),
        "ingest_seconds": round(ingest_s, 1), "group_seconds": round(group_s, 1),
        "groups_found": len(groups),
        "pure_groups": pure, "impure_groups": impure,
        "pair_precision": round(pair_prec, 3) if pair_prec is not None else None,
        "pair_recall": round(pair_rec, 3) if pair_rec is not None else None,
        "mergeable_pairs": total_pairs, "records_grouped": len(grouped),
        "impure_examples": impure_examples,
    }


def run_retrieval(be, corpus: List[dict], docs_root: str, k: int,
                  max_queries: int, verbose: bool) -> Dict[str, Any]:
    """Ingest real chunk bodies, query with the questions, score recall@k.

    Answers are not stored here. Storing an answer and querying it with the
    question that produced it shares too much vocabulary to mean anything — the
    number would flatter and would not compare with any other corpus.
    """
    wanted = collections.defaultdict(set)
    for rec in corpus:
        wanted[rec["chunk"]].add(rec["fact"])

    key_to_uuid: Dict[str, str] = {}
    missing = 0
    t0 = time.time()
    for key in wanted:
        body = recover_body(key, docs_root)
        if not body:
            missing += 1
            continue
        uuid = _get_uuid(be.add(body, source="redis-docs", force=True))
        if uuid:
            key_to_uuid[key] = uuid
    ingest_s = time.time() - t0

    queries = [(key, q) for key, qs in wanted.items() if key in key_to_uuid
               for q in qs]
    random.Random(1).shuffle(queries)
    queries = queries[:max_queries]

    hits = 0
    rr: List[float] = []
    lat: List[float] = []
    misses = []
    for key, question in queries:
        t = time.time()
        res = be.retrieve(question, max_layer=2, limit=k)
        lat.append(time.time() - t)
        want = key_to_uuid[key]
        got = [r.get("uuid") for r in res]
        if want in got:
            hits += 1
            rr.append(1.0 / (got.index(want) + 1))
        else:
            rr.append(0.0)
            if len(misses) < 5:
                misses.append({"q": question[:78],
                               "heading": key.split(":::")[1][:38]})
    return {
        "mode": "retrieval", "k": k,
        "chunks_ingested": len(key_to_uuid), "chunks_unrecoverable": missing,
        "queries": len(queries), "ingest_seconds": round(ingest_s, 1),
        f"recall@{k}": round(hits / len(queries), 3) if queries else None,
        "mrr": round(statistics.mean(rr), 3) if rr else None,
        "latency_p50": round(statistics.median(lat), 3) if lat else None,
        "misses": misses,
    }


# ── Main ────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qa", default=DEFAULT_QA, help="QA jsonl (default: %(default)s)")
    ap.add_argument("--docs", default=DEFAULT_DOCS,
                    help="root holding redis-docs-in/ (retrieval mode)")
    ap.add_argument("--mode", choices=("dedup", "compaction", "retrieval"),
                    default="dedup")
    ap.add_argument("--groups", type=int, default=200,
                    help="positive groups to sample; an equal number of "
                         "hard-negative chunks is added (~3 records each)")
    ap.add_argument("--threshold", type=float, default=0.90,
                    help="compaction similarity threshold (default: compact()'s own)")
    ap.add_argument("-k", type=int, default=5, help="cutoff for recall@k")
    ap.add_argument("--max-queries", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42, help="corpus sample seed")
    ap.add_argument("--config", action="append", metavar="KEY=VALUE", default=[],
                    help="override a backend config key, e.g. "
                         "--config dedup_threshold=0.93. Repeatable.")
    ap.add_argument("--sweep", action="store_true",
                    help="run --mode dedup across a threshold ladder on one "
                         "corpus and print the precision/recall trade")
    ap.add_argument("--json", metavar="PATH", help="write results as JSON")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if not os.path.isfile(args.qa):
        print(f"QA file not found: {args.qa}", file=sys.stderr)
        return 2
    if not _check_qdrant_alive():
        print("Qdrant is not reachable — this harness needs it", file=sys.stderr)
        return 2
    if not os.environ.get("HLM_EMBED_MODEL"):
        print("warning: HLM_EMBED_MODEL unset — running on the 384-dim local "
              "fallback, which measures a different system (see CLAUDE.md)",
              file=sys.stderr)

    print(f"loading {args.qa} …", flush=True)
    corpus = load_corpus(args.qa, args.groups, args.seed)
    n_records = len(corpus)
    n_facts = len({r["fact"] for r in corpus})
    dupes = n_records - n_facts
    print(f"corpus: {n_records} records over {n_facts} distinct facts "
          f"({dupes} duplicate writes to catch), seed {args.seed}\n")

    def parse_overrides(pairs: List[str]) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {}
        for pair in pairs:
            if "=" not in pair:
                raise SystemExit(f"--config expects KEY=VALUE, got {pair!r}")
            key, raw = pair.split("=", 1)
            try:
                cfg[key.strip()] = float(raw) if "." in raw else int(raw)
            except ValueError:
                cfg[key.strip()] = {"true": True, "false": False}.get(raw.lower(), raw)
        return cfg

    def run_point(overrides: Dict[str, Any]) -> Dict[str, Any]:
        """One measurement in its own backend.

        A fresh store per point is not optional: the corpus is ingested during
        the run, so reusing a backend would let the previous point's records
        answer this one's dedup checks.
        """
        _align_test_collection_dims()
        config = {"enrich_llm": False}
        config.update(overrides)
        point_id = f"{EVAL_ID}{abs(hash(frozenset(overrides.items()))) % 10000}"
        be = _make_backend(point_id, config=config)
        try:
            if args.mode == "dedup":
                res = run_dedup(be, corpus, args.verbose)
            elif args.mode == "compaction":
                res = run_compaction(be, corpus, args.threshold, args.verbose)
            else:
                res = run_retrieval(be, corpus, args.docs, args.k,
                                    args.max_queries, args.verbose)
        finally:
            _cleanup_qdrant_coll(be)
            be.close()
            _cleanup_db(point_id)
        res["config"] = dict(overrides)
        return res

    if args.sweep:
        # dedup_threshold blocks the write; dedup_warning_threshold only warns.
        # Both move together — warning above dedup would never fire.
        ladder = [(0.97, 0.95), (0.95, 0.93), (0.93, 0.90), (0.90, 0.87)]
        rows = []
        for dedup_t, warn_t in ladder:
            print(f"  running dedup_threshold={dedup_t} warning={warn_t} …", flush=True)
            r = run_point({"dedup_threshold": dedup_t,
                           "dedup_warning_threshold": warn_t})
            rows.append(r)
        print("\n" + "=" * 68)
        print(f"THRESHOLD SWEEP — {os.environ.get('HLM_EMBED_MODEL') or 'local-fallback'}")
        print("=" * 68)
        print(f"  {'dedup':>6} {'warn':>6} {'precision':>10} {'recall':>8} "
              f"{'TP':>4} {'FP':>4} {'FN':>4} {'blocked':>8}")
        for r in rows:
            c = r["config"]
            blocked = r["statuses"].get("duplicate", 0)
            print(f"  {c['dedup_threshold']:>6} {c['dedup_warning_threshold']:>6} "
                  f"{str(r['precision']):>10} {str(r['recall']):>8} "
                  f"{r['true_positive']:>4} {r['false_positive']:>4} "
                  f"{r['false_negative']:>4} {blocked:>8}")
        out = {"mode": "sweep", "points": rows}
    else:
        out = run_point(parse_overrides(args.config))

    out["corpus"] = {"records": n_records, "facts": n_facts,
                     "seed": args.seed, "qa_file": os.path.basename(args.qa)}
    out["embedder"] = os.environ.get("HLM_EMBED_MODEL") or "local-fallback"

    if out.get("mode") == "sweep":
        if args.json:
            with open(args.json, "w") as fh:
                json.dump(out, fh, indent=2, sort_keys=True)
                fh.write("\n")
            print(f"\nwrote {args.json}")
        return 0

    print("=" * 68)
    print(f"{args.mode.upper()} — {out['embedder']}")
    print("=" * 68)
    for key, value in out.items():
        if key in ("false_positive_examples", "false_negative_examples",
                   "impure_examples", "misses", "corpus", "points"):
            continue
        print(f"  {key:<22} {value}")
    for label in ("false_positive_examples", "false_negative_examples",
                  "impure_examples", "misses"):
        if out.get(label):
            print(f"\n  {label}:")
            for e in out[label]:
                print(f"    {e}")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=2, sort_keys=True)
            fh.write("\n")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
