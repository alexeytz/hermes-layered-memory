# Tool Reference

<!-- GENERATED FILE — do not edit by hand.
     Regenerate with: python3 scripts/gen-tool-reference.py
     Source of truth: META_DISPATCH and META_*_SCHEMA in __init__.py -->

Every HLM operation is dispatched through a meta-tool plus an `action`
parameter. This file is generated from the schemas the agent actually sees.


**6 meta-tools, 43 actions.**

| Meta-tool | Actions |
|---|---|
| `layered_advanced` | `enrich`, `feedback`, `compact`, `traces`, `reenrich`, `stats`, `graph_health` |
| `layered_config` | `get`, `set`, `delete`, `register_taxonomy`, `get_taxonomy`, `unregister_taxonomy` |
| `layered_io` | `backup`, `export`, `import`, `obsidian_ingest` |
| `layered_maintenance` | `rebuild`, `purge`, `sync_check`, `sleep`, `review`, `decay`, `test_cleanup`, `resolve_conflicts` |
| `layered_memory` | `retrieve`, `peek`, `add`, `update`, `delete`, `delete_many`, `list`, `list_profiles`, `discover` |
| `layered_summaries` | `summarize`, `list`, `get`, `search`, `delete`, `update`, `batch_delete`, `list_expiring`, `sync` |

## `layered_advanced`

Advanced ops. Use: enrich (regenerates both topic AND keywords on older records — NOTE: automatic LLM enrichment on add is OFF by default since v0.3.0 (enrich_on_add="heuristics_only"), so this action is now the normal way to get LLM-quality metadata; set enrich_on_add="low_confidence" to restore the old on-write behaviour), feedback (adjust trust_score ±0.1 when user confirms memory was correct/wrong), compact (merge near-duplicate records from repeated adds — use when too many similar records exist; DRY-RUN unless execute=true), traces (debug: show retrieval pipeline trace for recent queries — use after retrieval to understand why results were returned), reenrich (targeted backfill: fixes only the MISSING field — use when records have topic but no keywords or vice versa, cheaper than enrich), stats (check prefetch vs explicit retrieval counts — use when investigating tool dispatch behavior), graph_health (diagnose: list records that cluster with nothing — either a fact captured once and never reinforced, or one worded so unlike any question that retrieval will never surface it; read-only).

**Actions:** `enrich`, `feedback`, `compact`, `traces`, `reenrich`, `stats`, `graph_health`

## `layered_config`

Runtime configuration management. Use: get (view current config values — all or a single key), set (change a config key at runtime — persists to DB and syncs to JSON file), delete (remove a custom config key), register_taxonomy (add a new data_type or data_id to the classification system — collection auto-mapped), get_taxonomy (list registered taxonomy entries), unregister_taxonomy (remove a taxonomy entry).

**Actions:** `get`, `set`, `delete`, `register_taxonomy`, `get_taxonomy`, `unregister_taxonomy`

## `layered_io`

Import/Export/Backup. Use: backup (save memories to file for safekeeping), export (extract memories as JSON or Markdown for external use), import (restore memories from a JSON export), obsidian_ingest (import Obsidian vault notes into layered memory — use when user wants to bulk-import notes with frontmatter/tags).

**Actions:** `backup`, `export`, `import`, `obsidian_ingest`

## `layered_maintenance`

Database maintenance. Use: rebuild (fix: regenerate Qdrant embeddings after sync_check reports stale data, or after bulk import/mass edits), purge (permanently remove soft-deleted/archived records — use min_age_hours=0 for immediate), sync_check (diagnose: compare SQLite vs Qdrant counts — returns in_sync boolean), sleep (archive TTL-expired, low-trust-duplicate, and very-old low-trust records — run periodically on large DBs), review (spawn LLM to classify records as keep/delete — run when DB is bloated), decay (age-based trust_score reduction — run monthly), test_cleanup (remove [HLM-TEST] records after E2E tests), resolve_conflicts (after import from another profile: resolve records flagged as divergent — keeps highest trust_score, newest on a tie, soft-deletes the rest; DRY-RUN unless execute=true).

**Actions:** `rebuild`, `purge`, `sync_check`, `sleep`, `review`, `decay`, `test_cleanup`, `resolve_conflicts`

## `layered_memory`

Core memory operations. Use: add (store a fact/rule/preference), retrieve (search for memories matching a question), update (correct/extend a stored memory — needs tag from retrieve), delete (forget something), list (browse recent memories), peek (debug: inspect a single layer's raw output when retrieval seems wrong — raw means raw: unlike retrieve it does not attach a conflict_alert, so divergent records show up as ordinary results), list_profiles (discover which Hermes profiles have layered memory data and their record counts), discover (ask whether *another* profile knows something — returns metadata only: which profile, topic, data_type, score. No content crosses the profile boundary; follow up with retrieve(cross_profile=true) if you need it).

**Actions:** `retrieve`, `peek`, `add`, `update`, `delete`, `delete_many`, `list`, `list_profiles`, `discover`

## `layered_summaries`

Summary CRUD. Use: summarize (store summary of a URL/video), get (read back a specific summary), search (find previously summarized content), list (browse summaries), update (edit a summary), delete/batch_delete (remove summaries), list_expiring (find old summaries), sync (re-index summaries DB against .md files on disk — run when summaries appear stale after file-level edits).

**Actions:** `summarize`, `list`, `get`, `search`, `delete`, `update`, `batch_delete`, `list_expiring`, `sync`
