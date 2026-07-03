# GitHub Release Contents

This is the public release of the **hermes-layered-memory** plugin for Hermes Agent.

## What's Included

| File | Description |
|------|-------------|
| `__init__.py` | Hermes MemoryProvider plugin — 28 tool schemas and dispatch (1402 lines) |
| `backend.py` | LayeredBackend engine — CRUD, 5-layer retrieval, Qdrant, SQLite, enrichment, compaction, export/import, decay, backup (3105 lines) |
| `summaries.py` | SummariesBackend — SQLite + FTS5 + portable `.md` files, profile-scoped, WAL mode (828 lines) |
| `plugin.yaml` | Plugin manifest (v0.2.0) |
| `hermes-layered-memory.example.json` | Example JSON config (collections, scoring) |
| `.env.example` | Example environment variables (URLs, models, paths) |
| `entity-patterns/default.json` | Externalized regex patterns for entity extraction |
| `docker-compose.yml` | Qdrant Docker compose (bind mount, data survives restarts) |
| `LICENSE` | MIT License |
| `README.md` | Full documentation and tool reference |
| `.gitignore` | Git ignore rules |

## Setup

See [README.md](README.md) for installation, configuration, and tool reference.

## Original Repository

This project was originally created as a private Gitea repository.

Original: https://github.com/alexeytz/hermes-layered-memory