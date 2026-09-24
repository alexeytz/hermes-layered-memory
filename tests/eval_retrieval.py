#!/usr/bin/env python3
"""Retrieval quality evaluation harness.

The regression suite proves the pipeline *runs*. This proves it *retrieves*.

Builds a fixed corpus in an isolated DB, runs a set of golden queries whose
correct answers are known, and reports recall@k, MRR and latency per layer.
Without this, none of the scoring weights, layer thresholds or embedding
choices can be evaluated — a change that halves retrieval quality looks
identical to one that doubles it.

Usage:
    python3 tests/eval_retrieval.py                    # evaluate layers 0-2
    python3 tests/eval_retrieval.py --max-layer 3      # include LLM rerank
    python3 tests/eval_retrieval.py --query-instruction "Instruct: Retrieve facts relevant to the query\\nQuery: "
    python3 tests/eval_retrieval.py --compare          # A/B raw vs instructed queries
    python3 tests/eval_retrieval.py --filler 500       # realistic corpus size
    python3 tests/eval_retrieval.py --json out.json    # machine-readable

Requires Qdrant and an embedding endpoint (same as the regression suite).
"""

from __future__ import annotations

import argparse
import json
import os
import pwd
import re
import statistics
from datetime import datetime, timedelta, timezone
import sys
import time
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conftest import (  # noqa: E402
    QDRANT_URL,
    _align_test_collection_dims,
    _check_qdrant_alive,
    _cleanup_db,
    _cleanup_qdrant_coll,
    _get_uuid,
    _make_backend,
)

# Unique per process: every run used the same id, so two concurrent
# evaluations shared one Qdrant profile namespace and deleted each other's
# records mid-flight. Observed as an unreproducible recall of 0.409.
EVAL_ID = "eval%d" % os.getpid()

# ── Corpus ──────────────────────────────────────────────────────────────────
# Each record has a stable key used by the golden queries. Content is written
# the way a real extraction would write it — a single durable fact per record.
CORPUS: List[Dict[str, Any]] = [
    # Hardware / environment
    ("gpu",        "The workstation has an NVIDIA RTX 4090 with 24GB of VRAM.", "ENV-DATA", "hw", "hardware"),
    ("cpu",        "The laptop is a ThinkPad T440p with an i7-4700MQ and 16GB RAM.", "ENV-DATA", "hw", "hardware"),
    ("disk",       "Primary storage is a 2TB NVMe drive mounted at /data.", "ENV-DATA", "hw", "storage"),
    ("qdrant",     "Qdrant runs in Docker on localhost port 6333 with a bind mount for storage.", "ENV-DATA", "sw", "vector database"),
    ("ollama",     "Ollama serves the qwen3-embedding:8b model producing 4096-dimensional vectors.", "ENV-DATA", "sw", "embeddings"),
    ("vllm",       "vLLM serves Qwen3-27B for chat completions with tensor parallelism across two GPUs.", "ENV-DATA", "sw", "inference"),
    ("postgres",   "PostgreSQL 16 runs on port 5432 and holds the analytics warehouse.", "ENV-DATA", "sw", "database"),
    ("redis",      "Redis is configured with 512MB maxmemory and an allkeys-lru eviction policy.", "ENV-DATA", "sw", "cache"),
    # Preferences
    ("editor",     "The user prefers Neovim over VS Code for all editing work.", "USER-DATA", "preferences", "editor"),
    ("theme",      "The user prefers dark mode in every application.", "USER-DATA", "preferences", "appearance"),
    ("lang",       "The user writes backend services in Python and avoids TypeScript.", "USER-DATA", "preferences", "languages"),
    ("commits",    "The user wants commit messages in imperative mood with no emoji.", "USER-DATA", "preferences", "git"),
    ("tests",      "The user requires tests to accompany every code change, no exceptions.", "USER-DATA", "preferences", "testing"),
    # Rules / system
    ("nopush",     "Never push to the remote repository without explicit approval.", "SYSTEM", "rules", "git safety"),
    ("noprint",    "Use the logger for diagnostics; never add print statements to the codebase.", "SYSTEM", "rules", "logging"),
    ("secrets",    "API keys must be read from environment variables, never committed to source.", "SYSTEM", "rules", "security"),
    # Project facts
    ("hlm_layers", "The layered memory plugin runs a five-layer pipeline: Qdrant ANN, SQLite filters, fusion scoring, LLM rerank, gap detection.", "CUSTOM", None, "architecture"),
    ("hlm_sqlite", "SQLite is the source of truth for memories and Qdrant is a rebuildable index.", "CUSTOM", None, "architecture"),
    ("hlm_dedup",  "Semantic dedup blocks a write when cosine similarity to an existing memory exceeds 0.97.", "CUSTOM", None, "deduplication"),
    ("hlm_decay",  "Trust score decays for memories older than 30 days unless they are pinned.", "CUSTOM", None, "lifecycle"),
    # Distractors — plausible neighbours that must NOT win
    ("gpu_old",    "The previous build used a GTX 1080 that was sold in 2024.", "ENV-DATA", "hw", "hardware history"),
    ("editor_old", "Years ago the user used Emacs before switching away from it.", "USER-DATA", "preferences", "editor history"),
    ("meeting",    "The Tuesday standup was moved to 10am starting in March.", "CUSTOM", None, "scheduling"),
    ("groceries",  "Buy oat milk, sourdough bread and coffee beans on the way home.", "CUSTOM", None, "errands"),

    # ── Competition set ─────────────────────────────────────────────────────
    # Near-duplicates of the golden records: same domain, same vocabulary,
    # answering a *different* question. Without these the corpus asks only
    # "did you land in the right neighbourhood"; with them a query has to
    # discriminate between several plausible answers, which is the only
    # condition under which a reranker can demonstrate anything (see the
    # saturation warning in _report).
    ("port_grafana", "Grafana is served on port 3000 behind the reverse proxy.", "ENV-DATA", "sw", "monitoring"),
    ("port_ingress", "The ingress controller terminates TLS on port 8443.", "ENV-DATA", "net", "networking"),
    ("port_minio",   "MinIO object storage listens on port 9000 for the S3 API.", "ENV-DATA", "sw", "storage"),
    ("vram_server",  "The inference server pools 48GB of VRAM across two cards.", "ENV-DATA", "hw", "hardware"),
    ("ram_laptop",   "The laptop tops out at 16GB of system RAM and cannot be upgraded.", "ENV-DATA", "hw", "hardware"),
    ("embed_local",  "A local all-MiniLM-L6-v2 fallback produces 384-dimensional vectors.", "ENV-DATA", "sw", "embeddings"),
    ("rerank_model", "A BGE reranker model is installed but not wired into the pipeline.", "ENV-DATA", "sw", "reranking"),
    ("editor_ide",   "JetBrains IDEs are installed but used only for Java work.", "USER-DATA", "preferences", "editor"),
    ("theme_term",   "The terminal colour scheme is Gruvbox at all times.", "USER-DATA", "preferences", "appearance"),
    ("lang_scripts", "Shell scripts are written in bash, never fish or zsh.", "USER-DATA", "preferences", "languages"),
    ("commits_pr",   "Pull request titles follow conventional commits and bodies stay short.", "USER-DATA", "preferences", "git"),
    ("tests_ci",     "CI runs the full suite on every push and blocks merge on failure.", "ENV-DATA", "workflow", "testing"),
    ("nopush_force", "Force-pushing to a shared branch is forbidden under any circumstance.", "SYSTEM", "rules", "git safety"),
    ("secrets_vault", "Production secrets live in Vault and are injected at deploy time.", "SYSTEM", "rules", "security"),
    ("hlm_fusion",   "Layer 2 fuses BM25, recency and trust into a single ranking score.", "CUSTOM", None, "architecture"),
    ("hlm_super",    "A superseded memory stays stored but drops out of normal retrieval.", "CUSTOM", None, "lifecycle"),

    # ── Vendor-diverse hardware and infrastructure (2026-09-17) ─────────────
    # GPU / accelerator — vendor-diverse (2026-09-17)
    ("gpu_render_amd",   "The render node uses an AMD Radeon RX 7900 XTX with 24GB of VRAM.", "ENV-DATA", "hw", "graphics"),
    ("gpu_infer_h100",   "The inference node uses an NVIDIA H100 SXM with 80GB of HBM3 memory.", "ENV-DATA", "hw", "graphics"),
    ("gpu_arc_a770",     "The media node uses an Intel Arc A770 with 16GB of GDDR6 memory.", "ENV-DATA", "hw", "graphics"),
    ("gpu_apple_m3max",  "The macOS build host uses an Apple M3 Max with 128GB of unified memory.", "ENV-DATA", "hw", "graphics"),
    ("gpu_rtx_a6000",    "The visualization node uses an NVIDIA RTX A6000 with 48GB of GDDR6 memory.", "ENV-DATA", "hw", "graphics"),
    ("gpu_mi250",        "The research node uses an AMD Instinct MI250 with 128GB of HBM2e memory.", "ENV-DATA", "hw", "graphics"),
    ("gpu_max1550",      "The compute node uses an Intel Data Center GPU Max 1550 with 128GB of HBM2e memory.", "ENV-DATA", "hw", "graphics"),
    ("gpu_l40s",         "The serving node uses an NVIDIA L40S with 48GB of GDDR6 memory.", "ENV-DATA", "hw", "graphics"),
    ("gpu_rocm_stack",   "The AMD training node uses ROCm 6.2.3 for GPU compute workloads.", "ENV-DATA", "sw", "graphics"),
    ("gpu_cuda_stack",   "The H100 training node uses CUDA 12.6 for accelerator workloads.", "ENV-DATA", "sw", "graphics"),
    ("gpu_oneapi_stack", "The Intel compute node uses Intel oneAPI 2025.0 for SYCL workloads.", "ENV-DATA", "sw", "graphics"),
    ("gpu_metal_stack",  "The Apple build host uses Metal 3 for GPU-accelerated workloads.", "ENV-DATA", "sw", "graphics"),
    ("gpu_compound",     "The render node runs Linux, uses an RX 7900 XTX, exposes 24GB of VRAM, drives two displays, and uses ROCm.", "ENV-DATA", "hw", "graphics"),

    # CPU stack — vendor-diverse (2026-09-17)
    ("cpu_threadripper", "The render node uses an AMD Ryzen Threadripper 7970X with 32 cores and 64 threads.", "ENV-DATA", "hw", "processor"),
    ("cpu_xeon_6430",    "The database node uses an Intel Xeon Gold 6430 with 32 cores and 64 threads.", "ENV-DATA", "hw", "processor"),
    ("cpu_epyc_9354",    "The server node uses an AMD EPYC 9354 with 32 cores and 64 threads.", "ENV-DATA", "hw", "processor"),
    ("cpu_graviton3",    "The ARM service node uses an AWS Graviton3 processor with 64 vCPUs.", "ENV-DATA", "hw", "processor"),
    ("cpu_ryzen_7950x",  "The build node uses an AMD Ryzen 9 7950X with 16 cores and 32 threads.", "ENV-DATA", "hw", "processor"),
    ("cpu_xeon_8462y",   "The analytics node uses an Intel Xeon Platinum 8462Y+ with 32 cores and 64 threads.", "ENV-DATA", "hw", "processor"),
    ("cpu_ampere_altra", "The ARM server uses an Ampere Altra Q80-30 with 80 cores.", "ENV-DATA", "hw", "processor"),
    ("cpu_apple_m3pro",  "The macOS service host uses an Apple M3 Pro with 12 CPU cores.", "ENV-DATA", "hw", "processor"),
    ("cpu_compound",     "The render node uses Zen 4, has 32 cores, reaches 5.3GHz boost, carries 128MB L3 cache, and has a 350W TDP.", "ENV-DATA", "hw", "processor"),

    # Memory and storage — vendor-diverse (2026-09-17)
    ("mem_ddr5_ecc",     "The database node uses 128GB of DDR5-4800 ECC memory across four channels.", "ENV-DATA", "hw", "memory"),
    ("mem_lpddr5x",      "The mobile build host uses 64GB of LPDDR5X-6400 memory.", "ENV-DATA", "hw", "memory"),
    ("disk_nvme_gen4",   "The build node stores projects on a 4TB PCIe 4.0 NVMe SSD.", "ENV-DATA", "hw", "storage"),
    ("disk_sata_ssd",    "The archive node uses an 8TB SATA SSD for local scratch storage.", "ENV-DATA", "hw", "storage"),
    ("mem_ddr4_non_ecc", "The legacy test node has 32GB of DDR4-3200 non-ECC memory.", "ENV-DATA", "hw", "memory"),
    ("disk_nvme_gen5",   "The benchmark node uses a 2TB PCIe 5.0 NVMe SSD.", "ENV-DATA", "hw", "storage"),
    ("disk_raid10",      "The database host stores its primary dataset on a four-drive RAID10 array.", "ENV-DATA", "hw", "storage"),
    ("disk_zfs",         "The storage host uses ZFS for its bulk data filesystem.", "ENV-DATA", "hw", "storage"),
    ("storage_compound", "The database node uses NVMe storage, formats it with ext4, mounts it at /srv/data, enables noatime, and reserves five percent.", "ENV-DATA", "hw", "storage"),

    # Network — vendor-diverse (2026-09-17)
    ("net_10gbe",        "The storage node connects through a 10GbE Ethernet interface.", "ENV-DATA", "net", "networking"),
    ("net_25gbe",        "The compute node connects through a 25GbE Ethernet interface.", "ENV-DATA", "net", "networking"),
    ("net_jumbo",        "The backup network uses an MTU of 9000 for jumbo frames.", "ENV-DATA", "net", "networking"),
    ("net_wireguard",    "The remote administration overlay uses WireGuard for encrypted tunnels.", "ENV-DATA", "net", "networking"),
    ("net_2_5gbe",       "The desktop node connects through a 2.5GbE Ethernet interface.", "ENV-DATA", "net", "networking"),
    ("net_bonding",      "The storage server combines two 10GbE interfaces with an active-backup bond.", "ENV-DATA", "net", "networking"),
    ("net_vlan30",       "The lab network places infrastructure services on VLAN 30.", "ENV-DATA", "net", "networking"),
    ("net_dns_resolver", "The internal resolver forwards external DNS requests to Cloudflare DNS.", "ENV-DATA", "net", "networking"),
    ("network_compound", "The storage host uses 25GbE, assigns a static address, carries production traffic on VLAN 30, uses MTU 9000, and permits SSH.", "ENV-DATA", "net", "networking"),

    # Platform / runtime — vendor-diverse (2026-09-17)
    ("os_ubuntu_2404",     "The build host runs Ubuntu 24.04 LTS with the 6.8 GA kernel.", "ENV-DATA", "sw", "platform"),
    ("os_debian_12",       "The database host runs Debian 12 with its stock kernel.", "ENV-DATA", "sw", "platform"),
    ("os_fedora_40",       "The workstation runs Fedora 40 with a 6.8 series kernel.", "ENV-DATA", "sw", "platform"),
    ("os_windows_wsl2",    "The Windows development host uses WSL2 with Ubuntu 24.04.", "ENV-DATA", "sw", "platform"),
    ("os_rocky_9",         "The service host runs Rocky Linux 9.4 for its server workload.", "ENV-DATA", "sw", "platform"),
    ("os_arch_linux",      "The experimental host runs Arch Linux with a rolling kernel package.", "ENV-DATA", "sw", "platform"),
    ("runtime_python_312", "The application environment uses Python 3.12.4 with uv for package management.", "ENV-DATA", "sw", "runtime"),
    ("runtime_node_22",    "The frontend environment uses Node.js 22 with npm for package management.", "ENV-DATA", "sw", "runtime"),
    ("platform_compound",  "The build host runs Ubuntu 24.04, uses Python 3.12, installs packages with uv, limits containers with cgroups, and starts services through systemd.", "ENV-DATA", "sw", "platform"),

    # Inference and data services — vendor-diverse (2026-09-17)
    ("infer_tensorrtllm", "The inference host serves Llama 3.1 70B with TensorRT-LLM.", "ENV-DATA", "sw", "inference"),
    ("infer_llamacpp",    "The CPU inference host serves GGUF models through llama.cpp.", "ENV-DATA", "sw", "inference"),
    ("vector_milvus",     "The retrieval cluster stores embeddings in Milvus 2.4.", "ENV-DATA", "sw", "vector"),
    ("vector_pgvector",   "The analytics database stores embeddings with pgvector 0.8.0.", "ENV-DATA", "sw", "vector"),
    ("infer_tgi",         "The model gateway serves transformer models through Text Generation Inference 3.0.", "ENV-DATA", "sw", "inference"),
    ("vector_weaviate",   "The semantic search service stores vectors in Weaviate 1.25.", "ENV-DATA", "sw", "vector"),
    ("vector_chroma",     "The notebook environment stores local embeddings with Chroma 0.5.", "ENV-DATA", "sw", "vector"),
    ("rerank_bge",        "The retrieval service reranks candidates with BGE-reranker-v2-m3.", "ENV-DATA", "sw", "retrieval"),
    ("services_compound", "The retrieval host uses Milvus, stores 1024-dimensional vectors, exposes HTTP on port 19530, runs under Docker, and persists data locally.", "ENV-DATA", "sw", "services"),
]

# ── Golden queries ──────────────────────────────────────────────────────────
# expect: any of these keys in the top-k counts as a hit (first one found
#         determines reciprocal rank)
# forbid: these must NOT appear in the top-k
QUERIES: List[Dict[str, Any]] = [
    {"q": "what GPU is in the workstation",           "expect": ["gpu"],        "forbid": ["gpu_old"]},
    {"q": "how much VRAM do I have",                  "expect": ["gpu"]},
    {"q": "which laptop am I using",                  "expect": ["cpu"]},
    {"q": "where is the vector database running",     "expect": ["qdrant"]},
    {"q": "what embedding model and how many dimensions", "expect": ["ollama"]},
    {"q": "what serves chat completions",             "expect": ["vllm"]},
    {"q": "redis eviction policy",                    "expect": ["redis"]},
    {"q": "what port does postgres use",              "expect": ["postgres"]},
    {"q": "which editor does the user prefer",        "expect": ["editor"],     "forbid": ["editor_old"]},
    {"q": "light mode or dark mode",                  "expect": ["theme"]},
    {"q": "what programming language for backend work", "expect": ["lang"]},
    {"q": "how should commit messages be written",    "expect": ["commits"]},
    {"q": "do I need to write tests",                 "expect": ["tests"]},
    {"q": "am I allowed to push to remote",           "expect": ["nopush"]},
    {"q": "how should the code do logging",           "expect": ["noprint"]},
    {"q": "where do API keys come from",              "expect": ["secrets"]},
    {"q": "describe the retrieval pipeline layers",   "expect": ["hlm_layers"]},
    {"q": "which store is authoritative for memories", "expect": ["hlm_sqlite"]},
    {"q": "what similarity blocks a duplicate write", "expect": ["hlm_dedup"]},
    {"q": "when does trust score decay",              "expect": ["hlm_decay"]},
    {"q": "what is on the shopping list",             "expect": ["groceries"]},
    {"q": "when is the standup",                      "expect": ["meeting"]},

    # Vendor-diverse additions (2026-09-17)
    {"q": "which graphics card is in the render node", "expect": ["gpu_render_amd"]},
    {"q": "how much memory does the h100 inference node have", "expect": ["gpu_infer_h100"]},
    {"q": "what graphics card does the media node use", "expect": ["gpu_arc_a770"]},
    {"q": "how much unified memory does the apple build host have", "expect": ["gpu_apple_m3max"]},
    {"q": "how many cores does the render node processor have", "expect": ["cpu_threadripper"]},
    {"q": "what processor is in the database node", "expect": ["cpu_xeon_6430"]},
    {"q": "how many threads does the server node have", "expect": ["cpu_epyc_9354"]},
    {"q": "how many vcpus does the arm service node have", "expect": ["cpu_graviton3"]},
    {"q": "what memory does the database node use", "expect": ["mem_ddr5_ecc"]},
    {"q": "what kind of memory does the mobile build host have", "expect": ["mem_lpddr5x"]},
    {"q": "what nvme generation is in the build node", "expect": ["disk_nvme_gen4"]},
    {"q": "how much scratch storage does the archive node have", "expect": ["disk_sata_ssd"]},
    {"q": "what network speed does the storage node have", "expect": ["net_10gbe"]},
    {"q": "what network speed does the compute node have", "expect": ["net_25gbe"]},
    {"q": "what mtu does the backup network use", "expect": ["net_jumbo"]},
    {"q": "what vpn technology does the remote administration overlay use", "expect": ["net_wireguard"]},
    {"q": "what operating system runs on the build host", "expect": ["os_ubuntu_2404"]},
    {"q": "what operating system runs on the database host", "expect": ["os_debian_12"]},
    {"q": "what operating system does the workstation use", "expect": ["os_fedora_40"]},
    {"q": "what does the windows development host use for linux", "expect": ["os_windows_wsl2"]},
    {"q": "what runtime serves llama 3.1 70b", "expect": ["infer_tensorrtllm"]},
    {"q": "what serves gguf models on the cpu host", "expect": ["infer_llamacpp"]},
    {"q": "which vector database does the retrieval cluster use", "expect": ["vector_milvus"]},
    {"q": "what extension stores embeddings in the analytics database", "expect": ["vector_pgvector"]},
    {"q": "how large is the high bandwidth memory allocation on the inference machine", "expect": ["gpu_infer_h100"]},
    {"q": "which graphics adapter is installed in the media machine", "expect": ["gpu_arc_a770"]},
    {"q": "what capacity does the mac build system expose to applications", "expect": ["gpu_apple_m3max"]},
    {"q": "what parallel capacity is available on the server machine", "expect": ["cpu_epyc_9354"]},
    {"q": "how many virtual execution slots does the arm service machine expose", "expect": ["cpu_graviton3"]},
    {"q": "what portable system memory technology is used on the mobile build machine", "expect": ["mem_lpddr5x"]},
    {"q": "what local capacity is assigned to the archive machine", "expect": ["disk_sata_ssd"]},
]

# ── Hard queries ────────────────────────────────────────────────────────────
# The set above asks lexically easy questions of a 24-record corpus, which is
# why L2 scores recall@5 = 1.000 on it — a ceiling, not a result. _report's
# saturation warning names the three shapes that can actually discriminate,
# and these are those three:
#
#   1. PARAPHRASE  — the query shares no content word with the gold record, so
#                    BM25 contributes nothing and only the embedding can find it.
#   2. COMPETITION — several near-duplicates from the competition set are
#                    plausible; exactly one answers the question asked.
#   3. MULTI       — more than one record is genuinely correct, so the metric
#                    measures ranking rather than presence.
#
# Marked `hard` so the report can break them out: a change that moves the easy
# set is usually noise, while a change that moves this set is a real signal.
HARD_QUERIES: List[Dict[str, Any]] = [
    # 1. Paraphrase — deliberately no shared content words with the gold record
    {"q": "how much graphics memory is available for training", "expect": ["gpu"], "hard": True},
    {"q": "am I permitted to publish work upstream on my own",  "expect": ["nopush"], "hard": True},
    {"q": "what is the sanctioned way to emit diagnostics",     "expect": ["noprint"], "hard": True},
    {"q": "where are credentials supposed to live",             "expect": ["secrets", "secrets_vault"], "hard": True},
    {"q": "is it acceptable to ship an untested patch",         "expect": ["tests", "tests_ci"], "hard": True},
    {"q": "which datastore can be regenerated from the other",  "expect": ["hlm_sqlite"], "hard": True},
    {"q": "at what closeness is a new note rejected as redundant", "expect": ["hlm_dedup"], "hard": True},
    {"q": "when does confidence in an old note start to erode", "expect": ["hlm_decay"], "hard": True},
    {"q": "what happens to a fact after it is replaced",        "expect": ["hlm_super"], "hard": True},
    {"q": "how are several ranking signals combined into one",  "expect": ["hlm_fusion"], "hard": True},
    {"q": "what typeface colours does the console use",         "expect": ["theme_term", "theme"], "hard": True},
    {"q": "which tool does the user reach for to write code",   "expect": ["editor"], "forbid": ["editor_old"], "hard": True},

    # 2. Competition — the competition set makes several records plausible
    {"q": "what port does the vector database listen on",  "expect": ["qdrant"],
     "forbid": ["port_grafana", "port_ingress", "port_minio", "postgres"], "hard": True},
    {"q": "what port does the object store use",           "expect": ["port_minio"],
     "forbid": ["qdrant", "postgres", "port_grafana"], "hard": True},
    {"q": "which port serves dashboards",                  "expect": ["port_grafana"],
     "forbid": ["qdrant", "postgres", "port_minio"], "hard": True},
    {"q": "where is TLS terminated",                       "expect": ["port_ingress"], "hard": True},
    {"q": "how much VRAM does the inference host have",    "expect": ["vram_server"],
     "forbid": ["gpu"], "hard": True},
    {"q": "how much memory does the portable machine have", "expect": ["ram_laptop", "cpu"],
     "forbid": ["gpu", "vram_server"], "hard": True},
    {"q": "what dimensionality does the fallback embedder produce", "expect": ["embed_local"],
     "forbid": ["ollama"], "hard": True},
    {"q": "is there a reranking model available",          "expect": ["rerank_model"], "hard": True},
    {"q": "what shell do scripts use",                     "expect": ["lang_scripts"],
     "forbid": ["lang"], "hard": True},
    {"q": "how should a pull request be titled",           "expect": ["commits_pr", "commits"], "hard": True},
    {"q": "is force pushing allowed",                      "expect": ["nopush_force"], "hard": True},
    {"q": "when does the test suite run automatically",    "expect": ["tests_ci"],
     "forbid": ["tests"], "hard": True},
    {"q": "which IDE is used for java",                    "expect": ["editor_ide"],
     "forbid": ["editor"], "hard": True},

    # 3. Multi — more than one record is genuinely correct; ranking is the test
    {"q": "what are the rules about git",   "expect": ["nopush", "nopush_force", "commits", "commits_pr"], "hard": True},
    {"q": "tell me about the user's preferences", "expect": ["editor", "theme", "lang", "commits", "tests"], "hard": True},
    {"q": "what runs on which port",        "expect": ["qdrant", "postgres", "port_grafana", "port_ingress", "port_minio"], "hard": True},
    {"q": "what do I know about embeddings", "expect": ["ollama", "embed_local"], "hard": True},

    # ── Vendor-diverse additions (2026-09-17) ───────────────────────────
    {"q": "which display accelerator is fitted to the scene machine", "expect": ["gpu_render_amd"], "hard": True},  # PARAPHRASE
    {"q": "how many execution units does the render processor provide", "expect": ["cpu_threadripper"], "forbid": ["cpu_compound", "gpu_compound", "gpu_render_amd"], "hard": True},  # COMPETITION
    {"q": "which silicon package powers the database machine", "expect": ["cpu_xeon_6430"], "forbid": ["mem_ddr5_ecc", "storage_compound"], "hard": True},  # COMPETITION
    {"q": "which volatile memory configuration belongs to the database machine", "expect": ["mem_ddr5_ecc"], "forbid": ["cpu_xeon_6430", "storage_compound"], "hard": True},  # COMPETITION
    {"q": "which flash bus generation serves the build machine", "expect": ["disk_nvme_gen4"], "forbid": ["cpu_ryzen_7950x"], "hard": True},  # COMPETITION
    {"q": "how fast can the storage machine communicate on its wired link", "expect": ["net_10gbe"], "forbid": ["net_25gbe", "net_2_5gbe"], "hard": True},  # COMPETITION
    {"q": "how fast is the compute machine's wired connection", "expect": ["net_25gbe"], "forbid": ["net_10gbe", "net_2_5gbe"], "hard": True},  # COMPETITION
    {"q": "what frame size is configured on the backup fabric", "expect": ["net_jumbo"], "forbid": ["net_10gbe", "net_25gbe"], "hard": True},  # COMPETITION
    {"q": "which encrypted tunnel technology protects remote administration", "expect": ["net_wireguard"], "forbid": ["net_vlan30", "net_bonding"], "hard": True},  # COMPETITION
    {"q": "which distribution release is installed on the build machine", "expect": ["os_ubuntu_2404"], "forbid": ["os_debian_12", "os_fedora_40", "os_rocky_9"], "hard": True},  # COMPETITION
    {"q": "which distribution is installed on the database machine", "expect": ["os_debian_12"], "forbid": ["os_ubuntu_2404", "os_fedora_40", "os_rocky_9"], "hard": True},  # COMPETITION
    {"q": "which package manager handles dependencies in the application environment", "expect": ["runtime_python_312"], "forbid": ["runtime_node_22"], "hard": True},  # COMPETITION
    {"q": "which package manager belongs to the frontend environment", "expect": ["runtime_node_22"], "forbid": ["runtime_python_312"], "hard": True},  # COMPETITION
    {"q": "which service handles transformer generation on the model gateway", "expect": ["infer_tgi"], "forbid": ["infer_tensorrtllm", "infer_llamacpp"], "hard": True},  # COMPETITION
    {"q": "which runtime handles gguf inference without a gpu", "expect": ["infer_llamacpp"], "forbid": ["infer_tgi", "infer_tensorrtllm"], "hard": True},  # COMPETITION
    {"q": "where are embeddings stored for the retrieval cluster and analytics database", "expect": ["vector_milvus", "vector_pgvector"], "hard": True},  # MULTI
    {"q": "which machines have a discrete amd or nvidia accelerator", "expect": ["gpu", "gpu_cuda_stack", "gpu_infer_h100", "gpu_l40s", "gpu_mi250", "gpu_old", "gpu_render_amd", "gpu_rtx_a6000"], "hard": True},  # MULTI
    {"q": "which machines use server-class processors with at least 32 cores", "expect": ["cpu_ampere_altra", "cpu_epyc_9354", "cpu_graviton3", "cpu_xeon_6430", "cpu_xeon_8462y"], "hard": True},  # MULTI
    {"q": "which hosts use high-speed ethernet above 10 gigabits", "expect": ["net_25gbe"], "forbid": ["gpu_max1550"], "hard": True},  # COMPETITION
    {"q": "which environments run linux distributions rather than windows or macos", "expect": ["os_ubuntu_2404", "os_debian_12", "os_fedora_40", "os_rocky_9", "os_arch_linux"], "hard": True},  # MULTI
    {"q": "which vector systems are used for semantic retrieval", "expect": ["qdrant", "services_compound", "vector_chroma", "vector_milvus", "vector_pgvector", "vector_weaviate"], "hard": True},  # MULTI
]

QUERIES = QUERIES + HARD_QUERIES

# ── Dense entities ──────────────────────────────────────────────────────────
#
# Every other record in this corpus is a *distinct* entity, so it could not see
# the failure the behaviour harness found: a question about one attribute of an
# entity that owns many records. The answering record sat at ANN position 12-15
# of a 30-candidate pool and never survived L2 fusion into the top 5, because
# its siblings match the entity name *and the attribute vocabulary* just as
# well, and only one of them is the attribute.
#
# The first version of this block scored dense@5 = 1.0 and gated nothing. Its
# siblings named the entity and nothing else — "restarted during the January
# maintenance window" — so "how much memory can worker-node-7 use" had exactly
# one lexical and semantic match and the crowd never competed. What makes the
# telecom corpus hard is that the siblings look like *answers*: asked for a
# pool's TPS, twenty-nine records about that pool talk about signalling
# connections and link capacity. So the decoys here carry GB figures, log
# policies and counts — every one a plausible reply to the question, none of
# them the fact.
#
# The siblings are not "forbidden": they are legitimately about the entity and
# returning them is unhelpful, not wrong. `forbid` names the *other* entities'
# record for the same attribute, which is a genuinely wrong answer.
DENSE_ENTITIES = [
    ("wnode7",  "worker-node-7",   96, 45),
    ("irelay",  "ingest-relay",    72, 21),
    ("atlsch",  "atlas-scheduler", 48, 90),
    ("vproxy",  "vault-proxy",     16, 14),
    ("bcache",  "beacon-cache",   128, 60),
]

#: Written once and formatted per entity: thirty near-identical lines repeated
#: five times is one decision expressed a hundred and fifty times.
#:
#: Sized from the failure, not from taste. The telecom pool whose TPS went
#: missing owned twenty-nine records; at seven siblings this block scored
#: dense@5 = 1.0 twice, once with neutral siblings and once with
#: plausible-answer decoys. Crowding is a function of how many competitors
#: share the query's vocabulary, and seven is not a crowd.
_DENSE_SIBLINGS = (
    # Memory-flavoured: every one carries a GB figure *and* the word memory,
    # so neither token discriminates and only the sense of "total allocation"
    # separates the answer from the crowd.
    [("kubelet",  "{e} reserves 8 GB of memory for the kubelet and system daemons."),
     ("ctrcap",   "{e} allocates at most 4 GB of memory to any single container."),
     ("pagecache","{e} holds 2 GB of memory as page cache under steady load."),
     ("heap",     "{e} gives its JVM 12 GB of memory as an initial heap."),
     ("hugepage", "{e} pins 6 GB of memory as hugepages for the data plane."),
     ("swap",     "{e} is configured with 3 GB of swap and memory overcommit off."),
     ("cache",    "{e} keeps a 9 GB in-memory result cache."),
     ("burst",    "{e} may burst 5 GB of memory above its request."),
     ("evict",    "{e} begins evicting pods at 1 GB of remaining memory."),
     ("scratch",  "{e} keeps a 500 GB local scratch disk for memory spill files.")]
    # Log-flavoured: log policies that are not retention.
    + [("rotate", "{e} rotates its access log every 12 hours."),
       ("ship",   "{e} ships logs to the central collector every 30 seconds."),
       ("dumps",  "{e} keeps crash dumps for 3 days before deleting them."),
       ("sample", "{e} samples debug logs at 1 percent of requests."),
       ("level",  "{e} logs at INFO and drops DEBUG lines at the agent."),
       ("audit",  "{e} writes audit log entries to a separate append-only stream.")]
    # Neutral bulk: names the entity, answers nothing.
    + [(f"op{_i:02d}", t) for _i, t in enumerate([
        "{e} was restarted during the January maintenance window.",
        "{e} is deployed in the eu-west availability zone.",
        "{e} emits metrics to the central Prometheus endpoint.",
        "{e} is owned by the platform team.",
        "{e} has a liveness probe on the /healthz path.",
        "{e} is excluded from the nightly rebalance.",
        "{e} mounts its configuration from a read-only volume.",
        "{e} drains connections for 30 seconds before shutdown.",
        "{e} is tainted so only fleet workloads schedule onto it.",
        "{e} reports its uptime to the fleet dashboard hourly.",
        "{e} runs the hardened kernel build used across the fleet.",
        "{e} was last patched in the February rollout.",
        "{e} participates in the weekly failover drill.",
        "{e} has its serial console disabled in firmware.",
     ])]
)

for _key, _name, _mem, _days in DENSE_ENTITIES:
    for _skey, _tmpl in _DENSE_SIBLINGS:
        CORPUS.append((f"{_key}_{_skey}", _tmpl.format(e=_name),
                       "ENV-DATA", "fleet", "fleet"))
    # Compound, because that is the shape the answer had when it went missing.
    # The telecom record read "pool 480 serves region WEST, runs on Dell
    # PowerEdge R740, handling 7170 TPS" — the queried figure was one clause in
    # a record mostly about other things, so its embedding sat closer to
    # "region and vendor" than to "TPS", while thirty single-purpose siblings
    # sat squarely on the entity. A gold record that *is* the question is the
    # one case retrieval never struggles with, and all three earlier versions
    # of this block wrote exactly that.
    CORPUS.append((f"{_key}_mem",
                   f"{_name} sits in rack B12 behind the fleet load balancer, "
                   f"runs the hardened kernel build, is owned by the platform "
                   f"team, and has a memory ceiling of {_mem} GB.",
                   "ENV-DATA", "fleet", "fleet capacity"))
    CORPUS.append((f"{_key}_logs",
                   f"{_name} was commissioned in the autumn hardware refresh, "
                   f"carries the fleet-standard agent set, reports to the "
                   f"eu-west collector, and retains logs for {_days} days.",
                   "ENV-DATA", "fleet", "fleet retention"))

DENSE_QUERIES: List[Dict[str, Any]] = []
for _key, _name, _mem, _days in DENSE_ENTITIES:
    DENSE_QUERIES.append({
        "q": f"what is the memory ceiling for {_name}",
        "expect": [f"{_key}_mem"], "dense": True,
        "forbid": [f"{o}_mem" for o, *_ in DENSE_ENTITIES if o != _key]})
    DENSE_QUERIES.append({
        "q": f"how long does {_name} retain logs",
        "expect": [f"{_key}_logs"], "dense": True,
        "forbid": [f"{o}_logs" for o, *_ in DENSE_ENTITIES if o != _key]})

QUERIES = QUERIES + DENSE_QUERIES

# Queries that carry no retrievable intent. Nothing in the corpus answers
# them, so a well-behaved prefetch should return nothing. Measured separately
# as a false-positive rate, not as recall.
NOISE_QUERIES = ["ok", "thanks", "sure", "got it", "yes please", "hmm"]

# Well-formed questions about facts the corpus demonstrably does not hold.
#
# Deliberately NOT appended to QUERIES: these have no gold record, so they are a
# false-positive measurement, not recall, and adding them would move the
# baseline fingerprint.
#
# NOISE_QUERIES above cannot stand in for this. "ok" and "thanks" carry no
# retrievable intent and prefetch already skips them; these are grammatical
# questions with real content words that simply have no answer here — the shape
# a user actually types when they overestimate what the store knows. The
# distinction had teeth on 2026-08-13: `low_relevance` fired on 5 of 6
# NOISE_QUERIES and **0 of 6** of these, because "what color is the sky on Mars"
# scores non-zero BM25 through `is`/`the`/`on` alone (455, 83 and 563 matching
# rows). A signal measured only against contentless filler read as working.
#
# Every term here must be absent from the corpus, and that is not something a
# static list can guarantee: 0.7.38 shipped "how do I make sourdough starter from
# scratch" and the corpus answers it — `sourdough` appears in the groceries
# record ("Buy oat milk, sourdough bread and coffee beans") and `scratch` in five
# filler records ("a 500 GB local scratch disk"). DF0 correctly declined to flag
# it and the probe scored the rule 1/6 worse than it deserved. `_offtopic_rate`
# now reports contamination rather than leaving it to be noticed by accident.
#
# Reported, not baselined. Nothing in the repo yet moves this number — the
# stopword-filtered rule that would was measured and rejected in 0.7.37,
# because it also fired on 7 of 61 golden queries and all 7 had the correct
# record in the returned set. This exists so the next candidate is measured
# against a corpus that can express the failure.
OFFTOPIC_QUERIES = [
    "what color is the sky on Mars",
    "who won the 1998 world cup final",
    "which composer wrote the Brandenburg concertos",
    "what is the boiling point of mercury",
    "which dynasty built the terracotta army",
    "how long is a marathon in kilometres",
]

# ── Haystack ────────────────────────────────────────────────────────────────
# recall@5 over 24 records is close to meaningless: five slots for 24
# candidates is a one-in-five shot. Real profiles hold hundreds to thousands.
# These filler records are plausible neighbours of the golden set — same
# domains, same vocabulary — so they compete for slots rather than padding the
# corpus with obvious noise.
_FILLER_TEMPLATES = [
    ("Service {i} listens on port {port} behind the ingress controller.", "ENV-DATA", "net", "networking"),
    ("Container image app-{i}:v{minor} is pinned in the deployment manifest.", "ENV-DATA", "sw", "containers"),
    ("Node worker-{i} has {gb}GB RAM and {cores} cores allocated.", "ENV-DATA", "hw", "cluster hardware"),
    ("Backup job {i} runs at 0{h}:00 UTC and retains {days} days.", "ENV-DATA", "sw", "backups"),
    ("The user prefers tabs over spaces in project-{i} configuration files.", "USER-DATA", "preferences", "formatting"),
    ("Meeting notes {i}: reviewed the roadmap and deferred item {minor}.", "CUSTOM", None, "meetings"),
    ("Rule {i}: rotate credentials for service-{i} every {days} days.", "SYSTEM", "rules", "credential policy"),
    ("Dataset {i} contains {gb} million rows partitioned by ingest date.", "CUSTOM", None, "data"),
    ("Alert {i} fires when p99 latency exceeds {days}00ms for {cores} minutes.", "ENV-DATA", "sw", "monitoring"),
    ("Library dep-{i} was upgraded to {minor}.{cores}.0 without breaking changes.", "ENV-DATA", "sw", "dependencies"),
]


def build_filler(n: int):
    """Generate n plausible distractor records, deterministically."""
    out = []
    for i in range(n):
        tpl, dtype, did, topic = _FILLER_TEMPLATES[i % len(_FILLER_TEMPLATES)]
        content = tpl.format(i=i, port=8000 + (i % 900), minor=(i % 9) + 1,
                             gb=(i % 12 + 1) * 8, cores=(i % 8) + 2,
                             h=(i % 9) + 1, days=(i % 30) + 1)
        out.append((f"filler{i}", content, dtype, did, topic))
    return out


# ── Metrics ─────────────────────────────────────────────────────────────────
def _assert_llm_live(be, layer: int):
    """Refuse to evaluate an LLM layer that cannot reach an LLM.

    L3 and L4 fall through as no-ops when _call_llm() returns "", so the
    harness reports "recall +0.000, no measurable gain" for layers that never
    executed — a recommendation to delete working features on evidence that
    was never gathered. Observed: L3 measured at 0.358s/query here against
    7.1s in a real session. The 20x gap only surfaced because someone noticed
    the GPUs were idle.

    Configuration presence is not sufficient: layer3_reasoning_effort
    defaulting wrong made every call return "" while the config looked
    complete. So this probes the endpoint.
    """
    if layer < 3:
        return
    if not be.llm_configured():
        raise SystemExit(
            "ABORT: L%d needs an LLM provider and none is configured.\n"
            "       layer3_model / layer3_provider_config.base_url are unset here.\n"
            "       Profile config lives in <profile>/.env and only reaches\n"
            "       os.environ under `hermes -p <profile>`.\n"
            "       Re-run with --profile <name> to load it." % layer)
    if not be._call_llm("Reply with the single word OK."):
        raise SystemExit(
            "ABORT: L%d provider is configured but returned nothing.\n"
            "       Evaluating it would measure a no-op. Check the endpoint and\n"
            "       layer3_reasoning_effort (reasoning models return null content\n"
            "       unless it is set to 'none')." % layer)


def _assert_healthy(be, layer: int):
    """Refuse to report numbers from a degraded system.

    If the circuit breaker is open or retrieval fell back to FTS5-only, the
    metrics would measure a broken system and produce false findings.
    """
    if be._degraded:
        raise SystemExit(
            f"ABORT: L{layer} retrieval is degraded (Qdrant circuit breaker open)."
            " Metrics from a degraded system are meaningless. Fix Qdrant and retry."
        )
    if be._qdrant_broken_until and time.time() < be._qdrant_broken_until:
        remaining = int(be._qdrant_broken_until - time.time())
        raise SystemExit(
            f"ABORT: Qdrant circuit open ({remaining}s remaining). "
            f"Cannot evaluate L{layer} while vector search is unavailable."
        )


def _evaluate(be, queries, key_to_uuid, uuid_to_key, max_layer: int, k: int,
              progress: bool = True) -> Dict[str, Any]:
    hits, rr, latencies, violations, misses = 0, [], [], [], []
    gold_ranks: List[Optional[int]] = []
    # Tracked separately: the easy set saturates, so a headline number that
    # mixes the two hides movement in the only queries able to show it.
    hard_n = hard_hits = 0
    dense_n = dense_hits = 0

    _assert_healthy(be, max_layer)
    _assert_llm_live(be, max_layer)

    if progress:
        # L3 and L4 call an LLM per query. Without a heartbeat the harness
        # looks hung for minutes with nothing but backend logging on screen.
        print(f"  L{max_layer}: {len(queries)} queries ", end="", flush=True)

    for case in queries:
        want = {key_to_uuid[key] for key in case["expect"]}
        forbid = {key_to_uuid[key] for key in case.get("forbid", [])}

        t0 = time.time()
        results = be.retrieve(case["q"], max_layer=max_layer, limit=k)
        latencies.append(time.time() - t0)

        if progress:
            print(".", end="", flush=True)
        got = [r.get("uuid") for r in results][:k]

        # Headroom probe. `retrieve(limit=k)` truncates to k, so reading the
        # gold rank off that list can never exceed k-1 and `headroom` was
        # structurally 0 for every corpus — which is the number the saturation
        # warning uses to decide whether L3 is testable at all. It therefore
        # always claimed "nothing for a reranker to promote", regardless of the
        # data. Ask a second, wider question to find where the answer really
        # sits. Skipped for L3/L4: it would double the LLM calls, and the check
        # only ever reads L2's value.
        if max_layer <= 2:
            wide = be.retrieve(case["q"], max_layer=max_layer, limit=max(k * 4, 20))
            wide_uuids = [r.get("uuid") for r in wide]
        else:
            wide_uuids = got
        gold_ranks.append(next((i for i, u in enumerate(wide_uuids) if u in want), None))
        rank = next((i for i, u in enumerate(got) if u in want), None)
        if case.get("hard"):
            hard_n += 1
        if case.get("dense"):
            dense_n += 1
        if rank is None:
            misses.append((case["q"], [uuid_to_key.get(u, u[:8]) for u in got]))
        else:
            hits += 1
            rr.append(1.0 / (rank + 1))
            if case.get("hard"):
                hard_hits += 1
            if case.get("dense"):
                dense_hits += 1
        if forbid & set(got):
            violations.append(case["q"])

    if progress:
        print(f" {sum(latencies):.0f}s")
    n = len(queries)
    return {
        "n": n,
        # Where the gold record sat in the FULL candidate list, not just the
        # returned top-k. L3 can only add value by promoting a record from
        # beyond `limit` into it — if the gold answer is never outside the
        # top-k at L2, no reranker can demonstrate anything, and a +0.000
        # result says the corpus is unable to test the layer.
        "gold_ranks": gold_ranks,
        "headroom": sum(1 for r in gold_ranks if r is not None and r >= k),
        f"recall@{k}": round(hits / n, 3) if n else 0.0,
        # The discriminating subset — paraphrase / competition / multi-answer.
        # This is the number to watch; the headline mixes it with an easy set
        # that sits at ceiling regardless of what the pipeline does.
        "hard_n": hard_n,
        f"hard_recall@{k}": round(hard_hits / hard_n, 3) if hard_n else 0.0,
        # Attribute-of-a-crowded-entity. Kept out of `hard` deliberately: hard
        # measures paraphrase and competition, this measures surviving a pile
        # of siblings, and averaging them together hides both.
        "dense_n": dense_n,
        f"dense_recall@{k}": round(dense_hits / dense_n, 3) if dense_n else 0.0,
        "mrr": round(sum(rr) / n, 3) if n else 0.0,
        "forbidden_hits": len(violations),
        "latency_p50": round(statistics.median(latencies), 3) if latencies else 0.0,
        "latency_max": round(max(latencies), 3) if latencies else 0.0,
        "misses": misses,
        "violations": violations,
    }


def _noise_rate(be, max_layer: int, k: int, floor: float) -> Dict[str, Any]:
    """Fraction of contentless queries that still return something."""
    _assert_healthy(be, max_layer)
    returned, scores = 0, []
    for q in NOISE_QUERIES:
        results = be.retrieve(q, max_layer=max_layer, limit=k)
        top = [r for r in results if r.get("score", 0.0) >= floor]
        if top:
            returned += 1
        scores.extend(round(r.get("score", 0.0), 3) for r in results[:1])
    return {
        "n": len(NOISE_QUERIES),
        "returned_something": returned,
        "noise_rate": round(returned / len(NOISE_QUERIES), 3),
        "top_scores": scores,
    }


def _offtopic_rate(be, max_layer: int, k: int) -> Dict[str, Any]:
    """How often a well-formed unanswerable question is flagged as such.

    `signalled` is the number carrying `low_relevance` — the only thing in the
    response that tells a caller the returned records are not answers. It is the
    number a relevance candidate has to move. `returned_something` is reported
    alongside it because retrieval returning five records here is correct
    behaviour, not a fault: the funnel ranks, it does not adjudicate.
    """
    _assert_healthy(be, max_layer)
    returned = signalled = 0
    for q in OFFTOPIC_QUERIES:
        results = be.retrieve(q, max_layer=max_layer, limit=k)
        if results:
            returned += 1
            flags = results[0].get("layer3_flags")
            if isinstance(flags, dict) and flags.get("low_relevance"):
                signalled += 1
    n = len(OFFTOPIC_QUERIES)
    out = {
        "n": n,
        "returned_something": returned,
        "signalled": signalled,
        "signalled_rate": round(signalled / n, 3) if n else 0.0,
    }
    contaminated = _offtopic_contamination(be)
    if contaminated:
        out["contaminated"] = contaminated
        print("    WARNING: off-topic probes overlap the corpus and no longer "
              "measure what they claim to:")
        for q, overlap in contaminated.items():
            print(f"      {q!r} -> {overlap}")
    return out


#: Words carrying no topical signal. Only used to decide which query terms must
#: be absent from the corpus for an off-topic probe to be honest — never to
#: change retrieval.
_OFFTOPIC_STOPWORDS = frozenset("""
a an and are as at be by can did do does for from has have how i in into is it
its many much of on or shall that the their there these they this to was were
what when where which who whom why will with would you your does long
""".split())


def _offtopic_contamination(be) -> Dict[str, Dict[str, int]]:
    """Off-topic probes whose terms the corpus actually contains.

    An off-topic probe only measures anything if the store demonstrably cannot
    answer it. A term the corpus holds makes the query answerable-in-part and
    understates any rule being tested. Corpus-dependent, so it is checked at run
    time rather than asserted once: `scratch` became a corpus word because a
    filler template mentions a scratch disk.
    """
    bad: Dict[str, Dict[str, int]] = {}
    for q in OFFTOPIC_QUERIES:
        terms = {t for t in re.findall(r"[a-z0-9]+", q.lower())
                 if t not in _OFFTOPIC_STOPWORDS and len(t) > 1}
        overlap = {}
        for t in sorted(terms):
            try:
                df = be._get_conn().execute(
                    "SELECT count(*) FROM memories_fts WHERE memories_fts MATCH ?",
                    (f'"{t}"',)).fetchone()[0]
            except Exception:
                continue  # unparseable token — cannot judge, do not accuse
            if df:
                overlap[t] = df
        if overlap:
            bad[q] = overlap
    return bad


def _apply_metadata_spread(be, uuids: List[str]) -> None:
    """Give records varied age and trust, deterministically and *uncorrelated*
    with whether they are the right answer.

    Two of Layer 2's five signals — recency and importance(trust) — are flat in
    the base corpus, because every record is written in one pass. Tuning the
    fusion against that measures three signals and calls it five.

    The spread is deliberately uncorrelated with correctness. A corpus where the
    gold record is always the newest and most trusted would flatter the design:
    the boosts would appear to work because the fixture was built to reward
    them. Real profiles hold plenty of recent, trusted records that do not
    answer the question in front of you, so "varied but uninformative" is both
    the honest default and the case that reveals whether the boosts do harm
    when they have nothing to say.
    """
    if not uuids:
        return
    rows = []
    for i, uid in enumerate(uuids):
        age_days = (i * 37) % 400                  # 0..399, no relation to gold
        trust = 0.2 + ((i * 17) % 70) / 100.0      # 0.20..0.89
        created = (datetime.now(timezone.utc) - timedelta(days=age_days)).isoformat()
        rows.append((created, round(trust, 4), uid))
    be._get_conn().executemany(
        "UPDATE memories SET created_at = ?, trust_score = ? WHERE uuid = ?", rows)
    be._get_conn().commit()


def _build(profile_suffix: str, query_instruction: Optional[str], filler: int = 0,
           scoring: Optional[Dict[str, float]] = None, spread: bool = False):
    config = {"enrich_llm": False}
    if query_instruction is not None:
        config["query_instruction"] = query_instruction
    # Layer-2 fusion weights are the least-examined tuning surface in the
    # system — DEFAULT_WEIGHTS has never been measured against an alternative.
    # Exposing them here is what makes "does this weight help?" answerable.
    if scoring:
        # `fusion_base` is a top-level config key, not a scoring weight; accept
        # it through the same flag so one sweep can vary both.
        base = scoring.pop("fusion_base_score", None)
        if base is not None:
            config["fusion_base"] = "score" if base else "rrf"
        if scoring:
            config["scoring"] = dict(scoring)
    # Realign the disposable test collections with whatever embedder is in
    # play before building — otherwise switching between the local fallback and
    # a configured endpoint aborts the run on a dimension error.
    _align_test_collection_dims()
    be = _make_backend(f"{EVAL_ID}{profile_suffix}", config=config)
    key_to_uuid, uuid_to_key = {}, {}
    records = list(CORPUS) + build_filler(filler)
    # Embed the whole corpus in batches first. add() embeds one record per HTTP
    # round trip, which took 1145s for 524 records — 19 minutes to set up a
    # 3-minute measurement makes iterating on the corpus impractical.
    vectors = be._embed_batch([r[1] for r in records])
    assert len(vectors) == len(records), "batch embedding returned the wrong count"
    for n, ((key, content, dtype, did, topic), vec) in enumerate(zip(records, vectors)):
        uid = _get_uuid(be.add(content, data_type=dtype, data_id=did, topic=topic,
                               force=True, embedding=vec))
        key_to_uuid[key] = uid
        uuid_to_key[uid] = key
        if filler and n and n % 250 == 0:
            print(f"    indexed {n}/{len(records)}...", flush=True)
    if spread:
        _apply_metadata_spread(be, list(uuid_to_key.keys()))
        print("    applied age/trust spread (uncorrelated with correctness)")
    return be, key_to_uuid, uuid_to_key


def _run(query_instruction: Optional[str], layers: List[int], k: int,
         floor: float, label: str, suffix: str, filler: int = 0,
         scoring: Optional[Dict[str, float]] = None,
         spread: bool = False) -> Dict[str, Any]:
    print(f"\n=== {label} ===")
    total = len(CORPUS) + filler
    print(f"Indexing {total} records ({len(CORPUS)} golden + {filler} distractors)...")
    t_index = time.time()
    be, key_to_uuid, uuid_to_key = _build(suffix, query_instruction, filler, scoring, spread)
    print(f"  indexed in {time.time() - t_index:.0f}s")
    out: Dict[str, Any] = {"label": label, "corpus_size": total,
                           "query_instruction": query_instruction,
                           "scoring": scoring, "layers": {}}
    try:
        for layer in layers:
            m = _evaluate(be, QUERIES, key_to_uuid, uuid_to_key, layer, k)
            noise = _noise_rate(be, layer, k, floor)
            offtopic = _offtopic_rate(be, layer, k)
            out["layers"][str(layer)] = {"quality": m, "noise": noise,
                                         "offtopic": offtopic}
            print(f"  L{layer}: recall@{k}={m[f'recall@{k}']:<6} mrr={m['mrr']:<6} "
                  f"forbidden={m['forbidden_hits']:<3} noise={noise['noise_rate']:<6} "
                  f"p50={m['latency_p50']}s max={m['latency_max']}s")
            print(f"       off-topic: {offtopic['signalled']}/{offtopic['n']} "
                  f"flagged low_relevance "
                  f"({offtopic['returned_something']}/{offtopic['n']} returned records)")
            for q, got in m["misses"][:5]:
                print(f"       MISS {q!r} -> {got}")
    finally:
        _cleanup_qdrant_coll(be)
        be.close()
        _cleanup_db(f"{EVAL_ID}{suffix}")
    _report(out, k)
    return out


def _report(out: Dict[str, Any], k: int) -> None:
    """Print the summary the run exists to produce.

    Per-layer lines scroll away under backend logging; the question this
    harness answers — does each extra layer earn its latency — is only legible
    side by side.
    """
    layers = out["layers"]
    if not layers:
        return
    print(f"\n{'=' * 68}")
    print(f"SUMMARY — {out['label']}  ({out.get('corpus_size', '?')} records)")
    print(f"{'=' * 68}")
    print(f"{'layer':<7}{'recall@' + str(k):<11}{'hard@' + str(k):<10}"
          f"{'dense@' + str(k):<11}{'MRR':<9}"
          f"{'forbidden':<11}{'p50':<9}{'total':<8}")
    for name in sorted(layers, key=int):
        q = layers[name]["quality"]
        print(f"L{name:<6}{q['recall@' + str(k)]:<11}"
              f"{q.get('hard_recall@' + str(k), '-'):<10}"
              f"{q.get('dense_recall@' + str(k), '-'):<11}{q['mrr']:<9}"
              f"{q['forbidden_hits']:<11}{str(q['latency_p50']) + 's':<9}"
              f"{str(round(q['latency_p50'] * q['n'], 1)) + 's':<8}")

    # Does each expensive layer beat the cheap default?
    base = layers.get("2")
    if base:
        bq = base["quality"]
        print("\nAgainst L2 (the default path):")
        for name in sorted(layers, key=int):
            if name == "2":
                continue
            q = layers[name]["quality"]
            d_recall = q[f"recall@{k}"] - bq[f"recall@{k}"]
            d_hard = q.get(f"hard_recall@{k}", 0.0) - bq.get(f"hard_recall@{k}", 0.0)
            d_mrr = q["mrr"] - bq["mrr"]
            d_lat = q["latency_p50"] - bq["latency_p50"]
            # Judge on the hard subset first: the easy set cannot move, so a
            # verdict driven by the headline is a verdict driven by noise.
            verdict = ("no measurable gain"
                       if abs(d_hard) < 1e-9 and abs(d_recall) < 1e-9 and abs(d_mrr) < 0.005
                       else "better" if (d_hard > 0 or d_recall > 0 or d_mrr > 0.005)
                       else "worse")
            cost = f"{d_lat:+.2f}s/query"
            print(f"  L{name}: recall {d_recall:+.3f}  hard {d_hard:+.3f}  "
                  f"mrr {d_mrr:+.3f}  latency {cost:<14} -> {verdict}")
        # A saturated baseline cannot show improvement. Saying "no gain" when
        # the metric is already at ceiling is a claim the data cannot support.
        headroom = bq.get("headroom", 0)
        saturated = (bq[f"recall@{k}"] >= 1.0 and bq["mrr"] >= 0.999) or headroom == 0
        if saturated:
            print("\n  WARNING: this corpus cannot test L3/L4.")
            print("  L2: recall@%d=%.3f, MRR=%.3f, and the gold answer falls "
                  "outside the\n  top-%d for %d of %d queries."
                  % (k, bq[f"recall@{k}"], bq["mrr"], k, headroom, bq["n"]))
            print("  A reranker can only help by promoting a record from beyond "
                  "rank %d\n  into it. With no such queries there is nothing for "
                  "it to promote," % k)
            print("  so +0.000 is structurally guaranteed and says nothing about "
                  "the layer.")
            print("  Build queries where L2 ranks the answer 6-15: paraphrases "
                  "that miss\n  the lexical overlap, near-duplicate distractors, "
                  "several plausible\n  answers competing for one query.")
        else:
            print("\n  A layer that costs latency for no measurable gain is a "
                  "deletion candidate,\n  not a tuning problem.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-layer", type=int, default=2,
                    help="highest layer to evaluate (default 2; 3+ calls an LLM)")
    ap.add_argument("-k", type=int, default=5, help="cutoff for recall@k (default 5)")
    ap.add_argument("--floor", type=float, default=0.0,
                    help="score floor applied when measuring the noise rate")
    ap.add_argument("--query-instruction", default=None,
                    help="prefix applied to the query before embedding")
    ap.add_argument("--compare", action="store_true",
                    help="A/B raw queries against an instruction-prefixed variant")
    # Defaults to a haystack. At 0 the corpus is ~40 records and recall@5 sits
    # at ceiling for any pipeline, which is how "L3 shows no gain" got recorded
    # as a property of L3 rather than of the measurement. Use --filler 0 only
    # to iterate quickly on the corpus itself.
    ap.add_argument("--filler", type=int, default=500, metavar="N",
                    help="add N plausible distractor records to the corpus "
                         "(default 500; recall over ~40 records is not a "
                         "meaningful number)")
    ap.add_argument("--spread", action="store_true",
                    help="give records varied age/trust, uncorrelated with "
                         "correctness, so Layer 2's recency and importance "
                         "terms are exercised instead of constant")
    ap.add_argument("--baseline", metavar="PATH",
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "eval-baseline.json"),
                    help="baseline file for --check/--write-baseline")
    ap.add_argument("--write-baseline", action="store_true",
                    help="write this run's metrics to --baseline and exit 0")
    ap.add_argument("--check", action="store_true",
                    help="compare against --baseline and exit 1 on a regression "
                         "beyond --tolerance")
    ap.add_argument("--scoring", action="append", metavar="KEY=VALUE", default=[],
                    help="override a Layer-2 fusion weight, repeatable "
                         "(e.g. --scoring bm25_weight=0.2)")
    ap.add_argument("--tolerance", type=float, default=0.05, metavar="F",
                    help="relative drop tolerated by --check (default 0.05 = 5%%)")
    ap.add_argument("--profile", metavar="NAME",
                    help="load this profile's .env (LAYER3/EMBED settings) before "
                         "building the backend — required to evaluate L3/L4")
    ap.add_argument("--quiet-backend", action="store_true", default=True,
                    help="suppress backend INFO/DEBUG logging so the report is "
                         "readable (default: on; use --verbose-backend to keep it)")
    ap.add_argument("--verbose-backend", dest="quiet_backend", action="store_false",
                    help="keep backend logging on screen")
    ap.add_argument("--json", metavar="PATH", help="write results as JSON")
    args = ap.parse_args()

    if args.profile:
        env_path = os.path.join(pwd.getpwuid(os.getuid()).pw_dir, ".hermes",
                                "profiles", args.profile, ".env")
        if not os.path.exists(env_path):
            print("no .env for profile %r at %s" % (args.profile, env_path),
                  file=sys.stderr)
            return 2
        loaded = []
        for line in open(env_path, encoding="utf-8"):
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if key.startswith("HLM_"):
                os.environ[key] = value.strip().strip('"').strip("'")
                loaded.append(key)
        print("loaded %d settings from profile %r" % (len(loaded), args.profile))

    if args.quiet_backend:
        import logging
        logging.getLogger("backend").setLevel(logging.WARNING)
        for name in list(logging.root.manager.loggerDict):
            if "layered" in name or "backend" in name:
                logging.getLogger(name).setLevel(logging.WARNING)

    # The embedder is the single biggest determinant of these numbers, and
    # without profile settings the backend silently falls back to a local
    # 384-dim model instead of production's 4096-dim endpoint. Numbers from
    # the two are not comparable, so say which one is in play.
    embed_url = os.environ.get("HLM_EMBED_URL")
    embed_model = os.environ.get("HLM_EMBED_MODEL")
    if embed_url and embed_model:
        print(f"embedder: {embed_model} via {embed_url}")
    else:
        print("embedder: LOCAL FALLBACK (no HLM_EMBED_URL/MODEL set)\n"
              "          Different model and dimensionality from a configured\n"
              "          endpoint, so these numbers are not comparable with runs\n"
              "          that had one. Use --profile <name> to load the profile's.")

    if not _check_qdrant_alive():
        print(f"Qdrant not reachable at {QDRANT_URL} — cannot evaluate retrieval.")
        return 2

    scoring = {}
    for item in args.scoring:
        if "=" not in item:
            print(f"bad --scoring {item!r}, expected KEY=VALUE", file=sys.stderr)
            return 2
        key, value = item.split("=", 1)
        try:
            scoring[key.strip()] = float(value)
        except ValueError:
            print(f"bad --scoring value {value!r} for {key!r}", file=sys.stderr)
            return 2
    if scoring:
        print("scoring overrides:", scoring)

    layers = list(range(0, args.max_layer + 1))
    results = []

    if args.compare:
        default_instruction = ("Instruct: Given a question, retrieve stored facts "
                               "that answer it\nQuery: ")
        results.append(_run(None, layers, args.k, args.floor, "baseline (raw query)",
                            "-a", args.filler, scoring, args.spread))
        results.append(_run(args.query_instruction or default_instruction, layers,
                            args.k, args.floor, "query instruction prefix", "-b",
                            args.filler, scoring, args.spread))
        print("\n=== comparison ===")
        for layer in map(str, layers):
            a = results[0]["layers"][layer]["quality"]
            b = results[1]["layers"][layer]["quality"]
            d_recall = b[f"recall@{args.k}"] - a[f"recall@{args.k}"]
            d_mrr = b["mrr"] - a["mrr"]
            print(f"  L{layer}: recall {a[f'recall@{args.k}']:.3f} -> {b[f'recall@{args.k}']:.3f} "
                  f"({d_recall:+.3f})   mrr {a['mrr']:.3f} -> {b['mrr']:.3f} ({d_mrr:+.3f})")
    else:
        results.append(_run(args.query_instruction, layers, args.k, args.floor,
                            "retrieval quality", "", args.filler, scoring, args.spread))

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nWrote {args.json}")

    if args.write_baseline:
        return _write_baseline(results[0], args.baseline, args.k, args.filler,
                               args.spread)
    if args.check:
        return _check_baseline(results[0], args.baseline, args.k, args.tolerance,
                               args.filler, args.spread)
    return 0


# ── Baseline / regression floor ─────────────────────────────────────────────
# A number nobody compares against is a number nobody notices moving. The
# metrics here are only meaningful against the same corpus and the same
# embedder, so both are recorded and a mismatch is reported rather than
# silently compared.

#: Gated. `dense_recall` is ten queries, so one flipping moves it 0.1 — that
#: is sensitivity, not flakiness, and a single dense query regressing is worth
#: failing on: the whole subset exists because nothing else in this corpus
#: could see that class of failure at all.
_BASELINE_METRICS = ("recall@{k}", "hard_recall@{k}", "dense_recall@{k}", "mrr")


def _baseline_fingerprint(k: int, filler: int, spread: bool = False) -> Dict[str, Any]:
    return {
        "corpus_records": len(CORPUS) + filler,
        "golden_records": len(CORPUS),
        "filler": filler,
        "queries": len(QUERIES),
        "hard_queries": sum(1 for q in QUERIES if q.get("hard")),
        "dense_queries": sum(1 for q in QUERIES if q.get("dense")),
        "k": k,
        # --spread is part of what the numbers mean, not a display option: it
        # varies age and trust so Layer 2's recency and importance terms do
        # something, and L2 measures 0.980/0.966/0.882 with it against
        # 0.941/0.897/0.873 without. That is a 4-point gap on recall@5 —
        # comfortably outside the 5% default tolerance in one direction and
        # invisible inside it in the other, so a baseline written one way and
        # checked the other reports a regression that is really a
        # configuration difference. It was absent from the fingerprint, which
        # is exactly how this file came to hold --spread numbers with nothing
        # recording that fact.
        "spread": bool(spread),
        "embedder": os.environ.get("HLM_EMBED_MODEL") or "local-fallback",
    }


#: Largest L2-minus-L0 recall gap a baseline may record. See the refusal in
#: `_write_baseline` for the measurements behind it; the short version is that
#: the two layers differ only by the fusion arm, so a wide gap means the vector
#: arm was dead rather than that fusion is good.
_MAX_FUSION_RECALL_GAP = 0.15


def _write_baseline(result: Dict[str, Any], path: str, k: int, filler: int,
                    spread: bool = False) -> int:
    # L3 and L4 are measured on demand and never baselined. They call an LLM,
    # so their numbers move between runs for reasons that are not code changes,
    # and pinning them produces a flaky gate rather than a strict one — the
    # reasoning is in docs/testing.md and it was documented but not enforced.
    # `--write-baseline` alongside `--max-layer 4` would have written them in
    # silently, and the gate would then have failed on model variance.
    gated = {name: data for name, data in result["layers"].items()
             if int(name) <= 2}
    skipped = sorted(set(result["layers"]) - set(gated), key=int)
    payload = {
        "fingerprint": _baseline_fingerprint(k, filler, spread),
        "layers": {
            name: {m.format(k=k): data["quality"].get(m.format(k=k))
                   for m in _BASELINE_METRICS}
            for name, data in gated.items()
        },
    }
    # Refuse to anchor the gate on a run whose vector arm was dead.
    #
    # 2026-09-17: a `--write-baseline` run recorded L0 recall@5 = 0.177 where
    # the same command twenty minutes earlier measured 0.903 — and wrote it.
    # Nothing objected. The fingerprint matched, because the run used the
    # correct 4096-dim embedder and the degradation was transient and mid-run,
    # so it left no trace there. `T364` would then have gated every future
    # release against a broken reference and passed. **A baseline recorded
    # from a degraded run is worse than no baseline**, which is this repo's
    # standing objection to a green result from a check that did not run.
    #
    # The test is relative rather than a floor, because a floor is a number
    # nobody can justify and would need moving for every corpus change. L0 and
    # L2 differ *only* by the fusion arm, and across both measured corpora that
    # arm has never bought more than 0.02 recall:
    #
    #     healthy 2026-08-12 (700 records)   L0 0.869   L2 0.885   gap 0.016
    #     healthy 2026-09-17 (758 records)   L0 0.903   L2 0.903   gap 0.000
    #     degraded 2026-09-17                L0 0.177   L2 0.558   gap 0.381
    #
    # The degraded signature is that gap blowing open: the vector-only layers
    # collapse while fusion's lexical arm carries the run. The bound sits an
    # order of magnitude above every healthy gap measured and far below the
    # degraded one, so it discriminates the failure and not the corpus.
    _r = "recall@%d" % k
    _l0 = (payload["layers"].get("0") or {}).get(_r)
    _l2 = (payload["layers"].get("2") or {}).get(_r)
    if _l0 is not None and _l2 is not None and (_l2 - _l0) > _MAX_FUSION_RECALL_GAP:
        print(f"\nREFUSING to write a baseline: L2 {_r} ({_l2:.3f}) exceeds "
              f"L0 ({_l0:.3f}) by {_l2 - _l0:.3f}, above the "
              f"{_MAX_FUSION_RECALL_GAP} bound.\n"
              f"  L0 and L2 differ only by the fusion arm, which has never "
              f"bought more than 0.02 recall. A gap this wide means the vector "
              f"arm was not working during this run and the lexical arm "
              f"carried it.\n"
              f"  Check the embedder and Qdrant, then re-run. The existing "
              f"baseline is untouched.")
        return 1

    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"\nWrote baseline to {path}")
    if skipped:
        print("  (measured but not baselined: "
              + ", ".join(f"L{n}" for n in skipped)
              + " — LLM layers are not gated, see docs/testing.md)")
    for name, metrics in sorted(payload["layers"].items()):
        print(f"  L{name}: " + "  ".join(f"{m}={v}" for m, v in sorted(metrics.items())))
    return 0


def _check_baseline(result: Dict[str, Any], path: str, k: int, tol: float,
                    filler: int, spread: bool = False) -> int:
    if not os.path.exists(path):
        print(f"\nNo baseline at {path} — run with --write-baseline first.",
              file=sys.stderr)
        return 2
    with open(path) as fh:
        base = json.load(fh)

    got_fp = _baseline_fingerprint(k, filler, spread)
    want_fp = base.get("fingerprint", {})
    # Compare every field that changes what the numbers mean, `filler`
    # included. This used to build the "got" fingerprint from the *baseline's*
    # filler value, so corpus size could never differ from itself and the two
    # size fields were left out of the comparison below as well — the comment
    # here claimed a different --filler was "caught as a fingerprint mismatch"
    # while the code guaranteed it could not be. Running --filler 0 against a
    # 540-record baseline then compared recall over 40 records with numbers
    # measured over 540 and printed "No regression": five slots for 40
    # candidates is an easier test than five slots for 540, so the gate read
    # green precisely when it had stopped measuring anything.
    for field in ("golden_records", "queries", "hard_queries", "dense_queries",
                  "k", "embedder", "filler", "corpus_records", "spread"):
        if want_fp.get(field) != got_fp.get(field):
            hint = ("  Re-run with --write-baseline to re-anchor "
                    "(and say so in the commit).")
            # An embedder mismatch usually means the wrong baseline file, not a
            # corpus change — the numbers are per-model and there is a baseline
            # per model. Suggesting --write-baseline here is actively harmful:
            # it overwrites the *other* model's anchor with these numbers, and
            # the operator following the advice destroys the file they were not
            # measuring. `--check` with no --baseline defaults to the
            # local-fallback anchor, so on any host configured the way CLAUDE.md
            # instructs (production embedder exported) the documented bare
            # command lands here. T364 dodges it by naming the prod baseline
            # explicitly, so the suite stays green and only a human following
            # the docs meets the trap.
            # 2026-08-27 external MCP validation, F-3.
            if field == "embedder":
                import glob as _glob
                here = os.path.dirname(os.path.abspath(__file__))
                for cand in sorted(_glob.glob(os.path.join(here, "eval-baseline*.json"))):
                    try:
                        with open(cand, encoding="utf-8") as fh:
                            fp = json.load(fh).get("fingerprint", {})
                    except Exception:
                        continue
                    if fp.get("embedder") == got_fp.get("embedder"):
                        hint = ("  This looks like the wrong baseline file rather "
                                "than a real change.\n  Re-run with --baseline %s"
                                % os.path.relpath(cand, os.getcwd()))
                        break
            print(f"\nBASELINE MISMATCH: {field} was {want_fp.get(field)!r}, "
                  f"now {got_fp.get(field)!r}.\n"
                  f"  The corpus, query set or embedder changed, so the numbers are "
                  f"not comparable.\n{hint}", file=sys.stderr)
            return 2

    failures = []
    print(f"\n{'=' * 68}\nBASELINE CHECK (tolerance {tol:.0%})\n{'=' * 68}")
    for name, data in sorted(result["layers"].items()):
        want = base.get("layers", {}).get(name)
        if not want:
            print(f"  L{name}: no baseline entry — skipped")
            continue
        for metric in _BASELINE_METRICS:
            key = metric.format(k=k)
            old, new = want.get(key), data["quality"].get(key)
            if old is None or new is None:
                continue
            floor = old * (1 - tol)
            ok = new >= floor
            print(f"  L{name} {key:<16} {old:.3f} -> {new:.3f} "
                  f"(floor {floor:.3f})  {'ok' if ok else 'REGRESSION'}")
            if not ok:
                failures.append(f"L{name} {key}: {old:.3f} -> {new:.3f}")

    if failures:
        print("\nRETRIEVAL REGRESSION:", file=sys.stderr)
        for f in failures:
            print("  " + f, file=sys.stderr)
        return 1
    print("\nNo regression.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
