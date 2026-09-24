#!/usr/bin/env python3
"""Ask what you want, then print the install plan. Change nothing.

`docs/consider-features.md` #41's second half. `scripts/check-environment.py`
answers *is this working*; this answers *what do I have to do*. It asks a short
series of questions, then emits a numbered plan: the commands to run, the exact
`.env` block to write, and what to verify at the end.

**It is advisory on purpose.** It creates no profile, writes no file, starts no
container. The deferral note that opened #41 observed that an unrun script
rots; an installer that mutates a working machine rots dangerously, and this
repo's own history has two incidents that began with a helper acting on a
profile it had resolved wrongly. Output you can read before running is worth
more than a step saved.

Every `HLM_*` key it can emit is checked against `.env.example` by `T691`, so
the plan cannot quietly recommend a variable that was renamed out of the code —
which is exactly what the `HERMES_LAYERED_*` → `HLM_*` rename did to eight
files (`T630`).

    python3 scripts/plan-install.py                      # interactive
    python3 scripts/plan-install.py --profile p --embed local --qdrant docker \
        --no-llm                                         # scripted
"""
import argparse
import os
import pwd
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_HOME = pwd.getpwuid(os.getuid()).pw_dir


def ask(prompt, default=None, choices=None):
    if not sys.stdin.isatty():
        raise SystemExit(
            f"need an answer for {prompt!r} but stdin is not a terminal — "
            f"pass it as a flag (see --help)")
    suffix = f" [{default}]" if default else ""
    if choices:
        suffix = f" ({'/'.join(choices)})" + suffix
    while True:
        got = input(f"{prompt}{suffix}: ").strip() or (default or "")
        if choices and got not in choices:
            print(f"  choose one of: {', '.join(choices)}")
            continue
        if got:
            return got


def collect(args):
    """Every answer the plan depends on, from flags or from the operator."""
    a = {}
    a["profile"] = args.profile or ask("Hermes profile name")

    a["embed"] = args.embed or ask(
        "Embeddings: 'local' 384-dim sentence-transformers (no service), "
        "'fastembed' 1024-dim offline, or 'remote' HTTP endpoint",
        default="local", choices=("local", "fastembed", "remote"))
    if a["embed"] == "remote":
        a["embed_url"] = args.embed_url or ask(
            "  embedding endpoint URL", default="http://localhost:11434/api/embed")
        a["embed_model"] = args.embed_model or ask(
            "  embedding model", default="qwen3-embedding:8b")
    elif a["embed"] == "fastembed":
        a["local_model"] = args.local_embed_model or ask(
            "  FastEmbed model", default="intfloat/multilingual-e5-large")

    a["qdrant"] = args.qdrant or ask(
        "Qdrant: 'docker' start one here, 'url' use an existing server, "
        "or 'off' for SQLite-only",
        default="docker", choices=("docker", "url", "off"))
    if a["qdrant"] == "url":
        a["qdrant_url"] = args.qdrant_url or ask(
            "  Qdrant URL", default="http://localhost:6333")

    if args.no_llm:
        a["llm"] = False
    elif args.llm_model:
        a["llm"] = True
        a["llm_model"] = args.llm_model
        a["llm_url"] = args.llm_url or ask("  LLM base URL (OpenAI-compatible)")
        a["llm_key"] = args.llm_key or ""
    else:
        a["llm"] = ask("Configure the layer-3 LLM (reranking, gap detection, "
                       "enrichment)? Without it retrieval is capped at depth 2",
                       default="n", choices=("y", "n")) == "y"
        if a["llm"]:
            a["llm_model"] = ask("  LLM model name")
            a["llm_url"] = ask("  LLM base URL (OpenAI-compatible)",
                               default="http://localhost:8000/v1")
            a["llm_key"] = ask("  API key (blank if the endpoint needs none)",
                               default="none")
            if a["llm_key"] == "none":
                a["llm_key"] = ""
    a["exists"] = os.path.isdir(os.path.join(REAL_HOME, ".hermes", "profiles",
                                             a["profile"]))
    return a


def env_block(a):
    """The .env lines, as (key, value, why) — the plan's centrepiece."""
    p = a["profile"]
    out = [("HLM_DB_PATH", f"~/.hermes/hermes-layered-memory-dbs/{p}.db",
            "this profile's SQLite store; it is the source of truth")]

    if a["qdrant"] == "off":
        out.append(("HLM_QDRANT_ENABLED", "false",
                    "SQLite-only: retrieval falls back to a cosine scan over the "
                    "embeddings SQLite already holds, and write-time dedup is bypassed"))
    else:
        out.append(("HLM_QDRANT_URL", a.get("qdrant_url", "http://localhost:6333"),
                    "vector search, the primary retrieval path"))

    if a["embed"] == "remote":
        out.append(("HLM_EMBED_URL", a["embed_url"], "remote embedding endpoint"))
        out.append(("HLM_EMBED_MODEL", a["embed_model"],
                    "MUST be set together with the URL — one without the other "
                    "silently falls back to the 384-dim local model"))
    elif a["embed"] == "fastembed":
        out.append(("HLM_LOCAL_EMBED_MODEL", a["local_model"],
                    "offline FastEmbed; overrides the sentence-transformers fallback"))

    out.append(("HLM_MAX_LAYER", "2",
                "fusion scoring. Below 2 the lexical arm cannot reach the caller"))

    if a["llm"]:
        out.append(("HLM_LAYER3_MODEL", a["llm_model"], "layer-3 reranker model"))
        out.append(("HLM_LAYER3_BASE_URL", a["llm_url"], "OpenAI-compatible endpoint"))
        if a["llm_key"]:
            out.append(("HLM_LAYER3_API_KEY", a["llm_key"], "endpoint credential"))
        out.append(("HLM_REASONING_EFFORT", "none",
                    "reasoning models otherwise spend the whole budget thinking "
                    "and return content: null"))
    out.append(("HLM_LOG", "INFO", "DEBUG for layer-by-layer pipeline visibility"))
    return out


def render(a):
    p = a["profile"]
    prof_dir = f"~/.hermes/profiles/{p}"
    n = [0]

    def step(title):
        n[0] += 1
        print(f"\n{n[0]}) {title}")

    print("=" * 74)
    print(f"INSTALL PLAN — profile {p!r}")
    print("=" * 74)
    print("\nNothing below has been done. Run the steps yourself, in order.")

    if a["exists"]:
        step(f"The profile already exists at {prof_dir} — skip creation.")
    else:
        step("Create the profile.")
        print(f"     hermes profile create {p} --no-skills")
        print("   A profile does NOT inherit the root model config. Until you set")
        print("   one, the first chat exits with \"No inference provider configured\"")
        print("   before HLM is reached, which looks like a plugin failure and is not:")
        print(f"     hermes -p {p} config set model.default  <model-id>")
        print(f"     hermes -p {p} config set model.provider custom")
        print(f"     hermes -p {p} config set model.base_url <http://host:8000/v1>")
        print(f"     hermes -p {p} config set model.api_mode chat_completions")

    step("Symlink this checkout into the profile's plugins directory.")
    print(f"     mkdir -p {prof_dir}/plugins")
    print(f"     ln -sf {ROOT} \\")
    print(f"        {prof_dir}/plugins/hermes-layered-memory")
    print("   Symlink, not a copy: a copy drifts and nothing in the repo notices.")

    if a["qdrant"] == "docker":
        step("Start Qdrant.")
        print(f"     cd {ROOT} && docker compose up -d && sleep 3")
        print("     curl -s http://localhost:6333/    # returns version info")
        print("   Data persists in ./qdrant-data/. Ports 6333 (HTTP), 6334 (gRPC).")
    elif a["qdrant"] == "url":
        step("Confirm the existing Qdrant answers.")
        print(f"     curl -s {a['qdrant_url']}/collections")
    else:
        step("No Qdrant. Nothing to start — the .env below disables it.")
        print("   If you enable Qdrant later, run layered_advanced(action=\"compact\")")
        print("   once: write-time dedup was bypassed while it was off.")

    step("Enable the plugin and point Hermes' memory at it.")
    for line in (f"hermes -p {p} plugins enable hermes-layered-memory",
                 f"hermes -p {p} config set memory.memory_enabled false",
                 f"hermes -p {p} config set memory.user_profile_enabled false",
                 f"hermes -p {p} config set memory.write_approval false",
                 f"hermes -p {p} config set memory.provider hermes-layered-memory"):
        print(f"     {line}")
    print("   `plugins enable` may report it could not grant a tool override.")
    print("   Harmless here: HLM registers a memory provider and overrides no tool.")

    step("Copy the JSON config (collections, scoring — rarely edited).")
    print(f"     cp {ROOT}/hermes-layered-memory.example.json \\")
    print(f"        {prof_dir}/hermes-layered-memory.json")

    step(f"Create {prof_dir}/.env with exactly these lines:")
    print()
    width = max(len(k) for k, _, _ in env_block(a))
    for k, v, why in env_block(a):
        print(f"     {k}={v}")
    print("\n   What each one is for:")
    for k, _, why in env_block(a):
        print(f"     {k.ljust(width)}  {why}")
    if a["embed"] == "local":
        print("\n   NOTE: no HLM_EMBED_* here — you chose the built-in 384-dim")
        print("   sentence-transformers model. That is a real choice, not a gap.")
        print("   If you later add a remote endpoint, the physical Qdrant")
        print("   collection changes with the model, so run")
        print("   layered_maintenance(action=\"rebuild\") after switching.")

    step("Verify, before trusting it.")
    print("     python3 scripts/check-environment.py")
    if a["embed"] == "remote":
        print(f"     # export HLM_EMBED_URL / HLM_EMBED_MODEL in the shell first —")
        print(f"     # the check reads the environment, not the profile's .env.")
    print("   Then, for the full suite (~7-18 min, background it):")
    print("     python3 tests/run-regression.py")

    step("Start a session.")
    print(f"     hermes -p {p} chat -q \"remember that the deploy host is host-01\"")
    print("   The plugin initialises lazily on the first tool call.")
    print("\n" + "=" * 74)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile")
    ap.add_argument("--embed", choices=("local", "fastembed", "remote"))
    ap.add_argument("--embed-url")
    ap.add_argument("--embed-model")
    ap.add_argument("--local-embed-model")
    ap.add_argument("--qdrant", choices=("docker", "url", "off"))
    ap.add_argument("--qdrant-url")
    ap.add_argument("--llm-model")
    ap.add_argument("--llm-url")
    ap.add_argument("--llm-key", default="")
    ap.add_argument("--no-llm", action="store_true")
    render(collect(ap.parse_args()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
