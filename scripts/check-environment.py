#!/usr/bin/env python3
"""Fresh clone to working state, and say which part is not working.

`docs/consider-features.md` #41 asked for one-command setup and deferred it
with the right reason: the value is not the install step, it is the
**validation** step. This repo's most expensive documented trap is a suite that
reports green while every vector call runs on the 384-dim MiniLM fallback,
because two environment variables were not exported — and the counter-measure
has been a paragraph in `AGENTS.md` telling the reader to export them.

So this is the validation half, written first. It is a script and not a skill
on purpose: a skill is instructions an agent may skip or misread, and **only a
script can fail**.

**Two kinds of question, and they must not share a verdict.** `AGENTS.md` is
explicit about this after paying for it three times in three releases: a check
that answers "about the repo" (same on every machine — assert it) and one that
answers "about this machine" (print it) cannot both drive one exit code.

    --check   this machine's readiness. Exits 1 if something the suite needs
              is missing, so a human or a CI step can gate on it.
    --repo    the repo's own invariants. Exits 1 on those. Safe anywhere.

With no flag, both run and the exit code is the worse of the two, which is what
a person typing one command wants.

Deliberately does **not** install anything. The deferral note observed that an
unrun script rots; an installer that mutates a working machine rots dangerously.
`T690` keeps the checks honest by requiring each named symbol to resolve.
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

OK, WARN, BAD = "ok  ", "warn", "FAIL"


def _say(state, label, detail=""):
    print(f"  [{state}] {label}" + (f" — {detail}" if detail else ""))
    return state != BAD


def repo_checks() -> bool:
    """About the repo. The answer is the same on every machine."""
    print("repo:")
    good = True

    want = ["plugin.yaml", "backend/constants.py", ".env.example",
            "docker-compose.yml", "tests/run-regression.py"]
    missing = [f for f in want if not os.path.exists(os.path.join(ROOT, f))]
    good &= _say(OK if not missing else BAD, "expected files present",
                 "" if not missing else f"missing {missing}")

    try:
        from backend import LayeredBackend  # noqa: F401
        good &= _say(OK, "backend imports")
    except Exception as e:
        good &= _say(BAD, "backend imports", f"{type(e).__name__}: {e}")

    try:
        import subprocess
        r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "check-version.py")],
                           capture_output=True, text=True)
        good &= _say(OK if r.returncode == 0 else BAD, "version files agree",
                     r.stdout.strip() or r.stderr.strip())
    except Exception as e:
        good &= _say(BAD, "version files agree", str(e))
    return good


def machine_checks() -> bool:
    """About this machine. Never asserted into the repo's verdict."""
    print("machine:")
    good = True

    url = os.environ.get("HLM_EMBED_URL")
    model = os.environ.get("HLM_EMBED_MODEL")
    if not (url and model):
        good &= _say(BAD, "HLM_EMBED_URL / HLM_EMBED_MODEL exported",
                     "unset — the suite would run on the 384-dim MiniLM "
                     "fallback and report green while exercising the fallback "
                     "paths. This is the trap this script exists for.")
    else:
        _say(OK, "embedding env exported", f"{model} via {url}")
        try:
            import json
            import urllib.request
            req = urllib.request.Request(
                url, data=json.dumps({"model": model, "input": ["probe"]}).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                body = json.loads(resp.read())
            vec = (body.get("embeddings") or [body.get("embedding")])[0]
            good &= _say(OK if vec else BAD, "embedder answers",
                         f"{len(vec)} dimensions" if vec else "no vector in the response")
        except Exception as e:
            good &= _say(BAD, "embedder answers", f"{type(e).__name__}: {e}")

    qurl = os.environ.get("HLM_QDRANT_URL", "http://localhost:6333")
    try:
        import urllib.request
        with urllib.request.urlopen(qurl + "/collections", timeout=10) as resp:
            n = len((__import__("json").loads(resp.read()))["result"]["collections"])
        _say(OK, "qdrant reachable", f"{qurl}, {n} collections")
    except Exception as e:
        good &= _say(BAD, "qdrant reachable", f"{qurl}: {type(e).__name__}")

    # Reported, never asserted: an optional capability whose absence is not a
    # broken environment. `T215` gates what the suite actually needs.
    if os.environ.get("HLM_LAYER3_MODEL") and os.environ.get("HLM_LAYER3_BASE_URL"):
        _say(OK, "layer-3 LLM configured", os.environ["HLM_LAYER3_MODEL"])
    else:
        _say(WARN, "layer-3 LLM not configured",
             "L3/L4 and enrichment are unavailable; the suites still pass")
    return good


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="this machine only")
    ap.add_argument("--repo", action="store_true", help="the repo only")
    args = ap.parse_args()

    run_repo = args.repo or not args.check
    run_machine = args.check or not args.repo

    ok = True
    if run_repo:
        ok &= repo_checks()
    if run_machine:
        if run_repo:
            print()
        ok &= machine_checks()

    print()
    print("READY" if ok else "NOT READY — fix the FAIL lines above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
