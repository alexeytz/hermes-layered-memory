#!/usr/bin/env python3
"""Measure which reasoning-off spellings actually work on an endpoint.

`backend/llm.py` sends a superset of five keys to permissive endpoints, on the
theory that a server ignores what it does not implement. That theory says
nothing about which key *works*, and the distinction is the whole game: a key
that is accepted and ignored looks exactly like success, so a "try each until
one doesn't error" loop stops at the first inert one and the model keeps
reasoning at full cost.

This turns that question into a command. Nothing here encodes knowledge about
any model — it asks the endpoint in front of you, which is the only thing that
cannot go stale. Re-run it when the served model changes.

    python3 scripts/measure-reasoning-suppression.py \
        --base-url http://your-vllm-host:8000/v1 --model YOUR-MODEL

Reads the signal from **completion tokens**, not from `reasoning_content`.
Measured 2026-08-20 on a Qwen served by vLLM: the model emitted its reasoning
as ordinary content and left `reasoning_content` empty, so a check for that
field would have scored every spelling — including the three inert ones — as a
success. The token count separated them cleanly (96 -> 5).

Exit status is 1 when nothing suppressed, so this can gate a deployment.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

#: A question the model answers instantly but is tempted to reason about. The
#: bat-and-ball is chosen because the intuitive answer is wrong, which is what
#: provokes a reasoning model into showing its work.
PROMPT = ("A bat and a ball cost $1.10 in total. The bat costs $1.00 more than "
          "the ball. How much does the ball cost? Reply with just the number.")

#: Mirrors _PERMISSIVE_REASONING_OFF in backend/llm.py, plus the spellings that
#: are plausible but deliberately not shipped — the point is to find out.
TRIALS: list[tuple[str, dict]] = [
    ("baseline (no keys)", {}),
    ("reasoning_effort='none'", {"reasoning_effort": "none"}),
    ("thinking=null", {"thinking": None}),
    ("chat_template_kwargs.enable_thinking", {"chat_template_kwargs": {"enable_thinking": False}}),
    ("think=false", {"think": False}),
    ("reasoning={'exclude':True}", {"reasoning": {"exclude": True}}),
    ("thinking='none' (string, not shipped)", {"thinking": "none"}),
    ("reasoning_effort='minimal' (strict hosts)", {"reasoning_effort": "minimal"}),
    ("ALL — what HLM sends", {"reasoning_effort": "none", "thinking": None,
                              "chat_template_kwargs": {"enable_thinking": False},
                              "think": False, "reasoning": {"exclude": True}}),
]


def _call(base_url: str, model: str, extra: dict, timeout: float) -> dict:
    body = {"model": model, "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": 700, "temperature": 0}
    body.update(extra)
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    started = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:110].replace("\n", " ")
        return {"ok": False, "error": f"HTTP {e.code}: {detail}"}
    except Exception as e:  # network, timeout, malformed JSON
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    message = payload.get("choices", [{}])[0].get("message", {}) or {}
    return {
        "ok": True,
        "latency": round(time.time() - started, 2),
        "tokens": (payload.get("usage") or {}).get("completion_tokens"),
        "reasoning_chars": len(message.get("reasoning_content") or ""),
        "answer": (message.get("content") or "").strip().replace("\n", " ")[:32],
    }


def verdict(results: list[dict], base_tokens: int) -> tuple[list[str], dict | None]:
    """Turn measured results into the change an operator should actually make.

    Split out of main() so all four outcomes can be tested. Only the
    "nothing to do" branch had ever executed — the other three were written
    from reasoning about outcomes that had not happened, and one of them named
    a setting that does not exist (there is no `layer3_reasoning_style` value
    that sends `chat_template_kwargs` alone; the four values are auto, openai,
    permissive, off).

    Returns `(lines, recommendation)`. The recommendation is what `--apply`
    acts on: `{"style": ..., "auto": bool}`, or None when there is nothing to
    change. `auto` is False where a human has to decide first — "nothing
    suppressed" can mean a model that cannot be switched off *or* a spelling
    this script does not know, and those want opposite responses, so --apply
    refuses rather than guessing.

    Lines name the exact call and env var rather than the setting, because
    "use layer3_reasoning_style=openai" still leaves the reader to work out
    where to put it.
    """
    def _how(style: str) -> list[str]:
        return [
            f'    layered_config(action="set", key="layer3_reasoning_style", value="{style}")',
            f"    …or set HLM_REASONING_STYLE={style} in the profile's .env and restart.",
        ]

    shipped = next((r for r in results if r["trial"].startswith("ALL")), None)
    singles = [r for r in results if r["effective"]
               and not r["trial"].startswith(("baseline", "ALL"))]
    keys = {k for r in singles for k in r["keys"]}
    out: list[str] = []
    rec: dict | None = None

    if singles:
        out.append("Effective spellings on this endpoint:")
        for r in singles:
            out.append(f"  - {r['trial']}  ({r['tokens']} tok vs {base_tokens} baseline)")
    else:
        out.append("No single spelling suppressed reasoning here.")
    out.append("")
    out.append("What to do about it:")

    if shipped and shipped.get("effective"):
        out.append("  Nothing — HLM's default already suppresses on this endpoint.")
        out.append("  layer3_reasoning_style=auto sends the permissive superset to a")
        out.append("  non-OpenAI host, and that is what you measured above.")
        if keys:
            out.append(f"  Carried by: {', '.join(sorted(keys))}. The rest of the superset is")
            out.append("  inert here and costs nothing — keep it; it is what makes the same")
            out.append("  default work on other providers.")
        rec = None
    elif singles and keys == {"reasoning_effort"}:
        out.append("  Switch to the strict style — reasoning_effort is the only key that")
        out.append("  works here, and sending the rest risks a rejection that takes it")
        out.append("  down with them:")
        out.extend(_how("openai"))
        rec = {"style": "openai", "auto": True}
    elif singles:
        # A key works alone but the shipped superset did not. That is not a
        # config problem: _get_new_conn's sibling, the [L005] repair path, is
        # supposed to drop only the key an error names and retry with the rest,
        # so a working key inside a failing superset means the repair did not
        # recover. Say that, rather than inventing a style to pin.
        out.append("  This is a bug report, not a config change. A key works on its own")
        out.append(f"  ({', '.join(sorted(keys))}) while the full superset did not, so the")
        out.append("  [L005] repair path — which should drop only the key the endpoint")
        out.append("  named and retry with the others — did not recover. Grep the HLM log")
        out.append("  for [L005] to see which key was blamed, and file that.")
        out.append("  Meanwhile, to keep suppression working:")
        _interim = "permissive" if "chat_template_kwargs" in keys else "openai"
        out.extend(_how(_interim))
        rec = {"style": _interim, "auto": True}
    else:
        out.append("  Nothing here suppressed reasoning. Two possibilities, and they")
        out.append("  need different responses:")
        out.append("   1. The model cannot be switched off. Qwen3-Thinking-2507 is")
        out.append("      documented that way, and Qwen3-Instruct-2507 does not support")
        out.append("      the flag because it never reasons in the first place — check")
        out.append("      which variant this is before assuming a defect.")
        out.append("   2. It wants a spelling this script does not know.")
        out.append("  If it is (1), stop paying for keys that do nothing:")
        out.extend(_how("off"))
        out.append("  and budget for the reasoning latency — it is a property of the")
        out.append("  model you chose, not a misconfiguration.")
        # Not auto-applicable: (1) and (2) above want opposite responses and
        # only a human can tell them apart. Setting `off` when the truth is (2)
        # would turn a fixable misconfiguration into a permanent one.
        rec = {"style": "off", "auto": False}
    return out, rec


def apply_style(style: str, db_path: str, *, dry_run: bool = False) -> int:
    """Write `layer3_reasoning_style` into a profile's runtime config.

    Behind `--apply` and off by default, deliberately. A measurement tool that
    silently rewrites config is one you have to be warned about before running,
    and it destroys its own reproducibility: re-running it would no longer
    report the state you were in, because running it changed that state.

    Uses `backend.set_config()`, the same path `layered_config(action="set")`
    takes, so validation and the JSON sync are not reimplemented here.
    """
    import os
    import sys as _sys
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo not in _sys.path:
        _sys.path.insert(0, repo)
    try:
        from backend import LayeredBackend
    except Exception as e:
        print(f"--apply needs the backend importable from {repo}: {e}", file=_sys.stderr)
        return 2
    if not os.path.exists(db_path):
        print(f"--apply: no database at {db_path}", file=_sys.stderr)
        return 2

    be = LayeredBackend(db_path=db_path, profile_name=os.path.basename(db_path)[:-3])
    try:
        before = (be.get_config(key="layer3_reasoning_style") or {}).get("value")
        print(f"\n  current layer3_reasoning_style: {before!r}")
        print(f"  writing:                       {style!r}   ({db_path})")
        if dry_run:
            print("  --dry-run: nothing written.")
            return 0
        result = be.set_config("layer3_reasoning_style", style)
        if isinstance(result, dict) and result.get("error"):
            print(f"  REFUSED by set_config: {result['error']}", file=_sys.stderr)
            return 1
        after = (be.get_config(key="layer3_reasoning_style") or {}).get("value")
        print(f"  read back:                     {after!r}")
        if after != style:
            print("  write did not stick — check for an env override "
                  "(HLM_REASONING_STYLE wins over the DB).", file=_sys.stderr)
            return 1
        print("  applied.")
        # Config is cached per backend instance, so a Hermes session that is
        # already running keeps the old value until it re-initializes. Writing
        # the fix and watching the next call reason anyway is a confusing way
        # to learn that, so say it here.
        print("  NOTE: a session already running holds its own cached config —")
        print("        restart it (or start a new one) for this to take effect.")
        return 0
    finally:
        be.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", required=True, help="OpenAI-compatible /v1 base URL")
    ap.add_argument("--model", required=True, help="model id as the endpoint names it")
    ap.add_argument("--timeout", type=float, default=240.0)
    ap.add_argument("--json", action="store_true", help="emit machine-readable results")
    ap.add_argument("--apply", action="store_true",
                    help="write the recommended layer3_reasoning_style into a profile's "
                         "config. Off by default; measurement never changes state.")
    ap.add_argument("--dry-run", action="store_true",
                    help="with --apply, show the write without performing it")
    ap.add_argument("--db-path", help="profile DB to write to with --apply "
                                      "(default: ~/.hermes/hermes-layered-memory-dbs/<profile>.db)")
    ap.add_argument("--profile", help="profile name, used to derive --db-path")
    args = ap.parse_args()

    results = []
    for label, extra in TRIALS:
        r = _call(args.base_url, args.model, extra, args.timeout)
        r["trial"] = label
        r["keys"] = sorted(extra)
        results.append(r)

    baseline = next((r for r in results if r["trial"].startswith("baseline")), None)
    if not baseline or not baseline.get("ok"):
        print("baseline call failed — cannot judge anything else:", file=sys.stderr)
        print(f"  {baseline.get('error') if baseline else 'no baseline'}", file=sys.stderr)
        return 2
    base_tokens = baseline.get("tokens") or 0

    # A spelling counts as effective when it cuts the answer to a fraction of
    # the baseline. Half is deliberately generous: suppression on a reasoning
    # model is a cliff (96 -> 5 when measured), not a slope, so anything in
    # between is worth a human look rather than a verdict.
    for r in results:
        r["effective"] = bool(r.get("ok") and base_tokens
                              and (r.get("tokens") or 0) <= base_tokens * 0.5)

    if args.json:
        print(json.dumps({"base_url": args.base_url, "model": args.model,
                          "baseline_tokens": base_tokens, "results": results}, indent=2))
    else:
        print(f"\n{args.model} @ {args.base_url}   baseline {base_tokens} tokens\n")
        print(f"{'trial':<42} {'ok':<4} {'lat':>6} {'tok':>5} {'reas':>6}  verdict")
        for r in results:
            if not r["ok"]:
                print(f"{r['trial']:<42} {'NO':<4} {'-':>6} {'-':>5} {'-':>6}  {r['error'][:44]}")
                continue
            # Named `mark`, not `verdict`: the latter is the module-level
            # function that turns these rows into an action, and a local of
            # the same name shadowed it — caught immediately by running the
            # script, which is the only reason to run it before shipping.
            mark = "EFFECTIVE" if r["effective"] else "inert (accepted, ignored)"
            if r["trial"].startswith("baseline"):
                mark = "—"
            print(f"{r['trial']:<42} {'yes':<4} {r['latency']:>6} "
                  f"{str(r['tokens']):>5} {r['reasoning_chars']:>6}  {mark}")
        lines, rec = verdict(results, base_tokens)
        for line in lines:
            print(line)

    if args.apply:
        _lines, rec = verdict(results, base_tokens)
        if not rec:
            print("\n--apply: nothing to change; the shipped default already works here.")
        elif not rec.get("auto"):
            print("\n--apply refused. The recommendation above depends on a judgement "
                  "this script cannot make —\n  whether the model can be switched off at "
                  "all, or whether it wants a spelling not tested here.\n  Decide, then set "
                  "it deliberately.", file=sys.stderr)
            return 1
        else:
            import os
            db = args.db_path
            if not db:
                prof = args.profile
                if not prof:
                    print("\n--apply needs --db-path or --profile.", file=sys.stderr)
                    return 2
                db = os.path.expanduser(
                    f"~/.hermes/hermes-layered-memory-dbs/{prof}.db")
            rc = apply_style(rec["style"], db, dry_run=args.dry_run)
            if rc:
                return rc

    return 0 if any(r["effective"] and not r["trial"].startswith("baseline")
                    for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
