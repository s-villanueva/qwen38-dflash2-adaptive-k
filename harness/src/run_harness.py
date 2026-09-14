#!/usr/bin/env python3
"""Bounded local-API runner for a GVS5H-style ledger workflow.

The GVS5H dependency is installed separately by harness/scripts/bootstrap_gvs5h.sh.
This wrapper must be used only with an isolated experimental endpoint.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import os
import pathlib
import sys
from typing import Any


HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent


def required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"{name} must be set; source harness/config/harness.env first")
    return value


def configure(run_root: pathlib.Path) -> dict[str, Any]:
    base_url = required("QWEN_HARNESS_BASE_URL").rstrip("/")
    model = required("QWEN_HARNESS_MODEL")
    escalation = pathlib.Path(required("GVS5H_ESCALATION_DIR")).expanduser().resolve()
    if not (escalation / "multiagent.py").is_file() or not (escalation / "orchestrator.py").is_file():
        raise SystemExit("GVS5H_ESCALATION_DIR is not a patched GVS5H escalation directory")
    max_iters = int(os.environ.get("QWEN_HARNESS_MAX_ITERS", "3"))
    max_tasks = int(os.environ.get("QWEN_HARNESS_MAX_TASKS", "8"))
    max_tokens = int(os.environ.get("QWEN_HARNESS_MAX_TOKENS", "2048"))
    timeout = int(os.environ.get("QWEN_HARNESS_TIMEOUT_SECONDS", "1800"))
    if not 1 <= max_iters <= 10:
        raise SystemExit("QWEN_HARNESS_MAX_ITERS must be in 1..10")
    if not 1 <= max_tasks <= 12:
        raise SystemExit("QWEN_HARNESS_MAX_TASKS must be in 1..12")
    if not 128 <= max_tokens <= 8192:
        raise SystemExit("QWEN_HARNESS_MAX_TOKENS must be in 128..8192")

    # The GVS5H generic OpenAI adapter is selected through its groq: prefix.
    # The placeholder is local-only and never represents provider credentials.
    os.environ.update(
        {
            "ESCALATION_OPENAI_BASE": f"{base_url}/chat/completions",
            "GROQ_API_KEY": "local-harness-placeholder",
            "ESCALATION_GROQ_REASONING": "",
            "ESCALATION_CHAT_TEMPLATE_KWARGS": os.environ.get(
                "QWEN_HARNESS_CHAT_TEMPLATE_KWARGS", '{"enable_thinking": false}'
            ),
            "ESCALATION_CLOUD_MAX_TOKENS": str(max_tokens),
            "ESCALATION_CLOUD_TIMEOUT": str(timeout),
            "MULTIAGENT_MODEL": f"groq:{model}",
            "MULTIAGENT_MAX_ITERS": str(max_iters),
            "MULTIAGENT_MAX_TASKS": str(max_tasks),
            "MULTIAGENT_STRICT_FORMAT": "1",
            "MULTIAGENT_WS": str(run_root / "workspaces"),
        }
    )
    if str(escalation) not in sys.path:
        sys.path.insert(0, str(escalation))
    return {
        "model": model,
        "max_iters": max_iters,
        "max_tasks": max_tasks,
        "max_tokens_per_call": max_tokens,
        # planner + ideation + manager, up to I*(worker + cutoff + manager), finalizer
        "max_calls_per_task": 4 + (3 * max_iters),
        "timeout_seconds": timeout,
        "dflash_schedule_expected": "[[1,1,12],[2,4,5],[5,64,0]]",
    }


def load_jobs(args: argparse.Namespace) -> list[dict[str, str]]:
    if bool(args.task) == bool(args.tasks_jsonl):
        raise SystemExit("provide exactly one of --task or --tasks-jsonl")
    if args.task:
        return [{"id": "single", "prompt": args.task, "kind": args.kind}]
    jobs: list[dict[str, str]] = []
    with open(args.tasks_jsonl, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            prompt = str(item.get("prompt", "")).strip()
            kind = str(item.get("kind", args.kind)).strip().lower()
            if not prompt or kind not in {"code", "math"}:
                raise SystemExit(f"invalid task at JSONL line {line_number}")
            jobs.append({"id": str(item.get("id", line_number)), "prompt": prompt, "kind": kind})
    if not jobs:
        raise SystemExit("no tasks found")
    return jobs


def safe_id(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value)[:80] or "task"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task")
    parser.add_argument("--tasks-jsonl")
    parser.add_argument("--kind", choices=("code", "math"), default="code")
    parser.add_argument("--max-active", type=int, default=int(os.environ.get("QWEN_HARNESS_MAX_ACTIVE", "2")))
    args = parser.parse_args()
    if not 1 <= args.max_active <= 4:
        raise SystemExit("--max-active must be in 1..4; 1-2 is recommended")

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = ROOT / "runs" / stamp
    run_root.mkdir(parents=True, exist_ok=False)
    config = configure(run_root)
    jobs = load_jobs(args)
    from multiagent import multiagent_solve
    from orchestrator import CODE_SPEC, MATH_SPEC

    manifest = {
        "started_utc": stamp,
        "mode": "isolated-qwen-ledger-harness",
        "source": "GVS5H-adapted",
        "config": config,
        "max_active": args.max_active,
        "task_count": len(jobs),
        "verification": "disabled: no generated code is executed by this wrapper",
    }
    (run_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    def run_job(job: dict[str, str]) -> dict[str, Any]:
        status: dict[str, Any] = {}
        spec = CODE_SPEC if job["kind"] == "code" else MATH_SPEC
        try:
            response = multiagent_solve(job["prompt"], spec, status_out=status, tests=None)
            result = {"id": job["id"], "kind": job["kind"], "ok": bool(response.strip()), "status": status}
            (run_root / f"{safe_id(job['id'])}.response.md").write_text(response, encoding="utf-8")
        except Exception as exc:
            result = {"id": job["id"], "kind": job["kind"], "ok": False, "error": f"{type(exc).__name__}: {exc}", "status": status}
        return result

    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_active) as pool:
        futures = [pool.submit(run_job, job) for job in jobs]
        for future in futures:
            result = future.result()
            results.append(result)
            print(json.dumps({"id": result["id"], "ok": result["ok"], "n_calls": result.get("status", {}).get("n_calls")}))

    summary = {"finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "results": results}
    (run_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0 if all(result["ok"] for result in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
