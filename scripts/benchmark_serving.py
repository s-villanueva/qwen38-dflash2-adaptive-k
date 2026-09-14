#!/usr/bin/env python3
"""Repeatable OpenAI-compatible serving benchmark with CSV output."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests


PROMPTS = [
    "Explain Rayleigh scattering and why the daytime sky appears blue.",
    "Describe how a distributed inference server coordinates tensor-parallel workers.",
    "Explain the difference between weather and climate with concrete examples.",
    "Describe how speculative decoding can accelerate autoregressive language models.",
    "Explain why memory bandwidth matters during large-language-model decoding.",
    "Describe the roles of NCCL, GLOO, and Ray in a multi-node inference cluster.",
    "Explain the tradeoffs between latency and throughput in batched model serving.",
    "Describe how an Ethernet RoCE network carries GPU collective communication.",
]

RAW_COLUMNS = [
    "run_id", "configuration", "model", "node_count", "target_tp", "draft_tp",
    "speculative_tokens", "concurrency", "repeat", "request_id", "prompt_id",
    "started_at_utc", "status_code", "success", "error", "prompt_tokens",
    "completion_tokens", "ttft_s", "latency_s", "decode_s", "request_output_tok_s",
]

SUMMARY_COLUMNS = [
    "run_id", "configuration", "model", "node_count", "target_tp", "draft_tp",
    "speculative_tokens", "concurrency", "repeats", "requested_requests",
    "successful_requests", "failed_requests", "prompt_tokens", "completion_tokens",
    "aggregate_wall_s", "request_throughput_req_s", "output_throughput_tok_s",
    "total_token_throughput_tok_s", "mean_ttft_s", "p50_ttft_s", "p95_ttft_s",
    "mean_latency_s", "p50_latency_s", "p95_latency_s", "draft_tokens",
    "accepted_tokens", "draft_acceptance_rate", "mean_accepted_tokens_per_draft",
]


def percentile(values: list[float], probability: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def metric_value(metrics_text: str, metric_name: str) -> float:
    pattern = re.compile(rf"^{re.escape(metric_name)}(?:\{{[^}}]*\}})?\s+([0-9.eE+-]+)$", re.MULTILINE)
    match = pattern.search(metrics_text)
    return float(match.group(1)) if match else 0.0


def speculative_counters(session: requests.Session, base_url: str) -> tuple[float, float, float]:
    try:
        response = session.get(f"{base_url}/metrics", timeout=10)
        response.raise_for_status()
        text = response.text
        drafts = metric_value(text, "vllm:spec_decode_num_drafts_total")
        drafted_tokens = metric_value(text, "vllm:spec_decode_num_draft_tokens_total")
        accepted_tokens = metric_value(text, "vllm:spec_decode_num_accepted_tokens_total")
        return drafts, drafted_tokens, accepted_tokens
    except requests.RequestException:
        return 0.0, 0.0, 0.0


def send_request(
    endpoint: str,
    model: str,
    prompt: str,
    request_id: str,
    prompt_id: int,
    max_tokens: int,
    start_event: threading.Event,
    timeout: float,
) -> dict[str, object]:
    start_event.wait()
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    first_token_at: float | None = None
    prompt_tokens = 0
    completion_tokens = 0
    status_code = 0
    error = ""

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "seed": 20260913 + prompt_id,
    }

    try:
        with requests.post(endpoint, json=payload, stream=True, timeout=timeout) as response:
            status_code = response.status_code
            response.raise_for_status()
            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                usage = chunk.get("usage")
                if usage:
                    prompt_tokens = int(usage.get("prompt_tokens") or prompt_tokens)
                    completion_tokens = int(usage.get("completion_tokens") or completion_tokens)
                choices = chunk.get("choices") or []
                if choices:
                    delta = choices[0].get("delta") or {}
                    if delta.get("content") and first_token_at is None:
                        first_token_at = time.perf_counter()
    except Exception as exc:  # Keep partial benchmark data instead of aborting the run.
        error = f"{type(exc).__name__}: {exc}"

    finished = time.perf_counter()
    latency = finished - started
    ttft = (first_token_at - started) if first_token_at is not None else math.nan
    decode = max(latency - ttft, 0.0) if not math.isnan(ttft) else math.nan
    output_rate = completion_tokens / latency if latency > 0 else math.nan
    success = not error and status_code == 200 and completion_tokens > 0

    return {
        "request_id": request_id,
        "prompt_id": prompt_id,
        "started_at_utc": started_at,
        "status_code": status_code,
        "success": success,
        "error": error,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "ttft_s": round(ttft, 6),
        "latency_s": round(latency, 6),
        "decode_s": round(decode, 6),
        "request_output_tok_s": round(output_rate, 6),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="experimental-vllm-decoding")
    parser.add_argument("--concurrency", default="1,2,4,8,16,32,60")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--run-id", default="")
    parser.add_argument(
        "--configuration", default="vllm-0.28.1-dflash2-target-tp4-draft-tp1-k10"
    )
    parser.add_argument("--draft-tp", type=int, default=1)
    parser.add_argument("--speculative-tokens", type=int, default=10)
    parser.add_argument(
        "--file-prefix", default="experimental_qwen3.8_27b_dflash2_tp4_dtp1"
    )
    args = parser.parse_args()

    concurrency_levels = [int(value) for value in args.concurrency.split(",")]
    if any(value < 1 or value > 64 for value in concurrency_levels):
        raise SystemExit("Concurrency must be between 1 and 64")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = args.run_id or f"{args.file_prefix}_{timestamp}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / f"{args.file_prefix}_{timestamp}_raw.csv"
    summary_path = args.output_dir / f"{args.file_prefix}_{timestamp}_summary.csv"
    endpoint = f"{args.base_url.rstrip('/')}/v1/chat/completions"
    common = {
        "run_id": run_id,
        "configuration": args.configuration,
        "model": "RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead",
        "node_count": 4,
        "target_tp": 4,
        "draft_tp": args.draft_tp,
        "speculative_tokens": args.speculative_tokens,
    }

    session = requests.Session()
    health = session.get(f"{args.base_url.rstrip('/')}/health", timeout=10)
    health.raise_for_status()

    # One short unrecorded warmup keeps first-request routing/JIT out of measured results.
    warmup_event = threading.Event()
    warmup_event.set()
    send_request(endpoint, args.model, PROMPTS[0], "warmup", 0, 32, warmup_event, args.timeout)

    with raw_path.open("w", newline="", encoding="utf-8") as raw_file, summary_path.open(
        "w", newline="", encoding="utf-8"
    ) as summary_file:
        raw_writer = csv.DictWriter(raw_file, fieldnames=RAW_COLUMNS)
        summary_writer = csv.DictWriter(summary_file, fieldnames=SUMMARY_COLUMNS)
        raw_writer.writeheader()
        summary_writer.writeheader()

        for concurrency in concurrency_levels:
            level_rows: list[dict[str, object]] = []
            level_wall = 0.0
            before = speculative_counters(session, args.base_url.rstrip("/"))

            for repeat in range(args.repeats):
                start_event = threading.Event()
                burst_started = time.perf_counter()
                with ThreadPoolExecutor(max_workers=concurrency) as executor:
                    futures = []
                    for index in range(concurrency):
                        prompt_id = (repeat * concurrency + index) % len(PROMPTS)
                        request_id = f"c{concurrency}-r{repeat + 1}-q{index + 1}"
                        futures.append(
                            executor.submit(
                                send_request,
                                endpoint,
                                args.model,
                                PROMPTS[prompt_id],
                                request_id,
                                prompt_id,
                                args.max_tokens,
                                start_event,
                                args.timeout,
                            )
                        )
                    start_event.set()
                    burst_rows = [future.result() for future in as_completed(futures)]
                burst_wall = time.perf_counter() - burst_started
                level_wall += burst_wall

                for row in burst_rows:
                    full_row = {
                        **common,
                        "concurrency": concurrency,
                        "repeat": repeat + 1,
                        **row,
                    }
                    raw_writer.writerow(full_row)
                    level_rows.append(full_row)
                raw_file.flush()
                successful = sum(bool(row["success"]) for row in burst_rows)
                tokens = sum(int(row["completion_tokens"]) for row in burst_rows)
                print(
                    f"concurrency={concurrency} repeat={repeat + 1}/{args.repeats} "
                    f"success={successful}/{concurrency} output_tokens={tokens} "
                    f"wall_s={burst_wall:.3f} output_tok_s={tokens / burst_wall:.3f}",
                    flush=True,
                )

            after = speculative_counters(session, args.base_url.rstrip("/"))
            successes = [row for row in level_rows if row["success"]]
            ttfts = [float(row["ttft_s"]) for row in successes]
            latencies = [float(row["latency_s"]) for row in successes]
            prompt_tokens = sum(int(row["prompt_tokens"]) for row in successes)
            completion_tokens = sum(int(row["completion_tokens"]) for row in successes)
            draft_count = max(after[0] - before[0], 0.0)
            draft_tokens = max(after[1] - before[1], 0.0)
            accepted_tokens = max(after[2] - before[2], 0.0)

            summary_writer.writerow(
                {
                    **common,
                    "concurrency": concurrency,
                    "repeats": args.repeats,
                    "requested_requests": concurrency * args.repeats,
                    "successful_requests": len(successes),
                    "failed_requests": len(level_rows) - len(successes),
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "aggregate_wall_s": round(level_wall, 6),
                    "request_throughput_req_s": round(len(successes) / level_wall, 6),
                    "output_throughput_tok_s": round(completion_tokens / level_wall, 6),
                    "total_token_throughput_tok_s": round((prompt_tokens + completion_tokens) / level_wall, 6),
                    "mean_ttft_s": round(statistics.fmean(ttfts), 6) if ttfts else math.nan,
                    "p50_ttft_s": round(percentile(ttfts, 0.50), 6),
                    "p95_ttft_s": round(percentile(ttfts, 0.95), 6),
                    "mean_latency_s": round(statistics.fmean(latencies), 6) if latencies else math.nan,
                    "p50_latency_s": round(percentile(latencies, 0.50), 6),
                    "p95_latency_s": round(percentile(latencies, 0.95), 6),
                    "draft_tokens": int(draft_tokens),
                    "accepted_tokens": int(accepted_tokens),
                    "draft_acceptance_rate": round(accepted_tokens / draft_tokens, 6) if draft_tokens else math.nan,
                    "mean_accepted_tokens_per_draft": round(accepted_tokens / draft_count, 6) if draft_count else math.nan,
                }
            )
            summary_file.flush()

    print(f"RAW_CSV={raw_path}")
    print(f"SUMMARY_CSV={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
