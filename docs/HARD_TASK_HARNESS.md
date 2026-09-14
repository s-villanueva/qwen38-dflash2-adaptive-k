# Hard-task harness: method and evidence

## Purpose

The ledger harness is an optional request-level workflow around the hybrid
DFlash2 service. It sends several bounded roles—planner, ideation, manager,
worker, and finalizer—to the same experimental model endpoint. It does not
alter DFlash2's target verification or scheduler policy.

The implementation deliberately keeps orchestration concurrency small. When
two workspaces are active, the service's scheduler may move from its K=12
single-request band to the K=5 band; as requests finish, its normal scheduler
policy can select a different band. This is not a client-side forced K value.

## Sanitized smoke test

`results/hard_task_harness_smoke_20260914.csv` records a run of two independent
algorithmic tasks with `max_active=2`, two workflow iterations, a per-call cap
of 2,048 output tokens, and no generated-code execution.

| Measure | Value |
| --- | ---: |
| Model calls | 18 |
| Completion tokens | 14,719 |
| Full-run wall time | 97.916 s |
| Aggregate completion rate | 150.32 tok/s |
| Non-empty final responses | 2/2 |
| Calls reaching the cap | 3 |

The matching direct hard-prompt request produced 2,048 tokens in 19.666 s:
104.14 tok/s. It reached the configured cap, so it measures sustained decoding
but not a complete answer length.

## Interpretation

These numbers establish that the isolated endpoint accepted concurrent ledger
work and that normal answer content was returned when request-scoped Qwen
thinking was disabled. They do **not** establish a quality improvement. The
hard-task output was not compiled or judged; at least one generated answer did
not respect its requested implementation language. It would be misleading to
call this a coding benchmark, a pass@1 result, or a community-ready accuracy
claim.

For a quality result, use a licensed benchmark, a matched one-call baseline,
fixed prompt/template and token budget, network-disabled code execution,
deterministic compiler/runtime limits, and publish aggregate scoring only.
