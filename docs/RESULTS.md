# Hybrid DFlash2 speculative decoding on four DGX Spark nodes

## Purpose

This experiment makes DFlash2 useful across mixed concurrency instead of
forcing one speculative width for every batch. It uses vLLM's native dynamic
speculative-decoding scheduler with a RadixArk/Qwen3.8-27B NVFP4 target and a
DFlash2 draft model, TP=4 across four nodes. Node addresses and hostnames are
intentionally omitted from this public report.

## Reproducible implementation

Use `scripts/run_hybrid_tp4.sh`. It starts the isolated experimental launcher
with:

```json
{
  "method": "dflash",
  "model": "<draft-model-path>",
  "num_speculative_tokens": 12,
  "draft_tensor_parallel_size": 1,
  "num_speculative_tokens_per_batch_size": [[1, 1, 12], [2, 4, 5], [5, 64, 0]]
}
```

The policy is deterministic at the scheduler-batch level. Scheduler batch size
is not the same thing as the number of client requests in flight:

| Scheduled batch size | Action | Reason |
| ---: | --- | --- |
| 1 | DFlash2 K=12 | Preserve the long-draft upside of high-acceptance tasks. |
| 2--4 | DFlash2 K=5 | Limit rejected-draft overhead and maintain interactive throughput. |
| 5--64 | K=0 | Disable speculation so target batching takes priority. |

This is not a per-request neural classifier. It is a TP-safe native scheduler
policy. It is deliberately incompatible with data parallelism because separate
DP ranks could select different K values.

## Work-in-progress policy

The three-point `{12, 5, 0}` schedule is intentionally conservative. Its
thresholds were selected from the available measurements, but they are coarse:
the scheduler changes width abruptly at batch size 2 and removes speculation
at batch size 5.
They should not be interpreted as a globally optimal shape. A more mature
policy should sweep intermediate widths, add hysteresis around load changes,
and use target-verified acceptance/cost telemetry to vary the verification
budget more smoothly.

## Hardware and serving configuration

- TP=4 through Ray across four nodes. Addresses, hostnames, and RoCE interface,
  HCA, and GID values are provided only through the local environment.
- GPU memory utilization: 0.60 because the cluster was contended.
- FP8 KV cache, `max_num_batched_tokens=8192`, `max_num_seqs=64`.
- Prefix caching enabled. It improves repeated-prefix prefill/TTFT, not
  decode-only output token rate.
- The matched C=1 observations below are single-request decode measurements.
  The hybrid table is aggregate throughput under concurrent client load; neither
  is an end-to-end TTFT claim.

## Evidence for the hybrid thresholds

Historical matched TP=4 C=1 observations used a 1,024-token prompt and a
256-token completion:

| Workload | K=5 decode tok/s | K=12 decode tok/s | Decision |
| --- | ---: | ---: | --- |
| Code | 95.6 | 97.0 | Near tie; K=12 has a 1.5% edge. |
| General prose | 52.4 | 43.2 | K=5 is 21.3% faster. |
| Exact reproduction | 145.4 | 275.3 | K=12 is 89.3% faster. |

The high 275.3 tok/s observation required C=1, K=12, warmed CUDA graphs, and
a highly predictable/repetitive continuation with 100% acceptance. Its raw
fixture is not included in this release, so it is context rather than technical
proof. It is a special-case peak, not a general-purpose latency promise.

The hybrid policy was then exercised with generic 200-token requests under
variable concurrency. All 369 requests completed:

| Concurrency | Aggregate output tok/s | p50 request latency (s) |
| ---: | ---: | ---: |
| 1 | 54.0 | 3.64 |
| 2 | 91.1 | 4.10 |
| 4 | 148.5 | 4.95 |
| 8 | 148.3 | 10.79 |
| 16 | 232.1 | 13.78 |
| 32 | 354.4 | 18.03 |
| 60 | 377.3 | 31.77 |

At high concurrency, occasional draft counters are expected as batches drain
below five requests; sustained batches use K=0 by design.

## Why this is useful, and what it is not

The useful contribution is an end-to-end, reproducible deployment and
measurement case for Qwen3.8-27B on a four-node DGX Spark/RoCE cluster,
including a workload-derived K schedule and the supporting failure/benchmark
documentation.

It is **not** a novel dynamic-speculative-decoding algorithm. vLLM already
documents the same scheduler mechanism and lists DFlash among its tested
methods. Do not present the native API or the basic concurrency-to-K concept
as original research.

The contribution is publishable as an engineering case study if the repository
contains the exact versions, launcher, hardware/network notes, raw and summary
CSV/JSONL files, methodology, warm-up procedure, acceptance counters, and
limitations. Keep results separated by target model, quantization, node count,
memory-utilization setting, and workload. Do not compare unlike configurations
as a speedup claim.

## Recommended GitHub contents

1. This launcher and report.
2. `TP4_INCIDENT_AND_RECOVERY.md` for the TP4 startup/graph-cache incident.
3. Raw and summarized benchmark artifacts with schemas.
4. A reproduction guide covering RoCE/NCCL/GLOO/Ray interfaces.
5. A correctness section: target verification remains authoritative.
6. Future work: acceptance-aware K policy and Qwen3.8 EAGLE-3, explicitly
   labelled as experimental and not included in the reported result.

## References

- vLLM dynamic speculative decoding documentation:
  <https://github.com/vllm-project/vllm/blob/main/docs/features/speculative_decoding/dynamic_speculative_decoding.md>
- DFlash2 documentation:
  <https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/dflash2/>
