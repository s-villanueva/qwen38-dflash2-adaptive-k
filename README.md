# Qwen3.8-27B DFlash2 hybrid speculative decoding on TP=4 DGX Spark

Reproducible deployment notes and benchmark evidence for a native vLLM dynamic
speculative-decoding policy on a four-node DGX Spark/RoCE cluster.

The policy adapts DFlash2 width to scheduler batch size:

| Scheduled batch size | Speculative width |
| ---: | ---: |
| 1 | K=12 |
| 2--4 | K=5 |
| 5--64 | K=0 |

It is aimed at mixed interactive and concurrent workloads. The target model
continues to verify all emitted output; this policy only changes how many draft
tokens are attempted.

> **Work in progress.** The current `{12, 5, 0}` policy has only three
> workload-derived operating points. Its step changes at scheduler batch sizes
> 2 and 5 are a conservative baseline, not a claim that three widths or abrupt transitions
> are optimal. The intended next phase is to collect target-verified traces and
> replace these coarse bands with a smoother, acceptance-aware draft budget.

## Technical proof

The primary evidence is [`results/hybrid_adaptive_k_concurrency_summary.csv`](results/hybrid_adaptive_k_concurrency_summary.csv).
It records 369 successful requests (zero failures) with the
`RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead` target, TP=4, GPU-memory utilization
0.60, FP8 KV cache, and DFlash2 draft. Historical CSVs use a family-level
`Qwen/Qwen3.8-27B` label; the RadixArk checkpoint above is the exact target.

| Concurrency | Aggregate output tok/s | p50 latency (s) |
| ---: | ---: | ---: |
| 1 | 54.0 | 3.64 |
| 2 | 91.1 | 4.10 |
| 4 | 148.5 | 4.95 |
| 16 | 232.1 | 13.78 |
| 32 | 354.4 | 18.03 |
| 60 | 377.3 | 31.77 |

The threshold rationale, including historical matched C=1 observations, is
documented in [`docs/RESULTS.md`](docs/RESULTS.md). Those observations are
context only until their raw fixtures are published.

## Repository layout

- `scripts/run_hybrid_tp4.sh`: reproducible hybrid-policy wrapper.
- `scripts/run_tp4_base.sh`: isolated TP=4/Ray/RoCE launcher.
- `scripts/benchmark_serving.py`: OpenAI-compatible benchmark driver.
- `results/`: static-DFlash, vanilla, and hybrid summary CSV evidence.
- `docs/RESULTS.md`: configuration, methodology, results, and limitations.
- `docs/TP4_INCIDENT_AND_RECOVERY.md`: startup incident and recovery details.
- `harness/`: bounded GVS5H-style hard-task orchestration adapter for the
  isolated experimental API, with a pinned upstream bootstrap and no copied
  benchmark data.

## Reproduction

This requires four nodes with one GPU each, Docker, Ray, the GB10 vLLM image,
the Qwen3.8 target checkpoint, DFlash2 draft checkpoint, and a working RoCE
fabric. Set the node/interface variables in `scripts/run_tp4_base.sh` before
running:

```bash
source config/cluster.env.example  # replace example values first
bash scripts/run_hybrid_tp4.sh
python3 scripts/benchmark_serving.py \
  --base-url http://127.0.0.1:8000 \
  --model experimental-radixark-dflash-tp4-k12
```

Run on a dedicated experimental cluster. The supplied results were contended
and should not be presented as a hardware maximum.

## Scope and limitations

- This is an engineering case study, not a new speculative-decoding algorithm.
- vLLM already provides `num_speculative_tokens_per_batch_size`; this project
  documents a workload-derived DFlash2 schedule and TP=4 deployment evidence.
- Dynamic K is scheduler-batch-level, not client-concurrency-level, and must
  not be used with data parallelism.
- Prefix caching helps repeated-prefix prefill/TTFT, not decode-only tok/s.
- The policy has only three K values and abrupt concurrency thresholds. More
  intermediate K levels, hysteresis, and an acceptance-aware controller remain
  future work.
- Do not compare these figures with different target revisions, quantization,
  node counts, memory limits, prompts, or warm-up state.

## Upstream work

- [timothystewart6/vllm-gb10](https://github.com/timothystewart6/vllm-gb10):
  the GB10 vLLM image/repository fork used as the experimental base.
- [vLLM dynamic speculative decoding](https://github.com/vllm-project/vllm/blob/main/docs/features/speculative_decoding/dynamic_speculative_decoding.md)
- [DFlash2 documentation](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/dflash2/)
- [slee-persis/GVS5H](https://github.com/slee-persis/GVS5H): MIT-licensed
  ledger-orchestration code used by the optional hard-task harness. Its paper
  and benchmark data are not included here.

## License

The repository material is released under Apache-2.0. Checkpoint licenses and
the upstream vLLM/GB10 image terms remain their respective owners' terms.
