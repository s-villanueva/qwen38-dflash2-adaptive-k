# DFlash2 distributed draft failure on the four-node DGX Spark cluster

## Executive summary

The first four-node DFlash2 configuration failed while vLLM was initializing
the speculative draft path. The Qwen3.8-27B target model, Ray placement,
four-rank NCCL group, RoCE transport, and model-weight loading had already
succeeded. The first meaningful failure was a CUDA `IndexKernel` out-of-bounds
assertion in DFlash2 candidate selection. Later cuBLAS, Ray actor, and engine
errors were consequences of the poisoned CUDA context, not independent root
causes.

The failure was avoided without returning to the vanilla deployment and
without removing four-node tensor parallelism from the target model. The
working topology is:

- Qwen3.8-27B NVFP4 target: tensor parallel size 4 across four DGX Sparks;
- DFlash2 draft: logical tensor parallel size 1;
- speculative window: 10 tokens;
- Ray distributed executor for the target;
- a cache directory dedicated to DFlash2 K=10 and draft TP=1.

This topology keeps all four nodes involved in every target-model verification
step while preventing DFlash2's unstable cross-rank candidate merge from being
used. It completed compilation, profiling, FlashInfer autotuning, target CUDA
graph capture, DFlash2 CUDA graph capture, live inference, and a 369-request
benchmark without a CUDA error.

## Scope and isolation

The experiment was deliberately isolated from the existing NVIDIA-guide
deployment. It has its own:

- Compose project and container names;
- image (`vllm-gb10:v0.28.0-gb10.2` release line);
- Ray runtime;
- vLLM compilation cache;
- served model name (`experimental-vllm-decoding`);
- a deployment directory local to each node (path intentionally omitted).

The vanilla implementation was not edited, stopped, replaced, or used as a
fallback while diagnosing this incident.

## Cluster topology

| Role | Address / hostname | TP rank |
| --- | --- | ---: |
| Head | intentionally omitted | 0 |
| Worker 1 | intentionally omitted | 1 |
| Worker 2 | intentionally omitted | 2 |
| Worker 3 | intentionally omitted | 3 |

Each node contributes one GB10 GPU. Ray reported four active nodes and four
reserved GPUs with no pending resource demand or failed node.

## Software and model configuration

The running API reported:

- vLLM: `0.28.1.dev0+g2cf0a6915.d20260828`;
- container release line: `v0.28.0-gb10.2`;
- target model: Qwen3.8-27B NVFP4 family checkpoint (exact local path omitted);
- target architecture: `Qwen3_5ForConditionalGeneration`;
- draft architecture: `DFlash2DraftModel`;
- dtype: BF16 target execution;
- KV cache: `fp8_e4m3`;
- distributed executor: Ray;
- target TP: 4;
- draft TP: 1;
- speculative method: `dflash`;
- speculative tokens: 10;
- maximum sequences: 64;
- maximum batched tokens: 8192;
- GPU memory utilization target: 0.60;
- automatic tool choice enabled with the parser available in the historical
  deployment.

The effective serving command is maintained in
`docker-compose.head.yaml`. Its material vLLM portion is:

```bash
vllm serve <target-model-path> \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 4 \
  --distributed-executor-backend ray \
  --gpu-memory-utilization 0.60 \
  --load-format fastsafetensors \
  --kv-cache-dtype fp8_e4m3 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 8192 \
  --enable-auto-tool-choice \
  --tool-call-parser <historical-parser> \
  --speculative-config \
    '{"method":"dflash","model":"<draft-model-path>","num_speculative_tokens":10,"draft_tensor_parallel_size":1}' \
  --served-model-name experimental-vllm-decoding
```

## Network configuration

The cluster retained the interfaces and RoCE parameters already proven on the
four Sparks:

| Purpose | Setting |
| --- | --- |
| NCCL socket interface | cluster-specific, intentionally omitted |
| GLOO socket interface | cluster-specific, intentionally omitted |
| UCX/OpenMPI interface | cluster-specific, intentionally omitted |
| NCCL RDMA device | cluster-specific, intentionally omitted |
| RoCE GID index | cluster-specific, intentionally omitted |
| NCCL IB/RoCE transport | enabled (`NCCL_IB_DISABLE=0`) |
| GPUDirect RDMA | disabled (`NCCL_NET_GDR_LEVEL=0`) |
| NCCL algorithm | Ring |
| Minimum NCCL channels | 4 |

Disabling GPUDirect RDMA is intentional on the GB10 unified-memory SoC. It
does not disable RoCE: NCCL continues to use the ConnectX device through the
host-accessible transport. During successful initialization, the four ranks
formed a PYNCCL group using NCCL 2.31.2.

## Original failure

### Primary error signature

With both the target and the DFlash2 draft distributed at TP=4, CUDA reported
an `IndexKernel` index-out-of-bounds assertion while the DFlash2 candidate
selection/top-k path was being exercised. The characteristic sequence was:

1. CUDA `IndexKernel` reports an index outside the valid tensor dimension.
2. A device-side assertion invalidates the CUDA context for that rank.
3. Subsequent cuBLAS operations fail because CUDA work is already in an error
   state.
4. The affected Ray worker exits or becomes unreachable.
5. The engine reports distributed worker/actor failures and aborts startup.

Only item 1 is the primary failure. Diagnosing from the final Ray or cuBLAS
message alone would incorrectly point toward networking, placement, or GEMM
execution.

### Location in the algorithm

DFlash2 reduces the draft vocabulary candidates in stages. In the failing
distributed-draft topology, each tensor-parallel rank first selects local
candidates. Candidate values and identifiers are then combined across ranks,
a second top-k operation chooses the global winners, and those returned
positions are used by a gather/indexing operation.

The observed assertion is consistent with an invalid position reaching that
final gather during the cross-rank merge on the GB10/SM12.x path. Logical
draft TP=1 does not need that cross-rank merge: the candidate indices remain
local, so the failing distributed indexing branch is bypassed.

This report calls that the root cause at the useful engineering level:
**DFlash2 distributed candidate selection is not reliable for this draft,
vLLM build, and GB10 CUDA path.** The available evidence does not prove which
lower-level component first produced the bad position. A complete upstream
fix still requires a minimal reproducer and instrumentation immediately before
the final gather.

## How alternative hypotheses were eliminated

### Ray placement was not the primary problem

Ray discovered four active nodes, reserved four GPUs, and assigned target TP
ranks 0 through 3. There were no pending placement demands. The failure
occurred only after distributed execution had been established.

### NCCL/RoCE initialization was not the primary problem

All ranks joined a four-rank NCCL group, and model weights loaded on the head
and remote nodes. A broken interface, GID, or unreachable worker would fail
during rendezvous or collectives rather than first surfacing as a CUDA tensor
index assertion inside candidate selection.

### CUDA graph capture was not the primary problem

An `--enforce-eager` experiment did not eliminate the failure. That removes
CUDA graph replay from the suspected path but does not change DFlash2's
distributed candidate-index calculation. The continued assertion therefore
ruled out graph capture as the initiating defect.

### The target model was not the primary problem

Qwen3.8-27B weights loaded successfully on all four ranks. The target also
completed TP=4 initialization under the working draft topology. The topology
change was limited to the draft path.

### Memory pressure was not the primary problem

The error was an invalid index, not an allocation failure. The final
configuration nevertheless uses 0.60 GPU memory utilization to retain safe
headroom for the target, draft, compilation, graph capture, Ray, and
multimodal warmup. Successful profiling reported approximately 48.6 to 49.5
GiB of available KV-cache memory per rank and a cluster KV-cache capacity of
about 4.08 million tokens.

## Attempts that did not solve it

### Keeping the DFlash2 draft at TP=4

This preserves a symmetric target/draft layout but also preserves the exact
cross-rank candidate merge that produces the invalid gather index. It is not
usable until the DFlash2 distributed selector is corrected upstream or
patched locally.

### Enforcing eager execution

Eager mode changed execution/capture behavior but did not repair the invalid
candidate index. It also sacrifices graph-based performance and therefore is
not an acceptable production workaround.

### Treating downstream cuBLAS or Ray errors as separate faults

Restarting or tuning those components would only reset the CUDA context. Once
the device-side assertion occurs, later errors cannot be used to identify the
initiating operation.

## Implemented solution

The solution changes only the draft topology:

```json
{
  "method": "dflash",
  "model": "<draft-model-path>",
  "num_speculative_tokens": 10,
  "draft_tensor_parallel_size": 1
}
```

The target remains TP=4. The draft can be logically TP=1 because it is much
smaller than the 27B target. Every speculative block is still verified by the
target across all four GPUs. This is a cluster-utilizing workaround, not a
single-node rollback.

Additional safeguards were applied:

1. `--max-num-seqs 64` matches the largest planned comparison load.
2. `--max-num-batched-tokens 8192` bounds scheduler work during high
   concurrency.
3. `--gpu-memory-utilization 0.60` leaves startup and graph-capture headroom.
4. The compilation cache is isolated in a dedicated local directory on every
   node.
   Draft topology and speculative K affect compiled shapes; reusing artifacts
   from another topology can produce invalid or misleading results.
5. The DFlash2 model directory is mounted read-only at a local path.
6. The head waits for four live Ray nodes before starting vLLM.

## Successful startup evidence

The working run completed all of the following:

- target weight loading on TP ranks 0, 1, 2, and 3;
- DFlash2 draft loading;
- target `torch.compile` on all ranks;
- DFlash head and candidate-selector compilation;
- initial profiling and warmup;
- FlashInfer autotuning;
- 51 piecewise CUDA graph captures;
- 39 full target CUDA graph captures;
- 39 full DFlash2 CUDA graph captures;
- API application startup;
- `/health` returning HTTP 200;
- `/v1/models` exposing `experimental-vllm-decoding`;
- automatic tool-choice initialization;
- a real `/v1/chat/completions` request returning HTTP 200.

The initial engine startup took about 226.6 seconds, including approximately
39.7 seconds attributed to compilation. Target/DFlash graph capture completed
without the previous `IndexKernel` assertion.

## Warnings that were observed but were not fatal

Several warnings are expected in this hardware/topology and should not be
confused with the incident:

- Ray warns that TP=4 exceeds the one GPU physically present on each node.
  This is expected because the placement group spans four one-GPU nodes.
- `SymmMemCommunicator` is unavailable for compute capability 12.1.
- MNNVL multicast custom collectives are unavailable in the multi-node group.
  vLLM correctly selects PYNCCL instead.
- The configured `VLLM_ATTENTION_BACKEND` environment variable is not
  recognized by this build. Automatic backend selection still selected
  FlashInfer for the supported model paths.
- DFlash2 warns that external multimodal embeddings are not passed to the
  draft. Text-only draft inputs are expected; the target remains responsible
  for multimodal correctness.
- Shared-memory broadcast may warn during long compilation or capture stages.
  Progress resumed once the slow stage finished.
- FlashInfer reports its own backend choices and autotuning status. These are
  informational unless followed by an actual CUDA error.

## Functional DFlash2 proof

The first live smoke request produced 70 draft tokens and accepted 32, an
acceptance rate of 45.7% for that short request. This proved that the server
was not merely running with speculative decoding configured: the DFlash2
draft path generated proposals and the target accepted some of them.

The longer benchmark is more representative. It generated proposals under
every tested concurrency and accepted roughly 17 to 18% of draft tokens. The
lower acceptance rate compared with the smoke request is workload-dependent
and is not a failure.

## Benchmark validation

The repeatable benchmark used:

- concurrency levels 1, 2, 4, 8, 16, 32, and 60;
- three bursts per concurrency level;
- exactly 200 requested output tokens per request (`ignore_eos=true`);
- deterministic prompt selection and seeds;
- streaming responses for time-to-first-token measurement;
- one unrecorded warmup request;
- 369 recorded requests in total.

All 369 requests succeeded. Post-benchmark health remained HTTP 200 and the
logs contained no `IndexKernel`, device-side assertion, traceback, or CUDA
error.

| Concurrency | Requests | Output throughput (tokens/s) | Median TTFT (s) | P95 latency (s) | Draft acceptance |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3 | 31.98 | 0.307 | 7.656 | 18.34% |
| 2 | 6 | 52.49 | 0.340 | 7.911 | 18.39% |
| 4 | 12 | 75.90 | 0.428 | 11.010 | 17.70% |
| 8 | 24 | 88.70 | 0.489 | 18.224 | 17.06% |
| 16 | 48 | 139.03 | 0.777 | 22.943 | 17.20% |
| 32 | 96 | 195.02 | 1.188 | 32.712 | 17.06% |
| 60 | 180 | 284.92 | 1.908 | 41.724 | 17.05% |

The raw and summarized evidence is stored in:

- `experimental_qwen3.8_27b_dflash2_tp4_dtp1_20260913T204731Z_raw.csv`;
- `experimental_qwen3.8_27b_dflash2_tp4_dtp1_20260913T204731Z_summary.csv`.

The harness is `benchmark_serving.py` and should be reused without workload
changes for the vanilla comparison.

## Fair vanilla comparison requirements

The next run should change only the serving implementation under test. Keep
all of these constant:

- exact target checkpoint identifier and revision;
- four nodes and target TP=4;
- BF16 target dtype;
- FP8 KV-cache dtype;
- GPU memory utilization;
- maximum sequence and batched-token limits;
- prompts, deterministic seeds, and 200-token outputs;
- concurrency sequence and three repeats;
- one unrecorded warmup;
- streaming client and timing definitions;
- network interfaces and NCCL/RoCE parameters.

For vanilla, remove `--speculative-config` and use a separate vLLM compilation
cache. Do not point vanilla at the DFlash2 cache. The comparison should report
absolute throughput and latency for both configurations plus percentage
change computed as:

```text
throughput improvement (%) =
  (DFlash2 output tokens/s / vanilla output tokens/s - 1) * 100

latency reduction (%) =
  (1 - DFlash2 latency / vanilla latency) * 100
```

No performance-improvement claim should be made until that matched vanilla
run is complete. The current benchmark establishes stability and the
experimental side of the comparison only.

## Remaining technical work for a true distributed-draft fix

Draft TP=1 is a safe operational workaround, not a source-level repair. A
proper TP>1 fix should:

1. reproduce the failure with a minimal candidate-selector test on GB10;
2. record candidate tensor shapes, vocabulary offsets, selected positions,
   and valid gather bounds on every rank;
3. synchronize before the final gather so the first failing operation is
   unambiguous;
4. replace or guard the small second-stage distributed top-k/gather if it can
   produce rank-local versus global index confusion;
5. test K values independently with isolated compile caches;
6. validate eager and captured execution;
7. test TP=2 and TP=4 before restoring distributed draft execution;
8. repeat the full concurrency benchmark and compare output correctness.

Until those checks pass, `draft_tensor_parallel_size=1` should remain pinned
for this Qwen3.8-27B/DFlash2/GB10 deployment.

## Operational state after validation

After the benchmark, the experimental head and all three experimental workers
were stopped. OpenWebUI and the monitoring services remained healthy. The
configuration, dedicated caches, model files, harness, and CSV evidence were
preserved for repeatable startup and comparison.
