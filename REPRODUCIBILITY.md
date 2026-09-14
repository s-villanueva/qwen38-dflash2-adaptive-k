# Reproducibility

## Tested configuration

- Target: `RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead`.
- Draft: DFlash2 checkpoint compatible with the target tokenizer.
- Serving base: `timothystewart6/vllm-gb10` image digest
  `sha256:60994dfa59451aabde6091d0391ac89fc7843a6a66f35c9d9535ad27477357f2`.
- vLLM: `0.28.1.dev0+g2cf0a6915.d20260828` for hybrid measurements.
- Topology: four nodes, TP=4, Ray distributed executor, one GPU per node.
- RoCE: a cluster-specific interface, HCA, and GID index supplied through the
  environment; those deployment values are intentionally not published.
- Runtime: FP8 E4M3 KV cache, GPU-memory utilization `0.60`,
  `max_num_batched_tokens=8192`, `max_num_seqs=64`.

## Procedure

1. Configure and review `config/cluster.env.example`.
2. Confirm all four nodes can reach the head over the selected RoCE and Ray
   interfaces. Do not publish their addresses, hostnames, or interface values.
3. Start `scripts/run_hybrid_tp4.sh` on the Ray head.
4. Wait for `/health` to return HTTP 200 and allow CUDA graphs to warm.
5. Run the benchmark driver with fixed prompt fixture, output cap, seed, and
   concurrency matrix.
6. Publish raw and summary CSVs together with model/image revisions.

Use the same contention state for every comparison. The included hybrid run was
contended and does not establish a dedicated-cluster ceiling.
