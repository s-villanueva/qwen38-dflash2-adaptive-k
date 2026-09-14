# Requirements

- Four Linux nodes with Docker, SSH connectivity from the head to workers, and
  one compatible GPU per node.
- A working Ray cluster network plus RoCE/NCCL/GLOO interfaces selected for
  the physical fabric.
- The GB10 vLLM image, Qwen3.8 target checkpoint, and compatible DFlash2 draft
  checkpoint available on every node or through shared storage.
- Enough local storage for checkpoints, vLLM compiled-graph cache, Ray object
  store, and benchmark artifacts.
- Python 3.12 with `requests` only for `scripts/benchmark_serving.py`.

For the optional `harness/`, Git and Python 3.12 are required to fetch the
pinned MIT-licensed GVS5H dependency. The harness must run against an isolated
OpenAI-compatible endpoint, not a production or vanilla service.

This launcher is TP=4 only. Do not enable data parallelism with the dynamic-K
schedule because independent DP schedulers can choose divergent K values.

The launcher uses host networking, device passthrough, and privileged Docker
containers. Run it only on an administrator-controlled experimental cluster.
