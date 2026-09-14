# Technical proof and comparability notes

The result CSVs are preserved as source evidence, not as universal performance
claims. The hybrid summary contains 369/369 successful generic workload
requests at C=1, 2, 4, 8, 16, 32, and 60. It was collected with the stated
hybrid K policy, TP=4, and 0.60 GPU-memory utilization on a contended cluster.

The static-DFlash and vanilla files are useful historical baselines, but use
them only for configuration-specific context because they use different vLLM
revisions and not every workload/configuration is matched. The matched
single-request workload comparisons and caveats are in `docs/RESULTS.md`.

To verify CSV integrity after cloning:

```bash
sha256sum results/*.csv results/raw/*.csv
```

The benchmark driver remains in `scripts/benchmark_serving.py`; reproduce with
the exact model/image revisions, workload fixture, warm-up procedure, and
memory-contended state described in the results documentation.
