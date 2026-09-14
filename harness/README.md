# Bounded ledger harness for hard tasks

This optional component combines the existing adaptive DFlash2 serving policy
with a deliberately small, filesystem-ledger orchestration loop. It is for
operational experiments on difficult code or math prompts. It does not change
model weights, the vLLM scheduler, DFlash2, or any vanilla deployment.

The wrapper dispatches planner, ideation, manager, worker, and finalizer calls
to one OpenAI-compatible **experimental** endpoint. The underlying service can
therefore select its existing scheduler-level K policy: K=12 at batch size 1,
K=5 at 2--4, and K=0 at 5 or more. `--max-active 2` is the recommended upper
bound for this TP=4 experiment.

## What is included

- `src/run_harness.py` is this project's bounded dispatcher and manifest
  writer. It never executes model-produced code.
- `patches/gvs5h-local-qwen.patch` adds request-scoped chat-template options to
  GVS5H's generic OpenAI-compatible adapter. It lets short coordination calls
  disable Qwen thinking without changing server defaults.
- `scripts/bootstrap_gvs5h.sh` fetches the exact upstream source and applies
  that small patch in a separate dependency directory.
- `results/hard_task_harness_smoke_20260914.csv` records only aggregate,
  sanitized operational counters.

The upstream orchestration code is not copied here. Bootstrap obtains
[`slee-persis/GVS5H`](https://github.com/slee-persis/GVS5H) at commit
`0bc99eae8c6ef0d49a91808272c60928113da1ce`. Its code is MIT-licensed; its
paper, figures, run data, and benchmark statements have separate terms. Read
its `LICENSE`, `LICENSE-CC-BY-4.0`, and `NOTICE.md` before redistribution.

## Setup

Run this on the experimental host, never against the vanilla service:

```bash
git clone https://github.com/<your-org>/qwen38-dflash2-hybrid-tp4.git
cd qwen38-dflash2-hybrid-tp4
bash harness/scripts/bootstrap_gvs5h.sh /opt/gvs5h
cp harness/config/harness.env.example harness/config/harness.env
# edit only local values in harness/config/harness.env
set -a; source harness/config/harness.env; set +a
python3 harness/src/run_harness.py --task "Prove a simple recurrence." --kind math
```

For a bounded concurrent run, supply JSON Lines with `id`, `prompt`, and an
optional `kind` (`code` or `math`):

```bash
python3 harness/src/run_harness.py --tasks-jsonl tasks.jsonl --max-active 2
```

The wrapper writes run ledgers to `harness/runs/`, which is ignored by Git.
Treat them as sensitive because prompts and completions can be private.

## Safety and evaluation limits

- Point `QWEN_HARNESS_BASE_URL` only to an isolated, access-controlled
  experimental API. Do not enter public addresses, credentials, or model paths
  in the example configuration.
- Keep `QWEN_HARNESS_MAX_ACTIVE` at 1--2 unless matched load testing justifies
  more. The hard limit is four.
- `ok=true` means a non-empty final response was returned. It does **not** mean
  the response is correct, compilable, safe, or in the requested language.
- Verify generated code later in a network-disabled sandbox with CPU, memory,
  process, and wall-clock limits.
- Compare quality against a matched one-call baseline before making community
  claims. The included smoke test is not a LiveCodeBench result.
