# Changelog

## Unreleased

- Added an optional, bounded GVS5H-style ledger harness for hard-task testing
  against the isolated experimental API.
- Added sanitized hard-task smoke-test evidence and a clean single-request
  decode-rate measurement. These are operational checks, not correctness or
  benchmark claims.

## 0.1.0 - 2026-09-14

- Initial experimental TP=4 DFlash2 hybrid-K deployment bundle.
- Native scheduler policy: K=12 for scheduler batches of 1, K=5 for batches
  of 2--4, and K=0 for batches of 5--64.
- Published benchmark summaries, raw historical fixtures, incident notes, and
  reproducibility guidance.
- Marked the policy as work in progress; acceptance-aware smoothing is future
  work.
