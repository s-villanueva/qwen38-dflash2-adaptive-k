# Pre-publication checklist

- Replace the `CITATION.cff` repository URL.
- Replace the abbreviated `LICENSE` placeholder with the complete Apache-2.0
  text or choose the license you intend to publish.
- Review all scripts for private IP addresses, usernames, paths, SSH details,
  access keys, tokens, and model-cache paths.
- Confirm benchmark CSVs contain no prompt, completion, user, or secret data.
- Publish exact vLLM/image/model revisions and hashes used for each result.
- Keep raw runs only when their schema and privacy review permit publication.
- State that results were contended at GPU-memory utilization 0.60.
- Do not claim algorithmic novelty for vLLM dynamic speculative decoding.
