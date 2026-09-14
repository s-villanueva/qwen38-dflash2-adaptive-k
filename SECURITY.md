# Security and publication safety

Never commit SSH keys, passwords, access tokens, model credentials, populated
environment files, private prompt/completion data, model weights, or internal
hostnames/IP addresses. Review all benchmark artifacts before publication.

The harness ignores its local environment file and all `runs/` ledgers. Its
published smoke results contain aggregate counters only; do not publish task
prompts, generated answers, transcripts, or OpenWebUI configuration exports.

Prefix caching can reveal whether requests share a prefix through timing. Use
per-tenant cache salting or disable cross-tenant cache reuse in multi-tenant
deployments. Report security issues privately to the repository maintainer; do
not use public issues for credential or infrastructure disclosures.
