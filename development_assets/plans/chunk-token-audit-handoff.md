# Chunk Token Audit Handoff

Status: approved for implementation on 2026-08-11.

Implement `development_assets/plans/chunk-token-audit-finalized-plan.md` end to end. Keep changes minimal and consistent with existing experiment, indexing, and test patterns.

Critical constraints:

- Use `uv` for Python/dependency commands and Ruff-format before finalizing.
- Do not mutate SQLite, Chroma, chunk IDs, or evaluation labels.
- Do not download model files implicitly.
- Do not compute embeddings during the audit.
- Do not invent or falsely label translations as human-reviewed. If a trustworthy 30-pair reviewed fixture cannot be produced, complete fixture support and tests but document that the paired audit is blocked pending review.
- Do not modify unrelated user changes, commit, or push.
- Run focused validation immediately after the first substantive edit, then full validation at the end.
