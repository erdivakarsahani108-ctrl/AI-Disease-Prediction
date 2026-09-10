# Security

## Scope

This is an academic/educational project. The notes below describe the
security posture as implemented, and what is explicitly out of scope.

## Data handling

- No personally identifiable information (PII) is collected or stored
  (see `docs/privacy_policy.md`).
- All data files (`data/`, `models/`, `reports/`) are local to the
  deployment; no data is transmitted to third-party services by default
  (see `docs/ai_transparency.md`).
- `.env.example` documents any environment variables a future optional
  integration (e.g. an LLM API key) would require; secrets must never be
  committed to source control - actual `.env` files are excluded via
  `.gitignore`.

## Application security

- The Streamlit app does not accept file uploads or execute arbitrary
  user-supplied code.
- User free-text input is only ever used for rule-based/fuzzy-matching
  NLP symptom extraction (`src/nlp/extraction.py`) - it is never
  evaluated, executed, or passed to a shell/SQL query.
- No SQL database is used for user-facing data (feedback is stored in a
  flat, append-only CSV - `src/database/feedback.py`), eliminating SQL
  injection risk for that path.
- Dependencies are pinned to compatible version ranges in
  `requirements.txt`; a Python virtual environment (`.venv/`) isolates
  project dependencies from the host system.

## Known out-of-scope items (Future Scope)

- Authentication / authorization (this is a single-tenant educational
  demo, not a multi-user production system).
- Rate limiting / abuse protection for a public deployment.
- Encryption at rest for `feedback/feedback_log.csv` (contains no PII, so
  risk is low, but this would be required before any production use).
- Formal security audit / penetration testing.

## Reporting a concern

For an academic project, please raise any concern via the repository's
issue tracker (see `docs/contact_feedback.md`).
