# Security Policy

## Supported Scope

This is an academic (B.Tech CSE Major Project, Group 08) Data Science/AI
project. It is not a production medical system - see
[`docs/medical_disclaimer.md`](docs/medical_disclaimer.md). Security
support is best-effort.

For the full security posture and design rationale (data handling,
application security, known out-of-scope items), see
[`docs/security.md`](docs/security.md).

## Reporting a Vulnerability

If you discover a security vulnerability in this repository (e.g. a
credential leak, an injection vector, or a dependency with a known CVE):

1. **Do not** open a public GitHub issue describing the exploit in detail.
2. Report it privately via the repository's GitHub Security Advisories
   feature ("Security" tab -> "Report a vulnerability"), or via the
   contact channel in [`docs/contact_feedback.md`](docs/contact_feedback.md).
3. Include: affected file(s)/component, steps to reproduce, and
   potential impact.

We will acknowledge reports and aim to address confirmed issues promptly
given this is a student project maintained on a best-effort basis.

## Secrets & Credentials Policy

- No passwords, API keys, tokens, or database credentials are ever
  committed to this repository.
- `.env` is git-ignored; `.env.example` documents required variable
  **names** only (with placeholder values), never real secrets.
- If a secret is ever accidentally committed, it must be treated as
  compromised: rotate/revoke it immediately and purge it from Git history
  (e.g. via `git filter-repo` or BFG Repo-Cleaner), not just delete it in
  a later commit.

## Dependencies

Dependencies are pinned to compatible version ranges in
`requirements.txt`. Run `pip list --outdated` periodically and review
release notes before upgrading, especially for `tensorflow`,
`scikit-learn`, and `streamlit`.
