# database/

Reserved for **MySQL schema definitions, migration scripts, and seed
data** (per the project's declared tech stack - see the root
`README.md -> Technology Stack`).

## Current status

The application's currently *implemented* persistence layer is a
lightweight, anonymous, flat-file feedback log
(`src/database/feedback.py` -> `feedback/feedback_log.csv`) - see
`docs/privacy_policy.md` for why a full relational database is not
required for the app's core functionality (no personal data is
collected, and there is no multi-user account system).

This folder is reserved so a MySQL-backed persistence layer (e.g. for
storing anonymized prediction logs, structured experiment tracking, or a
multi-user deployment) can be added later without restructuring the
repository - see "Future Scope" in the root `README.md`.

## Suggested future contents

```
database/
├── schema.sql            -- CREATE TABLE statements
├── migrations/           -- versioned schema migration scripts
└── seed/                 -- optional non-sensitive seed/reference data
```

No credentials are ever stored in this folder - see `.env.example` for
the expected MySQL connection environment variables.
