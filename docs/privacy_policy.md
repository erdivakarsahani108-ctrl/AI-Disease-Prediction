# Privacy Policy

## What this application collects

**Nothing that identifies you personally.** This application intentionally
does NOT collect:

- Name
- Phone number
- Address
- Medical ID / insurance number
- Any other personal identifier

## What is stored, and where

| Data | Stored? | Where | Retention |
|---|---|---|---|
| Chat messages / selected symptoms | In-memory only (`st.session_state`) | Your browser session, server RAM | Cleared on "Clear Session" or when the session ends |
| Thumbs up/down feedback + predicted disease + confidence | Yes (anonymous) | `feedback/feedback_log.csv` (local to the deployment) | Until manually deleted by the maintainer |
| Health Report PDF | Generated on-demand, sent to your browser | Not stored server-side | N/A |

No conversation content, symptoms, or predictions are linked to any
personal identifier at any point.

## Clear Session

Every page provides a **🗑️ Clear Session** button in the sidebar that
immediately erases all in-memory chat history and symptom state for your
current browser session.

## Third parties

This application does not call any third-party AI/LLM API by default (see
`docs/ai_transparency.md`). If a future deployment enables an optional LLM
integration, this policy will be updated accordingly and users will be
informed before any data leaves the local deployment.

## Feedback data usage

Feedback (👍/👎) is stored only to help the project maintainer
qualitatively review prediction usefulness. **It is never used to
automatically retrain the model** (see `docs/responsible_ai.md`).
