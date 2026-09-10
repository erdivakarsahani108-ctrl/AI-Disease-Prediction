# Contributing to AI-Based Disease Prediction System

Thank you for your interest in this project! This is a B.Tech CSE Major
Project (Group 08), and contributions/suggestions are welcome for
academic and educational purposes.

## Ground rules

1. **No fabricated medical data.** Never add invented patient records,
   fake clinical facts, or unverified disease/symptom/medicine
   associations. See [`docs/data_sources.md`](docs/data_sources.md) and
   [`docs/dataset_methodology.md`](docs/dataset_methodology.md) for the
   data-integrity standard this project follows.
2. **No secrets in code.** Never commit passwords, API keys, tokens, or
   database credentials. Use `.env` (git-ignored) with `.env.example` as
   the template - see [`SECURITY.md`](SECURITY.md).
3. **Respect the Medical Disclaimer.** Any change must keep this project
   framed as an educational/informational AI tool, never as a medical
   diagnosis system. See [`docs/medical_disclaimer.md`](docs/medical_disclaimer.md).
4. **Keep it reproducible.** Use the fixed `RANDOM_SEED` where randomness
   is involved (`src/utils/paths.py`), and prefer adding/updating a
   `scripts/*.py` pipeline step over one-off manual data manipulation.

## How to contribute

1. Fork the repository and create a feature branch:
   `git checkout -b feature/your-feature-name`
2. Set up the environment:
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
3. Make your changes. Add/update tests in `tests/` for any behavior
   change.
4. Run the test suite before opening a pull request:
   ```powershell
   pytest tests -v
   ```
5. Commit with a clear message and open a pull request describing what
   changed and why.

## Code style

- Follow existing module conventions (see `src/` for examples).
- Prefer clear docstrings over inline comments; only comment code that
  genuinely needs clarification.
- Keep functions small and single-purpose, consistent with the existing
  `src/preprocessing/`, `src/nlp/`, `src/ml/`, `src/prediction/` modules.

## Reporting issues

Please use the repository's Issues tab to report bugs or suggest
features. See [`docs/contact_feedback.md`](docs/contact_feedback.md) for
more on the project's feedback channels.

## Team / Group Information

- **Project:** AI-Based Disease Prediction System
- **Group No.:** 08
- **Domain:** Data Science + Artificial Intelligence + Machine Learning
