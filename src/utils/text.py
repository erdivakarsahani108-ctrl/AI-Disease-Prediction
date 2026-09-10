"""Shared text-cleaning helpers used by both the data-preprocessing pipeline
and the NLP symptom-extraction engine, so that a symptom name is always
standardized the exact same way regardless of where it is produced.
"""

import re

_MULTI_SPACE_RE = re.compile(r"\s+")
_MULTI_UNDERSCORE_RE = re.compile(r"_+")


def clean_symptom_token(raw: str) -> str:
    """Normalize a raw symptom string coming from the source CSV files or
    user free text into the canonical ``snake_case`` form used throughout
    the project, e.g. ``"  foul_smell_ofurine "`` -> ``"foul_smell_ofurine"``.

    Returns ``""`` for missing/NaN/"nan"-like values so callers can safely
    skip them instead of accidentally treating the literal string ``"nan"``
    as a symptom.
    """
    if raw is None:
        return ""
    try:
        import math

        if isinstance(raw, float) and math.isnan(raw):
            return ""
    except TypeError:
        pass
    token = str(raw).strip().lower()
    if token in ("", "nan", "none", "null"):
        return ""
    token = token.replace("-", "_")
    token = token.replace(" ", "_")
    token = _MULTI_UNDERSCORE_RE.sub("_", token)
    token = token.strip("_")
    return token


def to_display_name(symptom_key: str) -> str:
    """Turn a canonical snake_case symptom key into a human readable label,
    e.g. ``"high_fever"`` -> ``"High Fever"``.
    """
    words = symptom_key.replace("_", " ").split()
    return " ".join(w.capitalize() for w in words)


def normalize_free_text(text: str) -> str:
    """Lowercase and collapse whitespace of a free-text user message while
    keeping punctuation that matters for negation detection (e.g. ``"don't"``).
    """
    if text is None:
        return ""
    text = str(text).strip().lower()
    text = _MULTI_SPACE_RE.sub(" ", text)
    return text
