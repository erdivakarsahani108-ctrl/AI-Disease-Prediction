"""Rule-based + fuzzy-matching NLP symptom extraction engine.

Supports English, Hindi (Devanagari) and Hinglish (romanized Hindi) free
text such as:

    "Mujhe 3 din se bukhar hai, sir dard hai aur body pain ho raha hai."
    "I have fever, cough and body pain."
    "I don't have cough."

Pipeline:
    normalize_free_text
        -> build canonical phrase index (symptom synonyms + display names)
        -> direct phrase matching (longest phrase first)
        -> fuzzy matching (rapidfuzz) for spelling mistakes on remaining tokens
        -> negation scoping (do not activate a symptom mentioned in a
           negated context, e.g. "I don't have cough" / "mujhe khansi nahi hai")
        -> duration extraction (regex, English + Hindi/Hinglish)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rapidfuzz import fuzz, process

from src.nlp.symptom_synonyms import SYMPTOM_SYNONYMS
from src.utils.text import normalize_free_text, to_display_name

# ---------------------------------------------------------------------------
# Negation cues (English + Hindi/Hinglish). A symptom phrase found within a
# small word-window AFTER one of these cues (or immediately followed by a
# Hindi post-positioned negation like "nahi"/"nahin") is treated as negated.
# ---------------------------------------------------------------------------
NEGATION_CUES_BEFORE = [
    "no", "not", "dont have", "don't have", "do not have", "without", "never had",
    "denies", "absence of", "no sign of", "not experiencing", "not feeling",
]
NEGATION_CUES_AFTER = ["nahi", "nahin", "nahi hai", "nahin hai", "bilkul nahi"]

NEGATION_WINDOW = 4  # words to look back/forward for a negation cue

DURATION_PATTERNS = [
    # "3 days", "since 3 days", "for 5 days"
    re.compile(r"(?:since|for|from)?\s*(\d+)\s*(?:day|days)\b"),
    # "3 din", "3 din se", "teen din se" (numeric only supported for din)
    re.compile(r"(\d+)\s*din(?:o)?(?:\s*se)?"),
    re.compile(r"(\d+)\s*(?:week|weeks)\b"),
    re.compile(r"(\d+)\s*hafte?"),
]
DURATION_UNIT_MULTIPLIER = {0: 1, 1: 1, 2: 7, 3: 7}  # index in DURATION_PATTERNS -> multiply to days


@dataclass
class ExtractionResult:
    raw_text: str
    normalized_text: str
    extracted_symptoms: list[str] = field(default_factory=list)
    negated_symptoms: list[str] = field(default_factory=list)
    fuzzy_matched: dict = field(default_factory=dict)  # symptom -> matched phrase
    duration_days: int | None = None
    unmatched_fragments: list[str] = field(default_factory=list)


def _build_phrase_index() -> list[tuple[str, str]]:
    """Return a list of (phrase, canonical_symptom) sorted by phrase length
    descending, so longer/more specific phrases are matched before shorter
    generic ones (prevents e.g. "pain" swallowing "chest pain")."""
    phrases = []
    # Import lazily to avoid circular import at module load time.
    from src.utils.paths import SYMPTOM_VOCAB_JSON
    import json

    vocab = []
    if SYMPTOM_VOCAB_JSON.exists():
        vocab = json.loads(SYMPTOM_VOCAB_JSON.read_text(encoding="utf-8")).get("symptoms", [])

    all_symptoms = set(vocab) | set(SYMPTOM_SYNONYMS.keys())
    if vocab:
        # Guard against synonym-dictionary keys that don't correspond to a
        # real trained-model feature (would otherwise be extracted but
        # silently ignored/mismatched downstream by the ML pipeline).
        all_symptoms = set(vocab)
    for canonical in all_symptoms:
        display = to_display_name(canonical).lower()
        phrases.append((display, canonical))
        phrases.append((canonical.replace("_", " "), canonical))
        syn = SYMPTOM_SYNONYMS.get(canonical, {})
        for p in syn.get("en", []):
            phrases.append((p.lower(), canonical))
        for p in syn.get("hi", []):
            phrases.append((p.lower(), canonical))

    # dedupe, sort longest-first
    seen = set()
    unique_phrases = []
    for phrase, canonical in phrases:
        phrase = phrase.strip()
        key = (phrase, canonical)
        if phrase and key not in seen:
            seen.add(key)
            unique_phrases.append((phrase, canonical))
    unique_phrases.sort(key=lambda x: len(x[0]), reverse=True)
    return unique_phrases


_PHRASE_INDEX = None


def get_phrase_index():
    global _PHRASE_INDEX
    if _PHRASE_INDEX is None:
        _PHRASE_INDEX = _build_phrase_index()
    return _PHRASE_INDEX


def _is_negated(text: str, match_start: int, match_end: int) -> bool:
    before_text = text[:match_start]
    after_text = text[match_end:]
    before_words = before_text.split()[-NEGATION_WINDOW:]
    after_words = after_text.split()[:NEGATION_WINDOW]

    before_joined = " ".join(before_words)
    for cue in NEGATION_CUES_BEFORE:
        if cue in before_joined:
            return True

    after_joined = " ".join(after_words)
    for cue in NEGATION_CUES_AFTER:
        if cue in after_joined:
            return True
    return False


def extract_duration_days(text: str) -> int | None:
    for i, pattern in enumerate(DURATION_PATTERNS):
        m = pattern.search(text)
        if m:
            try:
                value = int(m.group(1))
            except (ValueError, IndexError):
                continue
            return value * DURATION_UNIT_MULTIPLIER.get(i, 1)
    return None


def extract_symptoms(text: str, fuzzy_threshold: int = 87) -> ExtractionResult:
    """Extract canonical symptoms (with negation + duration handling) from a
    free-text user message.
    """
    normalized = normalize_free_text(text)
    result = ExtractionResult(raw_text=text, normalized_text=normalized)

    if not normalized:
        return result

    # Strip punctuation for matching purposes (but keep the untouched
    # `normalized` text available for duration-regex extraction above).
    # Apostrophes are removed (not replaced with a space) so contractions
    # like "don't" -> "dont" still match our negation-cue list.
    match_text = normalized.replace("'", "")
    match_text = re.sub(r"[^a-z0-9\s]", " ", match_text)
    match_text = re.sub(r"\s+", " ", match_text).strip()

    working_text = f" {match_text} "
    matched_spans: list[tuple[int, int]] = []
    found_positive: set[str] = set()
    found_negative: set[str] = set()

    for phrase, canonical in get_phrase_index():
        needle = f" {phrase} "
        start = 0
        while True:
            idx = working_text.find(needle, start)
            if idx == -1:
                break
            span_start, span_end = idx + 1, idx + 1 + len(phrase)
            # Skip if this span overlaps an already-matched (longer) phrase.
            if any(not (span_end <= s or span_start >= e) for s, e in matched_spans):
                start = idx + 1
                continue
            matched_spans.append((span_start, span_end))
            if _is_negated(working_text, span_start, span_end):
                found_negative.add(canonical)
            else:
                found_positive.add(canonical)
            start = idx + 1

    # Fuzzy pass on remaining unmatched word n-grams (handles spelling mistakes)
    remaining = working_text
    for s, e in sorted(matched_spans, reverse=True):
        remaining = remaining[:s] + " " * (e - s) + remaining[e:]
    tokens = [t for t in remaining.split() if len(t) > 3]
    candidate_phrases = [p for p, _ in get_phrase_index() if " " not in p]
    for tok in set(tokens):
        best = process.extractOne(tok, candidate_phrases, scorer=fuzz.ratio)
        if best and best[1] >= fuzzy_threshold:
            phrase_matched = best[0]
            canonical = next(c for p, c in get_phrase_index() if p == phrase_matched)
            idx = working_text.find(tok)
            if idx != -1 and not _is_negated(working_text, idx, idx + len(tok)):
                if canonical not in found_positive and canonical not in found_negative:
                    found_positive.add(canonical)
                    result.fuzzy_matched[canonical] = f"{tok} ~ {phrase_matched}"

    # A symptom explicitly negated anywhere takes precedence over an
    # incidental positive match elsewhere in a different clause.
    found_positive -= found_negative

    result.extracted_symptoms = sorted(found_positive)
    result.negated_symptoms = sorted(found_negative)
    result.duration_days = extract_duration_days(normalized)
    return result
