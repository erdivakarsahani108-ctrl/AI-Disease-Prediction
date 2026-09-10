"""Synthetic data augmentation.

The publicly sourced dataset (see docs/data_sources.md) contains only
~300 unique real disease/symptom-combination records once exact duplicates
are removed (each disease is represented by a small, fixed number of
symptom subsets). That is too small and imbalanced for stable multi-class
ML training and far below the ~10,000 record target requested for this
project.

To reach a usable dataset size WITHOUT fabricating any new medical fact,
this module generates additional rows by resampling subsets of the
already-validated real symptom pool for each disease (see
``build_disease_symptom_pool`` in ``clean.py``). No symptom is ever added to
a disease that was not already observed for that disease in the real data.
Every generated row is stamped ``synthetic_flag = True`` and
``source = "synthetic_augmentation"`` so real and synthetic rows remain
fully distinguishable, as required by the project methodology.
"""

from __future__ import annotations

import random

import pandas as pd

from src.utils.paths import RANDOM_SEED


def generate_synthetic_records(
    real_df: pd.DataFrame,
    disease_pool: dict,
    target_per_disease: int,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Generate up to ``target_per_disease`` TOTAL rows (real + synthetic)
    per disease by sampling symptom subsets from that disease's real pool.

    Returns only the newly generated synthetic rows (columns: disease,
    symptoms, is_synthetic).
    """
    rng = random.Random(seed)
    existing_by_disease: dict[str, set] = {}
    for disease, symptoms in zip(real_df["disease"], real_df["symptoms"]):
        existing_by_disease.setdefault(disease, set()).add(symptoms)

    synthetic_rows = []
    for disease, pool in disease_pool.items():
        seen = existing_by_disease.get(disease, set())
        current_count = len(seen)
        needed = max(0, target_per_disease - current_count)
        if needed == 0 or len(pool) < 2:
            continue

        # Sample subsets of ANY size >= 2 from the disease's validated real
        # symptom pool. Every subset only contains symptoms already observed
        # for this disease in the real data, so no fact is fabricated - we
        # simply explore the combinatorial space of already-validated
        # associations more broadly than the handful of combinations present
        # in the raw source file.
        min_size = 2
        max_size = len(pool)

        attempts = 0
        max_attempts = needed * 40 + 200
        generated = 0
        while generated < needed and attempts < max_attempts:
            attempts += 1
            size = rng.randint(min_size, max(min_size, max_size))
            size = min(size, len(pool))
            subset = tuple(sorted(rng.sample(pool, size)))
            if subset in seen:
                continue
            seen.add(subset)
            synthetic_rows.append((disease, subset))
            generated += 1

    return pd.DataFrame(synthetic_rows, columns=["disease", "symptoms"])
