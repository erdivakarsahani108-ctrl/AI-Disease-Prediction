"""Curated disease -> body-system category mapping.

This is an ORGANIZATIONAL taxonomy only (used for grouping/filtering in the
dashboard, dictionary and knowledge graph). It does not add, remove or alter
any clinical fact about a disease - it simply classifies the 41 disease
labels that already exist in the public source dataset into standard,
widely-taught body-system categories.
"""

DISEASE_CATEGORY_MAP = {
    "fungal infection": "Dermatological",
    "allergy": "Immunological",
    "gerd": "Gastrointestinal",
    "chronic cholestasis": "Hepatobiliary",
    "drug reaction": "Immunological",
    "peptic ulcer diseae": "Gastrointestinal",
    "aids": "Infectious Disease",
    "diabetes": "Endocrine",
    "gastroenteritis": "Gastrointestinal",
    "bronchial asthma": "Respiratory",
    "hypertension": "Cardiovascular",
    "migraine": "Neurological",
    "cervical spondylosis": "Musculoskeletal",
    "paralysis (brain hemorrhage)": "Neurological",
    "jaundice": "Hepatobiliary",
    "malaria": "Infectious Disease",
    "chicken pox": "Infectious Disease",
    "dengue": "Infectious Disease",
    "typhoid": "Infectious Disease",
    "hepatitis a": "Hepatobiliary",
    "hepatitis b": "Hepatobiliary",
    "hepatitis c": "Hepatobiliary",
    "hepatitis d": "Hepatobiliary",
    "hepatitis e": "Hepatobiliary",
    "alcoholic hepatitis": "Hepatobiliary",
    "tuberculosis": "Respiratory",
    "common cold": "Respiratory",
    "pneumonia": "Respiratory",
    "dimorphic hemmorhoids(piles)": "Gastrointestinal",
    "heart attack": "Cardiovascular",
    "varicose veins": "Cardiovascular",
    "hypothyroidism": "Endocrine",
    "hyperthyroidism": "Endocrine",
    "hypoglycemia": "Endocrine",
    "osteoarthristis": "Musculoskeletal",
    "arthritis": "Musculoskeletal",
    "(vertigo) paroymsal  positional vertigo": "Neurological (ENT)",
    "acne": "Dermatological",
    "urinary tract infection": "Renal / Urinary",
    "psoriasis": "Dermatological",
    "impetigo": "Dermatological",
}


def get_category(disease_name: str) -> str:
    key = str(disease_name).strip().lower()
    return DISEASE_CATEGORY_MAP.get(key, "Other / Uncategorized")
