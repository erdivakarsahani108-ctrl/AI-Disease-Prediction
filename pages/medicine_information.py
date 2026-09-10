"""Medicine Information - educational reference only. NOT a prescription
or auto-recommendation system."""

import pandas as pd
import streamlit as st
from rapidfuzz import fuzz

from src.utils.paths import MEDICINE_DICTIONARY_CSV
from src.utils.ui import render_sidebar_common

render_sidebar_common()
st.title("💊 Medicine Information")
st.warning(
    "**Educational purposes only.** This module does NOT prescribe medicine or "
    "dosage based on symptoms. Always consult a qualified healthcare "
    "professional before taking, stopping, or changing any medication."
)

df = pd.read_csv(MEDICINE_DICTIONARY_CSV)
query = st.text_input("🔎 Search by medicine name, generic name or drug class", "")

filtered = df.copy()
if query:
    def score(row):
        haystack = f"{row['medicine_name']} {row['generic_name']} {row['drug_class']}".lower()
        return fuzz.partial_ratio(query.lower(), haystack)
    filtered["_score"] = filtered.apply(score, axis=1)
    filtered = filtered[filtered["_score"] >= 55].sort_values("_score", ascending=False)

st.caption(f"{len(filtered)} medicine(s) found · {len(df)} total curated entries")
st.info(
    "This is a small, hand-curated seed reference (general pharmacology "
    "knowledge, not a licensed drug database). **Future Scope:** integrate a "
    "licensed medicine API/dataset (e.g. RxNorm, openFDA) for comprehensive "
    "coverage - see docs/data_sources.md."
)

for _, row in filtered.iterrows():
    with st.expander(f"💊 {row['medicine_name']}  ({row['generic_name']})"):
        st.markdown(f"**Drug class:** {row['drug_class']}")
        st.markdown(f"**Common uses:** {row['common_uses']}")
        st.markdown(f"**Precautions:** {row['precautions']}")
        st.markdown(f"**Common side effects:** {row['common_side_effects']}")
        st.markdown(f"**⚠️ Important warnings:** {row['important_warnings']}")
        st.markdown(f"**Interactions:** {row['interactions']}")
        st.caption(f"Source: {row['source']} · Verified: {row['verification_date']}")
