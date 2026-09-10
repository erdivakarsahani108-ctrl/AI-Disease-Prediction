"""Disease Dictionary - searchable structured knowledge base."""

import pandas as pd
import streamlit as st
from rapidfuzz import fuzz

from src.utils.paths import DISEASE_DICTIONARY_CSV
from src.utils.ui import render_disclaimer_banner, render_sidebar_common

render_sidebar_common()
st.title("📖 Disease Dictionary")
st.caption("Search by disease name, category or symptom. Backed by a structured knowledge base, not solely an LLM.")
render_disclaimer_banner()

df = pd.read_csv(DISEASE_DICTIONARY_CSV)

col1, col2 = st.columns([2, 1])
query = col1.text_input("🔎 Search disease, alias or symptom", "")
category_filter = col2.selectbox("Filter by category", ["All"] + sorted(df["category"].unique().tolist()))

filtered = df.copy()
if category_filter != "All":
    filtered = filtered[filtered["category"] == category_filter]

if query:
    def match_score(row):
        haystack = f"{row['disease']} {row['common_symptoms']} {row['category']}".lower()
        return fuzz.partial_ratio(query.lower(), haystack)

    filtered["_score"] = filtered.apply(match_score, axis=1)
    filtered = filtered[filtered["_score"] >= 60].sort_values("_score", ascending=False)

st.caption(f"{len(filtered)} disease(s) found")

for _, row in filtered.iterrows():
    with st.expander(f"🩺 {row['disease']}  ·  {row['category']}"):
        st.markdown(f"**Description:** {row['description']}")
        st.markdown(f"**Common symptoms:** {row['common_symptoms']}")
        st.markdown(f"**Total known symptoms in dataset:** {row['total_known_symptoms']}")
        st.markdown(f"**General precautions:** {row['general_precautions']}")
        st.markdown(f"**Risk factors:** {row['risk_factors']}")
        st.markdown(f"**When to seek professional care:** {row['when_to_seek_professional_care']}")
        st.markdown(f"**Similar diseases (data-driven, symptom-overlap):** {row['similar_diseases']}")
        st.caption(f"Source: {row['source']} · Verified: {row['verification_date']}")
