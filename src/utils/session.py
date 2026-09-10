"""Shared Streamlit session-state helpers.

Keeping conversation-relevant state ONLY in `st.session_state` (in-memory,
per-browser-session) - nothing is persisted to disk beyond the anonymous
feedback log, per the project's Privacy requirements.
"""

from __future__ import annotations

import streamlit as st


def init_session_state() -> None:
    defaults = {
        "chat_history": [],  # list of {"role": "user"/"assistant", "content": str}
        "known_symptoms": set(),
        "denied_symptoms": set(),
        "last_result": None,
        "conversation_started": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_session() -> None:
    for key in ["chat_history", "known_symptoms", "denied_symptoms", "last_result", "conversation_started"]:
        if key in st.session_state:
            del st.session_state[key]
    init_session_state()
