import os

import streamlit as st


def get_configured_secret(name: str, default: str = "") -> str:
    """Read a Streamlit secret when available, then the environment; empty values count as unset."""
    try:
        return st.secrets.get(name) or os.getenv(name) or default
    except Exception:
        return os.getenv(name) or default
