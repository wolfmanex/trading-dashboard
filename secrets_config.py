import os

import streamlit as st


def get_configured_secret(name: str, default: str = "") -> str:
    """Read a Streamlit secret when available, then fall back to the environment."""
    try:
        return st.secrets.get(name) or os.getenv(name, default)
    except Exception:
        return os.getenv(name, default)
