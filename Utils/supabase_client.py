import os
from typing import Optional

import streamlit as st
from dotenv import load_dotenv


load_dotenv()


def _read_secret_or_env(name: str) -> Optional[str]:
    try:
        value = st.secrets.get(name)
    except Exception:
        value = None
    return value or os.getenv(name)


def is_supabase_configured() -> bool:
    url = _read_secret_or_env("SUPABASE_URL")
    key = _read_secret_or_env("SUPABASE_KEY")
    return bool(url and key)


def _normalize_supabase_url(url: str) -> str:
    cleaned = url.strip().rstrip("/")
    if cleaned.endswith("/rest/v1"):
        cleaned = cleaned[: -len("/rest/v1")]
    return cleaned


def get_supabase_client():
    url = _read_secret_or_env("SUPABASE_URL")
    key = _read_secret_or_env("SUPABASE_KEY")
    if not url or not key:
        return None

    try:
        from supabase import create_client
    except Exception:
        return None

    try:
        return create_client(_normalize_supabase_url(url), key)
    except Exception:
        return None
