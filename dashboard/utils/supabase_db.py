from __future__ import annotations
import streamlit as st
import psycopg2


# Expects a [supabase] section in st.secrets
# Example:
# [supabase]
# host = "db.ertcdieopoecjddamgkx.supabase.co"
# port = 6543
# user = "postgres"
# password = "..."
# dbname = "postgres"
# sslmode = "require"


def get_supabase_connection():
    cfg = st.secrets["supabase"]
    return psycopg2.connect(
        host=cfg["host"],
        port=cfg.get("port", 6543),
        user=cfg["user"],
        password=cfg["password"],
        dbname=cfg.get("dbname", "postgres"),
        sslmode=cfg.get("sslmode", "require"),
        )