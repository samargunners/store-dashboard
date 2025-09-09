from __future__ import annotations
import os
import re
from pathlib import Path
import pandas as pd
import streamlit as st
import psycopg2
import psycopg2.extras as pgu

# =========================
# Store lock via environment
# =========================
STORE_PC = os.getenv("STORE_PC")
if not STORE_PC:
    st.error("STORE_PC env var must be set for this deployment (e.g., 301290).")
    st.stop()

st.set_page_config(page_title=f"Store {STORE_PC} — HME", layout="wide")
st.title(f"🏪 Store {STORE_PC} — HME (Latest)")

# =========================
# DB connection helper
# =========================
# Expects a [supabase] section in st.secrets
# [supabase]
# host=..., port=6543, user=..., password=..., dbname=postgres, sslmode=require

def get_conn():
    cfg = st.secrets["supabase"]
    return psycopg2.connect(
        host=cfg["host"],
        port=cfg.get("port", 6543),
        user=cfg["user"],
        password=cfg["password"],
        dbname=cfg.get("dbname", "postgres"),
        sslmode=cfg.get("sslmode", "require"),
    )

# =========================
# Optional local ingester (CLI use only)
# =========================
# Mirrors your upload_hme_to_supabase.py so you can run `python app.py --ingest`
# locally or in a CI job. It is NO-OP during normal Streamlit runtime.

TARGET_COLS = [
    "date", "store", "time_measure", "total_cars", "menu_all",
    "greet_all", "service", "lane_queue", "lane_total"
]

INSERT_SQL = f"""
insert into public.hme_report ({",".join(TARGET_COLS)})
values (
    %(date)s, %(store)s, %(time_measure)s,
    %(total_cars)s, %(menu_all)s, %(greet_all)s,
    %(service)s, %(lane_queue)s, %(lane_total)s
)
ON CONFLICT (store, date, time_measure) DO UPDATE SET
    total_cars = EXCLUDED.total_cars,
    menu_all   = EXCLUDED.menu_all,
    greet_all  = EXCLUDED.greet_all,
    service    = EXCLUDED.service,
    lane_queue = EXCLUDED.lane_queue,
    lane_total = EXCLUDED.lane_total;
"""

THIS_FILE = Path(__file__).resolve()
DATA_DIR  = THIS_FILE.parent / "data" / "hme"

def _pc_number_from_store(store_text: str) -> int | None:
    if not isinstance(store_text, str):
        store_text = str(store_text) if store_text is not None else ""
    m = re.match(r"\s*(\d+)", store_text)
    return int(m.group(1)) if m else None

def _find_latest_transformed() -> Path | None:
    tdir = DATA_DIR / "transformed"
    cand = list(tdir.glob("hme_transformed.xlsx")) or list(tdir.glob("hme_transformed.csv"))
    if not cand:
        cand = sorted(tdir.glob("hme_transformed_*.xlsx"), reverse=True) + \
               sorted(tdir.glob("hme_transformed_*.csv"),  reverse=True)
    return max(cand, key=lambda p: p.stat().st_mtime) if cand else None

def ingest_latest_transformed():
    src = _find_latest_transformed()
    if not src:
        print("[INGEST] No transformed file found under data/hme/transformed/")
        return
    print(f"[INGEST] Loading {src.name}")
    if src.suffix.lower() == ".xlsx":
        df = pd.read_excel(src)
    else:
        df = pd.read_csv(src)

    needed = [
        "Date","store","time_measure","Total Cars","menu_all","greet_all",
        "service","lane_queue","lane_total"
    ]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in transformed file: {missing}")

    out = pd.DataFrame({
        "date": pd.to_datetime(df["Date"], errors="coerce").dt.date,
        "store": df["store"].apply(_pc_number_from_store),
        "time_measure": df["time_measure"].astype(str),
        "total_cars": pd.to_numeric(df["Total Cars"], errors="coerce").astype("Int64"),
        "menu_all":   pd.to_numeric(df["menu_all"],   errors="coerce").astype("Int64"),
        "greet_all":  pd.to_numeric(df["greet_all"],  errors="coerce").astype("Int64"),
        "service":    pd.to_numeric(df["service"],    errors="coerce").astype("Int64"),
        "lane_queue": pd.to_numeric(df["lane_queue"], errors="coerce").astype("Int64"),
        "lane_total": pd.to_numeric(df["lane_total"], errors="coerce").astype("Int64"),
    })
    out = out[out["store"].notna()].copy()
    out = out.astype(object).where(pd.notnull(out), None)

    with get_conn() as conn, conn.cursor() as cur:
        cur.executemany(INSERT_SQL, out.to_dict(orient="records"))
        conn.commit()
    print("[INGEST] Upsert complete.")

# =========================
# LIVE VIEW (latest date for STORE_PC)
# =========================
with get_conn() as conn, conn.cursor(cursor_factory=pgu.RealDictCursor) as cur:
    cur.execute(
        """
        SELECT max(date) AS max_date
        FROM public.hme_report
        WHERE store = %s
        """,
        (STORE_PC,),
    )
    row = cur.fetchone()
    if not row or not row["max_date"]:
        st.info("No HME data yet for this store.")
        st.stop()

    max_date = row["max_date"]
    st.caption(f"Latest date: {max_date}")

    cur.execute(
        """
        SELECT date, store, time_measure, total_cars, menu_all, greet_all, service, lane_queue, lane_total
        FROM public.hme_report
        WHERE store = %s AND date = %s
        ORDER BY time_measure
        """,
        (STORE_PC, max_date),
    )
    rows = cur.fetchall()

df = pd.DataFrame(rows)

# =========================
# KPIs
# =========================
left, mid1, mid2, right = st.columns(4)
left.metric("Total Cars (sum)", int(df["total_cars"].fillna(0).sum()))
mid1.metric("Menu All (avg)", round(df["menu_all"].mean(), 2))
mid2.metric("Greet All (avg)", round(df["greet_all"].mean(), 2))
right.metric("Service (avg)", round(df["service"].mean(), 2))

c1, c2, c3 = st.columns(3)
c1.metric("Lane Queue (avg)", round(df["lane_queue"].mean(), 2))
c2.metric("Lane Total (avg)", round(df["lane_total"].mean(), 2))
c3.metric("Time Buckets", df["time_measure"].nunique())

# =========================
# Tables & Charts
# =========================
st.subheader("Lane & Service by Time")
show_cols = ["time_measure", "lane_queue", "lane_total", "menu_all", "greet_all", "service", "total_cars"]
st.dataframe(df[show_cols], use_container_width=True)

st.subheader("Trends (by time_measure)")
line_df = df.copy()
try:
    line_df["_tm_sort"] = pd.to_datetime(line_df["time_measure"], format="%H:%M").dt.time
    line_df = line_df.sort_values("_tm_sort")
except Exception:
    line_df = line_df.sort_values("time_measure")

st.line_chart(line_df.set_index("time_measure")[
    ["total_cars", "lane_total", "lane_queue", "menu_all", "greet_all", "service"]
])

# =========================
# Footer
# =========================
st.caption("Data source: public.hme_report (Supabase)")

# =========================
# CLI entrypoint for ingestion
# =========================
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--ingest":
        ingest_latest_transformed()
        print("Done.")
    else:
        print("This module is meant to be run by Streamlit. For ingestion use: python app.py --ingest")
