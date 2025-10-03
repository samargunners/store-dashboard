# app.py — Store-Locked Metrics Dashboard (Labor, Sales, Guests, Voids, HME)

from __future__ import annotations
import os
from datetime import date, timedelta
import pandas as pd
import streamlit as st
import psycopg2
import psycopg2.extras as pgu

# =====================================================
# 1) STORE LOCK (no UI controls) + Page config
# =====================================================
STORE_PC = os.getenv("STORE_PC")
if not STORE_PC:
    # Fallback to Secrets -> [env]
    STORE_PC = (st.secrets.get("env", {}) or {}).get("STORE_PC")

# Normalize
if isinstance(STORE_PC, (int, float)):
    STORE_PC = str(int(STORE_PC))
if STORE_PC:
    STORE_PC = str(STORE_PC).strip()

if not STORE_PC:
    st.error("STORE_PC environment variable not set. Provide it under Secrets as:\n\n[env]\nSTORE_PC = \"343939\"\n\nOr set an OS env var named STORE_PC.")
    st.stop()


st.set_page_config(page_title=f"Store {STORE_PC} — Metrics Dashboard", layout="wide")
st.title(f"🏪 Store {STORE_PC} — Metrics Dashboard (Latest)")

# =====================================================
# 2) DB connection (Supabase Postgres via Streamlit secrets)
# =====================================================
# Expect st.secrets["supabase"] with keys: host, port, user, password, dbname, sslmode

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

# =====================================================
# 3) Helpers (dates, math, formatting)
# =====================================================

def get_period_dates(ref_date: date, period: str) -> tuple[date, date]:
    if period == "week":
        start = ref_date - timedelta(days=6)
        end = ref_date
    elif period == "month":
        start = ref_date.replace(day=1)
        end = ref_date
    elif period == "quarter":
        q = (ref_date.month - 1) // 3 + 1
        start = date(ref_date.year, 3 * q - 2, 1)
        end = ref_date
    elif period == "year":
        start = date(ref_date.year, 1, 1)
        end = ref_date
    else:
        start = ref_date
        end = ref_date
    return start, end


def get_prev_period_dates(ref_date: date, period: str) -> tuple[date, date]:
    if period == "week":
        end = ref_date - timedelta(days=7)
        start = end - timedelta(days=6)
    elif period == "month":
        first = ref_date.replace(day=1)
        end = first - timedelta(days=1)
        start = end.replace(day=1)
    elif period == "quarter":
        q = (ref_date.month - 1) // 3 + 1
        if q == 1:
            end = date(ref_date.year - 1, 12, 31)
            start = date(ref_date.year - 1, 10, 1)
        else:
            end = date(ref_date.year, 3 * (q - 1), 1) - timedelta(days=1)
            start = date(end.year, 3 * (q - 1) - 2, 1)
    elif period == "year":
        end = date(ref_date.year - 1, 12, 31)
        start = date(ref_date.year - 1, 1, 1)
    else:
        start = ref_date
        end = ref_date
    return start, end


def safe_div(a, b):
    try:
        return (a / b * 100.0) if b and b != 0 else None
    except Exception:
        return None


def weighted_avg(series, weights):
    if series is None or weights is None:
        return None
    s = pd.Series(series)
    w = pd.Series(weights)
    denom = w.sum(skipna=True)
    return float((s * w).sum(skipna=True) / denom) if denom and denom > 0 else None


def format_secs(x):
    return f"{x:.0f} sec" if x is not None else "N/A"

# =====================================================
# 4) Resolve latest business date for this store
#    (use the max date in sales_summary; fallback to labor_metrics)
# =====================================================
with get_supabase_connection() as conn, conn.cursor() as cur:
    cur.execute("SELECT max(date) FROM public.sales_summary WHERE pc_number = %s", (STORE_PC,))
    row = cur.fetchone()
    sales_max = row[0] if row else None

    cur.execute("SELECT max(date) FROM public.labor_metrics WHERE pc_number = %s", (STORE_PC,))
    row = cur.fetchone()
    labor_max = row[0] if row else None

if not sales_max and not labor_max:
    st.info("No data found yet for this store in sales_summary/labor_metrics.")
    st.stop()

END_DATE: date = date.today() - timedelta(days=1)
st.caption(f"Latest date (store-locked): {END_DATE} (fixed for layout testing)")

# =====================================================
# 5) LABOR % to SALES — Weekly / MTD / QTD / YTD
# =====================================================
periods = ["week", "month", "quarter", "year"]
labels = ["Weekly", "MTD", "QTD", "YTD"]

labor_metrics = []
with get_supabase_connection() as conn, conn.cursor(cursor_factory=pgu.RealDictCursor) as cur:
    for period in periods:
        s, e = get_period_dates(END_DATE, period)
        cur.execute(
              """
              SELECT COALESCE(SUM(total_pay),0) AS payroll_total
              FROM public.labor_metrics
              WHERE pc_number = %s AND date BETWEEN %s AND %s
              """,
              (STORE_PC, s, e),
        )
        payroll = float(cur.fetchone()["payroll_total"] or 0)

        cur.execute(
              """
              SELECT COALESCE(SUM(net_sales),0) AS sales_total
              FROM public.sales_summary
              WHERE pc_number = %s AND date BETWEEN %s AND %s
              """,
              (STORE_PC, s, e),
        )
        sales = float(cur.fetchone()["sales_total"] or 0)
        pct = safe_div(payroll, sales)
        labor_metrics.append((labels[periods.index(period)], pct))

st.markdown("## 💼 Labor Metrics")
cols = st.columns(4)
for i, (label, pct) in enumerate(labor_metrics):
    color = "#d4f7dc" if pct is not None and pct < 0.2 else ("#fff3cd" if pct is not None and pct < 0.3 else "#f8d7da")
    pct_display = f"{pct:.2f}%" if pct is not None else "N/A"
    cols[i].markdown(f"<div style='baackground-color:{color};padding:12px;border-radius:8px;text-align:center'>"
                    f"<b>Labor % to Sales ({label})</b><br><span style='font-size:1.5em'>{pct_display}</span>"
                    f"</div>", unsafe_allow_html=True)

# =====================================================
# 6) SALES % change — Weekly / MTD / QTD / YTD (vs previous)
# =====================================================
def period_sum(cur, table: str, col: str, s: date, e: date):
    cur.execute(
        f"SELECT COALESCE(SUM({col}),0) FROM public.{table} WHERE pc_number = %s AND date BETWEEN %s AND %s",
        (STORE_PC, s, e),
    )
    return float(cur.fetchone()[0] or 0)

sales_changes = []
with get_supabase_connection() as conn, conn.cursor() as cur:
    for period in periods:
        curr_s, curr_e = get_period_dates(END_DATE, period)
        prev_s, prev_e = get_prev_period_dates(END_DATE, period)
        curr_sales = period_sum(cur, "sales_summary", "net_sales", curr_s, curr_e)
        prev_sales = period_sum(cur, "sales_summary", "net_sales", prev_s, prev_e)
        change = safe_div(curr_sales - prev_sales, prev_sales)
        sales_changes.append((labels[periods.index(period)], change))

st.markdown("## 💵 Sales Metrics")
cols = st.columns(4)
for i, (label, change) in enumerate(sales_changes):
    display_change = change if change is not None else 0
    color = "#d4f7dc" if display_change > 0 else ("#f8d7da" if display_change < 0 else "#fff3cd")
    cols[i].markdown(f"<div style='background-color:{color};padding:12px;border-radius:8px;text-align:center'>"
                    f"<b>Sales % Change ({label})</b><br><span style='font-size:1.5em'>{display_change:.2f}%</span>"
                    f"</div>", unsafe_allow_html=True)

# =====================================================
# 7) GUEST COUNT % change — Weekly / MTD / QTD / YTD
# =====================================================
guest_changes = []
with get_supabase_connection() as conn, conn.cursor() as cur:
    for period in periods:
        curr_s, curr_e = get_period_dates(END_DATE, period)
        prev_s, prev_e = get_prev_period_dates(END_DATE, period)
        curr_guests = period_sum(cur, "sales_summary", "guest_count", curr_s, curr_e)
        prev_guests = period_sum(cur, "sales_summary", "guest_count", prev_s, prev_e)
        change = safe_div(curr_guests - prev_guests, prev_guests)
        guest_changes.append((labels[periods.index(period)], change))

st.markdown("## 👥 Guest Count Metrics")
cols = st.columns(4)
for i, (label, change) in enumerate(guest_changes):
    display_change = change if change is not None else 0
    color = "#d4f7dc" if display_change > 0 else ("#f8d7da" if display_change < 0 else "#fff3cd")
    cols[i].markdown(f"<div style='background-color:{color};padding:12px;border-radius:8px;text-align:center'>"
                    f"<b>Guest % Change ({label})</b><br><span style='font-size:1.5em'>{display_change:.2f}%</span>"
                    f"</div>", unsafe_allow_html=True)

# =====================================================
# 8) VOID COUNTS — Weekly / MTD / QTD / YTD
# =====================================================
void_counts = []
with get_supabase_connection() as conn, conn.cursor() as cur:
    for period in periods:
        s, e = get_period_dates(END_DATE, period)
        void_qty = period_sum(cur, "sales_summary", "void_qty", s, e)
        void_counts.append((labels[periods.index(period)], void_qty))

st.markdown("## 🧾 Void Counts")
cols = st.columns(4)
for i, (label, void_qty) in enumerate(void_counts):
    color = "#f8d7da" if void_qty is not None and void_qty > 0 else "#d4f7dc"
    cols[i].markdown(f"<div style='background-color:{color};padding:12px;border-radius:8px;text-align:center'>"
                    f"<b>Void Count ({label})</b><br><span style='font-size:1.5em'>{int(void_qty)}</span>"
                    f"</div>", unsafe_allow_html=True)

# --- METRICS BOX 5: Refund Metrics ---
st.markdown("## Refund Metrics")
# Use STORE_PC from environment
refund_values = []
with get_supabase_connection() as conn:
    for period in periods:
        s, e = get_period_dates(END_DATE, period)
        refund_total = pd.read_sql(
            "SELECT SUM(refund) as refund_total FROM public.sales_summary WHERE pc_number = %s AND date BETWEEN %s AND %s",
            conn, params=[STORE_PC, s, e])["refund_total"].iloc[0]
        refund_values.append((labels[periods.index(period)], refund_total))
cols = st.columns(4)
for i, (label, refund_total) in enumerate(refund_values):
    if refund_total is not None:
        cols[i].metric(f"Refunds ({label})", f"${refund_total:,.2f}")
    else:
        cols[i].metric(f"Refunds ({label})", "N/A")


# =====================================================
# 9) HME (Drive-Thru) Metrics — Weighted by Cars
#    • Period KPIs: Weekly / MTD / QTD / YTD (current vs previous)
#    • Daypart breakdown for the analysis window (last 7 days to END_DATE)
# =====================================================
st.markdown("## 🚗 HME (Drive-Thru) Metrics")

HME_TARGET_COLS = [
    "date", "time_measure", "total_cars", "menu_all", "greet_all",
    "service", "lane_queue", "lane_total"
]


def fetch_hme(store: str, start: date, end: date) -> pd.DataFrame:
    q = f"""
        SELECT {', '.join(HME_TARGET_COLS)}
        FROM public.hme_report
        WHERE store = %s AND date BETWEEN %s AND %s
    """
    with get_supabase_connection() as conn:
        df = pd.read_sql(q, conn, params=[store, start, end])
    return df

def summarize_hme(df: pd.DataFrame) -> dict[str, float | int | None]:
    if df is None or df.empty:
        return {"cars": 0, "menu_all": None, "greet_all": None, "service": None,
                "lane_queue": None, "lane_total": None}
    cars = df["total_cars"].fillna(0)
    return {
        "cars": int(cars.sum()),
        "menu_all": weighted_avg(df["menu_all"], cars),
        "greet_all": weighted_avg(df["greet_all"], cars),
        "service": weighted_avg(df["service"], cars),
        "lane_queue": weighted_avg(df["lane_queue"], cars),
        "lane_total": weighted_avg(df["lane_total"], cars),
    }

def pct_change(curr, prev):
    if prev is None or prev == 0 or curr is None:
        return None
    try:
        return (curr - prev) / prev * 100.0
    except Exception:
        return None

hme_labels = ["Weekly", "MTD", "QTD", "YTD"]
hme_periods = ["week", "month", "quarter", "year"]
hme_rows = []
for per, label in zip(hme_periods, hme_labels):
    curr_s, curr_e = get_period_dates(END_DATE, per)
    prev_s, prev_e = get_prev_period_dates(END_DATE, per)

    df_curr = fetch_hme(STORE_PC, curr_s, curr_e)
    df_prev = fetch_hme(STORE_PC, prev_s, prev_e)

    s_curr = summarize_hme(df_curr)
    s_prev = summarize_hme(df_prev)

    hme_rows.append({
        "label": label,
        "curr": s_curr,
        "prev": s_prev,
        "delta": {
            "cars": pct_change(s_curr["cars"], s_prev["cars"]) if s_prev["cars"] else None,
            "menu_all": pct_change(s_curr["menu_all"], s_prev["menu_all"]),
            "greet_all": pct_change(s_curr["greet_all"], s_prev["greet_all"]),
            "service": pct_change(s_curr["service"], s_prev["service"]),
            "lane_queue": pct_change(s_curr["lane_queue"], s_prev["lane_queue"]),
            "lane_total": pct_change(s_curr["lane_total"], s_prev["lane_total"]),
        }
    })

# KPI layout: Lane Total (primary), then Service, Greet, Menu, Cars
kpi_titles = [
    ("Lane Total (avg)", "lane_total"),
    ("Service (avg)", "service"),
    ("Greet (avg)", "greet_all"),
    ("Menu (avg)", "menu_all"),
    ("Cars (total)", "cars"),
]

for (title, key) in kpi_titles:
    cols = st.columns(4)
    for i, row in enumerate(hme_rows):
        curr_val = row["curr"][key]
        delta = row["delta"][key]

        if key == "cars":
            display = f"{int(curr_val) if curr_val is not None else 0}"
        else:
            display = format_secs(curr_val)

        if delta is None:
            cols[i].metric(f"{title} — {row['label']}", display)
        else:
            cols[i].metric(f"{title} — {row['label']}", display, f"{delta:.1f}%")

# Daypart breakdown for last 7 days up to END_DATE
ANALYSIS_START, ANALYSIS_END = END_DATE - timedelta(days=6), END_DATE
st.markdown("### Daypart Breakdown (last 7 days)")

df_sel = fetch_hme(STORE_PC, ANALYSIS_START, ANALYSIS_END)
if df_sel.empty:
    st.info("No HME records for the analysis window.")
else:
    agg = (
        df_sel
        .assign(total_cars=df_sel["total_cars"].fillna(0))
        .groupby("time_measure", dropna=False, as_index=False)
        .apply(lambda g: pd.Series({
            "total_cars": int(g["total_cars"].sum()),
            "menu_all_avg": weighted_avg(g["menu_all"], g["total_cars"]),
            "greet_all_avg": weighted_avg(g["greet_all"], g["total_cars"]),
            "service_avg": weighted_avg(g["service"], g["total_cars"]),
            "lane_queue_avg": weighted_avg(g["lane_queue"], g["total_cars"]),
            "lane_total_avg": weighted_avg(g["lane_total"], g["total_cars"]),
        }))
        .sort_values(by="time_measure")
        .reset_index(drop=True)
    )

    def fsec(v): return f"{v:.0f}" if v is not None else "—"
    agg_display = agg.copy()
    agg_display["menu_all_avg"] = agg_display["menu_all_avg"].map(fsec)
    agg_display["greet_all_avg"] = agg_display["greet_all_avg"].map(fsec)
    agg_display["service_avg"] = agg_display["service_avg"].map(fsec)
    agg_display["lane_queue_avg"] = agg_display["lane_queue_avg"].map(fsec)
    agg_display["lane_total_avg"] = agg_display["lane_total_avg"].map(fsec)
    agg_display.rename(columns={
        "time_measure": "Daypart",
        "total_cars": "Cars",
    }, inplace=True)

# ===============================
# Display all tables side-by-side
# ===============================
st.markdown("## 📊 All Tables Overview")
table_cols = st.columns(3)

# Sales Summary Table
with get_supabase_connection() as conn:
    sales_df = pd.read_sql(f"SELECT * FROM public.sales_summary WHERE pc_number = %s AND date BETWEEN %s AND %s ORDER BY date DESC", conn, params=[STORE_PC, END_DATE - timedelta(days=6), END_DATE])
table_cols[0].markdown("**Sales Summary**")
table_cols[0].dataframe(sales_df, height=250, use_container_width=True)

# Labor Metrics Table
with get_supabase_connection() as conn:
    labor_df = pd.read_sql(f"SELECT * FROM public.labor_metrics WHERE pc_number = %s AND date BETWEEN %s AND %s ORDER BY date DESC", conn, params=[STORE_PC, END_DATE - timedelta(days=6), END_DATE])
table_cols[1].markdown("**Labor Metrics**")
table_cols[1].dataframe(labor_df, height=250, use_container_width=True)

# HME Daypart Breakdown Table


# HME Daypart Breakdown Table (safe display)
import pandas as pd
if 'agg_display' in locals() and isinstance(agg_display, pd.DataFrame) and not agg_display.empty:
    agg_display.rename(columns={
        "menu_all_avg": "Menu (avg sec)",
        "greet_all_avg": "Greet (avg sec)",
        "service_avg": "Service (avg sec)",
        "lane_queue_avg": "Lane Queue (avg sec)",
        "lane_total_avg": "Lane Total (avg sec)",
    }, inplace=True)
    table_cols[2].markdown("**HME Daypart Breakdown**")
    table_cols[2].dataframe(agg_display, height=250, use_container_width=True)
else:
    table_cols[2].markdown("**HME Daypart Breakdown**")
    table_cols[2].info("No HME records for the analysis window.")

# Footer
st.caption("Data sources: public.sales_summary, public.labor_metrics, public.hme_report (Supabase)")

# ===============================
# Cards vs Charts Layout Switcher
# ===============================
st.markdown("# 📊 Dashboard Layout Options")
layout = st.radio("Choose dashboard layout:", ["Cards", "Charts"], horizontal=True)

if layout == "Cards":
    st.markdown("## 💼 Labor Metrics (Cards)")
    cols = st.columns(4)
    for i, (label, pct) in enumerate(labor_metrics):
        color = "#d4f7dc" if pct is not None and pct < 0.2 else ("#fff3cd" if pct is not None and pct < 0.3 else "#f8d7da")
        cols[i].markdown(f"<div style='background-color:{color};padding:12px;border-radius:8px;text-align:center'>"
                        f"<b>Labor % to Sales ({label})</b><br><span style='font-size:1.5em'>{pct:.2f}%</span>"
                        f"</div>", unsafe_allow_html=True)

    st.markdown("## 💵 Sales Metrics (Cards)")
    cols = st.columns(4)
    for i, (label, change) in enumerate(sales_changes):
        display_change = change if change is not None else 0
        color = "#d4f7dc" if display_change > 0 else ("#f8d7da" if display_change < 0 else "#fff3cd")
        cols[i].markdown(f"<div style='background-color:{color};padding:12px;border-radius:8px;text-align:center'>"
                        f"<b>Sales % Change ({label})</b><br><span style='font-size:1.5em'>{display_change:.2f}%</span>"
                        f"</div>", unsafe_allow_html=True)

    st.markdown("## 👥 Guest Count Metrics (Cards)")
    cols = st.columns(4)
    for i, (label, change) in enumerate(guest_changes):
        display_change = change if change is not None else 0
        color = "#d4f7dc" if display_change > 0 else ("#f8d7da" if display_change < 0 else "#fff3cd")
        cols[i].markdown(f"<div style='background-color:{color};padding:12px;border-radius:8px;text-align:center'>"
                        f"<b>Guest % Change ({label})</b><br><span style='font-size:1.5em'>{display_change:.2f}%</span>"
                        f"</div>", unsafe_allow_html=True)

    st.markdown("## 🧾 Void Counts (Cards)")
    cols = st.columns(4)
    for i, (label, void_qty) in enumerate(void_counts):
        color = "#f8d7da" if void_qty is not None and void_qty > 0 else "#d4f7dc"
        cols[i].markdown(f"<div style='background-color:{color};padding:12px;border-radius:8px;text-align:center'>"
                        f"<b>Void Count ({label})</b><br><span style='font-size:1.5em'>{int(void_qty)}</span>"
                        f"</div>", unsafe_allow_html=True)

    st.markdown("## 🚗 HME (Drive-Thru) Metrics (Cards)")
    kpi_titles = [
        ("Lane Total (avg)", "lane_total"),
        ("Service (avg)", "service"),
        ("Greet (avg)", "greet_all"),
        ("Menu (avg)", "menu_all"),
        ("Cars (total)", "cars"),
    ]
    for (title, key) in kpi_titles:
        cols = st.columns(4)
        for i, row in enumerate(hme_rows):
            curr_val = row["curr"][key]
            delta = row["delta"][key]
            if key == "cars":
                display = f"{int(curr_val) if curr_val is not None else 0}"
            else:
                display = format_secs(curr_val)
            if delta is None:
                cols[i].metric(f"{title} — {row['label']}", display)
            else:
                cols[i].metric(f"{title} — {row['label']}", display, f"{delta:.1f}%")

elif layout == "Charts":
    import numpy as np
    st.markdown("## 💼 Labor Metrics (Charts)")
    st.bar_chart({label: pct for label, pct in labor_metrics})

    st.markdown("## 💵 Sales Metrics (Charts)")
    st.bar_chart({label: change for label, change in sales_changes})

    st.markdown("## 👥 Guest Count Metrics (Charts)")
    st.bar_chart({label: change for label, change in guest_changes})

    st.markdown("## 🧾 Void Counts (Charts)")
    st.bar_chart({label: void_qty for label, void_qty in void_counts})

    st.markdown("## 🚗 HME (Drive-Thru) Metrics (Charts)")
    for (title, key) in kpi_titles:
        st.line_chart({row['label']: row['curr'][key] for row in hme_rows})

st.caption("Switch between Cards and Charts using the radio button above. Remove the layout you don't want.")
