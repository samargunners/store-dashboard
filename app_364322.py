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
st.title(f"🏪 Store {STORE_PC} — Metrics Dashboard (Rolling Periods)")

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
    """Get dynamic rolling period dates - always equal length periods"""
    if period == "week":
        # Last 7 days
        start = ref_date - timedelta(days=6)
        end = ref_date
    elif period == "month":
        # Last 30 days
        start = ref_date - timedelta(days=29)
        end = ref_date
    elif period == "quarter":
        # Last 90 days
        start = ref_date - timedelta(days=89)
        end = ref_date
    elif period == "year":
        # Last 365 days
        start = ref_date - timedelta(days=364)
        end = ref_date
    else:
        start = ref_date
        end = ref_date
    return start, end


def get_prev_period_dates(ref_date: date, period: str) -> tuple[date, date]:
    """Get previous dynamic rolling period dates - same length as current period"""
    if period == "week":
        # Previous 7 days (days 8-14 ago)
        end = ref_date - timedelta(days=7)
        start = end - timedelta(days=6)
    elif period == "month":
        # Previous 30 days (days 31-60 ago)
        end = ref_date - timedelta(days=30)
        start = end - timedelta(days=29)
    elif period == "quarter":
        # Previous 90 days (days 91-180 ago)
        end = ref_date - timedelta(days=90)
        start = end - timedelta(days=89)
    elif period == "year":
        # Previous 365 days (days 366-730 ago)
        end = ref_date - timedelta(days=365)
        start = end - timedelta(days=364)
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


def get_metric_color(value, is_change=False, lower_is_better=False, thresholds=None):
    """
    Get color for metric display based on whether lower or higher is better.
    
    Args:
        value: The metric value
        is_change: Whether this is a change percentage (vs absolute value)
        lower_is_better: If True, lower values get green, higher get red
        thresholds: Optional dict with 'good' and 'bad' thresholds for absolute values
    
    Returns:
        Color hex code for background
    """
    if value is None:
        return "#fff3cd"  # Yellow for N/A
    
    if is_change:
        # For percentage changes
        if lower_is_better:
            # Lower (negative) changes are better
            if value < -2:
                return "#d4f7dc"  # Green - significant improvement
            elif value < 0:
                return "#fff3cd"  # Yellow - slight improvement
            else:
                return "#f8d7da"  # Red - getting worse
        else:
            # Higher (positive) changes are better
            if value > 2:
                return "#d4f7dc"  # Green - significant improvement
            elif value > 0:
                return "#fff3cd"  # Yellow - slight improvement
            else:
                return "#f8d7da"  # Red - getting worse
    else:
        # For absolute values
        if thresholds:
            good_threshold = thresholds.get('good', 0)
            bad_threshold = thresholds.get('bad', 0)
            
            if lower_is_better:
                if value <= good_threshold:
                    return "#d4f7dc"  # Green - good
                elif value <= bad_threshold:
                    return "#fff3cd"  # Yellow - okay
                else:
                    return "#f8d7da"  # Red - bad
            else:
                if value >= good_threshold:
                    return "#d4f7dc"  # Green - good
                elif value >= bad_threshold:
                    return "#fff3cd"  # Yellow - okay
                else:
                    return "#f8d7da"  # Red - bad
        else:
            # Default behavior for absolute values without thresholds
            if lower_is_better:
                return "#d4f7dc" if value == 0 else "#f8d7da"
            else:
                return "#d4f7dc" if value > 0 else "#f8d7da"

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
labels = ["Last 7 Days", "Last 30 Days", "Last 90 Days", "Last 365 Days"]

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
    # Labor % to Sales: lower is better (pct is already in percentage form)
    color = get_metric_color(pct, is_change=False, lower_is_better=True, 
                           thresholds={'good': 20, 'bad': 30})
    pct_display = f"{pct:.2f}%" if pct is not None else "N/A"
    cols[i].markdown(f"<div style='background-color:{color};padding:12px;border-radius:8px;text-align:center'>"
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

# Collect sales dollar amounts for display
sales_amounts = []
with get_supabase_connection() as conn, conn.cursor() as cur:
    for period in periods:
        curr_s, curr_e = get_period_dates(END_DATE, period)
        curr_sales = period_sum(cur, "sales_summary", "net_sales", curr_s, curr_e)
        sales_amounts.append((labels[periods.index(period)], curr_sales))

st.markdown("## 💵 Sales Metrics")
cols = st.columns(4)
for i, ((label, change), (_, amount)) in enumerate(zip(sales_changes, sales_amounts)):
    display_change = change if change is not None else 0
    sales_display = f"${amount:,.0f}" if amount is not None else "$0"
    # Sales: higher is better
    color = get_metric_color(display_change, is_change=True, lower_is_better=False)
    cols[i].markdown(f"<div style='background-color:{color};padding:12px;border-radius:8px;text-align:center'>"
                    f"<b>Sales % Change ({label})</b><br><span style='font-size:1.5em'>{display_change:.2f}%</span>"
                    f"<br><small>{sales_display}</small>"
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

# Collect guest count numbers for display
guest_counts = []
with get_supabase_connection() as conn, conn.cursor() as cur:
    for period in periods:
        curr_s, curr_e = get_period_dates(END_DATE, period)
        curr_guests = period_sum(cur, "sales_summary", "guest_count", curr_s, curr_e)
        guest_counts.append((labels[periods.index(period)], curr_guests))

st.markdown("## 👥 Guest Count Metrics")
cols = st.columns(4)
for i, ((label, change), (_, count)) in enumerate(zip(guest_changes, guest_counts)):
    display_change = change if change is not None else 0
    count_display = f"{int(count):,}" if count is not None else "0"
    # Guest Count: higher is better
    color = get_metric_color(display_change, is_change=True, lower_is_better=False)
    cols[i].markdown(f"<div style='background-color:{color};padding:12px;border-radius:8px;text-align:center'>"
                    f"<b>Guest % Change ({label})</b><br><span style='font-size:1.5em'>{display_change:.2f}%</span>"
                    f"<br><small>{count_display} guests</small>"
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
    # Void Counts: lower is better (0 is best)
    color = get_metric_color(void_qty, is_change=False, lower_is_better=True,
                           thresholds={'good': 0, 'bad': 5})
    cols[i].markdown(f"<div style='background-color:{color};padding:12px;border-radius:8px;text-align:center'>"
                    f"<b>Void Count ({label})</b><br><span style='font-size:1.5em'>{int(void_qty)}</span>"
                    f"</div>", unsafe_allow_html=True)

# =====================================================
# 5) REFUND METRICS — Weekly / MTD / QTD / YTD
# =====================================================
refund_values = []
with get_supabase_connection() as conn:
    for period in periods:
        s, e = get_period_dates(END_DATE, period)
        refund_total = pd.read_sql(
            "SELECT SUM(refund) as refund_total FROM public.sales_summary WHERE pc_number = %s AND date BETWEEN %s AND %s",
            conn, params=[STORE_PC, s, e])["refund_total"].iloc[0]
        refund_values.append((labels[periods.index(period)], refund_total))

st.markdown("## 💳 Refund Metrics")
cols = st.columns(4)
for i, (label, refund_total) in enumerate(refund_values):
    # Refund Metrics: lower is better
    color = get_metric_color(refund_total, is_change=False, lower_is_better=True,
                           thresholds={'good': 0, 'bad': 100})
    refund_display = f"${refund_total:,.2f}" if refund_total is not None else "N/A"
    cols[i].markdown(f"<div style='background-color:{color};padding:12px;border-radius:8px;text-align:center'>"
                    f"<b>Refunds ({label})</b><br><span style='font-size:1.5em'>{refund_display}</span>"
                    f"</div>", unsafe_allow_html=True)


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

# KPI layout: Lane Total, Greet, Menu, Service, Cars
kpi_titles = [
    ("Lane Total (avg)", "lane_total", True),   # lower is better
    ("Greet (avg)", "greet_all", True),         # lower is better
    ("Menu (avg)", "menu_all", True),           # lower is better
    ("Service (avg)", "service", True),         # lower is better
    ("Cars (total)", "cars", False),            # higher is better
]

for (title, key, lower_is_better) in kpi_titles:
    cols = st.columns(4)
    for i, row in enumerate(hme_rows):
        curr_val = row["curr"][key]
        delta = row["delta"][key]

        if key == "cars":
            display = f"{int(curr_val) if curr_val is not None else 0}"
        else:
            display = format_secs(curr_val)

        if delta is None:
            # Use standard metric without delta
            cols[i].metric(f"{title} — {row['label']}", display)
        else:
            # Custom colored metric for proper "better" direction
            delta_color = get_metric_color(delta, is_change=True, lower_is_better=lower_is_better)
            delta_symbol = "📉" if (lower_is_better and delta < 0) or (not lower_is_better and delta > 0) else "📈"
            delta_display = f"{delta:.1f}%"
            
            # Create custom metric display with proper colors
            cols[i].markdown(f"""
                <div style='border:1px solid #ddd;padding:12px;border-radius:8px;background-color:white'>
                    <div style='font-size:0.9em;color:#666;margin-bottom:4px'>{title} — {row['label']}</div>
                    <div style='font-size:1.8em;font-weight:bold;margin-bottom:4px'>{display}</div>
                    <div style='background-color:{delta_color};padding:4px 8px;border-radius:4px;font-size:0.9em'>
                        {delta_symbol} {delta_display}
                    </div>
                </div>
            """, unsafe_allow_html=True)

