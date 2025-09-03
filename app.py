import os
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from predict_weekly import (
    train_and_forecast,
    _make_calendar_flags,
    _apply_calibration,
)

# Live fetch (Banxico)
try:
    from fetch_real_data import fetch_weekly_from_banxico
    HAS_LIVE = True
except Exception:
    HAS_LIVE = False

# -----------------------------
# Small helpers for display
# -----------------------------
def fmt_int(x):
    try:
        return f"{int(round(float(x))):,}"
    except Exception:
        return "-"

def fmt_mxn_billions_from_millions(mn_mxn):
    if pd.isna(mn_mxn):
        return "-"
    b = (float(mn_mxn) * 1_000_000) / 1_000_000_000
    return f"${b:,.2f} B MXN"

#def fmt_usd_from_millions_and_fx(mn_mxn, fx):
    #if pd.isna(mn_mxn) or pd.isna(fx) or float(fx) == 0:
     #   return "-"
    #usd_mn = float(mn_mxn) / float(fx)  # USD (millions)
   # return f"${usd_mn:,.1f} M USD"

def pct(x):
    if pd.isna(x):
        return "-"
    try:
        return f"{float(x)*100:.1f}%"
    except Exception:
        return "-"

def week_range_label(week_end_ts: pd.Timestamp) -> str:
    end = pd.Timestamp(week_end_ts).normalize()
    start = end - pd.Timedelta(days=6)
    start_str = start.strftime("%b %-d, %Y") if os.name != "nt" else start.strftime("%b %#d, %Y")
    end_str   = end.strftime("%b %-d, %Y")   if os.name != "nt" else end.strftime("%b %#d, %Y")
    return f"{start_str} – {end_str}"

def fmt_count_millions(n, decimals=1):
    try:
        return f"{float(n)/1_000_000:.{decimals}f}M"
    except Exception:
        return "-"

def fmt_mxn_compact_from_mn(mn_mxn):
    try:
        bn = float(mn_mxn) / 1_000
        return f"${bn:,.2f} B MXN"  # ✅ Always show billions
    except Exception:
        return "-"

def fmt_usd_in_millions_from_raw_mxn(mxn_val, fx):
    """
    Takes full MXN value + FX → returns formatted USD in millions.
    """
    try:
        if pd.isna(mxn_val) or pd.isna(fx) or float(fx) == 0:
            return "-"
        usd_total = float(mxn_val) / float(fx)             # full USD
        usd_millions = usd_total / 1_000_000               # in millions
        return f"${usd_millions:,.1f} M USD"
    except Exception:
        return "-"




# -----------------------------
# Streamlit config
# -----------------------------
st.set_page_config(page_title="Remittance Forecast (Weekly)", layout="wide")
st.title("Remittance Forecast — Weekly (Business View)")
st.caption(
    "This dashboard shows a weekly cash & volume forecast using a domestic transfer proxy (SPEI) "
    "with calendar and FX effects. It’s designed for business users. Technical details and model "
    "performance live in the project README."
)

# -----------------------------
# Token resolver
# -----------------------------
def get_banxico_token() -> str:
    """
    Prefer Streamlit secrets, then env vars.
    Expects `.streamlit/secrets.toml` to contain:
      DEFAULT_BANXICO_TOKEN = "your_token_here"
    """
    tok = None
    try:
        # Will raise KeyError if not present
        tok = st.secrets.get("DEFAULT_BANXICO_TOKEN")
    except Exception:
        # st.secrets may not exist outside Streamlit runtime; ignore
        pass

    if not tok:
        tok = os.getenv("BANXICO_TOKEN") or os.getenv("DEFAULT_BANXICO_TOKEN")

    if not tok:
        # Helpful message + optionally show available keys
        available = []
        try:
            available = list(st.secrets.keys())
        except Exception:
            pass
        msg = (
            'DEFAULT_BANXICO_TOKEN not found.\n'
            'Add it to `.streamlit/secrets.toml` like:\n\n'
            'DEFAULT_BANXICO_TOKEN = "your_token_here"\n\n'
            f"(Detected secrets keys: {available})"
        )
        st.error(msg)
        st.stop()
    return tok

# -----------------------------
# Cached loaders
# -----------------------------
@st.cache_data(ttl=24*60*60)
def load_sample_weekly():
    base = Path(__file__).parent
    sample_path = base / "sample_data" / "transactions_weekly.csv"
    if not sample_path.exists():
        raise FileNotFoundError(f"Sample file not found at: {sample_path}")
    return pd.read_csv(sample_path, parse_dates=["week_end"])

@st.cache_data(ttl=6*60*60)  # live cache: 6 hours
def load_live_weekly(start="2018-01-01"):
    if not HAS_LIVE:
        raise RuntimeError("Live mode not available: fetch_weekly_from_banxico not imported.")
    token = get_banxico_token()
    df = fetch_weekly_from_banxico(start=start, token=token)
    return df

# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.header("Controls")
    horizon = st.slider("Forecast horizon (weeks)", 1, 12, 4)
    use_cal = st.checkbox("Use calibration (cal_params.json if available)", value=True)



    refresh = st.button("↻ Refresh (re-fetch & clear cache)")

# Clear caches if user clicked refresh
if refresh:
    st.cache_data.clear()
    try:
        st.cache_resource.clear()  # clears any @st.cache_resource (e.g., models)
    except Exception:
        pass

# -----------------------------
# Load data (one source)
# -----------------------------
try:
    with st.spinner("Fetching latest data from Banxico…"):
        weekly_df = load_live_weekly(start="2018-01-01")
    st.caption(
        f"Live data • Latest week_end: {pd.to_datetime(weekly_df['week_end']).max().date()} • "
        f"Last updated: {pd.Timestamp.utcnow():%Y-%m-%d %H:%M} UTC"
    )
except Exception as e:
    st.error(f"Could not fetch live data: {e}")
    st.stop()


weekly_df["week_end"] = pd.to_datetime(weekly_df["week_end"]).dt.tz_localize(None)
weekly_df = weekly_df.sort_values("week_end").reset_index(drop=True)

# -----------------------------
# Forecast
# -----------------------------
fc = train_and_forecast(weekly_df, horizon_weeks=horizon, ticket_window=8)

# Optional calibration
if use_cal:
    try:
        cal_path = Path(__file__).parent / "cal_params.json"
        fc = _apply_calibration(fc, str(cal_path) if cal_path.exists() else None)
    except Exception as e:
        st.warning(f"Calibration skipped: {e}")

# -----------------------------
# Prepare future rows + flags
# -----------------------------
hist_last = weekly_df["week_end"].max()
future_only = fc[fc["week_end"] > hist_last].copy().sort_values("week_end")
if future_only.empty:
    future_only = fc.copy().sort_values("week_end")

cal_flags_future = _make_calendar_flags(future_only["week_end"])
future_only = future_only.merge(
    cal_flags_future[["week_end", "is_payweek", "is_holiday_us", "is_holiday_mx"]],
    on="week_end", how="left"
)
next_row = future_only.iloc[0]


# -----------------------------
# KPI cards
# -----------------------------
range_text = week_range_label(next_row["week_end"])
st.subheader(f"📅 Next Forecasted Week — {range_text}")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Transactions (M)", fmt_count_millions(next_row.get("pred_tx"), 1))
with c2:
    st.metric("Value (MXN)", fmt_mxn_compact_from_mn(next_row.get("pred_value_mn_mxn")))
    
with c3:
   st.metric("Value (USD)", fmt_usd_in_millions_from_raw_mxn(
    next_row.get("pred_value_mn_mxn"),
    next_row.get("fx_assumed")
))


with c4:
    # was: if {"pred_low", "pred_high", "pred_tx"}.issubset(fc.columns):
     if {"pred_low", "pred_high", "pred_tx"}.issubset(future_only.columns):
         rel = (next_row["pred_high"] - next_row["pred_low"]) / max(next_row["pred_tx"], 1.0)
         st.metric("Uncertainty (± relative)", pct(rel / 2.0))
     else:
          st.metric("Uncertainty (± relative)", "—")


c5, c6, c7, c8 = st.columns(4)
with c5:
    st.metric("US Payweek in window?", "Yes" if int(next_row.get("is_payweek", 0)) == 1 else "No")
with c6:
    st.metric("US Holiday in week?", "Yes" if int(next_row.get("is_holiday_us", 0)) == 1 else "No")
with c7:
    st.metric("MX Holiday in week?", "Yes" if int(next_row.get("is_holiday_mx", 0)) == 1 else "No")
with c8:
    fx_assumed = next_row.get("fx_assumed", np.nan)
    # was: "FX Assumed (USD/MXN)"
    st.metric("FX Assumed (MXN per USD)", f"{fx_assumed:,.2f}" if pd.notna(fx_assumed) else "—")


if "avg_ticket_mxn_used" in fc.columns and pd.notna(next_row.get("avg_ticket_mxn_used")):
    st.caption(f"Avg ticket used: ${next_row['avg_ticket_mxn_used']:,.0f} MXN")

# Sanity check: avg USD per transaction, catches unit/FX slips
if {"pred_value_mn_mxn","fx_assumed","pred_tx"}.issubset(next_row.index):
    try:
        usd_mn = float(next_row["pred_value_mn_mxn"]) / float(next_row["fx_assumed"])
        avg_tx_usd = (usd_mn * 1_000_000) / max(float(next_row["pred_tx"]), 1.0)
        if 50 <= avg_tx_usd <= 1000:
            st.caption(f"Avg per transaction (sanity): ${avg_tx_usd:,.0f} USD")
        else:
            st.caption("⚠️ Avg per transaction looks off; recheck units/FX.")
    except Exception:
        pass


st.divider()

# -----------------------------
# Plot 1: Weekly Trend — last 26w actuals + forecast
# -----------------------------
st.subheader("📈 Weekly Trend — Actuals vs Forecast")

lookback_weeks = 26
recent_hist = weekly_df.tail(lookback_weeks).copy()

fig1, ax1 = plt.subplots(figsize=(11, 4))
ax1.plot(recent_hist["week_end"], recent_hist["tx_count"], marker="o", linewidth=1.5, label="Actual Transactions")
ax1.plot(future_only["week_end"], future_only["pred_tx"], marker="x", linewidth=2.0, label="Forecast Transactions")
if {"pred_low", "pred_high"}.issubset(future_only.columns):
    ax1.fill_between(future_only["week_end"], future_only["pred_low"], future_only["pred_high"], alpha=0.2, label="Confidence band")
ax1.set_xlabel("Week Ending (Sundays)")
ax1.set_ylabel("Transactions")
ax1.grid(True)
ax1.legend(loc="upper left")
st.pyplot(fig1, clear_figure=True)

# -----------------------------
# Plot 2: Cash Needs — upcoming payout (USD millions; y-axis hidden)
# -----------------------------
st.subheader("Cash Needs — Upcoming Payout (USD, millions)")

cash_tbl = future_only.copy()
if {"pred_value_mn_mxn", "fx_assumed"}.issubset(cash_tbl.columns):
    cash_tbl["payout_usd_mn"] = cash_tbl["pred_value_mn_mxn"] / cash_tbl["fx_assumed"]
else:
    cash_tbl["payout_usd_mn"] = np.nan

fig2, ax2 = plt.subplots(figsize=(11, 3.8))
x_labels = cash_tbl["week_end"].dt.strftime("%Y-%m-%d")
ax2.bar(x_labels, cash_tbl["payout_usd_mn"])
ax2.set_yticks([])
ax2.set_xlabel("Week Ending (Sundays)")
ax2.set_ylabel("USD (mn)")


vals = cash_tbl["payout_usd_mn"]
y_offset = max(vals.max() * 0.01, 0.02) if pd.notna(vals.max()) else 0.02
for i, (x, y) in enumerate(zip(range(len(x_labels)), vals)):
    if pd.notna(y):
        ax2.text(i, y + y_offset, f"{y:,.0f}", ha="center", va="bottom", fontsize=9, weight="bold")


st.pyplot(fig2, clear_figure=True)

st.caption(
    "Notes: Total value (MXN, mn) = predicted transactions × recent avg ticket (MXN); "
    "Value (USD, mn) = Value (MXN, mn) ÷ FX (MXN per USD). "
    "Holidays/payweeks reflect whether those days fall within each forecasted week."
)

