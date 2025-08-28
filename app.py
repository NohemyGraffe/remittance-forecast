import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

st.set_page_config(page_title="US→MX Weekly Forecast", layout="wide")

DATA_CSV = "transactions_weekly.csv"
FORE_CSV = "weekly_forecast.csv"

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_CSV, parse_dates=["week_end"]).sort_values("week_end")
    fore = pd.read_csv(FORE_CSV, parse_dates=["week_end"]).sort_values("week_end")
    for df in (data, fore):
        df["week_end"] = pd.to_datetime(df["week_end"], utc=True).dt.tz_localize(None)
    return data, fore


def kpi_row(fore):
    nxt = fore.sort_values("week_end").iloc[0]
    tx_m  = float(nxt["pred_tx"])/1e6
    fx    = float(nxt["fx_assumed"]) if "fx_assumed" in nxt else None
    low_c = nxt.get("pred_low_cal", nxt.get("pred_low"))
    high_c= nxt.get("pred_high_cal", nxt.get("pred_high"))
    band  = f"{low_c/1e6:,.1f}–{high_c/1e6:,.1f}M" if pd.notna(low_c) and pd.notna(high_c) else "—"
    vcol  = "pred_value_mn_mxn"
    vlow  = nxt.get("pred_value_mn_mxn_low_cal", nxt.get("pred_value_mn_mxn_low"))
    vhigh = nxt.get("pred_value_mn_mxn_high_cal", nxt.get("pred_value_mn_mxn_high"))
    val   = f"MXN {float(nxt[vcol])/1e3:,.2f}B" if vcol in fore.columns else "—"
    vband = f"{vlow/1e3:,.2f}–{vhigh/1e3:,.2f}B" if vlow is not None and vhigh is not None else "—"
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Next week Tx", f"{tx_m:,.1f}M")
    c2.metric("Tx band", band)
    c3.metric("Next week Value", val)
    c4.metric("FX (USD/MXN)", f"{fx:,.2f}" if fx else "—")

def plot_tx(data, fore):
    fig, ax1 = plt.subplots(figsize=(11,5))
    ax1.plot(data["week_end"], data["tx_count"]/1e6, label="actual")
    ax1.plot(fore["week_end"], fore["pred_tx"]/1e6, label="forecast")
    low = fore.get("pred_low_cal", fore.get("pred_low"))
    high= fore.get("pred_high_cal", fore.get("pred_high"))
    if low is not None and high is not None:
        label = "forecast interval (calibrated)" if "pred_low_cal" in fore.columns else "forecast interval"
        ax1.fill_between(fore["week_end"], low/1e6, high/1e6, alpha=0.15, label=label)
    ax1.axvline(data["week_end"].max(), color="gray", ls=":", lw=1)
    ax1.set_xlabel("Week end (Sun)"); ax1.set_ylabel("Transactions (millions)")
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x,_: f"{x:.0f}M"))
    ax2 = ax1.twinx()
    fx_hist = data[["week_end","fx_rate"]]
    fx_fut = fore[["week_end","fx_assumed"]].rename(columns={"fx_assumed":"fx_rate"}) if "fx_assumed" in fore.columns else None
    fx_all = pd.concat([fx_hist, fx_fut]).drop_duplicates("week_end", keep="last").sort_values("week_end") if fx_fut is not None else fx_hist
    ax2.plot(fx_all["week_end"], fx_all["fx_rate"], ls="--", label="fx (USD/MXN)")
    ax2.set_ylabel("USD/MXN (weekly mean/assumed)")
    l1, lab1 = ax1.get_legend_handles_labels(); l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(l1+l2, lab1+lab2, loc="upper left")
    plt.title("Weekly transactions (SPEI proxy) vs forecast + USD/MXN FX")
    st.pyplot(fig)

def plot_value(data, fore):
    if "value_mn_mxn" not in data.columns or "pred_value_mn_mxn" not in fore.columns:
        st.info("Value fields not found – only transactions will be shown."); return
    fig, ax1 = plt.subplots(figsize=(11,5))
    ax1.plot(data["week_end"], data["value_mn_mxn"]/1_000, label="actual value")
    ax1.plot(fore["week_end"], fore["pred_value_mn_mxn"]/1_000, label="forecast value")
    low = fore.get("pred_value_mn_mxn_low_cal", fore.get("pred_value_mn_mxn_low"))
    high= fore.get("pred_value_mn_mxn_high_cal", fore.get("pred_value_mn_mxn_high"))
    if low is not None and high is not None:
        label = "forecast interval (calibrated)" if "pred_value_mn_mxn_low_cal" in fore.columns else "forecast interval"
        ax1.fill_between(fore["week_end"], low/1_000, high/1_000, alpha=0.15, label=label)
    ax1.axvline(data["week_end"].max(), color="gray", ls=":", lw=1)
    ax1.set_xlabel("Week end (Sun)"); ax1.set_ylabel("Value (MXN billions)")
    ax2 = ax1.twinx()
    fx_hist = data[["week_end","fx_rate"]]
    fx_fut = fore[["week_end","fx_assumed"]].rename(columns={"fx_assumed":"fx_rate"}) if "fx_assumed" in fore.columns else None
    fx_all = pd.concat([fx_hist, fx_fut]).drop_duplicates("week_end", keep="last").sort_values("week_end") if fx_fut is not None else fx_hist
    ax2.plot(fx_all["week_end"], fx_all["fx_rate"], ls="--", label="fx (USD/MXN)")
    ax2.set_ylabel("USD/MXN (weekly mean/assumed)")
    l1, lab1 = ax1.get_legend_handles_labels(); l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(l1+l2, lab1+lab2, loc="upper left")
    plt.title("Weekly value (MXN) vs forecast + USD/MXN FX")
    st.pyplot(fig)

data, fore = load_data()
# Trim history to 2023+ (matches your static plots)
data = data[data["week_end"] >= pd.Timestamp("2023-01-01")].copy()

st.title("US→MX weekly remittance proxy forecast")
kpi_row(fore)
plot_tx(data, fore)
plot_value(data, fore)
st.caption("Source: Banxico SIE (SPEI CF818, FIX SF43718). Model: Prophet + USD/MXN regressor, calibrated intervals.")
