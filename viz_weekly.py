#!/usr/bin/env python3
import argparse, pandas as pd, matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

DEF_DATA = "transactions_weekly.csv"
DEF_FORE = "weekly_forecast.csv"

def make_fx_series(data, fore):
    fx_hist = data[["week_end","fx_rate"]]
    if "fx_assumed" in fore.columns:
        fx_fut = fore[["week_end","fx_assumed"]].rename(columns={"fx_assumed":"fx_rate"})
        fx_all = pd.concat([fx_hist, fx_fut]).drop_duplicates("week_end", keep="last").sort_values("week_end")
    else:
        fx_all = fx_hist
    return fx_all

def next_week_row(fore: pd.DataFrame) -> pd.Series:
    """Return the first (earliest) future week row from forecast."""
    return fore.sort_values("week_end").iloc[0]

def choose_band_cols_tx(fore: pd.DataFrame):
    """Pick transaction band columns, preferring calibrated if available."""
    if {"pred_low_cal","pred_high_cal"}.issubset(fore.columns):
        return "pred_low_cal", "pred_high_cal", "forecast interval (calibrated)"
    if {"pred_low","pred_high"}.issubset(fore.columns):
        return "pred_low", "pred_high", "forecast interval"
    return None, None, None

def choose_band_cols_val(fore: pd.DataFrame):
    """Pick value band columns, preferring calibrated if available."""
    if {"pred_value_mn_mxn_low_cal","pred_value_mn_mxn_high_cal"}.issubset(fore.columns):
        return "pred_value_mn_mxn_low_cal", "pred_value_mn_mxn_high_cal", "forecast interval (calibrated)"
    if {"pred_value_mn_mxn_low","pred_value_mn_mxn_high"}.issubset(fore.columns):
        return "pred_value_mn_mxn_low", "pred_value_mn_mxn_high", "forecast interval"
    return None, None, None

def plot_tx(data, fore, out="weekly_plot_tx.png"):
    fig, ax1 = plt.subplots(figsize=(11,5))

    # Actual & forecast (scale to millions for readability)
    ax1.plot(data["week_end"], data["tx_count"]/1_000_000, label="actual")
    ax1.plot(fore["week_end"], fore["pred_tx"]/1_000_000, label="forecast")

    # Forecast interval (prefer calibrated)
    low_col, high_col, band_label = choose_band_cols_tx(fore)
    if low_col and high_col:
        ax1.fill_between(
            fore["week_end"], fore[low_col]/1_000_000, fore[high_col]/1_000_000,
            alpha=0.15, label=band_label
        )

    # Mark where history ends
    split = data["week_end"].max()
    ax1.axvline(split, color="gray", ls=":", lw=1)

    # Labels & y-axis formatter (millions)
    ax1.set_xlabel("Week end (Sun)")
    ax1.set_ylabel("Transactions (millions, weekly sum)")
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}M"))

    # FX on secondary axis
    ax2 = ax1.twinx()
    fx_all = make_fx_series(data, fore)
    ax2.plot(fx_all["week_end"], fx_all["fx_rate"], linestyle="--", label="fx (USD/MXN)")
    ax2.set_ylabel("USD/MXN (weekly mean/assumed)")

    # KPI BOX — next week's transactions (M) + interval + FX
    nxt = next_week_row(fore)
    tx_m   = float(nxt["pred_tx"])/1e6
    fx_ass = float(nxt["fx_assumed"]) if "fx_assumed" in nxt else None
    if low_col and high_col:
        lo_m = float(nxt[low_col])/1e6
        hi_m = float(nxt[high_col])/1e6
        band = f"\nBand: {lo_m:,.1f}–{hi_m:,.1f}M"
    else:
        band = ""
    fx_line = f"\nFX: {fx_ass:,.2f}" if fx_ass is not None else ""
    kpi_tx = f"Next week Tx ≈ {tx_m:,.1f}M{band}{fx_line}"

    ax1.text(
        0.98, 0.02, kpi_tx,
        transform=ax1.transAxes, va="bottom", ha="right",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.9, ec="gray"),
        fontsize=11
    )

    # Legend (combine both axes)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.title("Weekly transactions (SPEI proxy) vs forecast + USD/MXN FX")
    plt.tight_layout(); plt.gcf().autofmt_xdate()
    plt.savefig(out, dpi=150); plt.close()
    print(f"[viz] saved {out}")

def plot_value(data, fore, out="weekly_plot_value.png"):
    # Need value columns to plot this chart
    if "value_mn_mxn" not in data.columns or "pred_value_mn_mxn" not in fore.columns:
        print("[viz] value columns not found; skipping value plot.")
        return

    fig, ax1 = plt.subplots(figsize=(11,5))

    # Scale value to **billions** of MXN for readability (input is in millions)
    ax1.plot(data["week_end"], data["value_mn_mxn"]/1_000, label="actual value")
    ax1.plot(fore["week_end"], fore["pred_value_mn_mxn"]/1_000, label="forecast value")

    # Forecast band (prefer calibrated; scale to billions too)
    lowv, highv, band_label = choose_band_cols_val(fore)
    if lowv and highv:
        ax1.fill_between(
            fore["week_end"], fore[lowv]/1_000, fore[highv]/1_000,
            alpha=0.15, label=band_label
        )

    # Mark where history ends
    split = data["week_end"].max()
    ax1.axvline(split, color="gray", ls=":", lw=1)

    ax1.set_xlabel("Week end (Sun)")
    ax1.set_ylabel("Value (MXN billions, weekly sum)")

    # KPI BOX — next week's value (B MXN) + interval + Tx + FX
    nxt = next_week_row(fore)
    val_bn = float(nxt["pred_value_mn_mxn"])/1_000.0
    tx_m   = float(nxt["pred_tx"])/1e6
    fx_ass = float(nxt["fx_assumed"]) if "fx_assumed" in nxt else None

    if lowv and highv:
        lo_bn = float(nxt[lowv])/1_000.0
        hi_bn = float(nxt[highv])/1_000.0
        band  = f"\nBand: {lo_bn:,.2f}–{hi_bn:,.2f}B"
    else:
        band = ""
    fx_line = f"\nFX: {fx_ass:,.2f}" if fx_ass is not None else ""
    kpi_val = f"Next week Value ≈ MXN {val_bn:,.2f}B{band}\nTx ≈ {tx_m:,.1f}M{fx_line}"

    ax1.text(
        0.98, 0.02, kpi_val,
        transform=ax1.transAxes, va="bottom", ha="right",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.9, ec="gray"),
        fontsize=11
    )

    # FX on secondary axis
    ax2 = ax1.twinx()
    fx_all = make_fx_series(data, fore)
    ax2.plot(fx_all["week_end"], fx_all["fx_rate"], linestyle="--", label="fx (USD/MXN)")
    ax2.set_ylabel("USD/MXN (weekly mean/assumed)")

    # Legend (combine both axes)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.title("Weekly value (MXN) vs forecast + USD/MXN FX")
    plt.tight_layout(); plt.gcf().autofmt_xdate()
    plt.savefig(out, dpi=150); plt.close()
    print(f"[viz] saved {out}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEF_DATA)
    ap.add_argument("--forecast", default=DEF_FORE)
    args = ap.parse_args()

    data = pd.read_csv(args.data, parse_dates=["week_end"]).sort_values("week_end")
    fore = pd.read_csv(args.forecast, parse_dates=["week_end"]).sort_values("week_end")
    for df in (data, fore):
        df["week_end"] = pd.to_datetime(df["week_end"]).dt.tz_localize(None)

    # Show only history from 2022 onward (forecast still shown in full)
    data = data[data["week_end"] >= pd.Timestamp("2023-01-01")].copy()

    plot_tx(data, fore)
    plot_value(data, fore)

if __name__ == "__main__":
    main()
