#!/usr/bin/env python3
"""
predict_weekly.py — upgraded weekly forecaster

Improvements:
- Prophet: additive seasonality, yearly_seasonality=8, interval_width=0.90
- Adds week-of-year seasonality (period=52, fourier_order=6)
- Calendar regressors:
    • is_payweek (1st/15th in the week — U.S. pay cycle proxy)
    • is_holiday_us (U.S. sender-side holidays)
    • is_holiday_mx (Mexico holidays — receiver-side)
- FX features: fx_rate (level), fx_rate_lag1, fx_rate_lag2, dfx (WoW change), retfx (% change)
- Derives value forecast via recent avg_ticket_mxn (if available)

Reads:
  transactions_weekly.csv

Writes:
  weekly_forecast.csv
"""

import argparse
import pandas as pd
from prophet import Prophet
import json, os

# holidays is optional; if missing, holiday flags will be zeros
try:
    import holidays as _hol
    _HAS_HOLIDAYS = True
except Exception:
    _HAS_HOLIDAYS = False

DEF_IN  = "transactions_weekly.csv"
DEF_OUT = "weekly_forecast.csv"

# ---------- helpers ----------

def _week_window_dates(week_end):
    end = pd.Timestamp(week_end).normalize()
    start = end - pd.Timedelta(days=6)
    return pd.date_range(start, end, freq="D")

def _is_end_of_month(dt: pd.Timestamp) -> bool:
    return dt.is_month_end or (dt.day >= 28 and (dt + pd.Timedelta(days=1)).month != dt.month)

def _is_first_week_of_month(week_dates: pd.DatetimeIndex) -> bool:
    return any(d.day <= 7 for d in week_dates)

def _make_calendar_flags(week_ends: pd.Series):
    we = pd.to_datetime(week_ends, errors="coerce")
    is_payweek, is_mx_hol, is_us_hol, end_month, first_week = [], [], [], [], []

    mx_holidays = _hol.country_holidays("MX") if _HAS_HOLIDAYS else set()
    us_holidays = _hol.country_holidays("US") if _HAS_HOLIDAYS else set()

    for we_end in we:
        days = _week_window_dates(we_end)
        pay = any(day.day in (1, 15) for day in days)
        hol_mx = any(day.date() in mx_holidays for day in days)
        hol_us = any(day.date() in us_holidays for day in days)
        eom = any(_is_end_of_month(day) for day in days)
        fwm = _is_first_week_of_month(days)

        is_payweek.append(int(pay))
        is_mx_hol.append(int(hol_mx))
        is_us_hol.append(int(hol_us))
        end_month.append(int(eom))
        first_week.append(int(fwm))

    return pd.DataFrame({
        "week_end": we,
        "is_payweek": is_payweek,
        "is_holiday_mx": is_mx_hol,
        "is_holiday_us": is_us_hol,
        "end_of_month": end_month,
        "first_week_of_month": first_week
    })


def _fx_features(series: pd.Series):
    s = pd.Series(pd.to_numeric(series, errors="coerce"), index=series.index)
    df = pd.DataFrame({"fx_rate": s})
    df["fx_rate_lag1"] = s.shift(1)
    df["fx_rate_lag2"] = s.shift(2)
    df["fx_rate_lag7"] = s.shift(7)
    df["dfx"] = s.diff(1)
    df["retfx"] = s.pct_change(1)
    return df

def _recent_avg_ticket(df_hist, window=8):
    if "avg_ticket_mxn" in df_hist.columns and df_hist["avg_ticket_mxn"].notna().any():
        return float(df_hist["avg_ticket_mxn"].tail(window).mean())
    if "value_mn_mxn" in df_hist.columns and df_hist["value_mn_mxn"].notna().any():
        val = float(df_hist["value_mn_mxn"].tail(window).sum()) * 1_000_000
        tx  = float(df_hist["tx_count"].tail(window).sum())
        return val / max(tx, 1.0)
    return None

def _apply_calibration(fc_df, cal_path):
    if not cal_path or not os.path.exists(cal_path):
        return fc_df
    try:
        with open(cal_path, "r", encoding="utf-8") as f:
            cal = json.load(f)
        if cal.get("kind") == "relative":
            q = float(cal["q"])
            fc_df["pred_low_cal"]  = fc_df["pred_tx"] * (1.0 - q)
            fc_df["pred_high_cal"] = fc_df["pred_tx"] * (1.0 + q)
            if "avg_ticket_mxn_used" in fc_df.columns:
                at = fc_df["avg_ticket_mxn_used"]
                fc_df["pred_value_mn_mxn_low_cal"]  = (fc_df["pred_low_cal"]  * at) / 1_000_000
                fc_df["pred_value_mn_mxn_high_cal"] = (fc_df["pred_high_cal"] * at) / 1_000_000
    except Exception:
        pass
    return fc_df

# ---------- core ----------

def load_weekly(path):
    df = pd.read_csv(path, parse_dates=["week_end"])
    df["week_end"] = pd.to_datetime(df["week_end"]).dt.tz_localize(None)
    return df.sort_values("week_end")

def train_and_forecast(df, horizon_weeks=8, ticket_window=8):
    hist = df.copy()
    hist["week_end"] = pd.to_datetime(hist["week_end"]).dt.tz_localize(None)
    hist = hist.sort_values("week_end").reset_index(drop=True)

    cal_hist = _make_calendar_flags(hist["week_end"])
    fx_hist  = _fx_features(hist["fx_rate"])

    hist_feat = pd.concat(
    [hist, cal_hist[["is_payweek", "is_holiday_mx", "is_holiday_us"]], fx_hist],
    axis=1
     )
    hist_feat = hist_feat.loc[:, ~hist_feat.columns.duplicated(keep="first")].copy()


    wp = hist_feat.rename(columns={"week_end": "ds", "tx_count": "y"}).copy()
    desired = ["fx_rate", "is_payweek", "is_holiday_mx", "is_holiday_us", "end_of_month", "first_week_of_month"]
    regressors = [c for c in desired if c in wp.columns]

    for c in ["y"] + regressors:
        wp[c] = pd.to_numeric(wp[c], errors="coerce")
    wp.replace([float("inf"), float("-inf")], pd.NA, inplace=True)
    wp = wp.dropna(subset=["y"] + regressors).reset_index(drop=True)

    m = Prophet(
        weekly_seasonality=False,
        daily_seasonality=False,
        yearly_seasonality=8,
        seasonality_mode="additive",
        interval_width=0.90,
        changepoint_prior_scale=0.10
    )
    m.add_seasonality(name="woy", period=52, fourier_order=6)
    for col in regressors:
        m.add_regressor(col)

    m.fit(wp[["ds", "y"] + regressors])
    future = m.make_future_dataframe(periods=horizon_weeks, freq="W-SUN")

    last_fx = float(wp["fx_rate"].iloc[-1])
    fx_future_vals = [last_fx] * horizon_weeks
    fx_combo = pd.Series(list(wp["fx_rate"].values) + fx_future_vals)
    fx_feat_combo = _fx_features(fx_combo)
    cal_fut = _make_calendar_flags(future["ds"]).rename(columns={"week_end": "ds"}).reset_index(drop=True)
    fx_needed_all = fx_feat_combo.iloc[:len(future)].reset_index(drop=True)

    reg_all = pd.DataFrame(index=range(len(future)))
    for col in regressors:
        if col in ("fx_rate", "fx_rate_lag1", "fx_rate_lag2", "dfx", "retfx"):
            reg_all[col] = pd.to_numeric(fx_needed_all[col].values, errors="coerce")
        elif col in ("is_payweek", "is_holiday_mx", "is_holiday_us"):
            reg_all[col] = pd.to_numeric(cal_fut[col].values, errors="coerce")
        else:
            reg_all[col] = 0.0
    for col in regressors:
        future[col] = reg_all[col].ffill().bfill()

    fc = m.predict(future)[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(horizon_weeks)
    fc = fc.rename(columns={
        "ds": "week_end",
        "yhat": "pred_tx",
        "yhat_lower": "pred_low",
        "yhat_upper": "pred_high"
    })
    fc["fx_assumed"] = fx_future_vals

    avg_ticket = _recent_avg_ticket(hist, window=ticket_window)
    if avg_ticket is not None:
        fc["avg_ticket_mxn_used"] = avg_ticket
        fc["pred_value_mn_mxn"]      = (fc["pred_tx"]  * avg_ticket) / 1_000_000
        fc["pred_value_mn_mxn_low"]  = (fc["pred_low"] * avg_ticket) / 1_000_000
        fc["pred_value_mn_mxn_high"] = (fc["pred_high"]* avg_ticket) / 1_000_000

    return fc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEF_IN)
    ap.add_argument("--out",  default=DEF_OUT)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--ticket_window", type=int, default=8)
    ap.add_argument("--cal", default=None)
    args = ap.parse_args()

    df = load_weekly(args.data)
    fc = train_and_forecast(df, horizon_weeks=args.horizon, ticket_window=args.ticket_window)
    fc = _apply_calibration(fc, args.cal)

    fc.to_csv(args.out, index=False)
    print(f"[weekly] Saved forecast -> {args.out}")
    print(fc.to_string(index=False))

if __name__ == "__main__":
    main()
