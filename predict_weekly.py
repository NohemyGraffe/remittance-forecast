#!/usr/bin/env python3
"""
predict_weekly.py — upgraded weekly forecaster

Improvements:
- Prophet: multiplicative seasonality, yearly_seasonality=15, interval_width=0.90
- Adds week-of-year seasonality (period=52, fourier_order=6)
- Calendar regressors: is_payweek (1st/15th in the week), is_holiday_mx (MX holiday in the week)
- FX features: fx_rate (level), fx_rate_lag1, fx_rate_lag2, dfx (WoW change), retfx (% change)
- Derives value forecast via recent avg_ticket_mxn (if available)

Reads:
  transactions_weekly.csv  (required: week_end, tx_count, fx_rate; optional: value_mn_mxn, avg_ticket_mxn)

Writes:
  weekly_forecast.csv
    week_end, pred_tx, pred_low, pred_high, fx_assumed,
    (if ticket available) pred_value_mn_mxn, pred_value_mn_mxn_low/high, avg_ticket_mxn_used
"""

import argparse
import pandas as pd
from prophet import Prophet
import json, os


# holidays is optional; if missing, holiday flag will be zeros
try:
    import holidays as _hol
    _HAS_HOLIDAYS = True
except Exception:
    _HAS_HOLIDAYS = False

DEF_IN  = "transactions_weekly.csv"
DEF_OUT = "weekly_forecast.csv"

# ---------- helpers ----------

def _week_window_dates(week_end):
    """Return the 7 dates inside the weekly bucket that ends at `week_end` (assumes W-SUN)."""
    end = pd.Timestamp(week_end).normalize()
    start = end - pd.Timedelta(days=6)
    return pd.date_range(start, end, freq="D")

def _make_calendar_flags(week_ends: pd.Series):
    """Binary flags: pay-week (1st/15th in window), MX holiday in window."""
    we = pd.to_datetime(week_ends, errors="coerce")  # ensure datetime values (naive)
    is_payweek, is_holiday = [], []
    mx_holidays = _hol.country_holidays("MX") if _HAS_HOLIDAYS else set()

    for we_end in we:
        days = _week_window_dates(we_end)  # 7 dates in this weekly bucket
        pay = any(day.day in (1, 15) for day in days)
        hol = any(day.date() in mx_holidays for day in days) if _HAS_HOLIDAYS else False
        is_payweek.append(int(pay))
        is_holiday.append(int(hol))

    return pd.DataFrame({
        "week_end": we,
        "is_payweek": is_payweek,
        "is_holiday_mx": is_holiday
    })

def _fx_features(series: pd.Series) -> pd.DataFrame:
    """Build FX level + lags + changes on an aligned index (keeps 'fx_rate' level)."""
    s = pd.Series(pd.to_numeric(series, errors="coerce"), index=series.index)
    df = pd.DataFrame({"fx_rate": s})
    df["fx_rate_lag1"] = s.shift(1)
    df["fx_rate_lag2"] = s.shift(2)
    df["dfx"]          = s.diff(1)       # week-over-week abs change
    df["retfx"]        = s.pct_change(1) # week-over-week % change
    return df

def _recent_avg_ticket(df_hist: pd.DataFrame, window=8):
    if "avg_ticket_mxn" in df_hist.columns and df_hist["avg_ticket_mxn"].notna().any():
        return float(df_hist["avg_ticket_mxn"].tail(window).mean())
    if "value_mn_mxn" in df_hist.columns and df_hist["value_mn_mxn"].notna().any():
        val = float(df_hist["value_mn_mxn"].tail(window).sum()) * 1_000_000
        tx  = float(df_hist["tx_count"].tail(window).sum())
        return val / max(tx, 1.0)
    return None


def _apply_calibration(fc_df, cal_path):
    """If cal_params.json is provided, add calibrated bands:
       pred_low_cal / pred_high_cal (and value equivalents)."""
    if not cal_path or not os.path.exists(cal_path):
        return fc_df
    try:
        with open(cal_path, "r", encoding="utf-8") as f:
            cal = json.load(f)
        if cal.get("kind") == "relative":
            q = float(cal["q"])
            # widen/narrow around point forecast by relative error quantile q
            fc_df["pred_low_cal"]  = fc_df["pred_tx"] * (1.0 - q)
            fc_df["pred_high_cal"] = fc_df["pred_tx"] * (1.0 + q)
            if "avg_ticket_mxn_used" in fc_df.columns:
                at = fc_df["avg_ticket_mxn_used"]
                fc_df["pred_value_mn_mxn_low_cal"]  = (fc_df["pred_low_cal"]  * at) / 1_000_000
                fc_df["pred_value_mn_mxn_high_cal"] = (fc_df["pred_high_cal"] * at) / 1_000_000
    except Exception:
        # if anything goes wrong, just keep original bands
        pass
    return fc_df



# ---------- core ----------

def load_weekly(path):
    df = pd.read_csv(path, parse_dates=["week_end"])
    df["week_end"] = pd.to_datetime(df["week_end"]).dt.tz_localize(None)
    return df.sort_values("week_end")

def train_and_forecast(df, horizon_weeks=8, ticket_window=8):
    # ----- history base -----
    hist = df.copy()
    hist["week_end"] = pd.to_datetime(hist["week_end"]).dt.tz_localize(None)
    hist = hist.sort_values("week_end").reset_index(drop=True)

    # calendar flags (history)
    cal_hist = _make_calendar_flags(hist["week_end"])

    # FX features (history) – keep 'fx_rate' level and lags/changes
    fx_hist = _fx_features(hist["fx_rate"])

    # combine features (avoid duplicate column names)
    hist_feat = pd.concat(
        [hist, cal_hist[["is_payweek", "is_holiday_mx"]], fx_hist],
        axis=1
    )
    hist_feat = hist_feat.loc[:, ~hist_feat.columns.duplicated(keep="first")].copy()

    # Prophet training frame
    wp = hist_feat.rename(columns={"week_end": "ds", "tx_count": "y"}).copy()
    wp = wp.loc[:, ~wp.columns.duplicated(keep="first")].copy()

    # columns we *want* to use
    desired = ["fx_rate"]

    # keep only columns that really exist
    regressors = [c for c in desired if c in wp.columns]

    # ensure 1-D numeric; coerce bad values to NaN
    for c in ["y"] + regressors:
        wp[c] = pd.to_numeric(wp[c], errors="coerce")


    # drop rows made NaN by lags/diffs, or any bad values
    wp.replace([float("inf"), float("-inf")], pd.NA, inplace=True)
    wp = wp.dropna(subset=["y"] + regressors).reset_index(drop=True)

    # ----- model -----
    m = Prophet(
    weekly_seasonality=False,
    daily_seasonality=False,
    yearly_seasonality=8,        # from tuner
    seasonality_mode="additive", # from tuner
    interval_width=0.90,
    changepoint_prior_scale=0.10 # from tuner
   )
    m.add_seasonality(name="woy", period=52, fourier_order=6)


    # register only the regressors we actually have (and pass in fit df)
    for col in regressors:
        m.add_regressor(col)

    # Fit on the cleaned training frame
    m.fit(wp[["ds", "y"] + regressors])

    # ---------- future features ----------
    # Prophet's future includes the model's internal history length (len(wp)) + horizon.
    future = m.make_future_dataframe(periods=horizon_weeks, freq="W-SUN")

    # FX path: hold-last (replace with your own FX scenario later if you want)
    last_fx = float(wp["fx_rate"].iloc[-1])  # align with model's history
    fx_future_vals = [last_fx] * horizon_weeks

    # Build combined FX on the SAME history Prophet saw (wp), then append future path
    fx_combo = pd.Series(list(wp["fx_rate"].values) + fx_future_vals)
    fx_feat_combo = _fx_features(fx_combo)

    # Calendar flags for the 'future' ds
    cal_fut = _make_calendar_flags(future["ds"]).rename(columns={"week_end": "ds"}).reset_index(drop=True)

    # Slice the combined FX features to the entire future dataframe length
    # (history part that Prophet expects + horizon)
    fx_needed_all = fx_feat_combo.iloc[:len(future)].reset_index(drop=True)

    # Build the regressor matrix for all rows in 'future' (history+future)
    # Start from the training regressors used in wp, then append only the future tail
    hist_regs = wp[regressors].reset_index(drop=True)
    # Map FX columns from fx_needed_all, and calendar from cal_fut
    reg_all = pd.DataFrame(index=range(len(future)))
    for col in regressors:
        if col in ("fx_rate", "fx_rate_lag1", "fx_rate_lag2", "dfx", "retfx"):
            reg_all[col] = pd.to_numeric(fx_needed_all[col].values, errors="coerce")
        elif col in ("is_payweek", "is_holiday_mx"):
            reg_all[col] = pd.to_numeric(cal_fut[col].values, errors="coerce")
        else:
            # Fallback: if any unexpected regressor sneaks in, fill with zeros
            reg_all[col] = 0.0

    # Attach regressors to the full 'future' frame

    for col in regressors:
        # Fill NaNs in regressor columns for the future frame as well
        future[col] = reg_all[col].ffill().bfill()

    # predict for full frame, then keep only the last horizon rows
    fc = m.predict(future)[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(horizon_weeks)
    fc = fc.rename(columns={"ds": "week_end", "yhat": "pred_tx", "yhat_lower": "pred_low", "yhat_upper": "pred_high"})
    fc["fx_assumed"] = fx_future_vals

    # ----- derive value forecast via recent avg ticket -----
    # Use original hist (full history) to compute avg ticket
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
    ap.add_argument("--ticket_window", type=int, default=8, help="weeks to average for avg_ticket_mxn")
    ap.add_argument("--cal", default=None, help="optional cal_params.json for calibrated intervals")
    args = ap.parse_args()


    df = load_weekly(args.data)
    fc = train_and_forecast(df, horizon_weeks=args.horizon, ticket_window=args.ticket_window)

    # APPLY CALIBRATION **before** saving
    fc = _apply_calibration(fc, args.cal)  # <— added

    fc.to_csv(args.out, index=False)
    print(f"[weekly] Saved forecast -> {args.out}")
    print(fc.to_string(index=False))

if __name__ == "__main__":
    main()
