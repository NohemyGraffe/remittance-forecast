#!/usr/bin/env python3
"""
tune_weekly.py — small grid search for weekly remittance model
Tries a few combos:
  - seasonality_mode: ['additive','multiplicative']
  - yearly_seasonality: [8, 12, 15]
  - changepoint_prior_scale: [0.05, 0.1, 0.2]
  - feature set: ['fx', 'fx_cal']   # fx only vs fx + payweek + holidays
Always adds week-of-year seasonality (period=52, fourier=6).

Backtest: rolling 1-week-ahead over last --folds weeks (default 26).
Reports MAPE/WAPE/etc and prints the top configs.
"""

import argparse, pandas as pd, numpy as np
from prophet import Prophet

try:
    import holidays as _hol
    _HAS_HOLIDAYS = True
except Exception:
    _HAS_HOLIDAYS = False

DEF_PATH = "transactions_weekly.csv"

def mape(y, yhat):
    y, yhat = np.asarray(y,float), np.asarray(yhat,float)
    return float(np.mean(np.abs((y - yhat) / np.clip(np.abs(y),1e-9,None))) * 100)

def wape(y, yhat):
    y, yhat = np.asarray(y,float), np.asarray(yhat,float)
    return float(np.sum(np.abs(y - yhat)) / max(np.sum(np.abs(y)),1e-9) * 100)

def _week_window_dates(week_end):
    end = pd.Timestamp(week_end).normalize()
    start = end - pd.Timedelta(days=6)
    return pd.date_range(start, end, freq="D")

def _make_calendar_flags(week_ends: pd.Series):
    we = pd.to_datetime(week_ends, errors="coerce")
    is_payweek, is_holiday = [], []
    mx_holidays = _hol.country_holidays("MX") if _HAS_HOLIDAYS else set()
    for we_end in we:
        days = _week_window_dates(we_end)
        pay = any(day.day in (1, 15) for day in days)
        hol = any(day.date() in mx_holidays for day in days) if _HAS_HOLIDAYS else False
        is_payweek.append(int(pay))
        is_holiday.append(int(hol))
    return pd.DataFrame({"week_end": we, "is_payweek": is_payweek, "is_holiday_mx": is_holiday})

def _build_frame(hist, feature_set):
    # base
    feat = hist.copy()
    # features
    if feature_set in ("fx","fx_cal"):
        feat["fx_rate"] = pd.to_numeric(feat["fx_rate"], errors="coerce")
    if feature_set == "fx_cal":
        cal = _make_calendar_flags(feat["week_end"])
        feat = feat.merge(cal, on="week_end", how="left")
    # prophet frame
    wp = feat.rename(columns={"week_end":"ds","tx_count":"y"})
    for c in [c for c in ["y","fx_rate","is_payweek","is_holiday_mx"] if c in wp.columns]:
        wp[c] = pd.to_numeric(wp[c], errors="coerce")
    wp = wp.dropna(subset=["y"]).reset_index(drop=True)
    return wp

def _fit_forecast_one(wp, feature_set, mode, yearly_order, cp_scale, interval=0.90):
    m = Prophet(
        weekly_seasonality=False,
        daily_seasonality=False,
        yearly_seasonality=yearly_order,
        seasonality_mode=mode,
        interval_width=interval,
        changepoint_prior_scale=cp_scale,
        seasonality_prior_scale=10,
    )
    # 52-week cycle
    m.add_seasonality(name="woy", period=52, fourier_order=6)
    # regressors
    if feature_set in ("fx","fx_cal"):
        m.add_regressor("fx_rate")
    if feature_set == "fx_cal":
        m.add_regressor("is_payweek")
        m.add_regressor("is_holiday_mx")
    m.fit(wp[["ds","y"] + [c for c in ["fx_rate","is_payweek","is_holiday_mx"] if c in wp.columns]])

    future = m.make_future_dataframe(periods=1, freq="W-SUN")
    # hold-last fx + calendar for the 1 future row
    if "fx_rate" in wp.columns:
        fx_last = float(wp["fx_rate"].iloc[-1])
        future.loc[:, "fx_rate"] = list(wp["fx_rate"].values) + [fx_last]
    if "is_payweek" in wp.columns:
        # recompute payweek/holiday for all future ds
        cal_all = _make_calendar_flags(future["ds"]).rename(columns={"week_end":"ds"})
        future["is_payweek"]   = cal_all["is_payweek"].values
        future["is_holiday_mx"]= cal_all["is_holiday_mx"].values

    fc = m.predict(future).tail(1).iloc[0]
    return float(fc["yhat"])

def backtest(df, folds, min_train, mode, yearly_order, cp_scale, feature_set):
    res = []
    for i in range(folds, 0, -1):
        cut = len(df) - i
        train = df.iloc[:cut]
        if len(train) < min_train: continue
        test = df.iloc[cut]
        wp = _build_frame(train, feature_set)
        if len(wp) < 10: continue
        try:
            yhat = _fit_forecast_one(wp, feature_set, mode, yearly_order, cp_scale)
        except Exception:
            continue
        res.append((test["tx_count"], yhat))
    if not res: return None
    y, yhat = zip(*res)
    return {
        "folds": len(y),
        "mape": round(mape(y,yhat), 3),
        "wape": round(wape(y,yhat), 3),
        "mode": mode,
        "yearly": yearly_order,
        "cp": cp_scale,
        "feat": feature_set,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEF_PATH)
    ap.add_argument("--folds", type=int, default=26)
    ap.add_argument("--min-train", type=int, default=60)
    args = ap.parse_args()

    df = pd.read_csv(args.data, parse_dates=["week_end"]).sort_values("week_end")
    df["week_end"] = pd.to_datetime(df["week_end"]).dt.tz_localize(None)

    grids = []
    for feature_set in ["fx","fx_cal"]:
        for mode in ["additive","multiplicative"]:
            for yearly in [8,12,15]:
                for cp in [0.05,0.1,0.2]:
                    grids.append((feature_set, mode, yearly, cp))

    rows = []
    for feat, mode, yearly, cp in grids:
        r = backtest(df, args.folds, args.min_train, mode, yearly, cp, feat)
        if r: rows.append(r)

    if not rows:
        print("No results; try lowering --min-train.")
        return

    out = pd.DataFrame(rows).sort_values(["mape","wape","folds"])
    print("\nTop configs by MAPE:")
    print(out.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
