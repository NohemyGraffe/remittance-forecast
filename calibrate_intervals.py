#!/usr/bin/env python3
# Calibrate forecast intervals via conformal residuals (relative errors).
# Produces cal_params.json with q such that P(|(y - ŷ)/ŷ| <= q) ≈ 1 - alpha.

import argparse, json, numpy as np, pandas as pd
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

def _week_window_dates(we):
    end = pd.Timestamp(we).normalize(); start = end - pd.Timedelta(days=6)
    return pd.date_range(start, end, freq="D")

def _make_calendar_flags(week_ends: pd.Series):
    we = pd.to_datetime(week_ends, errors="coerce")
    is_payweek, is_holiday = [], []
    mx_holidays = _hol.country_holidays("MX") if _HAS_HOLIDAYS else set()
    for we_end in we:
        days = _week_window_dates(we_end)
        pay = any(d.day in (1,15) for d in days)
        hol = any(d.date() in mx_holidays for d in days) if _HAS_HOLIDAYS else False
        is_payweek.append(int(pay)); is_holiday.append(int(hol))
    return pd.DataFrame({"week_end": we, "is_payweek": is_payweek, "is_holiday_mx": is_holiday})

def _build_frame(df, feat):
    feat_df = df.copy()
    feat_df["fx_rate"] = pd.to_numeric(feat_df["fx_rate"], errors="coerce")
    if feat == "fx_cal":
        cal = _make_calendar_flags(feat_df["week_end"])
        feat_df = feat_df.merge(cal, on="week_end", how="left")
    wp = feat_df.rename(columns={"week_end":"ds","tx_count":"y"}).copy()
    for c in [c for c in ["y","fx_rate","is_payweek","is_holiday_mx"] if c in wp.columns]:
        wp[c] = pd.to_numeric(wp[c], errors="coerce")
    wp = wp.dropna(subset=["y"]).reset_index(drop=True)
    return wp

def fit_forecast_one(wp, feat, mode, yearly, cp, interval):
    m = Prophet(weekly_seasonality=False, daily_seasonality=False,
                yearly_seasonality=yearly, seasonality_mode=mode,
                interval_width=interval, changepoint_prior_scale=cp,
                seasonality_prior_scale=10)
    m.add_seasonality(name="woy", period=52, fourier_order=6)
    m.add_regressor("fx_rate")
    if feat == "fx_cal":
        m.add_regressor("is_payweek"); m.add_regressor("is_holiday_mx")
    cols = ["ds","y","fx_rate"] + (["is_payweek","is_holiday_mx"] if feat=="fx_cal" else [])
    m.fit(wp[cols])
    future = m.make_future_dataframe(periods=1, freq="W-SUN")
    fx_last = float(wp["fx_rate"].iloc[-1])
    future["fx_rate"] = list(wp["fx_rate"].values) + [fx_last]
    if feat == "fx_cal":
        cal_all = _make_calendar_flags(future["ds"]).rename(columns={"week_end":"ds"})
        future["is_payweek"] = cal_all["is_payweek"].values
        future["is_holiday_mx"] = cal_all["is_holiday_mx"].values
    fc = m.predict(future).tail(1).iloc[0]
    return float(fc["yhat"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEF_PATH)
    ap.add_argument("--feat", choices=["fx","fx_cal"], default="fx")
    ap.add_argument("--mode", choices=["additive","multiplicative"], default="additive")
    ap.add_argument("--yearly", type=int, default=8)
    ap.add_argument("--cp", type=float, default=0.10)
    ap.add_argument("--folds", type=int, default=26)
    ap.add_argument("--min-train", type=int, default=60)
    ap.add_argument("--alpha", type=float, default=0.10, help="1 - desired coverage")
    ap.add_argument("--out", default="cal_params.json")
    args = ap.parse_args()

    df = pd.read_csv(args.data, parse_dates=["week_end"]).sort_values("week_end")
    df["week_end"] = pd.to_datetime(df["week_end"]).dt.tz_localize(None)

    pairs = []
    for i in range(args.folds, 0, -1):
        cut = len(df) - i
        train = df.iloc[:cut]
        if len(train) < args.min_train: continue
        test = df.iloc[cut]
        wp = _build_frame(train, args.feat)
        if len(wp) < 10: continue
        try:
            yhat = fit_forecast_one(wp, args.feat, args.mode, args.yearly, args.cp, interval=0.90)
        except Exception:
            continue
        pairs.append((float(test["tx_count"]), yhat))

    if not pairs:
        raise SystemExit("No folds. Try lowering --min-train.")

    y_true = np.array([p[0] for p in pairs], float)
    y_hat  = np.array([p[1] for p in pairs], float)

    rel_err = np.abs((y_true - y_hat) / np.clip(y_hat,1e-9,None))
    q_rel = float(np.quantile(rel_err, 1 - args.alpha))  # e.g., alpha=0.10 → 90% coverage
    mape_val = mape(y_true, y_hat)

    out = {"kind":"relative", "q": q_rel, "alpha": args.alpha,
           "feat": args.feat, "mode": args.mode, "yearly": args.yearly, "cp": args.cp,
           "folds": int(len(y_true)), "mape_point": mape_val}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {args.out} with q={q_rel:.4f}, folds={len(y_true)}, MAPE={mape_val:.3f}")

if __name__ == "__main__":
    main()
