#!/usr/bin/env python3
"""
backtest_weekly.py — Evaluate your forecasting model

- Splits data into training and test (holdout)
- Runs train_and_forecast on training data
- Compares prediction to actuals
- Calculates MAPE, MAE
- Optionally plots forecast vs actual
"""

import pandas as pd
import matplotlib.pyplot as plt
import argparse
from predict_weekly import train_and_forecast, _make_calendar_flags, _fx_features, _is_end_of_month, _is_first_week_of_month, _week_window_dates


def calculate_metrics(pred, actual):
    errors = (pred - actual).abs()
    mae = errors.mean()
    mape = (errors / actual).mean() * 100
    return mae, mape


def add_all_regressors(df):
    cal_flags = _make_calendar_flags(df["week_end"])
    fx_feats = _fx_features(df["fx_rate"])

    df_feat = pd.concat([
        df.reset_index(drop=True),
        cal_flags[["is_payweek", "is_holiday_us", "is_holiday_mx"]],
        fx_feats
    ], axis=1)

    df_feat["end_of_month"] = df_feat["week_end"].apply(_is_end_of_month)

    # FIXED: use list comprehension instead of apply
    df_feat["first_week_of_month"] = [
        int(_is_first_week_of_month(_week_window_dates(we)))
        for we in df_feat["week_end"]
    ]

    return df_feat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="transactions_weekly.csv")
    parser.add_argument("--cal", default=None, help="Optional calibration file")
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--ticket_window", type=int, default=8)
    args = parser.parse_args()

    df = pd.read_csv(args.file, parse_dates=["week_end"])
    df = df.sort_values("week_end")
    df["week_end"] = pd.to_datetime(df["week_end"]).dt.tz_localize(None)

    # Split into train and test
    train_df = df.iloc[:-args.horizon].copy()
    test_df = df.iloc[-args.horizon:].copy()

    # Add regressors to full df (for value calc and visualization)
    df = add_all_regressors(df)

    # Forecast
    fc = train_and_forecast(train_df, horizon_weeks=args.horizon, ticket_window=args.ticket_window)
    fc = fc.set_index("week_end")
    test_df = test_df.set_index("week_end")

    # Compare: transaction count
    compare = fc[["pred_tx"]].join(test_df["tx_count"]).dropna()
    mae_tx, mape_tx = calculate_metrics(compare["pred_tx"], compare["tx_count"])

    print("\n📊 Transaction Volume Accuracy:")
    print(f" - MAE:  {mae_tx:,.2f} tx")
    print(f" - MAPE: {mape_tx:.2f}%")

    # Compare: transaction value (if available)
    if "pred_value_mn_mxn" in fc.columns and "value_mn_mxn" in test_df.columns:
        compare_val = fc[["pred_value_mn_mxn"]].join(test_df["value_mn_mxn"]).dropna()
        print("\n🔍 Prediction vs Actual (Transaction Value, MXN Millions):")
        print(compare_val)

        mae_val, mape_val = calculate_metrics(compare_val["pred_value_mn_mxn"], compare_val["value_mn_mxn"])
        print("\n💰 Transaction Value Accuracy (MXN Millions):")
        print(f" - MAE:  {mae_val:,.2f} M MXN")
        print(f" - MAPE: {mape_val:.2f}%")

    # Forecast vs Actual with CI bands
    plt.figure(figsize=(12,6))
    plt.plot(compare.index, compare["tx_count"], label="Actual Tx", marker="o")
    plt.plot(compare.index, compare["pred_tx"], label="Predicted Tx", marker="x")

    # Confidence interval shading
    if "pred_low" in fc.columns and "pred_high" in fc.columns:
        plt.fill_between(fc.index, fc["pred_low"], fc["pred_high"], alpha=0.2, label="Confidence Interval")

    plt.title("Weekly Remittance Proxy: Actual vs Predicted")
    plt.xlabel("Week")
    plt.ylabel("Transactions")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    main()
