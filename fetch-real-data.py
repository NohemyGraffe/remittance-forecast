#!/usr/bin/env python3
"""
fetch_real_data.py — SPEI (count + value) + USD/MXN FIX → daily & weekly CSVs

#LOCAL TEST VERSION: token and series IDs are hard-coded so you can run it
without flags. Do NOT commit this to git. Rotate your token after testing.

Outputs:
  - transactions_daily.csv  (date, tx_count, value_mn_mxn, avg_ticket_mxn, fx_rate)
  - transactions_weekly.csv (week_end, tx_count, value_mn_mxn, avg_ticket_mxn, fx_rate)
"""

import os, sys, argparse
from datetime import date
import requests
import pandas as pd

API_BASE = "https://www.banxico.org.mx/SieAPIRest/service/v1/series"

# >>> HARD-CODED FOR LOCAL TESTING (replace/rotate later) <<<
DEFAULT_BANXICO_TOKEN = MY_API_SECRET
DEFAULT_SPEI_COUNT_ID = "SF316454"   # Number of transactions (daily)
DEFAULT_SPEI_VALUE_ID = "SF316455"   # Value of transactions (millions MXN, daily)
# <<< ---------------------------------------------------- >>>

def _log(msg, v): 
    if v: print(msg)

def _need(x, msg): 
    if not x: raise RuntimeError(msg); 
    return x

def _fetch_series(series_id, start, end, token, v=False):
    url = f"{API_BASE}/{series_id}/datos/{start}/{end}?token={token}"
    _log(f"[http] {url}", v)
    r = requests.get(url, timeout=30)
    if not r.ok:
        raise RuntimeError(f"HTTP {r.status_code} for {series_id}: {r.text[:300]}")
    j = r.json()
    ser = j.get("bmx", {}).get("series", [])
    if not ser:
        raise RuntimeError(f"No 'series' in JSON for {series_id}")
    datos = ser[0].get("datos", [])
    rows = []
    for d in datos:
        fecha, dato = d.get("fecha"), d.get("dato")
        if not dato or dato in ("N/E","N/E*","n/d",""): 
            continue
        rows.append((fecha, dato))
    if not rows:
        return pd.DataFrame(columns=["date","value"])
    df = pd.DataFrame(rows, columns=["fecha","valor_raw"])
    df["date"] = pd.to_datetime(df["fecha"], dayfirst=True, errors="coerce")
    df["value"] = pd.to_numeric(df["valor_raw"].str.replace(",",""), errors="coerce")
    df = df.dropna(subset=["date","value"])[["date","value"]].sort_values("date")
    df["date"] = df["date"].dt.tz_localize(None)
    _log(f"[parse] {series_id}: {len(df)} rows", v)
    return df

def fetch_daily(token, count_id, value_id, start, end, v=False):
    # SPEI count
    df_cnt = _fetch_series(count_id, start, end, token, v=v).rename(columns={"value":"tx_count"})
    if df_cnt.empty:
        raise RuntimeError("SPEI count series returned no rows. Check series id/date range.")
    # SPEI value (millions MXN)
    df_val = _fetch_series(value_id, start, end, token, v=v).rename(columns={"value":"value_mn_mxn"})
    # USD/MXN FIX
    df_fx  = _fetch_series("SF43718", start, end, token, v=v).rename(columns={"value":"fx_rate"})

    # Join daily on exact date (all are daily)
    daily = df_cnt.merge(df_val, on="date", how="inner").merge(df_fx, on="date", how="inner").sort_values("date")
    # avg ticket in MXN (value is in MILLIONS)
    daily["avg_ticket_mxn"] = (daily["value_mn_mxn"] * 1_000_000) / daily["tx_count"]
    return daily

def to_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    w = (daily.set_index("date")
              .resample("W-SUN")
              .agg(tx_count=("tx_count","sum"),
                   value_mn_mxn=("value_mn_mxn","sum"),
                   fx_rate=("fx_rate","mean"))
              .reset_index()
              .rename(columns={"date":"week_end"}))
    w["avg_ticket_mxn"] = (w["value_mn_mxn"] * 1_000_000) / w["tx_count"]
    return w

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--banxico-token", default=DEFAULT_BANXICO_TOKEN)
    ap.add_argument("--spei-count",   default=DEFAULT_SPEI_COUNT_ID)
    ap.add_argument("--spei-value",   default=DEFAULT_SPEI_VALUE_ID)
    ap.add_argument("--start",        default="2018-01-01")
    ap.add_argument("--end",          default=date.today().isoformat())
    ap.add_argument("--verbose",      action="store_true")
    args = ap.parse_args()

    # Allow env/CLI overrides but fall back to defaults above
    token   = (args.banxico_token or os.getenv("BANXICO_TOKEN") or DEFAULT_BANXICO_TOKEN).strip()
    countId = (args.spei_count     or os.getenv("SPEI_COUNT_ID") or DEFAULT_SPEI_COUNT_ID).strip()
    valueId = (args.spei_value     or os.getenv("SPEI_VALUE_ID") or DEFAULT_SPEI_VALUE_ID).strip()

    try:
        _need(token,   "Missing Banxico token")
        _need(countId, "Missing SPEI count series id")
        _need(valueId, "Missing SPEI value series id")
        print(f"[fetch] SPEI(count={countId}, value={valueId}) + FIX(SF43718) {args.start}..{args.end}")
        daily = fetch_daily(token, countId, valueId, args.start, args.end, v=args.verbose)
        daily.to_csv("transactions_daily.csv", index=False)
        print(f"[ok] transactions_daily.csv  ({len(daily)} rows)")
        weekly = to_weekly(daily)
        weekly.to_csv("transactions_weekly.csv", index=False)
        print(f"[ok] transactions_weekly.csv ({len(weekly)} rows)")
        print(weekly.tail(3).to_string(index=False))
    except Exception as e:
        print("ERROR:", e); sys.exit(2)

if __name__ == "__main__":
    main()
