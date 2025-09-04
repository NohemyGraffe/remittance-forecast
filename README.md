
# Remittance forecast prototype

This model forecasts **weekly transaction flows** with ~**6% error** on backtests.

---

## 💡 Potential benefits for a remittance company

Although this project uses **proxy data** (Banxico SPEI transfers) instead of true remittance inflows, it demonstrates how a forecasting system can deliver real value to a remittance company:

- **Liquidity & treasury** – Anticipates weekly MXN/USD payout needs, reducing risks of underfunding or idle capital and improving cash efficiency.  
- **FX risk management** – Aligns hedging strategies with projected flows, lowering exposure to currency swings.  
- **Operational planning** – Flags high-volume weeks (paydays, holidays), enabling better staffing, agent readiness, and smoother customer experience.  
- **Strategic insights** – Reveals seasonal patterns and anomalies, supporting product launches and partnership decisions.  
- **Regulatory & investor reporting** – Provides validated forecasts and error metrics, strengthening reporting and internal controls.  

---

## Model & pipeline overview

This project forecasts **weekly remittance-like transactions (count and value)** using Banxico SPEI data as a proxy. The pipeline covers **data ingestion, feature engineering, modeling, calibration, and deployment** in a Streamlit app.

- **Data ingestion** – Fetches weekly SPEI data from Banxico’s API, with caching and token management.  
- **Feature engineering** – Adds calendar flags (payweeks, US/MX holidays), FX features, and month-end markers.  
- **Forecasting model** – Prophet-style time series with regressors; outputs transaction forecasts, confidence bands, and value in MXN millions.  
- **Calibration** – Optional adjustment with `cal_params.json` to reduce bias.  
- **Backtesting** – Holdout evaluation shows ~**6–7% MAPE** for both transaction volume and value.  
- **Streamlit App** – Interactive dashboard with KPIs (Transactions M, Value MXN B, Value USD M, Margin of Error %), plots (actual vs forecast with uncertainty band), and cash needs (USD M).  

⚠️ **Note:** Data is a proxy (SPEI transfers ≠ real remittances). The focus is on demonstrating a **production-style forecasting system** and **business-oriented dashboard**, not exact remittance magnitudes.

---

## Model reliability

To assess the accuracy of this forecast engine, I ran a backtest on the weekly dataset:

- **Setup** – Last 8 weeks of data held out, model trained on remaining history, forecasts compared against actuals.

### Transaction volume (weekly count of transfers)
- **MAE:** ~7 million transactions  
- **MAPE:** ~6.8%  
  *On average, weekly forecasts differ from actuals by ~6–7%.*

  

### Transaction value (MXN, millions)
- **MAE:** ~352,000 million MXN  
- **MAPE:** ~5.9%  
  *Absolute errors are large because totals are in trillions of MXN, but the **percentage error** shows the model tracks value dynamics closely.*

---

## Important notes

- Dataset used is **Banxico’s SPEI domestic transfer data**, applied here only as a proxy for remittance flows.  
- Actual **US→MX remittance data** is not public at weekly frequency.  
- Forecasts incorporate **calendar effects** (payweeks, holidays) and **FX regressors**, which helps capture realistic flow patterns.  
- While the absolute effect measured here applies to SPEI transfers, the **same methodology** could be applied to true remittance data.  

---

## Future work & potential

This project was built as a **one-week prototype** to demonstrate the end-to-end forecasting pipeline. With more time and real remittance data, it could be developed into a **production-grade tool**.

### Current limitations
- Uses **proxy data** (SPEI), which differs in behavior and ticket size from remittances.  
- Exogenous regressors limited to **FX and holiday/payweek effects**.  

### Planned improvements
- Integrate **real remittance inflow data** (e.g., proprietary datasets).  
- Expand features with **US payroll calendars, employment/migration data, and macro indicators**.  
- Refine **ticket size modeling**, segmenting by transfer channel (cash pickup vs digital wallets).  
- Enhance **backtesting** (multi-year, benchmark vs naïve and advanced models).  
- **Productionize** with CI/CD, error monitoring, and automated recalibration.  
