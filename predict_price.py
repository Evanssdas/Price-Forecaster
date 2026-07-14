# predict_price.py - SELF-GRADING GB day-ahead peak PRICE forecaster.
# Each run: (1) backfills actual prices for past-due forecasts, (2) predicts tomorrow.
# Two-stage chain: helper (weather -> residual demand) -> price model. Honest regime flag.
#
# NOTE ON TTF (European gas): the pipeline LOGS the TTF gas price every day, but the
# MODEL DOES NOT USE IT. Reason: over 135 days, TTF-vs-GB-price change correlation was
# only ~0.10 - power was pricing scarcity, not fuel cost. Training on that would fit noise.
# We capture the data anyway so that when the market normalises and gas starts mattering,
# the history is already there to retrain on. scorecard.py runs the test automatically.
import os
import datetime as dt
import numpy as np
import pandas as pd
import requests
import lightgbm as lgb
import holidays

LAT, LON = 51.51, -0.13
LOG = "price_predictions_log.csv"
RESID_FEATURES = ["t_mean", "t_max", "t_min", "wind_max", "solar_rad",
                  "dow", "is_we", "month", "doy", "is_hol"]
PRICE_FEATURES = ["resid_peak_gw", "t_mean", "t_max", "t_min", "wind_max", "HDD", "CDD",
                  "dow", "is_we", "month", "doy", "is_hol", "price_lag1", "sys_lag1"]
RESID_HI = 38.0
SYS_HI = 120.0
COLS = ["date_made", "target_date", "predicted_price", "actual_price", "error",
        "regime", "ttf_eur_mwh", "status"]

today = pd.Timestamp.now(tz="Europe/London").normalize()
tom = today + pd.Timedelta(days=1)

log = pd.read_csv(LOG) if os.path.exists(LOG) else pd.DataFrame(columns=COLS)
for c in COLS:
    if c not in log.columns:
        log[c] = ""


def daily_peak_price(start, end):
    """Daily peak day-ahead price (APX), paged in 7-day chunks (Elexon range limit)."""
    rows = []
    d = start
    while d <= end:
        d2 = min(d + dt.timedelta(days=6), end)
        u = ("https://data.elexon.co.uk/bmrs/api/v1/balancing/pricing/market-index"
             f"?from={d}T00:00Z&to={d2}T23:59Z&format=json")
        rows += requests.get(u, timeout=60).json().get("data", [])
        d = d2 + dt.timedelta(days=1)
    rows = [x for x in rows if x.get("dataProvider") == "APXMIDP"]
    if not rows:
        return pd.Series(dtype=float)
    df = pd.DataFrame(rows)
    df["settlementDate"] = pd.to_datetime(df["settlementDate"]).dt.date
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    return df.groupby("settlementDate")["price"].max()


def latest_ttf():
    """Latest TTF (European gas benchmark, EUR/MWh). Logged, NOT used by the model."""
    try:
        import yfinance as yf
        d = yf.download("TTF=F", period="10d", progress=False, auto_adjust=False)
        if len(d):
            return round(float(d["Close"].iloc[-1]), 2)
    except Exception as e:
        print("ttf fetch failed:", type(e).__name__)
    return ""


# ---------- 1. BACKFILL actual prices ----------
try:
    no_actual = pd.to_numeric(log["actual_price"], errors="coerce").isna()
    has_pred = pd.to_numeric(log["predicted_price"], errors="coerce").notna()
    past_due = pd.to_datetime(log["target_date"], errors="coerce").dt.date < today.date()
    due = log[no_actual & has_pred & past_due]

    print("rows needing an actual:", len(due))
    if len(due):
        lo = pd.to_datetime(due["target_date"]).min().date()
        hi = pd.to_datetime(due["target_date"]).max().date()
        act = daily_peak_price(lo, hi)
        print("actuals fetched for", len(act), "days")
        for i, r in due.iterrows():
            d = pd.to_datetime(r["target_date"]).date()
            if d in act.index:
                a = round(float(act.loc[d]), 2)
                log.at[i, "actual_price"] = a
                log.at[i, "error"] = round(float(r["predicted_price"]) - a, 2)
                print("  graded", d, "actual", a)
except Exception as e:
    print("backfill skipped:", type(e).__name__, e)


# ---------- 2. PREDICT tomorrow ----------
already = (log["target_date"].astype(str) == tom.date().isoformat()).any()
if not already:
    row = {"date_made": today.date().isoformat(), "target_date": tom.date().isoformat(),
           "predicted_price": "", "actual_price": "", "error": "", "regime": "",
           "ttf_eur_mwh": "", "status": ""}

    # log gas regardless of whether the forecast succeeds - the data must accumulate
    row["ttf_eur_mwh"] = latest_ttf()
    print("TTF logged (not used by model):", row["ttf_eur_mwh"])

    try:
        wu = ("https://api.open-meteo.com/v1/forecast"
              f"?latitude={LAT}&longitude={LON}"
              "&daily=temperature_2m_mean,temperature_2m_max,temperature_2m_min,"
              "wind_speed_10m_max,shortwave_radiation_sum"
              "&timezone=Europe%2FLondon&forecast_days=3")
        wdf = pd.DataFrame(requests.get(wu, timeout=60).json()["daily"])
        wdf["time"] = pd.to_datetime(wdf["time"])
        w = wdf.loc[wdf["time"].dt.date == tom.date()].iloc[0]
        t_mean = float(w["temperature_2m_mean"])
        t_max = float(w["temperature_2m_max"])
        t_min = float(w["temperature_2m_min"])
        wind_max = float(w["wind_speed_10m_max"])
        solar_rad = float(w["shortwave_radiation_sum"])

        y = today.date() - dt.timedelta(days=1)
        price_lag1 = float(daily_peak_price(y, y).iloc[-1])

        su = f"https://data.elexon.co.uk/bmrs/api/v1/balancing/settlement/system-prices/{y}?format=json"
        sj = [x["systemSellPrice"] for x in requests.get(su, timeout=60).json().get("data", [])
              if x.get("systemSellPrice") is not None]
        sys_lag1 = float(np.mean(sj))

        uk = holidays.country_holidays("GB", subdiv="ENG")
        cal = {"dow": tom.dayofweek, "is_we": int(tom.dayofweek >= 5), "month": tom.month,
               "doy": tom.dayofyear, "is_hol": int(tom.date() in uk)}

        helper = lgb.Booster(model_file="model_resid.txt")
        rrow = {"t_mean": t_mean, "t_max": t_max, "t_min": t_min,
                "wind_max": wind_max, "solar_rad": solar_rad}
        rrow.update(cal)
        resid_peak = float(helper.predict(pd.DataFrame([rrow])[RESID_FEATURES])[0])

        prow = {"resid_peak_gw": resid_peak, "t_mean": t_mean, "t_max": t_max, "t_min": t_min,
                "wind_max": wind_max, "HDD": max(0.0, 15.5 - t_mean), "CDD": max(0.0, t_mean - 22.0),
                "price_lag1": price_lag1, "sys_lag1": sys_lag1}
        prow.update(cal)
        pm = lgb.Booster(model_file="model_price.txt")
        pred = round(float(np.exp(pm.predict(pd.DataFrame([prow])[PRICE_FEATURES])[0])), 2)

        elevated = (resid_peak >= RESID_HI) or (sys_lag1 >= SYS_HI) or (price_lag1 >= 150)
        row["predicted_price"] = pred
        row["regime"] = "elevated" if elevated else "normal"
        row["status"] = "ok"
        print("predicted", pred, "for", tom.date(), "| regime", row["regime"])
    except Exception as e:
        row["status"] = "skipped: " + type(e).__name__
        print("prediction skipped:", e)
    log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)
else:
    print("already have", tom.date())

log[COLS].to_csv(LOG, index=False)


# ---------- 3. running accuracy ----------
err = pd.to_numeric(log["error"], errors="coerce").dropna()
if len(err):
    print("")
    print("--- live record:", len(err), "graded days ---")
    print("MAE:", round(err.abs().mean(), 2), "GBP/MWh | bias:", round(err.mean(), 2))
    if len(err) < 20:
        print("(sample small - not yet meaningful)")
