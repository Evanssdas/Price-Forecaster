# predict_price.py - GB day-ahead peak PRICE forecast via a two-stage chain.
# Stage 1: helper (model_resid.txt) forecasts tomorrow's residual demand from tomorrow's weather (temp, wind, SOLAR).
# Stage 2: price model (model_price.txt) forecasts price from that + weather + calendar + price/system lags.
# Honest regime flag, graceful failure, duplicate guard, evening schedule.
import os, datetime as dt
import numpy as np, pandas as pd, requests
import lightgbm as lgb, holidays

LAT, LON = 51.51, -0.13
LOG = "price_predictions_log.csv"
RESID_FEATURES = ["t_mean","t_max","t_min","wind_max","solar_rad","dow","is_we","month","doy","is_hol"]
PRICE_FEATURES = ["resid_peak_gw","t_mean","t_max","t_min","wind_max","HDD","CDD",
                  "dow","is_we","month","doy","is_hol","price_lag1","sys_lag1"]
RESID_HI, SYS_HI = 38.0, 120.0   # spike-risk thresholds

today = pd.Timestamp.now(tz="Europe/London").normalize()
tom   = today + pd.Timedelta(days=1)

def log_row(r): pd.DataFrame([r]).to_csv(LOG, mode="a", header=not os.path.exists(LOG), index=False); print("logged:", r)
def skip(reason):
    log_row({"date_made":today.date().isoformat(),"target_date":tom.date().isoformat(),
             "predicted_price":"","actual_price":"","error":"","regime":"","status":f"skipped: {reason}"})

# duplicate guard
if os.path.exists(LOG):
    prev=pd.read_csv(LOG)
    if "target_date" in prev and (prev["target_date"]==tom.date().isoformat()).any():
        print("already have", tom.date()); raise SystemExit(0)

# ---- 1. tomorrow's weather forecast: temp, wind, solar ----
try:
    wu=("https://api.open-meteo.com/v1/forecast"
        f"?latitude={LAT}&longitude={LON}"
        "&daily=temperature_2m_mean,temperature_2m_max,temperature_2m_min,wind_speed_10m_max,shortwave_radiation_sum"
        "&timezone=Europe%2FLondon&forecast_days=3")
    wdf=pd.DataFrame(requests.get(wu,timeout=60).json()["daily"]); wdf["time"]=pd.to_datetime(wdf["time"])
    w=wdf.loc[wdf["time"].dt.date==tom.date()].iloc[0]
    t_mean,t_max,t_min = float(w["temperature_2m_mean"]),float(w["temperature_2m_max"]),float(w["temperature_2m_min"])
    wind_max,solar_rad = float(w["wind_speed_10m_max"]),float(w["shortwave_radiation_sum"])
except Exception as e:
    skip(f"weather fetch failed ({type(e).__name__})"); raise SystemExit(0)

# ---- 2. yesterday's price (lag1) and system price (sys_lag1) ----
try:
    y=today.date()-dt.timedelta(days=1)
    pu=("https://data.elexon.co.uk/bmrs/api/v1/balancing/pricing/market-index"
        f"?from={y}T00:00Z&to={y}T23:59Z&format=json")
    pj=[x for x in requests.get(pu,timeout=60).json().get("data",[]) if x.get("dataProvider")=="APXMIDP"]
    price_lag1=max(float(x["price"]) for x in pj)
    su=f"https://data.elexon.co.uk/bmrs/api/v1/balancing/settlement/system-prices/{y}?format=json"
    sj=[x["systemSellPrice"] for x in requests.get(su,timeout=60).json().get("data",[]) if x.get("systemSellPrice") is not None]
    sys_lag1=float(np.mean(sj))
except Exception as e:
    skip(f"price/system fetch failed ({type(e).__name__})"); raise SystemExit(0)

# ---- 3. calendar + stage-1 helper: forecast tomorrow's residual demand ----
uk=holidays.country_holidays("GB", subdiv="ENG")
cal={"dow":tom.dayofweek,"is_we":int(tom.dayofweek>=5),"month":tom.month,"doy":tom.dayofyear,"is_hol":int(tom.date() in uk)}
resid_row={"t_mean":t_mean,"t_max":t_max,"t_min":t_min,"wind_max":wind_max,"solar_rad":solar_rad, **cal}
try:
    helper=lgb.Booster(model_file="model_resid.txt")
    resid_peak=float(helper.predict(pd.DataFrame([resid_row])[RESID_FEATURES])[0])
except Exception as e:
    skip(f"helper/model_resid failed ({type(e).__name__})"); raise SystemExit(0)

# ---- 4. stage-2: price ----
price_row={"resid_peak_gw":resid_peak,"t_mean":t_mean,"t_max":t_max,"t_min":t_min,"wind_max":wind_max,
           "HDD":max(0.0,15.5-t_mean),"CDD":max(0.0,t_mean-22.0),
           "price_lag1":price_lag1,"sys_lag1":sys_lag1, **cal}
booster=lgb.Booster(model_file="model_price.txt")
pred=round(float(np.exp(booster.predict(pd.DataFrame([price_row])[PRICE_FEATURES])[0])),2)

# ---- 5. honest regime flag ----
regime="elevated" if (resid_peak>=RESID_HI or sys_lag1>=SYS_HI or price_lag1>=150) else "normal"

log_row({"date_made":today.date().isoformat(),"target_date":tom.date().isoformat(),
         "predicted_price":pred,"actual_price":"","error":"","regime":regime,"status":"ok"})
