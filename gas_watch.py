"""
gas_watch.py - the automated 'is gas relevant yet?' test.

Appends a GAS WATCH section to SCORECARD.md. It measures the rolling correlation between
daily changes in TTF (European gas) and GB day-ahead power price, and states plainly
whether gas has become a feature worth adding to the price model.

Background: tested over 135 days in the 2026 war period, the change-correlation was ~0.06-0.10.
Power was pricing SCARCITY (system tightness), not FUEL COST, so gas was not a useful feature.
Theory says the link should return when the system loosens. This script watches for that,
so the decision is made by evidence, not by memory.
"""
from __future__ import annotations

import datetime as dt
import os

import numpy as np
import pandas as pd
import requests

SCORECARD = "SCORECARD.md"
ADD_THRESHOLD = 0.30       # above this, gas is worth adding as a feature
WATCH_THRESHOLD = 0.20     # above this, it is worth keeping an eye on


def gb_peak_price(start, end):
    rows = []
    d = start
    while d <= end:
        d2 = min(d + dt.timedelta(days=6), end)
        u = ("https://data.elexon.co.uk/bmrs/api/v1/balancing/pricing/market-index"
             f"?from={d}T00:00Z&to={d2}T23:59Z&format=json")
        try:
            rows += requests.get(u, timeout=60).json().get("data", [])
        except Exception:
            pass
        d = d2 + dt.timedelta(days=1)
    rows = [x for x in rows if x.get("dataProvider") == "APXMIDP"]
    if not rows:
        return pd.Series(dtype=float)
    df = pd.DataFrame(rows)
    df["settlementDate"] = pd.to_datetime(df["settlementDate"]).dt.date
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    return df.groupby("settlementDate")["price"].max().sort_index()


def ttf_history(days=140):
    import yfinance as yf
    start = (dt.date.today() - dt.timedelta(days=days)).isoformat()
    d = yf.download("TTF=F", start=start, progress=False, auto_adjust=False)
    s = d["Close"].squeeze()
    s.index = pd.to_datetime(s.index).date
    return s.dropna()


def main() -> None:
    end = dt.date.today() - dt.timedelta(days=2)
    L = []
    A = L.append
    A("")
    A("## Gas watch: is TTF worth adding to the price model yet?")
    A("")
    A("_The model currently does **not** use gas as a feature. The pipeline logs TTF daily "
      "so the history accumulates; this test decides when it earns its place._")
    A("")

    try:
        ttf = ttf_history(140)
        results = []
        for window in (30, 90):
            start = end - dt.timedelta(days=window + 10)
            gb = gb_peak_price(start, end)
            j = pd.DataFrame({"gb": gb}).join(pd.DataFrame({"ttf": ttf}), how="inner").dropna()
            if len(j) < 15:
                results.append((window, len(j), np.nan))
                continue
            corr = float(j["gb"].diff().corr(j["ttf"].diff()))
            results.append((window, len(j), corr))

        A("| window | overlapping days | TTF-vs-price change correlation |")
        A("|---|---:|---:|")
        for w, n, c in results:
            cs = f"{c:+.2f}" if c == c else "-"
            A(f"| {w} days | {n} | **{cs}** |")
        A("")

        latest = [c for _, _, c in results if c == c]
        headline = latest[-1] if latest else np.nan

        if headline != headline:
            A("Not enough overlapping data to judge yet.")
        elif headline >= ADD_THRESHOLD:
            A(f"### ACTION: correlation is {headline:+.2f}, above {ADD_THRESHOLD:.2f}.")
            A("")
            A("**Gas now looks like a useful feature.** The scarcity premium appears to have faded "
              "and fuel cost is transmitting into power price again. Retrain the price model with "
              "`ttf_lag1` and `ttf_change` added, and compare the holdout MAE against the current "
              "model before shipping it. Do not assume it helps: measure it.")
        elif headline >= WATCH_THRESHOLD:
            A(f"### WATCH: correlation is {headline:+.2f}.")
            A("")
            A("Gas is starting to matter but has not yet cleared the "
              f"{ADD_THRESHOLD:.2f} bar. Keep logging; re-check next month.")
        else:
            A(f"### NO ACTION: correlation is {headline:+.2f}.")
            A("")
            A("**Gas is not a useful feature right now.** Power is being priced by system "
              "scarcity rather than fuel cost, so adding TTF would fit noise. The data continues "
              "to accumulate in the price log; this test re-runs daily and will say so when that changes.")
        A("")
        A("_Why the threshold: a feature with a change-correlation near zero adds noise, and a "
          "gradient-booster will happily fit spurious patterns in it. The bar exists to stop that._")
    except Exception as e:
        A(f"_Gas watch unavailable ({type(e).__name__})._")

    block = "\n".join(L)
    if os.path.exists(SCORECARD):
        existing = open(SCORECARD).read()
        # replace an old gas-watch section rather than stacking duplicates
        marker = "## Gas watch:"
        if marker in existing:
            existing = existing.split(marker)[0].rstrip()
        open(SCORECARD, "w").write(existing + "\n" + block + "\n")
    else:
        open(SCORECARD, "w").write("# Performance Scorecard\n" + block + "\n")
    print("gas watch written to", SCORECARD)


if __name__ == "__main__":
    main()
