"""
Carbonwiser: forecast Great Britain grid carbon intensity 24h and 48h ahead with XGBoost,
then use the forecasts to pick the greenest start time for a flexible job (Scheduler.py).

Commands
    python CarbonwiserXGBoost.py train                 # train + evaluate (tuning on, takes a few minutes)
    python CarbonwiserXGBoost.py train --no-tune       # quick run with default parameters
    python CarbonwiserXGBoost.py refit                 # retrain on ALL data (needed once before 'live')
    python CarbonwiserXGBoost.py live                  # forecast the next 48h from NOW with the real weather forecast
    python CarbonwiserXGBoost.py check                 # score earlier live forecasts against what really happened
    python CarbonwiserXGBoost.py schedule --now "2026-10-20 12:00"   # replay on stored history
    python CarbonwiserXGBoost.py backtest              # does forecast-based scheduling really save CO2?

In a notebook (Colab/Jupyter) call main() with a list instead:
    main(["train", "--no-tune"])

Design (no data leakage)
    One model per horizon H (24h, 48h). For a target hour T, features only use what is known at T-H:
      - past carbon intensity up to T-H (lags, rolling stats)
      - calendar features of T
      - the weather forecast issued 1 day (24h model) or 2 days (48h model) before T
    ci_neso_forecast is NOT used as a feature (it is a short-range forecast, not available 24-48h ahead).

Split: tune and fit on data up to 2025-12-31, test on 2026 (never used for any choice).

Live mode
    'live' downloads the last ~5 weeks of GB carbon intensity (NESO) and the current Open-Meteo weather forecast
    for the same six GB points used in training, builds exactly the same features, and predicts the next 48 hours.
    Hours now .. now+23 use the 24h model, hours now+24 .. now+47 use the 48h model.
    Every run is appended to live_log.csv; 'check' scores those forecasts once the hours have passed.

Files used / written (same folder as this script)
    in : gb_carbon_weather_cleaned.csv, Scheduler.py
    out: predictions_v2_24h.csv, predictions_v2_48h.csv, models/, forecast_v2_*.png, carbonwise.csv,
         live_forecast.csv, live_log.csv, live_forecast.png

All times are UTC (GB data).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

# ---------------- settings ----------------
try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:                      # running inside a notebook cell
    HERE = os.getcwd()

TRAIN_CSV = os.path.join(HERE, "gb_carbon_weather_cleaned.csv")
P24 = os.path.join(HERE, "predictions_v2_24h.csv")
P48 = os.path.join(HERE, "predictions_v2_48h.csv")
OUT = os.path.join(HERE, "carbonwise.csv")
MODEL_DIR = os.path.join(HERE, "models")

HORIZONS = [24, 48]
VAL_END = pd.Timestamp("2025-12-31 23:00", tz="UTC")     # everything after this is the test set
SEED = 42

# expanding-window CV folds (start inclusive, end exclusive); each fold trains on everything before it
FOLDS = [
    (pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2025-05-01", tz="UTC")),
    (pd.Timestamp("2025-05-01", tz="UTC"), pd.Timestamp("2025-09-01", tz="UTC")),
    (pd.Timestamp("2025-09-01", tz="UTC"), pd.Timestamp("2026-01-01", tz="UTC")),
]
DEFAULT_PARAMS = dict(max_depth=6, learning_rate=0.03, min_child_weight=5,
                      subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, reg_alpha=0.0)
WEATHER_VARS = ["temperature_2m", "wind_speed_10m", "cloud_cover", "shortwave_radiation"]

# Live mode: the same six GB points used to build the training data (names must match the CSV column names)
LOCATIONS = {
    "south":     (51.5, -0.1),   # London: demand + temperature
    "southwest": (50.6, -4.2),   # Cornwall/Devon: solar
    "midlands":  (52.6, -1.5),   # demand
    "scotland":  (55.9, -3.2),   # onshore wind
    "northsea":  (53.5, 1.5),    # offshore wind
    "irishsea":  (53.8, -3.8),   # offshore wind
}
LIVE_HISTORY_DAYS = 35           # carbon history needed for the 30-day and 7-day features
CI_URL = "https://api.carbonintensity.org.uk/intensity/{start}/{end}"
CI_FW48_URL = "https://api.carbonintensity.org.uk/intensity/{start}/fw48h"
PREV_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
CI_FMT = "%Y-%m-%dT%H:%MZ"
LIVE_FORECAST = os.path.join(HERE, "live_forecast.csv")
LIVE_LOG = os.path.join(HERE, "live_log.csv")
# ------------------------------------------


# =====================================================================
# Data and features
# =====================================================================
def resolve_path(path):
    if os.path.isabs(path) or os.path.isfile(path):
        return path
    rel = os.path.join(HERE, path)
    return rel if os.path.isfile(rel) else path


def load(csv_path):
    df = pd.read_csv(csv_path, index_col=0)
    required = {"ci_actual"}
    if required.difference(df.columns):
        raise SystemExit("Training CSV needs a datetime index column and a numeric ci_actual column.")
    df.index = pd.to_datetime(df.index, utc=True)
    return df.sort_index().asfreq("1h")   # one row per hour, so shift() is time-based


def locations(df):
    return sorted({c.split("_previous_day")[1][2:] for c in df.columns if "_previous_day" in c})


def make_features(df, H, require_target=True):
    y = df["ci_actual"]
    d = H // 24                       # 24h -> day-1 weather forecast, 48h -> day-2 forecast
    locs = locations(df)
    idx = df.index

    X = pd.DataFrame(index=idx)
    # calendar
    X["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    X["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    X["dayofweek"] = idx.dayofweek
    X["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    X["month_sin"] = np.sin(2 * np.pi * idx.month / 12)
    X["month_cos"] = np.cos(2 * np.pi * idx.month / 12)
    X["doy_sin"] = np.sin(2 * np.pi * idx.dayofyear / 365.25)
    X["doy_cos"] = np.cos(2 * np.pi * idx.dayofyear / 365.25)
    try:
        import holidays
        hol = holidays.country_holidays("GB", subdiv="ENG", years=range(2024, 2028))
        X["is_holiday"] = [int(x in hol) for x in idx.date]
    except Exception:
        print("(holidays package not available, skipping holiday flag)")

    # carbon history, only what is known at forecast time (T - H)
    past = y.shift(H)
    X[f"ci_lag_{H}"] = past
    X[f"ci_lag_{H + 24}"] = y.shift(H + 24)
    X["ci_lag_168"] = y.shift(168)
    X["ci_mean_24h"] = past.rolling(24, min_periods=12).mean()
    X["ci_std_24h"] = past.rolling(24, min_periods=12).std()
    X["ci_mean_7d"] = past.rolling(168, min_periods=84).mean()
    if H == 24:
        X["ci_same_hour_7d"] = sum(y.shift(H + 24 * k) for k in range(7)) / 7
        X["ci_mean_30d"] = past.rolling(720, min_periods=360).mean()
        X["ci_min_24h"] = past.rolling(24, min_periods=12).min()
        X["ci_max_24h"] = past.rolling(24, min_periods=12).max()
        X["ci_trend_3h"] = past - y.shift(H + 3)
        X["ci_trend_24h"] = past - y.shift(H + 24)
        X["ci_dev_from_7d"] = past - X["ci_mean_7d"]
        X["ci_lag_h1"] = y.shift(H + 1)
        X["ci_lag_h2"] = y.shift(H + 2)

    # weather: per-location columns for the matching forecast age
    wcols = [c for c in df.columns if f"previous_day{d}" in c]
    X[wcols] = df[wcols]

    # weather: summaries across locations
    for v in WEATHER_VARS:
        X[f"{v}_mean"] = df[[f"{v}_previous_day{d}_{l}" for l in locs]].mean(axis=1)
    wind = df[[f"wind_speed_10m_previous_day{d}_{l}" for l in locs]]
    solar = df[[f"shortwave_radiation_previous_day{d}_{l}" for l in locs]]
    X["wind_cube_mean"] = (wind ** 3).mean(axis=1)
    X["wind_spread"] = wind.max(axis=1) - wind.min(axis=1)
    wm, sm = wind.mean(axis=1), solar.mean(axis=1)
    for k in (3, 6, 12):
        X[f"wind_change_{k}h"] = wm - wm.shift(k)
    X["solar_change_3h"] = sm - sm.shift(3)

    # 24h model only: the day-2 forecast is also known, and so is how the forecast was revised
    if H == 24:
        for v in ("wind_speed_10m", "shortwave_radiation", "cloud_cover"):
            c1 = df[[f"{v}_previous_day1_{l}" for l in locs]].mean(axis=1)
            c2 = df[[f"{v}_previous_day2_{l}" for l in locs]].mean(axis=1)
            X[f"{v}_day2_mean"] = c2
            X[f"{v}_revision"] = c1 - c2

    if not require_target:          # live mode: the target hours are in the future
        return X, y
    keep = y.notna() & X[f"ci_lag_{H}"].notna()
    return X[keep], y[keep]


# =====================================================================
# Training
# =====================================================================
def fit_xgb(params, Xtr, ytr, Xva=None, yva=None, n_estimators=2000, seed=0, early=True):
    import xgboost as xgb
    kw = dict(objective="reg:absoluteerror", eval_metric="mae", tree_method="hist",
              n_estimators=n_estimators, random_state=seed, **params)
    if early:
        kw["early_stopping_rounds"] = 100
    model = xgb.XGBRegressor(**kw)
    if early:
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    else:
        model.fit(Xtr, ytr, verbose=False)
    return model


def cv_score(params, X, y):
    """Mean MAE across expanding-window folds, and the average number of trees chosen."""
    maes, iters = [], []
    for start, end in FOLDS:
        tr = X.index < start
        va = (X.index >= start) & (X.index < end)
        m = fit_xgb(params, X[tr], y[tr], X[va], y[va])
        maes.append(float(np.abs(m.predict(X[va]) - y[va]).mean()))
        iters.append(m.best_iteration + 1)
    return float(np.mean(maes)), int(np.mean(iters))


def random_params(rng):
    return dict(
        max_depth=int(rng.randint(3, 9)),
        learning_rate=float(rng.choice([0.02, 0.03, 0.05, 0.08])),
        min_child_weight=int(rng.choice([1, 3, 5, 10, 20])),
        subsample=float(rng.uniform(0.6, 1.0)),
        colsample_bytree=float(rng.uniform(0.4, 1.0)),
        reg_lambda=float(rng.choice([0.5, 1, 3, 10])),
        reg_alpha=float(rng.choice([0, 0.5, 2])),
    )


def tune(X, y, trials):
    trainval = X.index <= VAL_END
    Xtv, ytv = X[trainval], y[trainval]
    rng = np.random.RandomState(SEED)
    best = None
    print(f"tuning {trials} parameter sets with {len(FOLDS)}-fold time-series CV ...")
    for t in range(trials):
        params = DEFAULT_PARAMS if t == 0 else random_params(rng)
        mae, n_trees = cv_score(params, Xtv, ytv)
        print(f"  trial {t + 1:2d}: CV MAE {mae:6.2f} {'(default)' if t == 0 else ''}")
        if best is None or mae < best[0]:
            best = (mae, params, n_trees)
    print(f"best CV MAE {best[0]:.2f} | trees {best[2]} | params {best[1]}")
    return best[1], best[2]


def plot_forecast(out, H):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    last = out.iloc[-24 * 14:]
    plt.figure(figsize=(12, 4))
    plt.plot(last.index, last["actual"], label="actual", lw=1.5)
    plt.plot(last.index, last["xgb"], label=f"XGBoost {H}h", lw=1.2)
    plt.plot(last.index, last["persistence"], label="persistence", lw=0.8, alpha=0.6)
    plt.ylabel("gCO2/kWh")
    plt.title(f"{H}h-ahead forecast, last 14 days of test data")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, f"forecast_v2_{H}h.png"), dpi=120)
    if "ipykernel" in sys.modules:
        plt.show()
    plt.close()


def train_cmd(csv_path, do_tune, trials, seeds):
    try:
        import xgboost  # noqa: F401
    except ImportError:
        raise SystemExit("xgboost is not installed. Run: pip install -r requirements.txt")

    df = load(csv_path)
    os.makedirs(MODEL_DIR, exist_ok=True)
    results, monthly = [], []

    for H in HORIZONS:
        print(f"\n===== horizon {H}h =====")
        X, y = make_features(df, H)
        trainval = X.index <= VAL_END
        test = X.index > VAL_END
        print(f"features: {X.shape[1]} | train+val rows {trainval.sum()} | test rows {test.sum()}")

        if do_tune:
            params, n_trees = tune(X, y, trials)
        else:
            params, n_trees = DEFAULT_PARAMS, 600
        n_trees = max(100, int(n_trees * 1.1))   # a little more data than in the CV folds

        # final ensemble: same parameters, different seeds, trained on everything before the test period
        models = [fit_xgb(params, X[trainval], y[trainval], n_estimators=n_trees, seed=s, early=False)
                  for s in range(seeds)]
        pred = pd.Series(np.mean([m.predict(X[test]) for m in models], axis=0), index=X[test].index)
        y_te = y[test]

        persist = X.loc[test, f"ci_lag_{H}"]
        seasonal = X.loc[test, "ci_lag_168"]
        ok = seasonal.notna()
        for name, p in [(f"XGBoost {H}h", pred), (f"Persistence {H}h", persist),
                        ("Same hour last week", seasonal[ok])]:
            err = p - y_te.loc[p.index]
            results.append({"horizon": f"{H}h", "model": name,
                            "MAE": round(float(err.abs().mean()), 1),
                            "RMSE": round(float(np.sqrt((err ** 2).mean())), 1),
                            "MAPE %": round(float((err.abs() / y_te.loc[p.index]).mean() * 100), 1)})

        out = pd.DataFrame({"actual": y_te, "xgb": pred, "persistence": persist})
        by_month = out.groupby(out.index.month).apply(
            lambda g: pd.Series({"xgb_MAE": (g.xgb - g.actual).abs().mean(),
                                 "persistence_MAE": (g.persistence - g.actual).abs().mean()}))
        by_month.insert(0, "horizon", f"{H}h")
        monthly.append(by_month.round(1))

        imp = pd.Series(np.mean([m.feature_importances_ for m in models], axis=0),
                        index=X.columns).sort_values(ascending=False)
        print("top 12 features:\n", imp.head(12).round(3).to_string())

        out.to_csv(P24 if H == 24 else P48)
        for s, m in enumerate(models):
            m.save_model(os.path.join(MODEL_DIR, f"xgb_{H}h_seed{s}.json"))
        with open(os.path.join(MODEL_DIR, f"model_info_{H}h.json"), "w") as f:
            json.dump({"horizon_hours": H, "params": params, "n_trees": n_trees,
                       "features": list(X.columns), "trained_through": str(VAL_END)}, f, indent=2)
        plot_forecast(out, H)

    print("\n===== test results (2026) =====")
    print(pd.DataFrame(results).to_string(index=False))
    print("\n===== MAE by month (2026) =====")
    print(pd.concat(monthly).rename_axis("month").to_string())
    print(f"\nSaved {os.path.basename(P24)}, {os.path.basename(P48)} and models/ in {HERE}")


# =====================================================================
# Scheduling link
# =====================================================================
def import_scheduler():
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    try:
        import Scheduler
    except ImportError:
        raise SystemExit(f"Scheduler.py not found in {HERE}. Put it in the same folder as this script.")
    return Scheduler


def load_predictions():
    out = []
    for path in (P24, P48):
        if not os.path.exists(path):
            raise SystemExit(f"Missing {path}\nRun 'python CarbonwiserXGBoost.py train' first.")
        d = pd.read_csv(path, index_col=0)
        d.index = pd.to_datetime(d.index, utc=True)
        out.append(d.sort_index())
    return out


def forecast_window(now, hours, p24, p48):
    """Forecast for the hours now, now+1, ..., now+hours-1, using only forecasts issued at or before now.
    Hours up to now+24 come from the 24h model, later hours from the 48h model."""
    idx = pd.date_range(now, periods=hours, freq="1h")
    f = pd.Series(np.nan, index=idx)
    for t in idx:
        if t <= now + pd.Timedelta(hours=24) and t in p24.index:
            f[t] = p24.at[t, "xgb"]
        elif t > now + pd.Timedelta(hours=24) and t in p48.index:
            f[t] = p48.at[t, "xgb"]
    f = f.interpolate(limit=3, limit_direction="both")   # a few hours are missing in the source data
    if f.isna().any():
        raise ValueError(f"No forecast available for some hours starting {now}")
    return f


def to_utc(text):
    t = pd.Timestamp(text)
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    return t.floor("1h")


def write_csv(f, path=OUT):
    pd.DataFrame({
        "timestamp": f.index.strftime("%Y-%m-%d %H:%M"),
        "carbon_intensity_gco2_per_kwh": f.round(1).values,
    }).to_csv(path, index=False)


def run_scheduler(S, job_hours, deadline, power):
    S.JOB_HOURS = S.JOB_HOURS if job_hours is None else job_hours
    S.DEADLINE_HOURS = S.DEADLINE_HOURS if deadline is None else deadline
    S.POWER_KW = S.POWER_KW if power is None else power
    S.main(OUT)


def schedule_cmd(now_text, hours, job_hours, deadline, power):
    S = import_scheduler()
    p24, p48 = load_predictions()
    now = to_utc(now_text) if now_text else (p48.index.max() - pd.Timedelta(hours=47)).floor("1h")
    f = forecast_window(now, min(hours, 48), p24, p48)
    write_csv(f)
    print(f"Wrote {OUT}: {len(f)} hourly forecasts from {now:%Y-%m-%d %H:%M} UTC\n")
    run_scheduler(S, job_hours, deadline, power)


def backtest_run(hour, job_hours, deadline, power):
    """Replay the scheduler once a day. Each choice is made from the forecast only, then scored on
    what the grid actually did. Returns (one row per day, (job_hours, deadline, power))."""
    S = import_scheduler()
    job_hours = S.JOB_HOURS if job_hours is None else job_hours
    deadline = min(S.DEADLINE_HOURS if deadline is None else deadline, 48)
    power = S.POWER_KW if power is None else power
    p24, p48 = load_predictions()

    actual = p24["actual"]
    rows = []
    for now in [t for t in p24.index if t.hour == hour]:
        try:
            f = forecast_window(now, deadline, p24, p48)
        except ValueError:
            continue
        a = actual.reindex(f.index)
        if a.isna().any():
            continue
        s_fc, _ = S.best_start(f.values, job_hours, deadline, power)
        s_or, _ = S.best_start(a.values, job_hours, deadline, power)   # perfect foresight
        row = {
            "date": now,
            "run_now": S.co2(a.values, 0, job_hours, power),
            "forecast": S.co2(a.values, s_fc, job_hours, power),
            "oracle": S.co2(a.values, s_or, job_hours, power),
        }
        if deadline <= 24:   # "same hour yesterday" is only a valid forecast inside 24h
            pers = p24["persistence"].reindex(f.index)
            if not pers.isna().any():
                s_p, _ = S.best_start(pers.values, job_hours, deadline, power)
                row["persistence"] = S.co2(a.values, s_p, job_hours, power)
        rows.append(row)

    if not rows:
        raise SystemExit("No usable days for the backtest.")
    return pd.DataFrame(rows), (job_hours, deadline, power)


def backtest_cmd(hour, job_hours, deadline, power):
    df, (job_hours, deadline, power) = backtest_run(hour, job_hours, deadline, power)
    print(f"Backtest: {len(df)} days, one job per day starting {hour:02d}:00 UTC, "
          f"{job_hours}h job, deadline {deadline}h, {power} kW\n")
    base = df["run_now"].sum()
    print(f"{'strategy':32s} {'total g CO2':>12s} {'saved vs run-now':>18s}")
    for key, label in [("run_now", "Run immediately"),
                       ("persistence", "Schedule on 'same hour yesterday'"),
                       ("forecast", "Schedule on XGBoost forecast"),
                       ("oracle", "Perfect foresight (upper limit)")]:
        if key in df:
            tot = df[key].sum()
            print(f"{label:32s} {tot:12.0f} {(base - tot) / base * 100:17.1f}%")
    share = (base - df["forecast"].sum()) / (base - df["oracle"].sum()) * 100
    print(f"\nThe forecast captures {share:.0f}% of the saving that perfect foresight would give.")
    print(f"Forecast-based choice beat running now on {(df['forecast'] < df['run_now']).mean() * 100:.0f}% of days "
          f"and was worse on {(df['forecast'] > df['run_now']).mean() * 100:.0f}%.")


# =====================================================================
# Live forecasting: latest carbon data + the real weather forecast for Great Britain
# =====================================================================
def get_json(url, params=None, tries=4):
    import time
    import requests
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=60)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            if i == tries - 1:
                raise
            print(f"  retrying after error: {e}")
            time.sleep(2 * (i + 1))


def fetch_carbon_hourly(start, end):
    """Hourly mean of actual GB carbon intensity between two UTC timestamps (NESO Carbon Intensity API)."""
    rows = []
    cur = start
    while cur < end:
        nxt = min(cur + pd.Timedelta(days=13), end)       # the API allows about 14 days per request
        data = get_json(CI_URL.format(start=cur.strftime(CI_FMT), end=nxt.strftime(CI_FMT)))["data"]
        for item in data:
            rows.append((item.get("from"), (item.get("intensity") or {}).get("actual")))
        cur = nxt
    if not rows:
        return pd.Series(dtype=float)
    d = pd.DataFrame(rows, columns=["t", "v"])
    d["t"] = pd.to_datetime(d["t"], utc=True)
    d["v"] = pd.to_numeric(d["v"], errors="coerce")
    return d.drop_duplicates("t").set_index("t")["v"].sort_index().resample("1h").mean()


def fetch_neso_forecast(now):
    """NESO's own 48h forecast, only shown for comparison. Never stops the run if it fails."""
    try:
        data = get_json(CI_FW48_URL.format(start=now.strftime(CI_FMT)))["data"]
        s = pd.Series({pd.Timestamp(i["from"]): (i.get("intensity") or {}).get("forecast") for i in data})
        s.index = pd.to_datetime(s.index, utc=True)
        return pd.to_numeric(s, errors="coerce").sort_index().resample("1h").mean()
    except Exception as e:
        print(f"(NESO comparison forecast not available: {e})")
        return None


def fetch_weather_live(now):
    """Weather forecasts for the six GB points in the same 'previous_day1 / previous_day2' form as the
    training data: for each target hour, the forecast issued 1 and 2 days before it."""
    hourly = []
    for v in WEATHER_VARS:
        hourly += [v, f"{v}_previous_day1", f"{v}_previous_day2"]
    cols, filled = {}, 0
    for name, (lat, lon) in LOCATIONS.items():
        data = get_json(PREV_RUNS_URL, params={
            "latitude": lat, "longitude": lon, "hourly": ",".join(hourly),
            "past_days": 3, "forecast_days": 4, "timezone": "GMT"})["hourly"]
        w = pd.DataFrame(data)
        w.index = pd.to_datetime(w.pop("time"), utc=True)
        for v in WEATHER_VARS:
            for d in (1, 2):
                col = f"{v}_previous_day{d}"
                s = w[col] if col in w else pd.Series(np.nan, index=w.index)
                needed = (w.index >= now - pd.Timedelta(hours=12)) & (w.index < now + pd.Timedelta(hours=24 * d))
                filled += int((s.isna() & needed).sum())
                cols[f"{col}_{name}"] = s.fillna(w[v])     # fall back to the current forecast if a run is missing
    if filled:
        print(f"(note: {filled} weather values were missing for the previous-day forecasts "
              f"and were replaced by the current forecast)")
    return pd.DataFrame(cols)


def build_live_frame(now):
    """One hourly table in the same layout as the training CSV, running from 35 days back to now+47h.
    ci_actual is empty for the future; weather columns hold the real forecasts."""
    ci = fetch_carbon_hourly(now - pd.Timedelta(days=LIVE_HISTORY_DAYS), now)
    if ci.dropna().empty:
        raise SystemExit("No carbon intensity data came back from the NESO API.")
    lag_h = (now - ci.dropna().index.max()) / pd.Timedelta(hours=1)
    if lag_h > 3:
        print(f"(warning: the latest carbon value is {lag_h:.0f}h old, forecasts may be less accurate)")
    weather = fetch_weather_live(now)
    idx = pd.date_range(now - pd.Timedelta(days=LIVE_HISTORY_DAYS), now + pd.Timedelta(hours=47), freq="1h")
    live = pd.DataFrame(index=idx)
    live["ci_actual"] = ci.reindex(idx)
    live.loc[live.index >= now, "ci_actual"] = np.nan        # only complete hours before now are known
    return live.join(weather.reindex(idx))


def load_models(H):
    """Prefer the models refit on all data ('refit'); fall back to the ones trained for the 2026 test."""
    import xgboost as xgb
    for prefix in ("final_", ""):
        info_path = os.path.join(MODEL_DIR, f"{prefix}model_info_{H}h.json")
        paths = sorted(glob.glob(os.path.join(MODEL_DIR, f"{prefix}xgb_{H}h_seed*.json")))
        if os.path.exists(info_path) and paths:
            with open(info_path) as f:
                info = json.load(f)
            models = []
            for p in paths:
                m = xgb.XGBRegressor()
                m.load_model(p)
                models.append(m)
            if prefix == "":
                print(f"(note: using the {H}h models trained through {info['trained_through'][:10]} only. "
                      f"Run 'refit' to include all your data.)")
            return models, info
    raise SystemExit(f"No saved {H}h models found in {MODEL_DIR}. Run 'train' first (then 'refit').")


def refit_cmd(csv_path, seeds):
    """Retrain both horizons on ALL rows (including 2026), reusing the parameters found by 'train'."""
    try:
        import xgboost  # noqa: F401
    except ImportError:
        raise SystemExit("xgboost is not installed. Run: pip install -r requirements.txt")
    df = load(csv_path)
    os.makedirs(MODEL_DIR, exist_ok=True)
    for H in HORIZONS:
        X, y = make_features(df, H)
        info_path = os.path.join(MODEL_DIR, f"model_info_{H}h.json")
        if os.path.exists(info_path):
            with open(info_path) as f:
                old = json.load(f)
            params, n_trees = old["params"], old["n_trees"]
        else:
            print(f"(no tuned parameters found for {H}h, using the defaults. Run 'train' first for better results.)")
            params, n_trees = DEFAULT_PARAMS, 600
        print(f"refitting {H}h model on {len(X)} rows through {X.index.max():%Y-%m-%d} ...")
        for s in range(seeds):
            m = fit_xgb(params, X, y, n_estimators=n_trees, seed=s, early=False)
            m.save_model(os.path.join(MODEL_DIR, f"final_xgb_{H}h_seed{s}.json"))
        with open(os.path.join(MODEL_DIR, f"final_model_info_{H}h.json"), "w") as f:
            json.dump({"horizon_hours": H, "params": params, "n_trees": n_trees,
                       "features": list(X.columns), "trained_through": str(X.index.max())}, f, indent=2)
    print(f"Saved final models in {MODEL_DIR}. Next: python CarbonwiserXGBoost.py live")


def live_forecast(now=None):
    """Forecast the next 48 hours from now. Returns (forecast table, forecast origin).
    Used by both the command line ('live') and the GUI. (The 'now' argument is only for testing.)"""
    now = (pd.Timestamp.now(tz="UTC") if now is None else now).floor("1h")
    models = {H: load_models(H) for H in HORIZONS}
    print(f"Forecast origin: {now:%Y-%m-%d %H:%M} UTC")

    live = build_live_frame(now)

    parts = []
    for H, first in ((24, 0), (48, 24)):                 # now..now+23 -> 24h model, now+24..now+47 -> 48h model
        ens, info = models[H]
        X, _ = make_features(live, H, require_target=False)
        targets = pd.date_range(now + pd.Timedelta(hours=first), periods=24, freq="1h")
        Xt = X.reindex(targets)
        missing = [c for c in info["features"] if c not in Xt.columns]
        if missing:
            raise SystemExit(f"Features missing in live data: {missing[:5]} ... "
                             "(if 'is_holiday' is listed, run: pip install holidays)")
        Xt = Xt[info["features"]]
        if Xt[f"ci_lag_{H}"].isna().any():
            print(f"(warning: {int(Xt[f'ci_lag_{H}'].isna().sum())} {H}h rows have no recent carbon value)")
        pred = np.mean([m.predict(Xt) for m in ens], axis=0)
        parts.append(pd.DataFrame({"forecast": np.clip(pred, 0, None), "model": f"{H}h"}, index=targets))
    f = pd.concat(parts)

    neso = fetch_neso_forecast(now)
    f["neso_forecast"] = neso.reindex(f.index) if neso is not None else np.nan
    f.index.name = "timestamp"
    f.round(1).to_csv(LIVE_FORECAST)

    # append to the log (once per forecast origin) so 'check' can score it later
    log = pd.DataFrame({"issued_at": now, "target": f.index, "lead_hours": range(len(f)),
                        "model": f["model"].values, "forecast": f["forecast"].round(1).values,
                        "neso_forecast": f["neso_forecast"].round(1).values})
    if os.path.exists(LIVE_LOG):
        prev = pd.to_datetime(pd.read_csv(LIVE_LOG)["issued_at"], utc=True)
        if (prev == now).any():
            log = None
    if log is not None:
        log.to_csv(LIVE_LOG, mode="a", header=not os.path.exists(LIVE_LOG), index=False)

    return f, now


def live_cmd(hours, job_hours, deadline, power, now=None):
    """Forecast the next 48 hours from now, print them and run the scheduler."""
    f, now = live_forecast(now)

    print("\nForecast (gCO2/kWh, UTC):")
    show = f.round(0).copy()
    show.index = show.index.strftime("%a %d %b %H:%M")
    print(show.to_string())

    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 4))
        plt.plot(f.index, f["forecast"], label="XGBoost forecast", lw=1.8)
        if f["neso_forecast"].notna().any():
            plt.plot(f.index, f["neso_forecast"], label="NESO forecast (comparison)", lw=1, alpha=0.7)
        plt.axvline(now + pd.Timedelta(hours=24), color="grey", ls=":", lw=1)
        plt.ylabel("gCO2/kWh")
        plt.title(f"Next 48 hours from {now:%Y-%m-%d %H:%M} UTC")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(HERE, "live_forecast.png"), dpi=120)
        plt.close()
    except ImportError:
        pass

    # hand over to the scheduler
    S = import_scheduler()
    write_csv(f["forecast"].iloc[:min(hours, 48)])
    print(f"\nWrote {OUT} ({min(hours, 48)} hourly forecasts from {now:%Y-%m-%d %H:%M} UTC)\n")
    run_scheduler(S, job_hours, deadline, power)
    return f


def check_table():
    """Score the logged live forecasts against what the grid actually did.
    Returns (summary table, hours scored, number of live runs, per-hour detail)."""
    if not os.path.exists(LIVE_LOG):
        raise SystemExit("No live_log.csv yet. Run 'live' first, then come back once those hours have passed.")
    log = pd.read_csv(LIVE_LOG)
    for c in ("issued_at", "target"):
        log[c] = pd.to_datetime(log[c], utc=True)
    now = pd.Timestamp.now(tz="UTC").floor("1h")
    log = log[log["target"] < now - pd.Timedelta(hours=1)].copy()
    if log.empty:
        raise SystemExit("None of the logged forecast hours have passed yet. Check again later.")

    actual = fetch_carbon_hourly(log["target"].min() - pd.Timedelta(hours=48), now)
    log["actual"] = log["target"].map(actual)
    lag = log["model"].map({"24h": 24, "48h": 48})
    log["persistence"] = [actual.get(t - pd.Timedelta(hours=int(h)), np.nan) for t, h in zip(log["target"], lag)]
    log = log.dropna(subset=["actual"])
    if log.empty:
        raise SystemExit("The real values for those hours are not available yet.")

    rows = []
    for model, g in log.groupby("model"):
        for name, col in (("XGBoost", "forecast"), ("Persistence", "persistence"), ("NESO forecast", "neso_forecast")):
            gg = g.dropna(subset=[col])
            if gg.empty:
                continue
            err = gg[col] - gg["actual"]
            rows.append({"horizon": model, "forecast": name, "hours": len(gg),
                         "MAE": round(float(err.abs().mean()), 1),
                         "RMSE": round(float(np.sqrt((err ** 2).mean())), 1),
                         "MAPE %": round(float((err.abs() / gg["actual"]).mean() * 100), 1)})
    return pd.DataFrame(rows), len(log), int(log["issued_at"].nunique()), log


def check_cmd():
    table, n_hours, n_runs, _ = check_table()
    print(f"Scored {n_hours} forecast hours from {n_runs} live run(s):\n")
    print(table.to_string(index=False))


# =====================================================================
def main(argv=None):
    if argv is None:
        argv = [] if "ipykernel" in sys.modules else sys.argv[1:]

    ap = argparse.ArgumentParser(description="Carbonwiser: GB carbon intensity forecasting + scheduling")
    sub = ap.add_subparsers(dest="command")

    t = sub.add_parser("train", help="train 24h and 48h models, evaluate on 2026, write prediction files")
    t.add_argument("--csv", default=TRAIN_CSV, help="training CSV (default: gb_carbon_weather_cleaned.csv)")
    t.add_argument("--no-tune", action="store_true", help="skip the parameter search")
    t.add_argument("--trials", type=int, default=20, help="random-search trials per horizon")
    t.add_argument("--seeds", type=int, default=5, help="models averaged in the final ensemble")

    for name, text in (("schedule", "write carbonwise.csv from the forecasts and run Scheduler.py"),
                       ("backtest", "replay the scheduler on 2026 and report CO2 saved")):
        s = sub.add_parser(name, help=text)
        s.add_argument("--job-hours", type=int)
        s.add_argument("--deadline", type=int)
        s.add_argument("--power", type=float)
        if name == "schedule":
            s.add_argument("--now", help='forecast start, UTC, e.g. "2026-10-20 12:00"')
            s.add_argument("--hours", type=int, default=48, help="rows to write to carbonwise.csv (max 48)")
        else:
            s.add_argument("--hour", type=int, default=12, help="daily job submission hour (UTC)")

    r = sub.add_parser("refit", help="retrain on ALL data (including 2026); run once before 'live'")
    r.add_argument("--csv", default=TRAIN_CSV)
    r.add_argument("--seeds", type=int, default=5)

    lv = sub.add_parser("live", help="forecast the next 48h from now using the real weather forecast")
    lv.add_argument("--hours", type=int, default=48, help="rows to write to carbonwise.csv (max 48)")
    lv.add_argument("--job-hours", type=int)
    lv.add_argument("--deadline", type=int)
    lv.add_argument("--power", type=float)

    sub.add_parser("check", help="score earlier live forecasts against what really happened")

    args = ap.parse_args(argv)
    if args.command == "refit":
        refit_cmd(resolve_path(args.csv), args.seeds)
    elif args.command == "live":
        live_cmd(args.hours, args.job_hours, args.deadline, args.power)
    elif args.command == "check":
        check_cmd()
    elif args.command == "train":
        train_cmd(resolve_path(args.csv), not args.no_tune, args.trials, args.seeds)
    elif args.command == "schedule":
        schedule_cmd(args.now, args.hours, args.job_hours, args.deadline, args.power)
    elif args.command == "backtest":
        backtest_cmd(args.hour, args.job_hours, args.deadline, args.power)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
