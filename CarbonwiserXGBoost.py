"""
Carbonwiser: forecast Great Britain grid carbon intensity 24h and 48h ahead with XGBoost,
then use the forecasts to pick the greenest start time for a flexible job (Scheduler.py).

Commands
    python CarbonwiserXGBoost.py train                 # train + evaluate (tuning on, takes a few minutes)
    python CarbonwiserXGBoost.py train --no-tune       # quick run with default parameters
    python CarbonwiserXGBoost.py schedule --now "2026-10-20 12:00"
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

Files used / written (same folder as this script)
    in : gb_carbon_weather_cleaned.csv, Scheduler.py
    out: predictions_v2_24h.csv, predictions_v2_48h.csv, models/, forecast_v2_*.png, carbonwise.csv

All times are UTC (GB data).
"""
import argparse
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


def make_features(df, H):
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


def schedule_cmd(now_text, hours, job_hours, deadline, power):
    S = import_scheduler()
    p24, p48 = load_predictions()
    now = to_utc(now_text) if now_text else (p48.index.max() - pd.Timedelta(hours=47)).floor("1h")
    f = forecast_window(now, min(hours, 48), p24, p48)
    write_csv(f)
    print(f"Wrote {OUT}: {len(f)} hourly forecasts from {now:%Y-%m-%d %H:%M} UTC\n")
    S.JOB_HOURS = S.JOB_HOURS if job_hours is None else job_hours
    S.DEADLINE_HOURS = S.DEADLINE_HOURS if deadline is None else deadline
    S.POWER_KW = S.POWER_KW if power is None else power
    S.main(OUT)


def backtest_cmd(hour, job_hours, deadline, power):
    """Replay the scheduler once a day. Each choice is made from the forecast only, then scored on
    what the grid actually did."""
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
    df = pd.DataFrame(rows)
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

    args = ap.parse_args(argv)
    if args.command == "train":
        train_cmd(resolve_path(args.csv), not args.no_tune, args.trials, args.seeds)
    elif args.command == "schedule":
        schedule_cmd(args.now, args.hours, args.job_hours, args.deadline, args.power)
    elif args.command == "backtest":
        backtest_cmd(args.hour, args.job_hours, args.deadline, args.power)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
