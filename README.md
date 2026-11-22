# Carbonwiser

Carbonwiser forecasts the carbon intensity of the Great Britain electricity grid 24 and 48 hours ahead with XGBoost, then uses the forecast to pick the greenest start time for a flexible job (for example video processing or a batch run). It has a command-line interface and a Streamlit GUI.

## How it works

**Data** (`gb_carbon_weather_cleaned.csv`, hourly, UTC, 2024-03 to 2026-10)
- Carbon intensity: Great Britain Carbon Intensity API (NESO), averaged to hourly values (`ci_actual`).
- Weather: Open-Meteo Previous Runs API. These are the forecasts that were issued 1 and 2 days before each hour, for six points across GB (temperature, wind speed, cloud cover, solar radiation).

**Models**
- One XGBoost ensemble (5 seeds averaged) per horizon: 24h and 48h.
- Features for a target hour T use only what is known at T minus the horizon: past carbon intensity (lags, rolling statistics), calendar features, and the weather forecast issued 1 day (24h model) or 2 days (48h model) before T.
- Weather summaries across locations (mean, wind speed cubed, spread) and short-term weather trends are added.
- `ci_neso_forecast` is deliberately **not** used as a feature. It is a short-range forecast and would not be available 24 to 48 hours ahead.
- The objective is mean absolute error. Hyper-parameters are chosen by random search with expanding-window time-series cross-validation.

**Evaluation**
- Time-based split: fit and tune on data up to 2025-12-31, test on 2026 (about 7,300 hours, never used to make a choice).

## Results (2026 test set)

| Horizon | Model | MAE | RMSE | MAPE |
|---|---|---|---|---|
| 24h | XGBoost | 20.8 | 26.8 | 19.3% |
| 24h | Persistence (same hour yesterday) | 39.5 | 51.5 | 39.2% |
| 48h | XGBoost | 24.2 | 31.5 | 22.0% |
| 48h | Persistence (same hour two days ago) | 47.2 | 60.2 | 47.7% |

MAE and RMSE are in gCO2/kWh. Against the persistence baseline the error is cut by about 47% at 24h and 49% at 48h. These figures come from one test year, and retraining changes them by a few tenths, so treat small differences as noise.

## Scheduling results (2026 backtest)

One flexible job per day was submitted at 12:00 UTC over about 300 days. Each start time was chosen from the forecast only and scored on the grid's actual carbon intensity.

**Scenario 1: 2-hour job, 0.5 kW, 24-hour deadline (303 days)**

| Strategy | Total gCO2 | Saved vs run-now |
|---|---|---|
| Run immediately | 28,554 | 0.0% |
| Schedule on "same hour yesterday" | 28,591 | -0.1% |
| Schedule on XGBoost forecast | 22,880 | 19.9% |
| Perfect foresight (upper limit) | 21,289 | 25.4% |

The forecast captures 78% of the saving that perfect foresight would give. It beat running immediately on 50% of days and was worse on 17%.

**Scenario 2: 2-hour job, 2 kW, 48-hour deadline (302 days)**

| Strategy | Total gCO2 | Saved vs run-now |
|---|---|---|
| Run immediately | 113,671 | 0.0% |
| Schedule on XGBoost forecast | 80,817 | 28.9% |
| Perfect foresight (upper limit) | 71,835 | 36.8% |

The forecast captures 79% of the best possible saving. A longer deadline gives the scheduler more hours to choose from, so the saving is larger. The two scenarios cover slightly different numbers of days.

## Usage

```
pip install -r requirements.txt

python CarbonwiserXGBoost.py train                      # train, evaluate, write predictions_v2_24h.csv / predictions_v2_48h.csv
python CarbonwiserXGBoost.py train --no-tune            # faster, default parameters
python CarbonwiserXGBoost.py refit                      # retrain on all data (once, before using live mode)
python CarbonwiserXGBoost.py live                       # forecast the next 48 hours from now
python CarbonwiserXGBoost.py check                      # score earlier live forecasts against what really happened
python CarbonwiserXGBoost.py schedule --now "2026-10-20 12:00"   # replay on stored history
python CarbonwiserXGBoost.py backtest                   # add --deadline 48 --power 2 etc. to change the scenario
```

In a notebook, call `main(["train", "--no-tune"])` instead.

## Live forecasting

`live` predicts the next 48 hours from the current hour, using real data:

1. Downloads the last 35 days of actual carbon intensity from the NESO Carbon Intensity API.
2. Downloads the current weather forecast for the six GB points from the Open-Meteo Previous Runs API, in the same "forecast issued 1 / 2 days before" form the models were trained on.
3. Builds exactly the same features as in training and predicts: the 24h models cover the next 24 hours, the 48h models cover the 24 hours after that.
4. Writes `live_forecast.csv` (with NESO's own forecast alongside for comparison) and `carbonwise.csv`, then runs the scheduler.

Every run is appended to `live_log.csv`. After the forecast hours have passed, `python CarbonwiserXGBoost.py check` compares those forecasts with what the grid actually did, so the live accuracy can be measured on genuinely unseen data.

Run `train` and then `refit` once first. `refit` retrains the models on all rows of the CSV, including 2026.

## GUI

```
python -m streamlit run app.py
```

Tabs: **Live forecast** (next 48 hours, the best start time for your job, and the CO2 saved), **Backtest**, **Model accuracy**, **Live accuracy** and **Setup** (train and refit from the browser). Job length, deadline and power draw are set in the sidebar, and times can be shown in UTC, GB time or IST.

## Scheduler

`Scheduler.py` takes an hourly forecast (`carbonwise.csv` with columns `timestamp` and `carbon_intensity_gco2_per_kwh`, first row = now) and finds the start time that minimises CO2 for a job that must finish before a deadline:

```
CO2 (g) = sum of carbon intensity over the job's hours x power (kW)
```

Settings (job length, deadline, power draw) are at the top of the file or can be passed on the command line.

## Limitations

- Great Britain only. Times are UTC.
- Forecast accuracy is limited mostly by weather forecast error, especially wind. Extra weather variables (100 m wind speed, direct and diffuse solar radiation) are the most likely next improvement.
- Live accuracy has not been measured yet. The results above come from the 2026 backtest; use `check` after a few days of `live` runs for the real figure.
- Live mode needs the NESO and Open-Meteo APIs to be reachable, and the latest carbon value can lag by a few hours.
- The 24h models cover the next 24 hours and the 48h models the 24 hours after that, so forecasts for later hours are less accurate than for earlier ones.
