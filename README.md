# Carbonwiser

Carbonwiser forecasts the carbon intensity of the Great Britain electricity grid 24 and 48 hours ahead with XGBoost, then uses the forecast to pick the greenest start time for a flexible job (for example video processing or a batch run).

## How it works

**Data** (`gb_carbon_weather_cleaned.csv`, hourly, UTC, 2024-03 to 2026-10)
- Carbon intensity: Great Britain Carbon Intensity API (NESO), averaged to hourly values (`ci_actual`).
- Weather: Open-Meteo Previous Runs API. These are the forecasts that were issued 1 and 2 days before each hour, for six points across GB (temperature, wind speed, cloud cover, solar radiation).

**Models**
- One XGBoost ensemble (5 seeds averaged) per horizon: 24h and 48h.
- Features for a target hour T use only what is known at T minus the horizon: past carbon intensity (lags, rolling statistics), calendar features, and the weather forecast issued 1 day (24h model) or 2 days (48h model) before T.
- Weather summaries across locations (mean, wind speed cubed, spread) and short-term weather trends are added.
- `ci_neso_forecast` is deliberately **not** used as a feature. It is a short-range forecast and would not be available 24 to 48 hours ahead.
- Objective is mean absolute error. Hyper-parameters are chosen by random search with expanding-window time-series cross-validation.

**Evaluation**
- Time-based split: fit and tune on data up to 2025-12-31, test on 2026 (about 7,300 hours, never used to make a choice).

## Results (2026 test set)

| Horizon | Model | MAE | RMSE | MAPE |
|---|---|---|---|---|
| 24h | XGBoost | 21.1 | 27.1 | 19.8% |
| 24h | Persistence (same hour yesterday) | 39.5 | 51.5 | 39.2% |
| 48h | XGBoost | 24.0 | 31.2 | 21.8% |
| 48h | Persistence (same hour two days ago) | 47.2 | 60.2 | 47.7% |

MAE and RMSE are in gCO2/kWh. Against the persistence baseline the error is cut by about 47% at 24h and 49% at 48h. The weakest months are June and January.

## Usage

```
pip install -r requirements.txt

python CarbonwiserXGBoost.py train                      # train, evaluate, write predictions_v2_24h.csv / predictions_v2_48h.csv
python CarbonwiserXGBoost.py train --no-tune            # faster, default parameters
python CarbonwiserXGBoost.py schedule --now "2026-10-20 12:00"
python CarbonwiserXGBoost.py backtest
```

In a notebook, call `main(["train", "--no-tune"])` instead.

## Scheduler

`Scheduler.py` takes an hourly forecast (`carbonwise.csv` with columns `timestamp` and `carbon_intensity_gco2_per_kwh`, first row = now) and finds the start time that minimises CO2 for a job that must finish before a deadline. Settings (job length, deadline, power draw) are at the top of the file or can be passed on the command line.

`schedule` builds `carbonwise.csv` from the saved forecasts, using only forecasts that already exist at the chosen "now": the 24h model for hours up to now+24 and the 48h model for the hours after that.

`backtest` replays one job per day through 2026. Each start time is chosen from the forecast and then scored on what the grid actually did. It compares running immediately, scheduling on a persistence forecast, scheduling on the XGBoost forecast, and perfect foresight.

## Scheduling results (2026 backtest)

One flexible job per day was submitted at 12:00 UTC (2-hour job, 0.5 kW, 24-hour deadline) over 303 days.
Each start time was chosen from the forecast only and scored on the grid's actual carbon intensity.

| Strategy | Total gCO2 | Saved vs run-now |
|---|---|---|
| Run immediately | 28,554 | 0.0% |
| Schedule on "same hour yesterday" | 28,591 | -0.1% |
| Schedule on XGBoost forecast | 23,198 | 18.8% |
| Perfect foresight (upper limit) | 21,289 | 25.4% |

The forecast captures 74% of the saving that perfect foresight would give. It beat running immediately on 49% of days and was worse on 18%.

## Limitations

- Great Britain only. Times are UTC.
- The scheduler runs on stored history. A live pipeline (latest carbon intensity plus the current weather forecast, then predict the next 48 hours) is not built yet.
- Forecast accuracy is limited mostly by weather forecast error, especially wind. Extra weather variables (100 m wind speed, direct and diffuse solar radiation) are the most likely next improvement.
- Results come from one test year, so treat the monthly figures as indicative.
 
 
 