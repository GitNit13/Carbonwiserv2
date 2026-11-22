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