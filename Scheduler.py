import csv
import os

# ---- Settings (change these) ----
JOB_HOURS = 2        # how long the job runs
DEADLINE_HOURS = 24  # job must FINISH within this many hours from now
POWER_KW = 0.5       # power use of the job (video processing default)
# carbonwise.csv is read from the same folder as this script
CSV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "carbonwise.csv")


# ---- 1. Read the forecast (row 0 = "now") ----
def read_forecast(path=None):
    times, vals = [], []
    with open(path or CSV_FILE) as f:
        for row in csv.DictReader(f):
            times.append(row["timestamp"])
            vals.append(float(row["carbon_intensity_gco2_per_kwh"]))
    return times, vals


# ---- 2. CO2 of a job starting at hour 'start' ----
def co2(vals, start, job_hours=None, power_kw=None):
    job_hours = JOB_HOURS if job_hours is None else job_hours
    power_kw = POWER_KW if power_kw is None else power_kw
    return sum(vals[start:start + job_hours]) * power_kw  # grams


# ---- 3. Pick the start time (job must finish before the deadline) ----
def best_start(vals, job_hours=None, deadline_hours=None, power_kw=None):
    """Returns (start_index, urgent). urgent=True means no spare time, run now."""
    job_hours = JOB_HOURS if job_hours is None else job_hours
    deadline_hours = DEADLINE_HOURS if deadline_hours is None else deadline_hours
    latest_start = min(deadline_hours, len(vals)) - job_hours
    if latest_start <= 0:
        return 0, True
    best = min(range(latest_start + 1), key=lambda s: co2(vals, s, job_hours, power_kw))
    return best, False


# ---- 4. Compare and report ----
def main(path=None):
    times, vals = read_forecast(path)
    best, urgent = best_start(vals)
    if urgent:
        print("Urgent job: running immediately")

    now_co2 = co2(vals, 0)
    best_co2 = co2(vals, best)
    saved = now_co2 - best_co2
    pct = saved / now_co2 * 100 if now_co2 else 0.0

    print("Run now      :", times[0], "->", round(now_co2), "g CO2")
    print("Best start   :", times[best], "->", round(best_co2), "g CO2")
    print("Saved        :", round(saved), "g (", round(pct, 1), "% )")


if __name__ == "__main__":
    main()
