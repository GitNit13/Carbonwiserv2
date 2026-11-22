"""
Carbonwiser GUI: live GB carbon intensity forecast + "when should I run my job?" recommendation.

Run:
    pip install streamlit
    streamlit run app.py

Tabs
    Live forecast   - downloads the latest data and the real weather forecast, predicts the next 48 hours,
                      shows the greenest start time for your job
    Backtest        - replays 2026 and shows how much CO2 forecast-based scheduling really saves
    Model accuracy  - test-set accuracy of the 24h and 48h models (from the saved prediction files)
    Live accuracy   - scores the live forecasts you made earlier against what the grid actually did
    Setup           - what is installed / trained, and buttons to train and refit the models
"""
import contextlib
import io
import os
import sys

try:                                   # headless backend: the GUI runs plotting code in a worker thread
    import matplotlib
    matplotlib.use("Agg")
except ImportError:
    pass

import numpy as np
import pandas as pd
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import CarbonwiserXGBoost as cw
import Scheduler as S

st.set_page_config(page_title="Carbonwiser", page_icon="🌱", layout="wide")

TZ_CHOICES = {
    "UTC": "UTC",
    "Great Britain local time": "Europe/London",
    "India (IST)": "Asia/Kolkata",
}


# ---------------------------------------------------------------- helpers
def run_quietly(fn, *args, **kwargs):
    """Run fn and capture what it prints. Returns (result, printed text, error message or None)."""
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            result = fn(*args, **kwargs)
        return result, buf.getvalue(), None
    except SystemExit as e:                       # the script reports problems with SystemExit("message")
        return None, buf.getvalue(), str(e.code) if e.code else "The command stopped."
    except Exception as e:
        return None, buf.getvalue(), f"{type(e).__name__}: {e}"


def fmt(ts, tz):
    return ts.tz_convert(tz).strftime("%a %d %b, %H:%M")


def status():
    def have(prefix):
        return all(os.path.exists(os.path.join(cw.MODEL_DIR, f"{prefix}model_info_{h}h.json")) for h in cw.HORIZONS)
    return {
        "Training data (gb_carbon_weather_cleaned.csv)": os.path.exists(cw.TRAIN_CSV),
        "Prediction files (used by Backtest and Model accuracy)": os.path.exists(cw.P24) and os.path.exists(cw.P48),
        "Trained models": have(""),
        "Models refit on all data (used by Live forecast)": have("final_"),
        "Live forecast log (used by Live accuracy)": os.path.exists(cw.LIVE_LOG),
    }


def forecast_chart(f, start, job_hours, tz, show_neso):
    """Forecast line with the recommended window (green) and the run-now window (red)."""
    times = f.index.tz_convert(tz).tz_localize(None)
    d = pd.DataFrame({"time": times, "XGBoost forecast": f["forecast"].values})
    if show_neso and f["neso_forecast"].notna().any():
        d["NESO forecast (comparison)"] = f["neso_forecast"].values
    try:
        import altair as alt
        long = d.melt("time", var_name="series", value_name="value")
        line = alt.Chart(long).mark_line().encode(
            x=alt.X("time:T", title=None),
            y=alt.Y("value:Q", title="gCO₂/kWh", scale=alt.Scale(zero=False)),
            color=alt.Color("series:N", title=None))
        step = pd.Timedelta(hours=job_hours)
        best = pd.DataFrame({"a": [times[start]], "b": [times[start] + step]})
        now = pd.DataFrame({"a": [times[0]], "b": [times[0] + step]})
        band_best = alt.Chart(best).mark_rect(opacity=0.25, color="green").encode(x="a:T", x2="b:T")
        band_now = alt.Chart(now).mark_rect(opacity=0.25, color="red").encode(x="a:T", x2="b:T")
        return ("altair", (band_now + band_best + line).properties(height=330))
    except Exception:
        return ("basic", d.set_index("time"))


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🌱 Carbonwiser")
    st.caption("Great Britain grid carbon intensity: forecast and green scheduling")

    st.header("Your job")
    job_hours = int(st.number_input("Job length (hours)", min_value=1, max_value=24, value=int(S.JOB_HOURS), step=1))
    deadline = int(st.number_input("Must finish within (hours from now)", min_value=job_hours, max_value=48,
                                   value=max(int(S.DEADLINE_HOURS), job_hours), step=1))
    power = float(st.number_input("Power draw (kW)", min_value=0.01, max_value=1000.0,
                                  value=float(S.POWER_KW), step=0.1))
    tz_label = st.selectbox("Show times in", list(TZ_CHOICES))
    tz = TZ_CHOICES[tz_label]

    st.header("Status")
    for name, ok in status().items():
        st.write(("✅ " if ok else "❌ ") + name)

st.title("Carbonwiser")

tab_live, tab_back, tab_acc, tab_check, tab_setup = st.tabs(
    ["Live forecast", "Backtest", "Model accuracy", "Live accuracy", "Setup"])


# ---------------------------------------------------------------- live forecast
with tab_live:
    st.subheader("Next 48 hours, forecast from now")
    st.caption("Uses the latest carbon data and the real weather forecast for Great Britain. "
               f"Times are shown in: {tz_label}.")

    if st.button("Get live forecast", type="primary"):
        with st.spinner("Downloading data and predicting..."):
            res, printed, err = run_quietly(cw.live_forecast)
        if err:
            st.session_state.pop("live", None)
            st.error(err)
        else:
            st.session_state["live"] = {"f": res[0], "now": res[1], "log": printed}

    live = st.session_state.get("live")
    if not live:
        st.info("Click **Get live forecast**. You need trained models first (see the Setup tab).")
    else:
        f, now = live["f"], live["now"]
        vals = f["forecast"].tolist()
        start, urgent = S.best_start(vals, job_hours, deadline, power)
        now_co2 = S.co2(vals, 0, job_hours, power)
        best_co2 = S.co2(vals, start, job_hours, power)
        saved = now_co2 - best_co2
        pct = saved / now_co2 * 100 if now_co2 else 0.0

        st.caption(f"Forecast made at {fmt(now, tz)}. Settings: {job_hours} h job, deadline {deadline} h, {power} kW.")
        if urgent:
            st.warning("There is no spare time before the deadline, so the job has to start now.")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Run now", f"{now_co2:.0f} g CO₂")
        c2.metric("Best start", fmt(f.index[start], tz), help=f"{start} hour(s) from now")
        c3.metric("CO₂ at best start", f"{best_co2:.0f} g",
                  delta=f"-{saved:.0f} g" if saved > 0 else "0 g", delta_color="inverse")
        c4.metric("Saved", f"{pct:.1f} %")

        show_neso = st.checkbox("Show NESO's own forecast for comparison", value=True)
        kind, chart = forecast_chart(f, start, job_hours, tz, show_neso)
        if kind == "altair":
            st.altair_chart(chart, use_container_width=True)
            st.caption("Green band: recommended window. Red band: running now. "
                       "The 24h models cover the first 24 hours, the 48h models the next 24.")
        else:
            st.line_chart(chart)

        latest = max(0, min(deadline, len(vals)) - job_hours)
        cands = pd.DataFrame({
            "Start time": [fmt(f.index[i], tz) for i in range(latest + 1)],
            "CO₂ (g)": [round(S.co2(vals, i, job_hours, power)) for i in range(latest + 1)],
        }).sort_values("CO₂ (g)").head(5)
        st.markdown("**Best 5 start times**")
        st.dataframe(cands, hide_index=True)

        with st.expander("All 48 hourly forecasts"):
            table = f.copy().round(0)
            table.index = table.index.tz_convert(tz).strftime("%a %d %b %H:%M")
            table.index.name = "time"
            st.dataframe(table)
        d1, d2 = st.columns(2)
        d1.download_button("Download forecast (CSV)", data=f.round(1).to_csv(), file_name="live_forecast.csv")
        sched = pd.DataFrame({"timestamp": f.index.strftime("%Y-%m-%d %H:%M"),
                              "carbon_intensity_gco2_per_kwh": f["forecast"].round(1).values})
        d2.download_button("Download for Scheduler.py (carbonwise.csv)", data=sched.to_csv(index=False),
                           file_name="carbonwise.csv")
        if live["log"].strip():
            with st.expander("Details from the run"):
                st.code(live["log"])


# ---------------------------------------------------------------- backtest
with tab_back:
    st.subheader("Does forecast-based scheduling really save CO₂?")
    st.caption("Replays 2026, one job per day. Each start time is chosen from the forecast only, then scored on "
               "what the grid actually did. Uses the job settings in the sidebar.")
    hour = st.slider("Daily submission time (UTC)", 0, 23, 12)

    if st.button("Run backtest"):
        with st.spinner("Replaying 2026..."):
            res, printed, err = run_quietly(cw.backtest_run, hour, job_hours, min(deadline, 48), power)
        if err:
            st.session_state.pop("bt", None)
            st.error(err)
        else:
            st.session_state["bt"] = {"df": res[0], "hour": hour, "params": res[1]}

    bt = st.session_state.get("bt")
    if bt:
        df = bt["df"]
        jh, dl, pw = bt["params"]
        st.caption(f"{len(df)} days, one job per day at {bt['hour']:02d}:00 UTC, {jh} h job, "
                   f"deadline {dl} h, {pw} kW")
        base = df["run_now"].sum()
        labels = [("Run immediately", "run_now"),
                  ("Schedule on 'same hour yesterday'", "persistence"),
                  ("Schedule on XGBoost forecast", "forecast"),
                  ("Perfect foresight (upper limit)", "oracle")]
        table = pd.DataFrame([{"Strategy": name, "Total gCO₂": round(df[k].sum()),
                               "Saved vs run-now (%)": round((base - df[k].sum()) / base * 100, 1)}
                              for name, k in labels if k in df])
        st.dataframe(table, hide_index=True)

        possible = base - df["oracle"].sum()
        share = (base - df["forecast"].sum()) / possible * 100 if possible else 0.0
        m1, m2, m3 = st.columns(3)
        m1.metric("Saved by the forecast", f"{(base - df['forecast'].sum()) / base * 100:.1f} %")
        m2.metric("Share of the best possible saving", f"{share:.0f} %")
        m3.metric("Days better / worse than running now",
                  f"{(df['forecast'] < df['run_now']).mean() * 100:.0f}% / {(df['forecast'] > df['run_now']).mean() * 100:.0f}%")

        cum = pd.DataFrame({"XGBoost forecast": (df["run_now"] - df["forecast"]).cumsum().values,
                            "Perfect foresight": (df["run_now"] - df["oracle"]).cumsum().values},
                           index=pd.to_datetime(df["date"]).dt.tz_localize(None))
        if "persistence" in df:
            cum["Same hour yesterday"] = (df["run_now"] - df["persistence"]).cumsum().values
        st.markdown("**CO₂ saved so far (grams, compared with running immediately)**")
        st.line_chart(cum)


# ---------------------------------------------------------------- model accuracy
with tab_acc:
    st.subheader("Model accuracy on the 2026 test set")
    preds, _, err = run_quietly(cw.load_predictions)
    if err:
        st.info("No prediction files yet. Train the models first (Setup tab) or run "
                "`python CarbonwiserXGBoost.py train`.")
    else:
        p24, p48 = preds
        horizon = st.radio("Horizon", ["24h", "48h"], horizontal=True)
        d = p24 if horizon == "24h" else p48
        rows = []
        for name, col in (("XGBoost", "xgb"), ("Persistence (same hour before)", "persistence")):
            e = d[col] - d["actual"]
            rows.append({"Forecast": name, "MAE": round(e.abs().mean(), 1),
                         "RMSE": round(float(np.sqrt((e ** 2).mean())), 1),
                         "MAPE (%)": round((e.abs() / d["actual"]).mean() * 100, 1)})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.caption("MAE and RMSE are in gCO₂/kWh. MAPE overstates errors in low-carbon hours.")

        by_month = pd.DataFrame({
            "XGBoost": (d["xgb"] - d["actual"]).abs().groupby(d.index.month).mean(),
            "Persistence": (d["persistence"] - d["actual"]).abs().groupby(d.index.month).mean()})
        by_month.index.name = "month"
        st.markdown("**MAE by month**")
        st.bar_chart(by_month)

        days = st.slider("Days to show", 3, 60, 14)
        window = d.iloc[-24 * days:][["actual", "xgb", "persistence"]].copy()
        window.index = window.index.tz_convert(tz).tz_localize(None)
        st.markdown(f"**Actual vs forecast, last {days} days of the test set ({horizon} ahead)**")
        st.line_chart(window)


# ---------------------------------------------------------------- live accuracy
with tab_check:
    st.subheader("How accurate were your live forecasts?")
    st.caption("Every time you press 'Get live forecast' the forecast is logged. Once those hours have passed, "
               "this compares them with what the grid actually did.")
    if st.button("Score logged forecasts"):
        with st.spinner("Downloading the real values..."):
            res, printed, err = run_quietly(cw.check_table)
        if err:
            st.session_state.pop("chk", None)
            st.warning(err)
        else:
            st.session_state["chk"] = res

    chk = st.session_state.get("chk")
    if chk:
        table, n_hours, n_runs, detail = chk
        st.caption(f"{n_hours} forecast hours scored from {n_runs} live run(s)")
        st.dataframe(table, hide_index=True)
        st.caption("Judge the XGBoost rows against Persistence. NESO's own forecast is a short-range "
                   "forecast, so it is expected to be more accurate.")
        latest = detail.sort_values("issued_at").drop_duplicates("target", keep="last").sort_values("target")
        view = latest.set_index(latest["target"].dt.tz_convert(tz).dt.tz_localize(None))[["actual", "forecast"]]
        st.line_chart(view)


# ---------------------------------------------------------------- setup
with tab_setup:
    st.subheader("Setup")
    for name, ok in status().items():
        st.write(("✅ " if ok else "❌ ") + name)
    st.markdown("**Order:** 1) train the models, 2) refit them on all data, 3) get a live forecast.")
    st.caption("The full tuned training run takes longer than the quick one. For the best accuracy run "
               "`python CarbonwiserXGBoost.py train` in a terminal.")

    c1, c2 = st.columns(2)
    setup_log = None
    if c1.button("Train models (quick, no tuning)"):
        with st.spinner("Training both horizons. This takes a few minutes..."):
            _, setup_log, err = run_quietly(cw.train_cmd, cw.TRAIN_CSV, False, 20, 3)
        st.error(err) if err else st.success("Training finished.")
    if c2.button("Refit on all data"):
        with st.spinner("Refitting both horizons..."):
            _, setup_log, err = run_quietly(cw.refit_cmd, cw.TRAIN_CSV, 5)
        st.error(err) if err else st.success("Models refit. You can now get a live forecast.")
    if setup_log:
        with st.expander("Output"):
            st.code(setup_log)
