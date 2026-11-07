"""Metrics and drought events for USGS observations, NWM and NextGen runs (daily flow).

Input: one parquet file with the columns date, Observed, NWM and one column per
       NextGen run (m^3/s), and drought_thresholds.parquet (daily thresholds
       Q95, Q90, Q80, Q70).

Every column other than Observed is treated as a model and compared with Observed,
so a second NextGen run only needs a new column in the parquet file.

A drought day is a day when the daily flow is below the daily threshold.

Steps for each series:
1. Find runs of drought days.
2. Merge runs separated by a gap of at most `gap_days` days.
3. Keep events lasting at least `persistence_days` days.

Duration = number of days from event start to end (including merged gaps).
Severity = mean percentage below the threshold on the drought days of the event:
           mean of (threshold - flow) / threshold * 100.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_FOLDER = Path(__file__).resolve().parent.parent / "data"   # data/ next to the notebooks
THRESHOLDS_PARQUET = DATA_FOLDER / "drought_thresholds.parquet"
DAILY_FLOWS_PARQUET = DATA_FOLDER / "daily_flows.parquet"

OBSERVED = "Observed"
METRICS = ("KGE", "NSE", "NNSE", "logNSE", "RMSE", "PBIAS")

# Best possible value of each metric (shown as a dashed line on the bar charts)
METRIC_BEST = {"KGE": 1, "NSE": 1, "NNSE": 1, "logNSE": 1, "RMSE": 0, "PBIAS": 0}
METRIC_LABELS = {"RMSE": "RMSE (m³/s)", "PBIAS": "PBIAS (%)"}

# One fixed color per model, in column order (NWM, NextGen run 1, NextGen run 2, ...)
MODEL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_daily_flows(path=DAILY_FLOWS_PARQUET, models=None):
    """Read the daily flows (m^3/s), indexed by date, with Observed as the first column.

    models: the model columns to keep, in the order you want them (default: all of them).
    A run that ends earlier than the others has no value (NaN) after its last day.
    """
    flows = pd.read_parquet(path)
    flows["date"] = pd.to_datetime(flows["date"])
    flows = flows.set_index("date").sort_index()
    return flows[[OBSERVED] + (list(models) if models else model_names(flows))]


def model_names(flows):
    """All model columns (everything except Observed), e.g. NWM and the NextGen runs."""
    return [name for name in flows.columns if name != OBSERVED]


def load_station_thresholds(site_no, thresholds_parquet=THRESHOLDS_PARQUET):
    """Read the daily thresholds of one station from the parquet file."""
    table = pd.read_parquet(thresholds_parquet)
    table = table[table["site_no"] == str(site_no)]
    if table.empty:
        raise ValueError(f"No thresholds for station {site_no} in {thresholds_parquet}.")
    return table.reset_index(drop=True)


def daily_threshold_series(dates, thresholds, q):
    """Threshold value for each date, matched on calendar day (MM-DD)."""
    lookup = thresholds.set_index("month_day")[f"Q{q}"]
    keys = pd.DatetimeIndex(dates).strftime("%m-%d")
    return pd.Series(lookup.reindex(keys).to_numpy(), index=dates)


# ---------------------------------------------------------------------------
# Metrics (each model vs Observed)
# ---------------------------------------------------------------------------
def calculate_metrics(obs, sim):
    """KGE, NSE, NNSE, logNSE, RMSE (m^3/s) and PBIAS (%) of sim against obs.

    PBIAS = 100 * sum(sim - obs) / sum(obs); positive = the model is too high.
    """
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    keep = ~(np.isnan(obs) | np.isnan(sim))
    obs, sim = obs[keep], sim[keep]
    if len(obs) < 2:
        return {k: np.nan for k in METRICS}

    def nse(o, s):
        denom = np.sum((o - o.mean()) ** 2)
        return np.nan if denom == 0 else 1 - np.sum((s - o) ** 2) / denom

    # KGE (Gupta et al., 2009): correlation, variability ratio, bias ratio
    r = np.corrcoef(obs, sim)[0, 1] if obs.std() > 0 and sim.std() > 0 else np.nan
    alpha = sim.std() / obs.std() if obs.std() > 0 else np.nan
    beta = sim.mean() / obs.mean()
    kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

    # Small constant so log() is defined on zero-flow days
    eps = 0.01 * obs.mean()
    nse_value = nse(obs, sim)
    return {
        "KGE": kge,
        "NSE": nse_value,
        "NNSE": 1 / (2 - nse_value),
        "logNSE": nse(np.log(obs + eps), np.log(sim + eps)),
        "RMSE": float(np.sqrt(np.mean((sim - obs) ** 2))),
        "PBIAS": float(100 * np.sum(sim - obs) / np.sum(obs)),
    }


def metrics_table(flows):
    """Metrics for all days: one column per model, each vs Observed."""
    return pd.DataFrame({
        f"{name} vs Observed": calculate_metrics(flows[OBSERVED], flows[name])
        for name in model_names(flows)
    })


def period_metrics(flows, start, end):
    """Metrics of every model against Observed between two dates (inclusive)."""
    window = flows.loc[pd.Timestamp(start):pd.Timestamp(end)]
    if window.empty:
        raise ValueError(f"No daily flows between {start} and {end}.")
    print(f"{pd.Timestamp(start).date()} to {pd.Timestamp(end).date()}: {len(window)} days")
    return metrics_table(window)


def plot_metrics_bars(metrics, title="Model performance against USGS observations"):
    """Horizontal bar chart of a metrics table: one panel per metric, one bar per model."""
    models = [col.replace(" vs Observed", "") for col in metrics.columns]
    colors = MODEL_COLORS[:len(models)]
    y = np.arange(len(models))

    fig, axes = plt.subplots(2, 3, figsize=(14, 2.2 + 1.1 * len(models)))
    for ax, metric in zip(axes.flat, METRICS):
        values = metrics.loc[metric].to_numpy(dtype=float)
        ax.barh(y, values, color=colors, height=0.6, edgecolor="white", linewidth=2)
        ax.axvline(0, color="#888888", linewidth=0.8)
        ax.axvline(METRIC_BEST[metric], color="#444444", linewidth=1, linestyle="--")
        ax.set_yticks(y)
        ax.set_yticklabels(models)
        ax.invert_yaxis()
        ax.set_title(f"{METRIC_LABELS.get(metric, metric)}  (best = {METRIC_BEST[metric]})", fontsize=11)
        ax.grid(axis="x", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

        # Leave room on both sides so the value labels fit
        low, high = min(0, np.nanmin(values), METRIC_BEST[metric]), max(0, np.nanmax(values), METRIC_BEST[metric])
        pad = 0.25 * (high - low or 1)
        ax.set_xlim(low - (pad if low < 0 else 0), high + pad)
        for yi, v in zip(y, values):
            if np.isfinite(v):
                ax.text(v, yi, f" {v:.2f} ", va="center", ha="left" if v >= 0 else "right", fontsize=9, color="#222222")
            else:  # the model has no flow in this period
                ax.text(0, yi, " no data", va="center", ha="left", fontsize=9, color="#888888")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    plt.show()


def plot_period_series(flows, start, end, title="Daily streamflow"):
    """Daily flow of Observed and every model between two dates (inclusive)."""
    window = flows.loc[pd.Timestamp(start):pd.Timestamp(end)]
    if window.empty:
        raise ValueError(f"No daily flows between {start} and {end}.")

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(window.index, window[OBSERVED], color="#222222", linewidth=2,
            marker="o", markersize=4, label="USGS observed")
    for name, color in zip(model_names(window), MODEL_COLORS):
        label = name if window[name].notna().any() else f"{name} (no data in this period)"
        ax.plot(window.index, window[name], color=color, linewidth=2, label=label)

    ax.set_ylabel("Streamflow (m³/s)")
    ax.set_title(f"{title} ({pd.Timestamp(start).date()} to {pd.Timestamp(end).date()})")
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.autofmt_xdate()
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Drought events
# ---------------------------------------------------------------------------
def find_drought_events(flow, threshold, persistence_days=7, gap_days=3):
    """Return a table of drought events for one daily series."""
    flow = flow.dropna()
    threshold = threshold.reindex(flow.index)
    below = (flow < threshold).to_numpy()
    deficit_pct = ((threshold - flow) / threshold * 100).to_numpy()
    dates = flow.index

    # 1. Runs of consecutive drought days (a missing day breaks a run)
    runs = []
    start = None
    for i, is_below in enumerate(below):
        if is_below and start is None:
            start = i
        elif start is not None and (not is_below or (dates[i] - dates[i - 1]).days > 1):
            runs.append([start, i - 1])
            start = i if is_below else None
    if start is not None:
        runs.append([start, len(below) - 1])

    # 2. Merge runs whose gap between them is at most gap_days
    merged = []
    for run in runs:
        if merged and (dates[run[0]] - dates[merged[-1][1]]).days - 1 <= gap_days:
            merged[-1][1] = run[1]
        else:
            merged.append(run)

    # 3. Keep events at least persistence_days long
    events = []
    for s, e in merged:
        duration = (dates[e] - dates[s]).days + 1
        if duration < persistence_days:
            continue
        events.append({
            "start": dates[s],
            "end": dates[e],
            "duration_days": duration,
            # Mean % below threshold over the drought days (merged gap days excluded)
            "severity_pct": float(deficit_pct[s:e + 1][below[s:e + 1]].mean()),
            "min_flow_cms": float(flow.iloc[s:e + 1].min()),
        })
    return pd.DataFrame(events, columns=["start", "end", "duration_days", "severity_pct", "min_flow_cms"])


def summarize_events(events):
    """Number of events and min / median / max of duration and severity."""
    if events.empty:
        return {"n_events": 0,
                **{f"duration_{s}_days": np.nan for s in ("min", "median", "max")},
                **{f"severity_{s}_pct": np.nan for s in ("min", "median", "max")}}
    return {
        "n_events": len(events),
        "duration_min_days": events["duration_days"].min(),
        "duration_median_days": events["duration_days"].median(),
        "duration_max_days": events["duration_days"].max(),
        "severity_min_pct": events["severity_pct"].min(),
        "severity_median_pct": events["severity_pct"].median(),
        "severity_max_pct": events["severity_pct"].max(),
    }


def run_drought_comparison(flows, site_no, q=80, persistence_days=7, gap_days=3,
                           thresholds_parquet=THRESHOLDS_PARQUET):
    """Find drought events in Observed and every model and compare them.

    Returns (comparison table, dict of event tables, threshold series).
    """
    if q not in (95, 90, 80, 70):
        raise ValueError("q must be one of 95, 90, 80, 70")

    thresholds = load_station_thresholds(site_no, thresholds_parquet)
    threshold = daily_threshold_series(flows.index, thresholds, q)

    events = {name: find_drought_events(flows[name], threshold, persistence_days, gap_days)
              for name in flows.columns}

    summary = pd.DataFrame({name: summarize_events(ev_df) for name, ev_df in events.items()})
    for name in model_names(flows):
        summary[f"{name} - Observed"] = summary[name] - summary[OBSERVED]

    print(f"Station {site_no} | threshold Q{q} | persistence {persistence_days} d | "
          f"gap {gap_days} d | {flows.index.min().date()} to {flows.index.max().date()} ({len(flows)} days)")
    return summary, events, threshold


# ---------------------------------------------------------------------------
# Metrics for the most severe / longest observed drought events
# ---------------------------------------------------------------------------
def key_observed_events(observed_events):
    """The observed events with the largest severity and the longest duration."""
    if observed_events.empty:
        return {}
    return {
        "max_severity": observed_events.loc[observed_events["severity_pct"].idxmax()],
        "max_duration": observed_events.loc[observed_events["duration_days"].idxmax()],
    }


def event_metrics(flows, event):
    """Metrics of every model against Observed during one event (start to end)."""
    window = flows.loc[pd.Timestamp(event["start"]):pd.Timestamp(event["end"])]
    return metrics_table(window)
