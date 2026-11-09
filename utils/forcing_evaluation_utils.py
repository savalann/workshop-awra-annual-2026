"""Functions for 02.forcing_evaluation.ipynb (hourly IMERG and AORC vs rain gauges).

The notebook keeps the settings; every step here gets them as arguments.
Only numpy, pandas, geopandas, shapely and matplotlib are needed.
"""

import importlib
import json
import sys

try:
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
except ImportError:  # check_packages() (Step 0) tells you what to install
    pass


# ---------------------------------------------------------------- Step 0

def check_packages():
    """Check that every package this notebook needs is installed."""
    print(f"Python {sys.version.split()[0]}")
    if sys.version_info < (3, 9):
        print("  -> this notebook needs Python 3.9 or newer")

    packages = {   # name you import : name you pip-install
        "numpy": "numpy",            # numbers and arrays
        "pandas": "pandas",          # tables and time series
        "geopandas": "geopandas",    # maps and shapes (basin outline, gauge locations)
        "shapely": "shapely",        # geometry, used by geopandas
        "matplotlib": "matplotlib",  # figures
    }

    missing = []
    for module, pip_name in packages.items():
        try:
            version = getattr(importlib.import_module(module), "__version__", "")
            print(f"  ok       {module} {version}")
        except Exception as error:
            print(f"  MISSING  {module}   ({type(error).__name__}: {error})")
            missing.append(pip_name)

    if missing:
        raise SystemExit("Some packages are missing. Install them, then run this cell again:\n"
                         f"    pip install {' '.join(missing)}\n"
                         f"or, with conda:  conda install -c conda-forge {' '.join(missing)}")
    print("\nAll packages are installed.")


# ---------------------------------------------------------------- helpers

def independent(product, network, networks_used_by):
    """True if the product cannot already contain gauges of this network (see NETWORKS_USED_BY)."""
    return network not in networks_used_by[product]


def kge_parts(gauge, product):
    """Correlation, bias (product mean / gauge mean), variability (product spread / gauge spread) and KGE."""
    r = np.corrcoef(gauge, product)[0, 1]
    bias = product.mean() / gauge.mean()
    variability = product.std() / gauge.std()
    return r, bias, variability, 1 - np.sqrt((r - 1) ** 2 + (bias - 1) ** 2 + (variability - 1) ** 2)


# ---------------------------------------------------------------- Step 2

def load_data(data_dir, start, end):
    """Load the ready-made data (hourly gauges only), keep the chosen period and print what each gauge has."""
    info = json.load(open(data_dir / "info.json"))
    gauges = pd.read_csv(data_dir / "sipsey_fork_gauges.csv", dtype={"id": str})
    gauges = gauges[gauges["step"] == "hourly"].reset_index(drop=True)
    hourly = pd.read_csv(data_dir / "sipsey_fork_hourly.csv", index_col="time", parse_dates=["time"])

    # keep only the period chosen in the settings
    hourly = hourly.loc[pd.Timestamp(start) + pd.Timedelta(hours=1):pd.Timestamp(end) + pd.Timedelta(days=1)]

    # how much data does each gauge have, and when?
    for g in gauges.itertuples():
        series = hourly[f"gauge_{g.id}"]
        years = series.notna().sum() / 8766
        when = f"{series.first_valid_index():%Y-%m} to {series.last_valid_index():%Y-%m}" if years else "no data"
        print(f"{g.network:5s} {g.id:16s} {g.km_outside_basin:4.1f} km outside basin   {years:4.1f} years of data ({when})")
    print(f"\n{len(hourly):,} hours from {hourly.index[0]:%Y-%m-%d} to {hourly.index[-1]:%Y-%m-%d}")
    return info, gauges, hourly


# ---------------------------------------------------------------- Step 3

def plot_map(data_dir, info, gauges, basin_id, basin_name):
    """Map of the basin, its streams, the search buffer and the hourly gauges."""
    basin = gpd.read_file(data_dir / "basin.geojson")
    states = gpd.read_file(f"zip://{(data_dir / 'states.zip').resolve()}")

    fig, (ax_state, ax) = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={"width_ratios": [0.6, 1]})
    states[states["STUSPS"] == info["state"]].boundary.plot(ax=ax_state, color="grey", linewidth=0.8)
    basin.plot(ax=ax_state, color="#1b6ca8")                       # where is the basin?
    ax_state.set_title("Alabama", fontsize=9)
    ax_state.set_axis_off()

    basin.to_crs(5070).buffer(info["buffer_km"] * 1000).to_crs(4326).boundary.plot(ax=ax, color="grey", linewidth=0.7,
                                                                                   linestyle="--")
    basin.plot(ax=ax, color="#eaf0f5", edgecolor="black", linewidth=0.8)
    gpd.read_file(data_dir / "flowlines.geojson").plot(ax=ax, color="#5b8db8", linewidth=0.6)
    for g in gauges.itertuples():
        ax.scatter(g.lon, g.lat, marker="^", s=110, color="#c0622b", edgecolor="black", zorder=3)   # every gauge: a triangle
        ax.annotate(f"{g.network} {g.id}", (g.lon, g.lat), xytext=(5, 3), textcoords="offset points", fontsize=7)
    ax.set_title(f"{basin_id} {basin_name}\n(dashed: {info['buffer_km']:.1f} km search buffer)", fontsize=9)
    ax.tick_params(labelsize=7)
    fig.legend(handles=[plt.Line2D([], [], marker="^", markersize=9, linestyle="", color="#c0622b",
                                   markeredgecolor="black", label="rain gauge")],
               loc="lower center", fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    plt.show()


# ---------------------------------------------------------------- Step 4

def quality_control(gauges, hourly, reference, max_rain_in_one_hour_mm, blocked_below_percent, clock_check_hours,
                    min_years_of_data):
    """The four gauge checks, against the reference product (here IMERG).
    Returns the QC table, the cleaned gauges, the gauges used and the main gauge."""
    qc_rows, clean = [], {}
    for g in gauges.itertuples():
        gauge = hourly[f"gauge_{g.id}"]
        ref = hourly[f"{reference}_{g.id}"]

        # 1. impossible values
        too_large = gauge > max_rain_in_one_hour_mm
        gauge = gauge.mask(too_large)

        # 2. blocked gauge: compare monthly totals with the reference product, on hours when both have data
        #    (only months with real rain, at least 25 mm in the reference product: a dry month says nothing about the gauge)
        monthly = pd.DataFrame({"gauge": gauge, "ref": ref}).dropna().resample("MS").sum()
        caught_percent = 100 * monthly["gauge"] / monthly["ref"]
        blocked = monthly[(monthly["ref"] >= 25) & (caught_percent < blocked_below_percent)].index
        gauge = gauge.mask(gauge.index.to_period("M").to_timestamp().isin(blocked))

        # 3. clock: correlation with the reference product on rainy hours, for each shift; the best should be no shift at all
        match = {}
        for shift in range(-clock_check_hours, clock_check_hours + 1):
            pair = pd.DataFrame({"gauge": gauge, "ref": ref.shift(shift)}).dropna()
            rainy = pair[(pair["gauge"] > 0.1) | (pair["ref"] > 0.1)]
            match[shift] = rainy["gauge"].corr(rainy["ref"])
        best_shift = max(match, key=lambda k: match[k])

        # 4. enough data
        years = gauge.notna().sum() / 8766

        clean[g.id] = gauge
        qc_rows.append({"network": g.network, "id": g.id, "hours_too_large": int(too_large.sum()),
                        "blocked_months": " ".join(f"{m:%Y-%m}" for m in blocked) or "none",
                        "best_clock_shift_h": best_shift, "years_of_data": round(years, 1),
                        "used": best_shift == 0 and years >= min_years_of_data})

    qc = pd.DataFrame(qc_rows)
    selected = qc[qc["used"]].sort_values("years_of_data", ascending=False)
    main = selected.iloc[0]
    print(f"Gauges used: {', '.join(selected['network'] + ' ' + selected['id'])}.  Main gauge: {main.network} {main.id}")
    return qc, clean, selected, main


# ---------------------------------------------------------------- Step 5

def scores(gauge, product, wet_hour_mm, heavy_hour_mm):
    """All hourly scores of one product against one gauge."""
    hit = ((gauge >= wet_hour_mm) & (product >= wet_hour_mm)).sum()
    miss = ((gauge >= wet_hour_mm) & (product < wet_hour_mm)).sum()
    false_alarm = ((gauge < wet_hour_mm) & (product >= wet_hour_mm)).sum()
    r, bias, variability, kge = kge_parts(gauge, product)
    heavy = gauge >= heavy_hour_mm
    return {"hours": len(gauge), "gauge_total_mm": gauge.sum(), "ratio": product.sum() / gauge.sum(), "r": r, "KGE": kge,
            "POD": hit / (hit + miss), "FAR": false_alarm / (hit + false_alarm), "CSI": hit / (hit + miss + false_alarm),
            "heavy_hours": int(heavy.sum()),
            "heavy_ratio": product[heavy].sum() / gauge[heavy].sum() if heavy.any() else np.nan}


def hourly_scores(selected, clean, hourly, products, label, networks_used_by, wet_hour_mm, heavy_hour_mm):
    """Hourly scores at every gauge that passed QC. Returns the score table and, per gauge, the hours all three have."""
    rows, hourly_at = [], {}
    for g in selected.itertuples():
        both = pd.DataFrame({"gauge": clean[g.id], **{p: hourly[f"{p}_{g.id}"] for p in products}}).dropna()
        hourly_at[g.id] = both                                   # hours where the gauge and both products have data
        for p in products:
            rows.append({"network": g.network, "gauge": g.id, "product": label[p],
                         "independent": independent(p, g.network, networks_used_by),
                         **scores(both["gauge"], both[p], wet_hour_mm, heavy_hour_mm)})
    return pd.DataFrame(rows), hourly_at


# ---------------------------------------------------------------- Step 6: figures

def plot_cumulative_rainfall(hourly_at, main, products, label, colour, networks_used_by, basin_id, basin_name):
    """Figure 1: cumulative rainfall at the main gauge."""
    both = hourly_at[main.id]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(both.index, both["gauge"].cumsum(), color=colour["gauge"], linewidth=1.8,
            label=f"{main.network} gauge {main.id}")
    for p in products:
        ok = independent(p, main.network, networks_used_by)
        ax.plot(both.index, both[p].cumsum(), color=colour[p], linewidth=1.3, linestyle="-" if ok else "--",
                label=f"{label[p]} ({'independent' if ok else 'not independent'} test)")
    ax.set_ylabel("cumulative rainfall (mm)")
    ax.set_title(f"{basin_name}: {main.network} gauge {main.id} ({len(both):,} hours with data from all three)",
                 fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(color="#e5e5e5", linewidth=0.6)
    fig.tight_layout()
    plt.show()


def plot_hourly_scatter(hourly_at, main, score_table, products, label, colour, basin_id):
    """Figure 2: hourly scatter at the main gauge (rainy hours only; the many dry hours would all sit at 0, 0)."""
    both = hourly_at[main.id]
    rainy = both[(both[["gauge"] + products] > 0.1).any(axis=1)]
    top = rainy.max().max() * 1.05
    fig, axes = plt.subplots(1, len(products), figsize=(4.6 * len(products), 4.4))
    for ax, p in zip(axes, products):
        ax.scatter(rainy["gauge"], rainy[p], s=5, alpha=0.3, color=colour[p])
        ax.plot([0, top], [0, top], "k--", linewidth=0.8)                          # 1:1 line = perfect agreement
        s = score_table[(score_table["gauge"] == main.id) & (score_table["product"] == label[p])].iloc[0]
        ax.text(0.04, 0.96, f"{'independent' if s['independent'] else 'not independent'} test\n"
                f"ratio {s['ratio']:.2f}\nr {s['r']:.2f}\nKGE {s['KGE']:.2f}", transform=ax.transAxes, va="top",
                fontsize=8)
        ax.set(xlim=(0, top), ylim=(0, top), xlabel=f"{main.network} gauge {main.id}, hourly (mm)",
               ylabel=f"{label[p]}, hourly (mm)")
    fig.tight_layout()
    plt.show()


def plot_kge_by_gauge(selected, score_table, products, label, colour, networks_used_by, basin_id):
    """Figure 3: hourly KGE of each product at each gauge, as horizontal bars (hollow = not independent)."""
    import numpy as np
    gauges = list(selected.itertuples())
    y = np.arange(len(gauges))
    height = 0.8 / len(products)
    fig, ax = plt.subplots(figsize=(8, 1.2 + 1.2 * len(gauges)))
    for k, p in enumerate(products):
        for i, g in enumerate(gauges):
            kge = score_table[(score_table["gauge"] == g.id) & (score_table["product"] == label[p])]["KGE"]
            if kge.empty:
                continue
            value = kge.iloc[0]
            ok = independent(p, g.network, networks_used_by)
            ypos = y[i] + (k - (len(products) - 1) / 2) * height
            ax.barh(ypos, value, height * 0.9, color=colour[p] if ok else "white",
                    edgecolor=colour[p], linewidth=0 if ok else 2, hatch=None if ok else "//")
            # product name next to its value on the first gauge (instead of a legend)
            text = f"{label[p]}  {value:.2f}" if i == 0 else f"{value:.2f}"
            ax.text(value + 0.01, ypos, text, va="center", ha="left", fontsize=9, color=colour[p])
    ax.set_yticks(y, [f"{g.network} {g.id}" for g in gauges], fontsize=8)
    ax.set_ylim(len(gauges) - 0.5, -0.5)
    ax.set_xlim(min(0, ax.get_xlim()[0]), 1.15)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="in")
    ax.set_xlabel("hourly KGE")
    hollow = plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="grey", hatch="//", linewidth=2,
                           label="not independent (product may contain this gauge)")
    ax.legend(handles=[hollow], fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    fig.tight_layout()
    plt.show()


def plot_wettest_storm(hourly_at, main, products, label, colour, basin_id, day=None):
    """Figure 4: the wettest day at the main gauge (or the day you give, e.g. pd.Timestamp("2020-04-12")), hour by hour."""
    both = hourly_at[main.id]
    if day is None:
        day_totals = both["gauge"].groupby((both.index - pd.Timedelta(hours=1)).floor("D")).sum()
        day = day_totals.idxmax()
    window = both.loc[day - pd.Timedelta(hours=23):day + pd.Timedelta(days=2)]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(window.index, window["gauge"], color=colour["gauge"], linewidth=2.5, linestyle="--",
            label=f"{main.network} gauge {main.id} ({window['gauge'].sum():.0f} mm)")
    for p in products:
        ax.plot(window.index, window[p], color=colour[p], linewidth=1.5, label=f"{label[p]} ({window[p].sum():.0f} mm)")
    ax.set_ylabel("rain in the hour (mm)")
    ax.set_title(f"Wettest day at {main.network} {main.id}: {day:%d %b %Y} (UTC; totals in the legend)", fontsize=10)
    ax.grid(color="#e5e5e5", linewidth=0.6)
    ax.legend(frameon=False, fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    plt.show()


def plot_kge_per_year(hourly_at, main, products, label, colour, basin_id):
    """Figure 5: hourly KGE in each year at the main gauge (a year needs at least half a year of hours)."""
    both = hourly_at[main.id]
    year_of = (both.index - pd.Timedelta(hours=1)).year
    yearly_kge = pd.DataFrame({label[p]: both.groupby(year_of).apply(
        lambda d: kge_parts(d["gauge"], d[p])[3] if len(d) >= 4383 else np.nan) for p in products})
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for p in products:
        ax.plot(yearly_kge.index, yearly_kge[label[p]], marker="o", color=colour[p], linewidth=1.5, label=label[p])
    ax.set_ylabel("hourly KGE")
    ax.set_title(f"{main.network} gauge {main.id}: hourly KGE per year", fontsize=10)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(color="#e5e5e5", linewidth=0.6)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    plt.show()
    return yearly_kge
