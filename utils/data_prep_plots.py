"""Plotting functions used by the NextGen Data Preparation notebook.

Each function makes one figure and shows it in the notebook.
"""

import geopandas as gpd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from dataretrieval import nwis
from IPython.display import display
from shapely.geometry import Point

from . import ngiab_utils
from .ngen_outputs_utils import get_flow_data_from_netcdf, process_usgs_streamflow
from .forcings_utils import (
    compute_mean_annual,
    get_var_full_name,
    get_var_label,
    join_to_gpkg,
    make_mean_annual_map,
    process_time_series,
)
from .workshop_paths import DATA_DIR, WORKSHOP_DIR

# Daily Observed (USGS), NWM and NextGen flows (m^3/s), stored next to the notebooks
DAILY_FLOWS_PARQUET = WORKSHOP_DIR / "data" / "daily_flows.parquet"
HOURLY_FLOW_PARQUET = WORKSHOP_DIR / "data" / "usgs_hourly_02450250.parquet"  # USGS 15-min data averaged to hours (UTC)

# Variables whose cumulative / annual values are precipitation depths (mm)
PRECIP_VARIABLES = ["precip_rate", "apcp_surface", "precipitation", "precip"]


def _month_tick_interval(index, target_ticks=24):
    """Months between x-axis labels so that about `target_ticks` labels are shown."""
    months_span = (index.max().year - index.min().year) * 12 + (index.max().month - index.min().month) + 1
    return max(1, months_span // target_ticks)


def _format_time_axis(ax, index):
    """Month ticks, grid and rotated labels for a time-series plot."""
    ax.set_xlim(index.min(), index.max())
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=_month_tick_interval(index)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=45)
    plt.grid(True, linestyle="--", linewidth=0.7)
    plt.tight_layout()


def plot_hourly_series(forcings_path, gpkg_path, var_name):
    """Plot the hourly basin-mean time series of a forcing variable."""
    series = process_time_series(forcings_path, gpkg_path, var_name)

    plt.figure(figsize=(12, 6))
    plt.plot(series.index, series.values, linewidth=1, label=get_var_full_name(var_name))
    plt.xlabel("Date/Time", labelpad=10)
    plt.ylabel(get_var_label(var_name) or "Value", labelpad=10)
    plt.title(f"Hourly {get_var_full_name(var_name)}", pad=14)
    _format_time_axis(plt.gca(), series.index)

    plt.show()


def plot_cumulative_series(forcings_path, gpkg_path, var_name):
    """Plot the cumulative basin-mean time series of a forcing variable."""
    series = process_time_series(forcings_path, gpkg_path, var_name)
    cum_series = series.cumsum()

    plt.figure(figsize=(12, 6))
    plt.plot(cum_series.index, cum_series.values, linewidth=1.4,
             label=get_var_full_name(var_name) + " (Cumulative)")

    # Precipitation gets its own title and axis label
    if var_name.lower() in PRECIP_VARIABLES:
        plt.title("Cumulative Precipitation", pad=14)
        plt.ylabel("Cumulative Value (mm)", labelpad=10)
    else:
        plt.title(f"Cumulative {get_var_full_name(var_name)}", pad=14)
        plt.ylabel(get_var_label(var_name) or "Value", labelpad=10)
    plt.xlabel("Date/Time", labelpad=10)
    _format_time_axis(plt.gca(), cum_series.index)

    plt.show()


def plot_mean_annual_basin_map(forcings_path, gpkg_path, var_name):
    """Map the mean annual value of a forcing variable over the HydroFabric divides."""
    ds = xr.open_dataset(forcings_path)

    # Normalize catchment dimension name if needed
    if "catchment-id" in ds.dims and "catchment" not in ds.dims:
        ds = ds.rename({"catchment-id": "catchment"})

    # compute_mean_annual detects "rate" vs "state" automatically
    stats_df, meta = compute_mean_annual(ds, var_name, force_kind=None)
    print("Detected Kind:", meta.get("kind"), "| Units:", meta.get("units"), "| Rate base:", meta.get("rate_base"))

    if stats_df is None or stats_df.empty:
        raise ValueError("compute_mean_annual returned an empty stats_df — nothing to plot.")

    # Join the statistics to the HydroFabric polygons
    gdf_map = join_to_gpkg(gpkg_path, stats_df)
    if gdf_map is None or gdf_map.empty:
        raise ValueError("join_to_gpkg produced an empty GeoDataFrame — check gpkg_path and join keys.")
    if "mean_annual" not in gdf_map.columns:
        raise ValueError("Column 'mean_annual' not found in gdf_map. Columns available: "
                         + ", ".join(map(str, gdf_map.columns)))
    if gdf_map.geometry.is_empty.all():
        raise ValueError("All geometries are empty — verify HydroFabric polygons in the GPKG.")

    plt.close("all")
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    gdf_map.plot(
        column="mean_annual",
        ax=ax,
        cmap="viridis_r",
        legend=True,
        edgecolor="white",
        linewidth=0.5,
        missing_kwds={"color": "lightgrey", "label": "No data"},
    )

    # Units on the color bar
    fig.axes[-1].set_ylabel("mm/yr", fontsize=10)

    # Write the mean annual value inside each polygon (integers)
    for _, row in gdf_map.iterrows():
        geom = row.geometry
        if geom is not None and not geom.is_empty:
            pt = geom.representative_point()  # safer than centroid
            val = row.get("mean_annual", np.nan)
            if np.isfinite(val):
                ax.text(pt.x, pt.y, f"{int(round(val))}", ha="center", va="center",
                        fontsize=7, color="black", zorder=10)

    ax.set_title("Mean Annual Precipitation (mm/yr)", fontsize=14)
    ax.set_axis_off()
    fig.tight_layout()

    plt.show()


def plot_raw_gridded_map(hydrofabric_id, var_name):
    """Map the mean annual value of the raw gridded forcing over the HydroFabric catchments."""
    nc_path = f"{DATA_DIR}/{hydrofabric_id}/forcings/raw_gridded_data.nc"
    gpkg_path = f"{DATA_DIR}/{hydrofabric_id}/config/{hydrofabric_id}_subset.gpkg"

    fig, _, _ = make_mean_annual_map(nc_path, gpkg_path, var_name)

    plt.show()


def plot_streamflow_series(start_date, end_date):
    """Plot the hourly USGS observed streamflow (UTC) from usgs_hourly_02450250.parquet for a period."""
    flows = pd.read_parquet(HOURLY_FLOW_PARQUET)
    flows["date"] = pd.to_datetime(flows["date"])
    observed = flows.set_index("date").sort_index()["Observed"].loc[start_date:end_date]
    if observed.empty:
        raise ValueError(f"No streamflow in {HOURLY_FLOW_PARQUET.name} between {start_date} and {end_date}.")

    plt.figure(figsize=(12, 6))
    plt.plot(observed.index, observed.values, linewidth=1.4, label="USGS observed")
    plt.xlabel("Date (UTC)", labelpad=10)
    plt.ylabel("Streamflow (m³/s)", labelpad=10)
    plt.title(f"Hourly USGS Streamflow ({start_date} to {end_date})", pad=14)
    plt.legend()
    _format_time_axis(plt.gca(), observed.index)

    plt.show()


def get_gage_lookup_table(data_dir):
    """Table linking each USGS gage to its NGEN routing segment (feature_id)."""
    # Read gage ↔ segment pairs from the hydrofabric run directory
    # Returns: list of (ngen_segment_id, usgs_gage_id)
    ngiab_usgs_gages = ngiab_utils.get_gages_from_hydrofabric(data_dir)

    # Build lookup table: USGS gage id → NGEN feature_id
    ngiab_gages_df = pd.DataFrame(ngiab_usgs_gages, columns=["feature_id", "usgs_gage"])

    # USGS ids as strings (for matching and NWIS downloads)
    ngiab_gages_df["usgs_gage"] = ngiab_gages_df["usgs_gage"].astype(str)

    # Remove "wb-" prefix → numeric routing id used in NetCDF / troute
    # Example: "wb-2861391" → 2861391
    ngiab_gages_df["feature_id"] = (
        ngiab_gages_df["feature_id"]
        .astype(str)
        .str.replace("wb-", "", regex=False)
        .astype(int)
    )

    # Crosswalk table (one row per gage in this subset)
    display(ngiab_gages_df)
    return ngiab_gages_df


def build_hydrograph_table(ngiab_gages_df, gpkg_path, troute_path, start_date, end_date):
    """One table with hourly time, NGEN simulated and USGS observed streamflow (m/hr per m² basin)."""
    # Step 1: IDs from the hydrofabric lookup table
    usgs_gage_id = str(ngiab_gages_df["usgs_gage"].iloc[0])           # e.g. 10109001
    ngen_routing_feature_id = int(ngiab_gages_df["feature_id"].iloc[0])  # e.g. 2861391

    # Step 2: Total basin drainage area (km² → m²)
    # Total drainage_area in sqkm = full watershed above the outlet gage
    watershed_area_m2 = (
        gpd.read_file(gpkg_path, layer="divides")["tot_drainage_areasqkm"].max() * 1e6
    )

    # Step 3: Simulated streamflow from troute NetCDF (m³/h)
    simulated_flow_m3_per_hour = get_flow_data_from_netcdf(troute_path, ngen_routing_feature_id)

    # Step 4: Observed streamflow from USGS
    observed_streamflow_df = process_usgs_streamflow(usgs_gage_id, start_date, end_date, gpkg_path)

    if "Streamflow (m/hr)" in observed_streamflow_df.columns:
        observed_flow_m_per_hour_m2 = observed_streamflow_df["Streamflow (m/hr)"]
    else:
        observed_flow_m_per_hour_m2 = (
            observed_streamflow_df["00060"] * 3600 / (35.3147 * watershed_area_m2)
        )

    # Step 5: Shortest record length (sim vs obs)
    number_of_timesteps = min(len(simulated_flow_m3_per_hour), len(observed_flow_m_per_hour_m2))

    # Step 6: One table: time, simulated, observed
    hydrograph_df = pd.DataFrame({
        "Time": pd.to_datetime(observed_streamflow_df["Time"].iloc[:number_of_timesteps], utc=True)
                  .dt.tz_convert(None).dt.floor("h"),
        "simulated_streamflow": simulated_flow_m3_per_hour[:number_of_timesteps] / watershed_area_m2,
        "observed_streamflow": observed_flow_m_per_hour_m2.iloc[:number_of_timesteps].values,
    }).dropna()

    print(f"USGS gage: {usgs_gage_id} | NGEN feature: {ngen_routing_feature_id} | rows: {len(hydrograph_df)}")
    display(hydrograph_df.head())
    return hydrograph_df


def plot_hydrograph(hydrograph_df, ngiab_gages_df):
    """Plot USGS observed vs NGEN simulated streamflow."""
    usgs_gage_id = str(ngiab_gages_df["usgs_gage"].iloc[0])

    plt.figure(figsize=(12, 4))

    plt.plot(
        hydrograph_df["Time"],
        hydrograph_df["observed_streamflow"],
        label=f"USGS {usgs_gage_id}",
    )
    plt.plot(
        hydrograph_df["Time"],
        hydrograph_df["simulated_streamflow"],
        label="NGEN simulated",
    )

    plt.ylabel("Streamflow (m/hr per m² basin)")
    plt.xlabel("Time")
    plt.title(f"Hydrograph — USGS gage {usgs_gage_id}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def register_usgs_gage(ev, usgs_gage, ngiab_gages_df):
    """Register a USGS gage and its NGEN segment in TEEHR and link them in the crosswalk."""
    # Inputs
    site_no = usgs_gage[5:]
    ngen_id = ngiab_gages_df.iloc[0, 0]

    # 1) Fetch USGS site metadata (lat/lon + station name) from NWIS
    site = nwis.get_record(sites=site_no, service="site").iloc[0]
    pt = Point(float(site["dec_long_va"]), float(site["dec_lat_va"]))

    # 2) Register locations in TEEHR (USGS gage + matching NGEN segment)
    loc_gdf = gpd.GeoDataFrame(
        [
            {"id": f"usgs-{site_no}", "name": str(site["station_nm"]), "geometry": pt},
            {"id": ngen_id,           "name": f"NGEN {ngen_id.split('-')[-1]}", "geometry": pt},  # same point is OK for now
        ],
        crs="EPSG:4326",
    )
    ev.locations.load_dataframe(loc_gdf)

    # 3) Define the location crosswalk (USGS = primary, NGEN = secondary)
    ev.location_crosswalks.load_dataframe(
        pd.DataFrame([{"primary_location_id": f"usgs-{site_no}", "secondary_location_id": ngen_id}])
    )

    print(f"Registered location: usgs-{site_no} ({site['station_nm']}) and crosswalk → {ngen_id}")


def plot_teehr_streamflow(ev, usgs_gage, nwm_feature_id, ngen_id):
    """Plot USGS observed, NWM and NextGen streamflow stored in a TEEHR evaluation."""
    # Query the three time series from TEEHR
    usgs_df = ev.primary_timeseries.query(
        filters=["configuration_name = 'usgs_observations'", f"location_id = '{usgs_gage}'"]
    ).to_pandas().sort_values("value_time")

    nwm_df = ev.secondary_timeseries.query(
        filters=["configuration_name = 'nwm30_retrospective'", f"location_id = 'nwm30-{nwm_feature_id}'"]
    ).to_pandas().sort_values("value_time")

    ngen_df = ev.secondary_timeseries.query(
        filters=["configuration_name = 'ngen_simulation'", f"location_id = '{ngen_id}'"]
    ).to_pandas().sort_values("value_time")

    plt.figure(figsize=(12, 5))
    plt.plot(usgs_df["value_time"], usgs_df["value"], color="black", linewidth=2, label=f"USGS observed ({usgs_gage})")
    plt.plot(nwm_df["value_time"], nwm_df["value"], color="#2a78d6", linewidth=1.5, label="NWM v3.0 retrospective")
    plt.plot(ngen_df["value_time"], ngen_df["value"], color="#eb6834", linewidth=1.5, label="NextGen simulation")

    plt.ylabel("Streamflow (m³/s)")
    plt.xlabel("Date")
    plt.title("Streamflow: USGS observed vs NWM vs NextGen")
    plt.grid(alpha=0.3)
    plt.legend(frameon=False)
    plt.tight_layout()

    plt.show()
