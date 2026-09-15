"""
Pakistan Multi-Index Satellite Remote Sensing Analysis Tool
=============================================================
Run locally:      streamlit run app.py
Deploy live:       push this folder to GitHub -> deploy on share.streamlit.io
                    (Streamlit Community Cloud) or any Streamlit-capable host.

Data sources:
  - Demo (synthetic)      : works instantly, no setup, no API keys.
  - Upload GeoTIFF bands  : compute indices on your own local satellite bands.
  - Google Earth Engine   : real, live Sentinel-2 / Landsat 8/9 imagery for
                             anywhere in Pakistan. Needs your own free GEE
                             account + credentials (see sidebar instructions).
"""

import datetime as dt

import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

from utils.indices import INDEX_REGISTRY, compute_index
from utils.raster_utils import array_to_png_bytes, build_result_zip
from utils.synthetic_data import generate_bands

st.set_page_config(
    page_title="Pakistan Remote Sensing Analysis Tool",
    page_icon="🛰️",
    layout="wide",
)

PAKISTAN_CENTER = (30.3753, 69.3451)
PAKISTAN_BOUNDS = [[23.5, 60.5], [37.5, 77.5]]  # rough SW / NE corners

ALL_INDEX_NAMES = list(INDEX_REGISTRY.keys())

# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
if "picked_latlon" not in st.session_state:
    st.session_state.picked_latlon = PAKISTAN_CENTER
if "results" not in st.session_state:
    st.session_state.results = None  # dict: index_name -> array
if "bounds" not in st.session_state:
    st.session_state.bounds = None
if "timeseries" not in st.session_state:
    st.session_state.timeseries = None
if "uploaded_snapshots" not in st.session_state:
    st.session_state.uploaded_snapshots = []  # list of {date, bands, bounds}

# ----------------------------------------------------------------------------
# Sidebar - configuration
# ----------------------------------------------------------------------------
st.sidebar.title("🛰️ Configuration")

data_source = st.sidebar.radio(
    "Data source",
    ["Demo (synthetic data)", "Upload GeoTIFF bands", "Google Earth Engine (live)"],
    help=(
        "Demo generates realistic-looking synthetic bands instantly — no setup. "
        "Upload lets you compute indices on your own band GeoTIFFs. "
        "Google Earth Engine pulls real Sentinel-2/Landsat imagery but needs "
        "your own free GEE credentials."
    ),
)

st.sidebar.markdown("### 📍 Location (Area of Interest)")
st.sidebar.caption("Click the map to set the AOI center, then set a radius.")

click_map = folium.Map(location=st.session_state.picked_latlon, zoom_start=5,
                        tiles="CartoDB positron")
folium.Marker(st.session_state.picked_latlon, tooltip="AOI center").add_to(click_map)
map_data = st_folium(click_map, height=280, width=None, key="picker_map")
if map_data and map_data.get("last_clicked"):
    st.session_state.picked_latlon = (
        map_data["last_clicked"]["lat"],
        map_data["last_clicked"]["lng"],
    )

lat = st.sidebar.number_input("Latitude", value=float(st.session_state.picked_latlon[0]), format="%.5f")
lon = st.sidebar.number_input("Longitude", value=float(st.session_state.picked_latlon[1]), format="%.5f")
radius_km = st.sidebar.slider("AOI half-width (km)", 1, 100, 10)

st.sidebar.markdown("### 📅 Date range")
default_end = dt.date.today()
default_start = default_end - dt.timedelta(days=365)
start_date = st.sidebar.date_input("Start date", value=default_start)
end_date = st.sidebar.date_input("End date", value=default_end)

st.sidebar.markdown("### 🛰️ Satellite")
satellite = st.sidebar.selectbox("Satellite / collection", ["Sentinel-2 SR", "Landsat 8 SR", "Landsat 9 SR"])

st.sidebar.markdown("### 📊 Indices to compute")
selected_indices = st.sidebar.multiselect(
    "Indices", ALL_INDEX_NAMES, default=ALL_INDEX_NAMES,
    format_func=lambda n: f"{n} — {INDEX_REGISTRY[n]['description']}",
)

max_cloud = st.sidebar.slider("Max cloud cover % (GEE only)", 0, 100, 30)

ee_key_file = None
if data_source == "Google Earth Engine (live)":
    st.sidebar.info(
        "Live mode needs your own Earth Engine credentials.\n\n"
        "1. Sign up free at earthengine.google.com\n"
        "2. Create a service account + JSON key in Google Cloud Console, "
        "grant it Earth Engine access\n"
        "3. Upload the JSON key below"
    )
    ee_key_file = st.sidebar.file_uploader("GEE service account JSON key", type=["json"])

uploaded_bands = {}
if data_source == "Upload GeoTIFF bands":
    st.sidebar.info(
        "Upload single-band GeoTIFFs for Blue, Green, Red, NIR, SWIR1, SWIR2 "
        "(SWIR1/2 optional — needed only for NDBI, BSI, MNDWI)."
    )
    for b in ["blue", "green", "red", "nir", "swir1", "swir2"]:
        f = st.sidebar.file_uploader(f"{b.upper()} band", type=["tif", "tiff"], key=f"upl_{b}")
        if f is not None:
            uploaded_bands[b] = f

run_clicked = st.sidebar.button("▶ Compute indices", type="primary", use_container_width=True)

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def bbox_from_point(lat, lon, half_width_km):
    dlat = half_width_km / 111.0
    dlon = half_width_km / (111.0 * max(0.1, np.cos(np.radians(lat))))
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)  # minx, miny, maxx, maxy


def compute_from_bands(band_arrays, selected):
    out = {}
    for name in selected:
        spec = INDEX_REGISTRY[name]
        if all(b in band_arrays for b in spec["bands"]):
            out[name] = compute_index(name, band_arrays)
    return out


def render_overlay_map(bounds, index_name, arr):
    minx, miny, maxx, maxy = bounds
    m = folium.Map(location=[(miny + maxy) / 2, (minx + maxx) / 2], zoom_start=11,
                    tiles="CartoDB positron")
    png_bytes = array_to_png_bytes(arr, cmap_name=INDEX_REGISTRY[index_name]["cmap"])
    import base64
    b64 = base64.b64encode(png_bytes).decode()
    folium.raster_layers.ImageOverlay(
        image=f"data:image/png;base64,{b64}",
        bounds=[[miny, minx], [maxy, maxx]],
        opacity=0.75,
    ).add_to(m)
    folium.Rectangle(bounds=[[miny, minx], [maxy, maxx]], color="black", weight=1, fill=False).add_to(m)
    return m


def demo_timeseries(lat, lon, start_date, end_date, index_name, size=64):
    dates = pd.date_range(start_date, end_date, freq="MS")
    if len(dates) == 0:
        dates = pd.DatetimeIndex([pd.Timestamp(start_date)])
    rows = []
    for d in dates:
        bands = generate_bands(lat, lon, d.strftime("%Y-%m-%d"), size=size)
        spec = INDEX_REGISTRY[index_name]
        if all(b in bands for b in spec["bands"]):
            arr = compute_index(index_name, bands)
            rows.append({"date": d, "mean_value": float(np.nanmean(arr))})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
st.title("🇵🇰 Pakistan Multi-Index Satellite Remote Sensing Analysis Tool")
st.caption(
    "Select a location and date range, choose a satellite, and compute NDVI, EVI, "
    "NDWI, MNDWI, NDBI, BSI and SAVI — view them as map layers, chart them over "
    "time, compare periods, and export GeoTIFF / PNG / CSV."
)

if data_source == "Demo (synthetic data)":
    st.warning(
        "**Demo mode** — indices below are computed from procedurally generated "
        "synthetic reflectance data (seeded by your chosen location & date), "
        "*not real satellite imagery*. Use this to test the full workflow "
        "instantly. Switch to 'Google Earth Engine (live)' for real Sentinel-2 / "
        "Landsat data once you have your own free GEE credentials.",
        icon="⚠️",
    )

if run_clicked:
    bounds = bbox_from_point(lat, lon, radius_km)
    st.session_state.bounds = bounds

    with st.spinner("Computing indices..."):
        if data_source == "Demo (synthetic data)":
            bands = generate_bands(lat, lon, str(end_date), size=200)
            st.session_state.results = compute_from_bands(bands, selected_indices)

        elif data_source == "Upload GeoTIFF bands":
            if len(uploaded_bands) < 2:
                st.error("Please upload at least two bands (e.g. Red + NIR for NDVI).")
            else:
                from utils.raster_utils import read_band_from_upload
                band_arrays = {}
                file_bounds = bounds
                for b, f in uploaded_bands.items():
                    arr, profile, rbounds = read_band_from_upload(f)
                    band_arrays[b] = arr
                    file_bounds = (rbounds.left, rbounds.bottom, rbounds.right, rbounds.top)
                st.session_state.bounds = file_bounds
                st.session_state.results = compute_from_bands(band_arrays, selected_indices)

        elif data_source == "Google Earth Engine (live)":
            try:
                from utils.gee_utils import init_ee, get_aoi_geometry, get_composite, image_to_band_arrays
                key_bytes = ee_key_file.read() if ee_key_file is not None else None
                ok, msg = init_ee(key_bytes)
                if not ok:
                    st.error(msg)
                else:
                    aoi = get_aoi_geometry(*bounds)
                    image, n = get_composite(satellite, aoi, str(start_date), str(end_date), max_cloud)
                    n_images = n.getInfo()
                    if n_images == 0:
                        st.error("No cloud-free images found for this AOI/date range/cloud threshold. Try widening them.")
                    else:
                        st.success(f"Found {n_images} images — composite built.")
                        band_arrays = image_to_band_arrays(image, aoi, satellite, size=200)
                        st.session_state.results = compute_from_bands(band_arrays, selected_indices)
            except Exception as e:
                st.error(
                    f"Earth Engine error: {e}\n\n"
                    "This sandbox cannot reach Earth Engine's servers directly — "
                    "run this app on your own machine or a cloud host with internet "
                    "access and your own GEE credentials for live data."
                )

# ----------------------------------------------------------------------------
# Display results
# ----------------------------------------------------------------------------
if st.session_state.results:
    st.subheader("🗺️ Index map layers")
    tabs = st.tabs(list(st.session_state.results.keys()))
    for tab, (name, arr) in zip(tabs, st.session_state.results.items()):
        with tab:
            c1, c2 = st.columns([3, 1])
            with c1:
                m = render_overlay_map(st.session_state.bounds, name, arr)
                st_folium(m, height=420, width=None, key=f"map_{name}")
            with c2:
                st.metric("Mean", f"{np.nanmean(arr):.3f}")
                st.metric("Min", f"{np.nanmin(arr):.3f}")
                st.metric("Max", f"{np.nanmax(arr):.3f}")
                st.caption(INDEX_REGISTRY[name]["description"])

    st.subheader("📈 Time series")
    ts_index = st.selectbox("Index for time series", list(st.session_state.results.keys()), key="ts_index")
    if st.button("Generate time series over selected date range"):
        with st.spinner("Building time series..."):
            if data_source == "Demo (synthetic data)":
                df = demo_timeseries(lat, lon, start_date, end_date, ts_index)
            elif data_source == "Google Earth Engine (live)":
                try:
                    from utils.gee_utils import init_ee, get_aoi_geometry, get_timeseries, SATELLITE_COLLECTIONS
                    from utils.ee_indices import EE_INDEX_REGISTRY
                    key_bytes = ee_key_file.read() if ee_key_file is not None else None
                    ok, msg = init_ee(key_bytes)
                    if ok:
                        aoi = get_aoi_geometry(*st.session_state.bounds)
                        df = get_timeseries(
                            satellite, aoi, EE_INDEX_REGISTRY[ts_index],
                            SATELLITE_COLLECTIONS[satellite]["bands"],
                            str(start_date), str(end_date), freq="MS", max_cloud=max_cloud,
                        )
                    else:
                        st.error(msg)
                        df = pd.DataFrame()
                except Exception as e:
                    st.error(f"Earth Engine time series error: {e}")
                    df = pd.DataFrame()
            else:
                if st.session_state.uploaded_snapshots:
                    rows = []
                    for snap in st.session_state.uploaded_snapshots:
                        if ts_index in snap["results"]:
                            rows.append({"date": snap["date"],
                                         "mean_value": float(np.nanmean(snap["results"][ts_index]))})
                    df = pd.DataFrame(rows)
                else:
                    st.info(
                        "Upload mode only has a single snapshot. For a real time "
                        "series, use Demo or Google Earth Engine mode, which can "
                        "sample multiple dates automatically."
                    )
                    df = pd.DataFrame()

            st.session_state.timeseries = df

    if st.session_state.timeseries is not None and not st.session_state.timeseries.empty:
        df = st.session_state.timeseries
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["date"], y=df["mean_value"], mode="lines+markers", name=ts_index))
        fig.update_layout(title=f"{ts_index} over time", xaxis_title="Date", yaxis_title=f"Mean {ts_index}")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Compare years/months")
        df2 = df.copy()
        df2["date"] = pd.to_datetime(df2["date"])
        df2["year"] = df2["date"].dt.year
        df2["month"] = df2["date"].dt.strftime("%b")
        pivot = df2.pivot_table(index="month", columns="year", values="mean_value")
        month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        pivot = pivot.reindex([m for m in month_order if m in pivot.index])
        fig2 = go.Figure()
        for year in pivot.columns:
            fig2.add_trace(go.Scatter(x=pivot.index, y=pivot[year], mode="lines+markers", name=str(year)))
        fig2.update_layout(title=f"{ts_index} — month-by-month, year over year",
                            xaxis_title="Month", yaxis_title=f"Mean {ts_index}")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("⬇️ Export results")
    st.caption("Bundles GeoTIFF + PNG for each computed index, plus the time series CSV if generated, into one ZIP.")
    zip_bytes = build_result_zip(
        st.session_state.results,
        st.session_state.bounds,
        timeseries_df=st.session_state.timeseries,
    )
    st.download_button(
        "Download results as ZIP",
        data=zip_bytes,
        file_name=f"pakistan_rs_results_{dt.date.today()}.zip",
        mime="application/zip",
        type="primary",
    )
else:
    st.info("Configure your location, date range, satellite and indices in the sidebar, then click **Compute indices**.")
