"""
Pakistan Multi-Index Satellite Remote Sensing Analysis Tool
=============================================================
Single-file build (no local package imports) to avoid deployment path
issues on Streamlit Cloud / GitHub.

Run locally:      streamlit run app.py
Deploy live:       push this file + requirements.txt to a GitHub repo root,
                    then deploy on share.streamlit.io (Streamlit Community Cloud).

Data sources:
  - Demo (synthetic)      : works instantly, no setup, no API keys.
  - Upload GeoTIFF bands  : compute indices on your own local satellite bands.
  - Google Earth Engine   : real, live Sentinel-2 / Landsat 8/9 imagery for
                             anywhere in Pakistan. Needs your own free GEE
                             account + credentials (see sidebar instructions).
"""

import base64
import datetime as dt
import io
import json
import tempfile
import zipfile

import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import rasterio
import streamlit as st
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds
from scipy.ndimage import gaussian_filter
from streamlit_folium import st_folium

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Pakistan Remote Sensing Analysis Tool",
    page_icon="🛰️",
    layout="wide",
)

# =============================================================================
# 1. SPECTRAL INDEX FORMULAS (NumPy — used by Demo & Upload modes)
# =============================================================================
EPS = 1e-8


def _safe_divide(numerator, denominator):
    denominator = np.where(denominator == 0, EPS, denominator)
    return numerator / denominator


def ndvi(red, nir):
    red, nir = red.astype(float), nir.astype(float)
    return _safe_divide(nir - red, nir + red)


def evi(blue, red, nir, G=2.5, C1=6.0, C2=7.5, L=1.0):
    blue, red, nir = blue.astype(float), red.astype(float), nir.astype(float)
    denom = nir + C1 * red - C2 * blue + L
    return G * _safe_divide(nir - red, denom)


def ndwi(green, nir):
    green, nir = green.astype(float), nir.astype(float)
    return _safe_divide(green - nir, green + nir)


def mndwi(green, swir1):
    green, swir1 = green.astype(float), swir1.astype(float)
    return _safe_divide(green - swir1, green + swir1)


def ndbi(swir1, nir):
    swir1, nir = swir1.astype(float), nir.astype(float)
    return _safe_divide(swir1 - nir, swir1 + nir)


def bsi(blue, red, nir, swir1):
    blue, red, nir, swir1 = (blue.astype(float), red.astype(float),
                              nir.astype(float), swir1.astype(float))
    num = (swir1 + red) - (nir + blue)
    den = (swir1 + red) + (nir + blue)
    return _safe_divide(num, den)


def savi(red, nir, L=0.5):
    red, nir = red.astype(float), nir.astype(float)
    return _safe_divide((nir - red) * (1 + L), (nir + red + L))


INDEX_REGISTRY = {
    "NDVI": {"func": ndvi, "bands": ["red", "nir"], "cmap": "RdYlGn",
             "description": "Vegetation / greenness health"},
    "EVI": {"func": evi, "bands": ["blue", "red", "nir"], "cmap": "RdYlGn",
            "description": "Enhanced vegetation (atmosphere/soil corrected)"},
    "NDWI": {"func": ndwi, "bands": ["green", "nir"], "cmap": "Blues",
             "description": "Surface water / moisture content"},
    "MNDWI": {"func": mndwi, "bands": ["green", "swir1"], "cmap": "Blues",
              "description": "Water extraction (robust in urban areas)"},
    "NDBI": {"func": ndbi, "bands": ["swir1", "nir"], "cmap": "OrRd",
             "description": "Built-up / urban areas"},
    "BSI": {"func": bsi, "bands": ["blue", "red", "nir", "swir1"], "cmap": "copper",
            "description": "Bare soil exposure"},
    "SAVI": {"func": savi, "bands": ["red", "nir"], "cmap": "RdYlGn",
             "description": "Soil-adjusted vegetation index"},
}


def compute_index(name, band_arrays):
    spec = INDEX_REGISTRY[name]
    args = [band_arrays[b] for b in spec["bands"]]
    result = spec["func"](*args)
    return np.clip(result, -1, 1)


# =============================================================================
# 2. SYNTHETIC DEMO DATA GENERATOR
# =============================================================================
def _fbm(shape, seed, octaves=4, base_scale=6):
    rng = np.random.default_rng(seed)
    field = np.zeros(shape, dtype=float)
    amp = 1.0
    total_amp = 0.0
    for o in range(octaves):
        scale = base_scale * (2 ** o)
        small = rng.random((max(2, shape[0] // scale + 2), max(2, shape[1] // scale + 2)))
        big = np.kron(small, np.ones((scale, scale)))[: shape[0], : shape[1]]
        big = gaussian_filter(big, sigma=scale / 3.0)
        field += amp * big
        total_amp += amp
        amp *= 0.5
    field /= total_amp
    field -= field.min()
    field /= (field.max() + 1e-8)
    return field


def generate_bands(lat, lon, date_str, size=128):
    seed = abs(hash((round(lat, 3), round(lon, 3), date_str))) % (2**32)
    shape = (size, size)

    veg = _fbm(shape, seed + 1, octaves=4, base_scale=5)
    water = _fbm(shape, seed + 2, octaves=3, base_scale=10)
    urban = _fbm(shape, seed + 3, octaves=4, base_scale=4)
    soil = 1.0 - veg

    try:
        month = int(date_str.split("-")[1])
    except Exception:
        month = 6
    seasonal = 0.5 + 0.5 * np.cos((month - 4) / 12 * 2 * np.pi)

    veg = np.clip(veg * (0.5 + 0.5 * seasonal), 0, 1)
    water_mask = (water > 0.82).astype(float)
    urban_mask = np.clip((urban - 0.55) * 2.2, 0, 1)

    blue = 0.05 + 0.10 * soil + 0.25 * urban_mask + 0.02 * water_mask
    green = 0.05 + 0.15 * veg + 0.12 * soil + 0.22 * urban_mask + 0.05 * water_mask
    red = 0.05 + 0.08 * veg + 0.20 * soil + 0.28 * urban_mask + 0.03 * water_mask
    nir = 0.10 + 0.55 * veg + 0.10 * soil + 0.15 * urban_mask - 0.15 * water_mask
    swir1 = 0.08 + 0.10 * veg + 0.30 * soil + 0.30 * urban_mask - 0.10 * water_mask
    swir2 = 0.06 + 0.08 * veg + 0.28 * soil + 0.25 * urban_mask - 0.08 * water_mask

    return {
        "blue": np.clip(blue, 0.01, 1),
        "green": np.clip(green, 0.01, 1),
        "red": np.clip(red, 0.01, 1),
        "nir": np.clip(nir, 0.01, 1),
        "swir1": np.clip(swir1, 0.01, 1),
        "swir2": np.clip(swir2, 0.01, 1),
    }


# =============================================================================
# 3. RASTER I/O + EXPORT HELPERS
# =============================================================================
def read_band_from_upload(uploaded_file):
    data = uploaded_file.read()
    with MemoryFile(data) as memfile:
        with memfile.open() as src:
            arr = src.read(1).astype(float)
            profile = src.profile
            bounds = src.bounds
    return arr, profile, bounds


def array_to_geotiff_bytes(array, bounds, crs="EPSG:4326"):
    h, w = array.shape
    transform = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], w, h)
    profile = {
        "driver": "GTiff", "height": h, "width": w, "count": 1,
        "dtype": "float32", "crs": crs, "transform": transform, "nodata": np.nan,
    }
    buf = io.BytesIO()
    with MemoryFile() as memfile:
        with memfile.open(**profile) as dst:
            dst.write(array.astype("float32"), 1)
        buf.write(memfile.read())
    buf.seek(0)
    return buf.getvalue()


def _get_cmap(cmap_name):
    """Compatibility shim: matplotlib.cm.get_cmap() was removed in mpl 3.11+."""
    try:
        return matplotlib.colormaps[cmap_name]
    except Exception:
        return cm.get_cmap(cmap_name)


def array_to_png_bytes(array, cmap_name="RdYlGn", vmin=-1, vmax=1):
    normed = (np.clip(array, vmin, vmax) - vmin) / (vmax - vmin)
    cmap = _get_cmap(cmap_name)
    rgba = cmap(normed)
    if np.isnan(array).any():
        rgba[..., 3] = np.where(np.isnan(array), 0, 1)
    buf = io.BytesIO()
    plt.imsave(buf, rgba, format="png")
    buf.seek(0)
    return buf.getvalue()


def build_result_zip(index_arrays, bounds, timeseries_df=None):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, arr in index_arrays.items():
            cmap = INDEX_REGISTRY.get(name, {}).get("cmap", "viridis")
            zf.writestr(f"{name}.png", array_to_png_bytes(arr, cmap_name=cmap))
            zf.writestr(f"{name}.tif", array_to_geotiff_bytes(arr, bounds))
        if timeseries_df is not None:
            csv_buf = io.StringIO()
            timeseries_df.to_csv(csv_buf, index=False)
            zf.writestr("timeseries.csv", csv_buf.getvalue())
        zf.writestr("README.txt",
            "Pakistan Multi-Index Satellite Remote Sensing Analysis Tool\n"
            "-------------------------------------------------------------\n"
            "Each <INDEX>.tif is a georeferenced single-band float32 GeoTIFF "
            "(EPSG:4326) with values in the index's natural range (typically -1..1).\n"
            "Each <INDEX>.png is a quick-look colored preview of the same data.\n"
            "timeseries.csv (if present) contains the mean index value per "
            "selected date/period for the chosen area of interest.\n")
    buf.seek(0)
    return buf.getvalue()


# =============================================================================
# 4. GOOGLE EARTH ENGINE (live) — optional backend
# =============================================================================
SATELLITE_COLLECTIONS = {
    "Sentinel-2 SR": {
        "id": "COPERNICUS/S2_SR_HARMONIZED",
        "bands": {"blue": "B2", "green": "B3", "red": "B4", "nir": "B8", "swir1": "B11", "swir2": "B12"},
        "scale": 10, "cloud_prop": "CLOUDY_PIXEL_PERCENTAGE", "reflectance_scale": 1e-4,
    },
    "Landsat 8 SR": {
        "id": "LANDSAT/LC08/C02/T1_L2",
        "bands": {"blue": "SR_B2", "green": "SR_B3", "red": "SR_B4", "nir": "SR_B5", "swir1": "SR_B6", "swir2": "SR_B7"},
        "scale": 30, "cloud_prop": "CLOUD_COVER", "reflectance_scale": 2.75e-5,
    },
    "Landsat 9 SR": {
        "id": "LANDSAT/LC09/C02/T1_L2",
        "bands": {"blue": "SR_B2", "green": "SR_B3", "red": "SR_B4", "nir": "SR_B5", "swir1": "SR_B6", "swir2": "SR_B7"},
        "scale": 30, "cloud_prop": "CLOUD_COVER", "reflectance_scale": 2.75e-5,
    },
}


def init_ee(service_account_json_bytes=None):
    try:
        import ee
        if service_account_json_bytes:
            info = json.loads(service_account_json_bytes)
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
                f.write(service_account_json_bytes)
                key_path = f.name
            credentials = ee.ServiceAccountCredentials(info["client_email"], key_path)
            ee.Initialize(credentials)
        else:
            ee.Initialize()
        return True, "Earth Engine initialized."
    except Exception as e:
        return False, f"Earth Engine init failed: {e}"


def get_aoi_geometry(min_lon, min_lat, max_lon, max_lat):
    import ee
    return ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])


def get_composite(satellite, aoi, start_date, end_date, max_cloud=30):
    import ee
    cfg = SATELLITE_COLLECTIONS[satellite]
    coll = (ee.ImageCollection(cfg["id"]).filterBounds(aoi).filterDate(start_date, end_date)
            .filter(ee.Filter.lt(cfg["cloud_prop"], max_cloud)))
    image = coll.median().clip(aoi)
    if satellite == "Sentinel-2 SR":
        image = image.multiply(cfg["reflectance_scale"])
    else:
        image = image.multiply(cfg["reflectance_scale"]).add(-0.2)
    return image, coll.size()


def image_to_band_arrays(image, aoi, satellite):
    cfg = SATELLITE_COLLECTIONS[satellite]
    band_map = cfg["bands"]
    renamed = image.select(list(band_map.values()), list(band_map.keys()))
    sampled = renamed.sampleRectangle(region=aoi, defaultValue=0)
    return {name: np.array(sampled.get(name).getInfo(), dtype=float) for name in band_map.keys()}


def ndvi_ee(img):
    return img.normalizedDifference(["nir", "red"]).rename("index")


def evi_ee(img):
    return img.expression("2.5 * ((NIR - RED) / (NIR + 6.0 * RED - 7.5 * BLUE + 1.0))",
        {"NIR": img.select("nir"), "RED": img.select("red"), "BLUE": img.select("blue")}).rename("index")


def ndwi_ee(img):
    return img.normalizedDifference(["green", "nir"]).rename("index")


def mndwi_ee(img):
    return img.normalizedDifference(["green", "swir1"]).rename("index")


def ndbi_ee(img):
    return img.normalizedDifference(["swir1", "nir"]).rename("index")


def bsi_ee(img):
    return img.expression("((SWIR1 + RED) - (NIR + BLUE)) / ((SWIR1 + RED) + (NIR + BLUE))",
        {"SWIR1": img.select("swir1"), "RED": img.select("red"),
         "NIR": img.select("nir"), "BLUE": img.select("blue")}).rename("index")


def savi_ee(img, L=0.5):
    return img.expression("((NIR - RED) * (1 + L)) / (NIR + RED + L)",
        {"NIR": img.select("nir"), "RED": img.select("red"), "L": L}).rename("index")


EE_INDEX_REGISTRY = {"NDVI": ndvi_ee, "EVI": evi_ee, "NDWI": ndwi_ee, "MNDWI": mndwi_ee,
                      "NDBI": ndbi_ee, "BSI": bsi_ee, "SAVI": savi_ee}


def get_timeseries(satellite, aoi, index_func, start_date, end_date, freq="MS", max_cloud=30):
    import ee
    cfg = SATELLITE_COLLECTIONS[satellite]
    periods = pd.date_range(start_date, end_date, freq=freq)
    rows = []
    for i in range(len(periods) - 1):
        p_start = periods[i].strftime("%Y-%m-%d")
        p_end = periods[i + 1].strftime("%Y-%m-%d")
        coll = (ee.ImageCollection(cfg["id"]).filterBounds(aoi).filterDate(p_start, p_end)
                .filter(ee.Filter.lt(cfg["cloud_prop"], max_cloud)))
        n = coll.size().getInfo()
        if n == 0:
            rows.append({"date": p_start, "mean_value": None, "n_images": 0})
            continue
        image = coll.median().clip(aoi)
        if satellite == "Sentinel-2 SR":
            image = image.multiply(cfg["reflectance_scale"])
        else:
            image = image.multiply(cfg["reflectance_scale"]).add(-0.2)
        renamed = image.select(list(cfg["bands"].values()), list(cfg["bands"].keys()))
        idx_image = index_func(renamed)
        stat = idx_image.reduceRegion(reducer=ee.Reducer.mean(), geometry=aoi,
                                       scale=cfg["scale"], maxPixels=1e9).getInfo()
        val = list(stat.values())[0] if stat else None
        rows.append({"date": p_start, "mean_value": val, "n_images": n})
    return pd.DataFrame(rows)


# =============================================================================
# 5. STREAMLIT APP
# =============================================================================
PAKISTAN_CENTER = (30.3753, 69.3451)
ALL_INDEX_NAMES = list(INDEX_REGISTRY.keys())

if "picked_latlon" not in st.session_state:
    st.session_state.picked_latlon = PAKISTAN_CENTER
if "results" not in st.session_state:
    st.session_state.results = None
if "bounds" not in st.session_state:
    st.session_state.bounds = None
if "timeseries" not in st.session_state:
    st.session_state.timeseries = None

st.sidebar.title("🛰️ Configuration")

data_source = st.sidebar.radio(
    "Data source",
    ["Demo (synthetic data)", "Upload GeoTIFF bands", "Google Earth Engine (live)"],
    help=("Demo generates realistic-looking synthetic bands instantly — no setup. "
          "Upload lets you compute indices on your own band GeoTIFFs. "
          "Google Earth Engine pulls real Sentinel-2/Landsat imagery but needs "
          "your own free GEE credentials."),
)

st.sidebar.markdown("### 📍 Location (Area of Interest)")
st.sidebar.caption("Click the map to set the AOI center, then set a radius.")
click_map = folium.Map(location=st.session_state.picked_latlon, zoom_start=5, tiles="OpenStreetMap")
folium.Marker(st.session_state.picked_latlon, tooltip="AOI center").add_to(click_map)
map_data = st_folium(click_map, height=280, width=None, key="picker_map")
if map_data and map_data.get("last_clicked"):
    st.session_state.picked_latlon = (map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"])

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
        "2. Create a service account + JSON key in Google Cloud Console, grant it Earth Engine access\n"
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

run_clicked = st.sidebar.button("▶ Compute indices", type="primary", width="stretch")


def bbox_from_point(lat, lon, half_width_km):
    dlat = half_width_km / 111.0
    dlon = half_width_km / (111.0 * max(0.1, np.cos(np.radians(lat))))
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)


def compute_from_bands(band_arrays, selected):
    out = {}
    for name in selected:
        spec = INDEX_REGISTRY[name]
        if all(b in band_arrays for b in spec["bands"]):
            out[name] = compute_index(name, band_arrays)
    return out


def render_overlay_map(bounds, index_name, arr):
    minx, miny, maxx, maxy = bounds
    m = folium.Map(location=[(miny + maxy) / 2, (minx + maxx) / 2], zoom_start=11, tiles="OpenStreetMap")
    png_bytes = array_to_png_bytes(arr, cmap_name=INDEX_REGISTRY[index_name]["cmap"])
    b64 = base64.b64encode(png_bytes).decode()
    folium.raster_layers.ImageOverlay(image=f"data:image/png;base64,{b64}",
        bounds=[[miny, minx], [maxy, maxx]], opacity=0.75).add_to(m)
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
                        band_arrays = image_to_band_arrays(image, aoi, satellite)
                        st.session_state.results = compute_from_bands(band_arrays, selected_indices)
            except Exception as e:
                st.error(
                    f"Earth Engine error: {e}\n\n"
                    "Make sure you're running this on a host with internet access "
                    "and valid Earth Engine credentials."
                )

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
                    key_bytes = ee_key_file.read() if ee_key_file is not None else None
                    ok, msg = init_ee(key_bytes)
                    if ok:
                        aoi = get_aoi_geometry(*st.session_state.bounds)
                        df = get_timeseries(satellite, aoi, EE_INDEX_REGISTRY[ts_index],
                                             str(start_date), str(end_date), freq="MS", max_cloud=max_cloud)
                    else:
                        st.error(msg)
                        df = pd.DataFrame()
                except Exception as e:
                    st.error(f"Earth Engine time series error: {e}")
                    df = pd.DataFrame()
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
        st.plotly_chart(fig, width="stretch")

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
        st.plotly_chart(fig2, width="stretch")

    st.subheader("⬇️ Export results")
    st.caption("Bundles GeoTIFF + PNG for each computed index, plus the time series CSV if generated, into one ZIP.")
    zip_bytes = build_result_zip(st.session_state.results, st.session_state.bounds,
                                  timeseries_df=st.session_state.timeseries)
    st.download_button("Download results as ZIP", data=zip_bytes,
        file_name=f"pakistan_rs_results_{dt.date.today()}.zip", mime="application/zip", type="primary")
else:
    st.info("Configure your location, date range, satellite and indices in the sidebar, then click **Compute indices**.")
