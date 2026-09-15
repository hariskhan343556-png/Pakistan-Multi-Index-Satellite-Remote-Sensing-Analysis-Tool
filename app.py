import datetime as dt
import json

import streamlit as st

st.set_page_config(
    page_title="Pakistan Multi-Index Satellite Remote Sensing Tool",
    page_icon="\U0001F6F0\uFE0F",
    layout="wide",
)

try:
    import ee
    import folium
    from folium import LayerControl
    import pandas as pd
    import plotly.graph_objects as go
    from streamlit_folium import st_folium
except ModuleNotFoundError as exc:
    st.error(
        f"Startup failed: {exc}\n\n"
        "A required package did not install. Check 'Manage app -> Logs' on Streamlit "
        "Cloud for the pip install output and confirm requirements.txt is present at "
        "the repository root."
    )
    st.stop()
except Exception as exc:
    st.error(f"Startup failed while importing dependencies: {exc}")
    st.stop()

# =========================================================================
# Earth Engine authentication
# =========================================================================

@st.cache_resource(show_spinner=False)
def initialize_earth_engine():
    try:
        if "gee_service_account" in st.secrets:
            info = dict(st.secrets["gee_service_account"])
            credentials = ee.ServiceAccountCredentials(
                info["client_email"], key_data=json.dumps(info)
            )
            ee.Initialize(credentials, project=info.get("project_id"))
            return True, "Authenticated using service account."
        ee.Initialize()
        return True, "Authenticated using local Earth Engine credentials."
    except Exception:
        try:
            ee.Authenticate()
            ee.Initialize()
            return True, "Authenticated via interactive Earth Engine login."
        except Exception as exc2:
            return False, (
                "Earth Engine authentication failed. Run 'earthengine authenticate' "
                "locally, or add a service account under st.secrets['gee_service_account']. "
                f"Details: {exc2}"
            )


# =========================================================================
# Administrative boundaries (FAO GAUL)
# =========================================================================

COUNTRY_NAME = "Pakistan"
LEVEL0 = "FAO/GAUL/2015/level0"
LEVEL1 = "FAO/GAUL/2015/level1"
LEVEL2 = "FAO/GAUL/2015/level2"


@st.cache_data(show_spinner=False, ttl=86400)
def get_country_geometry():
    fc = ee.FeatureCollection(LEVEL0).filter(ee.Filter.eq("ADM0_NAME", COUNTRY_NAME))
    return fc.geometry()


@st.cache_data(show_spinner=False, ttl=86400)
def get_provinces():
    fc = ee.FeatureCollection(LEVEL1).filter(ee.Filter.eq("ADM0_NAME", COUNTRY_NAME))
    return fc.aggregate_array("ADM1_NAME").distinct().sort().getInfo()


@st.cache_data(show_spinner=False, ttl=86400)
def get_districts(province_name):
    fc = ee.FeatureCollection(LEVEL2).filter(
        ee.Filter.And(
            ee.Filter.eq("ADM0_NAME", COUNTRY_NAME),
            ee.Filter.eq("ADM1_NAME", province_name),
        )
    )
    return fc.aggregate_array("ADM2_NAME").distinct().sort().getInfo()


def get_geometry(area_level, province_name=None, district_name=None):
    if area_level == "Pakistan":
        return get_country_geometry()
    if area_level == "Province":
        fc = ee.FeatureCollection(LEVEL1).filter(
            ee.Filter.And(
                ee.Filter.eq("ADM0_NAME", COUNTRY_NAME),
                ee.Filter.eq("ADM1_NAME", province_name),
            )
        )
        return fc.geometry()
    fc = ee.FeatureCollection(LEVEL2).filter(
        ee.Filter.And(
            ee.Filter.eq("ADM0_NAME", COUNTRY_NAME),
            ee.Filter.eq("ADM1_NAME", province_name),
            ee.Filter.eq("ADM2_NAME", district_name),
        )
    )
    return fc.geometry()


def get_feature(area_level, province_name=None, district_name=None):
    if area_level == "Pakistan":
        return ee.FeatureCollection(LEVEL0).filter(ee.Filter.eq("ADM0_NAME", COUNTRY_NAME))
    if area_level == "Province":
        return ee.FeatureCollection(LEVEL1).filter(
            ee.Filter.And(
                ee.Filter.eq("ADM0_NAME", COUNTRY_NAME),
                ee.Filter.eq("ADM1_NAME", province_name),
            )
        )
    return ee.FeatureCollection(LEVEL2).filter(
        ee.Filter.And(
            ee.Filter.eq("ADM0_NAME", COUNTRY_NAME),
            ee.Filter.eq("ADM1_NAME", province_name),
            ee.Filter.eq("ADM2_NAME", district_name),
        )
    )


# =========================================================================
# Sentinel-2 retrieval and cloud masking
# =========================================================================

S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
SCL_CLOUD_CLASSES = [3, 8, 9, 10]


def mask_s2_clouds(image):
    scl = image.select("SCL")
    mask = scl.remap(SCL_CLOUD_CLASSES, [0, 0, 0, 0], 1)
    return image.updateMask(mask)


def scale_s2(image):
    optical = image.select("B.*").multiply(0.0001)
    return image.addBands(optical, None, True)


def get_sentinel2_composite(aoi, start_date, end_date, cloud_cover=20):
    collection = (
        ee.ImageCollection(S2_COLLECTION)
        .filterBounds(aoi)
        .filterDate(str(start_date), str(end_date))
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", cloud_cover))
        .map(mask_s2_clouds)
        .map(scale_s2)
    )
    size = collection.size()
    composite = collection.median().clip(aoi)
    return composite, size


def get_true_color(composite):
    return composite.select(["B4", "B3", "B2"])


# =========================================================================
# Spectral indices
# =========================================================================

INDEX_INFO = {
    "NDVI": {"formula": "(NIR - RED) / (NIR + RED)", "purpose": "Vegetation health, crop monitoring, forest monitoring, drought analysis.", "resolution": "10 m"},
    "EVI": {"formula": "2.5 * (NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1)", "purpose": "Dense vegetation and agricultural monitoring with improved sensitivity.", "resolution": "10 m"},
    "SAVI": {"formula": "((NIR - RED) / (NIR + RED + 0.5)) * 1.5", "purpose": "Sparse vegetation, agricultural land, semi-arid regions.", "resolution": "10 m"},
    "NDWI": {"formula": "(GREEN - NIR) / (GREEN + NIR)", "purpose": "Surface water, water bodies, flood analysis, moisture monitoring.", "resolution": "10 m"},
    "MNDWI": {"formula": "(GREEN - SWIR1) / (GREEN + SWIR1)", "purpose": "Water extraction, urban water detection, flood mapping.", "resolution": "20 m"},
    "NDBI": {"formula": "(SWIR1 - NIR) / (SWIR1 + NIR)", "purpose": "Built-up area detection, urban expansion, urban growth analysis.", "resolution": "20 m"},
    "BSI": {"formula": "((SWIR1 + RED) - (NIR + BLUE)) / ((SWIR1 + RED) + (NIR + BLUE))", "purpose": "Bare land, soil exposure, land degradation, agricultural land analysis.", "resolution": "20 m"},
}


def compute_index(composite, index_name):
    if index_name == "NDVI":
        return composite.normalizedDifference(["B8", "B4"]).rename("NDVI")
    if index_name == "EVI":
        return composite.expression(
            "2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))",
            {"NIR": composite.select("B8"), "RED": composite.select("B4"), "BLUE": composite.select("B2")},
        ).rename("EVI")
    if index_name == "SAVI":
        return composite.expression(
            "((NIR - RED) / (NIR + RED + L)) * (1 + L)",
            {"NIR": composite.select("B8"), "RED": composite.select("B4"), "L": 0.5},
        ).rename("SAVI")
    if index_name == "NDWI":
        return composite.normalizedDifference(["B3", "B8"]).rename("NDWI")
    if index_name == "MNDWI":
        return composite.normalizedDifference(["B3", "B11"]).rename("MNDWI")
    if index_name == "NDBI":
        return composite.normalizedDifference(["B11", "B8"]).rename("NDBI")
    if index_name == "BSI":
        return composite.expression(
            "((SWIR1 + RED) - (NIR + BLUE)) / ((SWIR1 + RED) + (NIR + BLUE))",
            {"SWIR1": composite.select("B11"), "RED": composite.select("B4"), "NIR": composite.select("B8"), "BLUE": composite.select("B2")},
        ).rename("BSI")
    raise ValueError(f"Unsupported index: {index_name}")


# =========================================================================
# Environmental datasets: LST, rainfall, night lights, air pollution
# =========================================================================

ENV_INFO = {
    "LST": {"dataset": "MODIS/061/MOD11A2", "resolution": "1000 m", "formula": "LST(C) = LST_Kelvin * 0.02 - 273.15", "purpose": "Thermal environment, urban heat islands, drought stress.", "unit": "\u00b0C"},
    "Rainfall": {"dataset": "UCSB-CHG/CHIRPS/DAILY", "resolution": "5566 m (~0.05\u00b0)", "formula": "Accumulated daily precipitation over date range", "purpose": "Agriculture, flood monitoring, drought analysis, water resources.", "unit": "mm"},
    "NightLights": {"dataset": "NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG", "resolution": "463 m", "formula": "Average radiance composite", "purpose": "Urban development, economic activity proxy, electrification.", "unit": "nW/cm\u00b2/sr"},
    "NO2": {"dataset": "COPERNICUS/S5P/OFFL/L3_NO2", "resolution": "1113 m", "formula": "Mean tropospheric NO2 column", "purpose": "Air quality monitoring - satellite column observation, not ground concentration.", "unit": "mol/m\u00b2"},
    "SO2": {"dataset": "COPERNICUS/S5P/OFFL/L3_SO2", "resolution": "1113 m", "formula": "Mean SO2 column density", "purpose": "Air quality, industrial/volcanic emission monitoring.", "unit": "mol/m\u00b2"},
    "CO": {"dataset": "COPERNICUS/S5P/OFFL/L3_CO", "resolution": "1113 m", "formula": "Mean CO column density", "purpose": "Air quality, combustion and fire emission monitoring.", "unit": "mol/m\u00b2"},
}


def get_lst(aoi, start_date, end_date):
    collection = (
        ee.ImageCollection("MODIS/061/MOD11A2")
        .filterBounds(aoi)
        .filterDate(str(start_date), str(end_date))
        .select("LST_Day_1km")
    )
    size = collection.size()
    image = collection.mean().multiply(0.02).subtract(273.15).rename("LST").clip(aoi)
    return image, size


def get_rainfall(aoi, start_date, end_date):
    collection = (
        ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
        .filterBounds(aoi)
        .filterDate(str(start_date), str(end_date))
        .select("precipitation")
    )
    size = collection.size()
    image = collection.sum().rename("Rainfall").clip(aoi)
    return image, size


def get_night_lights(aoi, start_date, end_date):
    collection = (
        ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
        .filterBounds(aoi)
        .filterDate(str(start_date), str(end_date))
        .select("avg_rad")
    )
    size = collection.size()
    image = collection.mean().rename("NightLights").clip(aoi)
    return image, size


def get_air_pollution(aoi, start_date, end_date, pollutant):
    band_map = {
        "NO2": ("COPERNICUS/S5P/OFFL/L3_NO2", "tropospheric_NO2_column_number_density"),
        "SO2": ("COPERNICUS/S5P/OFFL/L3_SO2", "SO2_column_number_density"),
        "CO": ("COPERNICUS/S5P/OFFL/L3_CO", "CO_column_number_density"),
    }
    dataset, band = band_map[pollutant]
    collection = (
        ee.ImageCollection(dataset)
        .filterBounds(aoi)
        .filterDate(str(start_date), str(end_date))
        .select(band)
    )
    size = collection.size()
    image = collection.mean().rename(pollutant).clip(aoi)
    return image, size


# =========================================================================
# Visualization: palettes, folium map, legends
# =========================================================================

VIS_PARAMS = {
    "NDVI": {"min": -1, "max": 1, "palette": ["#a50026", "#d73027", "#f46d43", "#fdae61", "#fee08b", "#d9ef8b", "#a6d96a", "#66bd63", "#1a9850", "#006837"]},
    "EVI": {"min": -1, "max": 1, "palette": ["#a50026", "#f46d43", "#fee08b", "#a6d96a", "#1a9850", "#006837"]},
    "SAVI": {"min": -1, "max": 1, "palette": ["#a50026", "#f46d43", "#fee08b", "#a6d96a", "#1a9850", "#006837"]},
    "NDWI": {"min": -1, "max": 1, "palette": ["#8c510a", "#d8b365", "#f6e8c3", "#c7eae5", "#5ab4ac", "#01665e"]},
    "MNDWI": {"min": -1, "max": 1, "palette": ["#8c510a", "#d8b365", "#f6e8c3", "#c7eae5", "#5ab4ac", "#01665e"]},
    "NDBI": {"min": -1, "max": 1, "palette": ["#ffffcc", "#ffeda0", "#feb24c", "#fc4e2a", "#e31a1c", "#800026"]},
    "BSI": {"min": -1, "max": 1, "palette": ["#ffffe5", "#fff7bc", "#fee391", "#fec44f", "#d95f0e", "#993404"]},
    "LST": {"min": 10, "max": 50, "palette": ["#313695", "#4575b4", "#74add1", "#abd9e9", "#fee090", "#fdae61", "#f46d43", "#d73027", "#a50026"]},
    "Rainfall": {"min": 0, "max": 500, "palette": ["#ffffcc", "#c7e9b4", "#7fcdbb", "#41b6c4", "#1d91c0", "#225ea8", "#0c2c84"]},
    "NightLights": {"min": 0, "max": 60, "palette": ["#000000", "#3b0f70", "#8c2981", "#de4968", "#fe9f6d", "#fcfdbf"]},
    "NO2": {"min": 0, "max": 0.0002, "palette": ["#f7fcf0", "#ccebc5", "#7bccc4", "#2b8cbe", "#08589e"]},
    "SO2": {"min": 0, "max": 0.0005, "palette": ["#fff7ec", "#fdbb84", "#fc8d59", "#d7301f", "#7f0000"]},
    "CO": {"min": 0, "max": 0.05, "palette": ["#f7fcfd", "#bfd3e6", "#8c96c6", "#88419d", "#4d004b"]},
}

LEGEND_LABELS = {
    "NDVI": ("Low vegetation", "High vegetation"), "EVI": ("Low vegetation", "High vegetation"),
    "SAVI": ("Low vegetation", "High vegetation"), "NDWI": ("Dry", "Water"), "MNDWI": ("Dry", "Water"),
    "NDBI": ("Non built-up", "Built-up"), "BSI": ("Vegetated/other", "Bare soil"),
    "LST": ("Cool", "Hot"), "Rainfall": ("Low", "High"), "NightLights": ("Dark", "Bright"),
    "NO2": ("Low", "High"), "SO2": ("Low", "High"), "CO": ("Low", "High"),
}


def get_tile_url(ee_image, vis_params):
    map_id = ee.Image(ee_image).getMapId(vis_params)
    return map_id["tile_fetcher"].url_format


def build_map(center_lat=30.3753, center_lon=69.3451, zoom=6):
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=zoom, tiles="CartoDB positron", control_scale=True)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(fmap)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="Satellite Basemap",
    ).add_to(fmap)
    return fmap


def add_ee_layer(fmap, ee_image, vis_params, name):
    tile_url = get_tile_url(ee_image, vis_params)
    folium.TileLayer(tiles=tile_url, attr="Google Earth Engine", name=name, overlay=True, control=True).add_to(fmap)


def add_aoi_layer(fmap, geojson, name="Selected AOI"):
    folium.GeoJson(
        geojson, name=name,
        style_function=lambda x: {"fillColor": "transparent", "color": "red", "weight": 3},
    ).add_to(fmap)


def add_legend(fmap, index_name, unit=""):
    if index_name not in VIS_PARAMS:
        return
    palette = VIS_PARAMS[index_name]["palette"]
    vmin = VIS_PARAMS[index_name]["min"]
    vmax = VIS_PARAMS[index_name]["max"]
    low_label, high_label = LEGEND_LABELS.get(index_name, ("Low", "High"))
    gradient = ", ".join(palette)
    legend_html = f"""
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999;
                background: white; padding: 10px 14px; border-radius: 6px;
                box-shadow: 0 1px 6px rgba(0,0,0,0.3); font-size: 13px; font-family: sans-serif;">
        <b>{index_name} {f'({unit})' if unit else ''}</b><br>
        <div style="width: 200px; height: 14px; margin-top: 6px;
                    background: linear-gradient(to right, {gradient});
                    border: 1px solid #999;"></div>
        <div style="display:flex; justify-content: space-between; width: 200px; margin-top: 2px;">
            <span>{vmin} {low_label}</span><span>{vmax} {high_label}</span>
        </div>
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend_html))


# =========================================================================
# Exports: GeoTIFF, HTML, CSV
# =========================================================================

def start_geotiff_export(image, aoi, description, scale, folder="EarthEngineExports"):
    task = ee.batch.Export.image.toDrive(
        image=image, description=description[:100], folder=folder,
        fileNamePrefix=description[:100], region=aoi, scale=scale,
        maxPixels=1e13, fileFormat="GeoTIFF",
    )
    task.start()
    return task


def map_to_html(fmap):
    return fmap.get_root().render()


def df_to_csv_bytes(df: pd.DataFrame):
    return df.to_csv(index=False).encode("utf-8")


# =========================================================================
# Application constants and state
# =========================================================================

SPECTRAL_INDICES = ["NDVI", "EVI", "SAVI", "NDWI", "MNDWI", "NDBI", "BSI"]
ENV_ANALYSES = ["LST", "Rainfall", "NightLights", "NO2", "SO2", "CO"]
ALL_ANALYSES = SPECTRAL_INDICES + ENV_ANALYSES


def init_session_state():
    defaults = {"results": {}, "cfg": None, "last_map": None}
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def header():
    st.title("\U0001F6F0\uFE0F Pakistan Multi-Index Satellite Remote Sensing Tool")
    st.caption("Automated satellite-based environmental and land-surface analysis for Pakistan")


def about_section():
    st.markdown(
        "This application provides automated satellite-based environmental and "
        "land-surface analysis for Pakistan using Google Earth Engine. Select a "
        "district, province, or the entire country, choose a date range and analysis, "
        "and generate interactive maps without manually downloading satellite imagery."
    )
    cols = st.columns(4)
    cards = [
        ("\U0001F6F0\uFE0F Satellite Data", "Access large Earth observation datasets through Google Earth Engine."),
        ("\U0001F331 Vegetation", "NDVI, EVI, SAVI"),
        ("\U0001F4A7 Water", "NDWI, MNDWI"),
        ("\U0001F3D9\uFE0F Urban", "NDBI, BSI"),
        ("\U0001F321\uFE0F Climate", "LST and rainfall"),
        ("\U0001F32B\uFE0F Air Quality", "NO2, SO2, CO"),
        ("\U0001F4A1 Development", "Night-time lights"),
    ]
    for i, (title, desc) in enumerate(cards):
        with cols[i % 4]:
            st.markdown(f"**{title}**")
            st.caption(desc)
    st.info(
        "This tool is intended for research, education, visualization, and exploratory "
        "geospatial analysis. Satellite-derived indicators are subject to spatial, temporal, "
        "atmospheric, and sensor-related limitations. Results should be validated against "
        "ground observations before operational or high-stakes use."
    )


def sidebar_controls():
    st.sidebar.header("Area of Interest")
    area_level = st.sidebar.radio("Area Level", ["Pakistan", "Province", "District"], horizontal=False)

    province_name = None
    district_name = None

    if area_level in ("Province", "District"):
        with st.spinner("Loading provinces..."):
            provinces = get_provinces()
        province_name = st.sidebar.selectbox("Select Province", provinces)

    if area_level == "District" and province_name:
        with st.spinner("Loading districts..."):
            districts = get_districts(province_name)
        district_name = st.sidebar.selectbox("Select District", districts)

    st.sidebar.header("Date Range")
    default_end = dt.date.today()
    default_start = default_end - dt.timedelta(days=90)
    start_date = st.sidebar.date_input("Start Date", value=default_start, max_value=default_end)
    end_date = st.sidebar.date_input("End Date", value=default_end, max_value=default_end)

    st.sidebar.header("Analysis")
    selected = st.sidebar.multiselect("Select Analysis (multiple allowed)", ALL_ANALYSES, default=["NDVI"])

    air_pollutants = [a for a in selected if a in ("NO2", "SO2", "CO")]
    spectral_selected = [a for a in selected if a in SPECTRAL_INDICES]
    env_selected = [a for a in selected if a in ("LST", "Rainfall", "NightLights")]

    st.sidebar.header("Settings")
    cloud_cover = st.sidebar.slider("Cloud Cover (%)", 0, 100, 20)
    resolution = st.sidebar.selectbox("Resolution (m)", [10, 20, 30, 60], index=0)
    true_color = st.sidebar.checkbox("Include Sentinel-2 True Color Preview", value=True)

    generate = st.sidebar.button("\U0001F680 GENERATE ANALYSIS", use_container_width=True)

    return {
        "area_level": area_level, "province_name": province_name, "district_name": district_name,
        "start_date": start_date, "end_date": end_date, "spectral_selected": spectral_selected,
        "env_selected": env_selected, "air_pollutants": air_pollutants, "cloud_cover": cloud_cover,
        "resolution": resolution, "true_color": true_color, "generate": generate,
    }


def validate_inputs(cfg):
    if cfg["start_date"] >= cfg["end_date"]:
        st.error("Start date must be before end date.")
        return False
    if not (cfg["spectral_selected"] or cfg["env_selected"] or cfg["air_pollutants"]):
        st.error("Select at least one analysis from the sidebar.")
        return False
    if cfg["area_level"] == "Pakistan":
        st.warning("Country-wide processing may take longer to render and export.")
    return True


def resolve_area_label(cfg):
    if cfg["area_level"] == "Pakistan":
        return "Pakistan"
    if cfg["area_level"] == "Province":
        return cfg["province_name"]
    return cfg["district_name"]


def run_analysis(cfg):
    aoi = get_geometry(cfg["area_level"], cfg["province_name"], cfg["district_name"])
    feature = get_feature(cfg["area_level"], cfg["province_name"], cfg["district_name"])
    area_label = resolve_area_label(cfg)

    results = {"area_label": area_label, "start": str(cfg["start_date"]), "end": str(cfg["end_date"]), "layers": {}}

    try:
        geojson = feature.getInfo()
    except Exception as exc:
        st.error(f"Unable to retrieve the selected boundary. {exc}")
        return None

    needs_s2 = bool(cfg["spectral_selected"]) or cfg["true_color"]
    if needs_s2:
        composite, size = get_sentinel2_composite(aoi, cfg["start_date"], cfg["end_date"], cfg["cloud_cover"])
        try:
            n_images = size.getInfo()
        except Exception as exc:
            st.error(f"Earth Engine request failed: {exc}")
            return None
        if n_images == 0:
            st.error(
                "No suitable satellite imagery was found for the selected area/date/cloud "
                "settings.\n\nTry:\n- Increasing the date range\n- Increasing the cloud threshold\n"
                "- Selecting another dataset"
            )
            return None
        results["s2_image_count"] = n_images

        for idx in cfg["spectral_selected"]:
            img = compute_index(composite, idx)
            results["layers"][idx] = {"image": img, "scale": cfg["resolution"], "type": "spectral"}

        if cfg["true_color"]:
            results["layers"]["TrueColor"] = {"image": get_true_color(composite), "scale": cfg["resolution"], "type": "truecolor"}

    for env in cfg["env_selected"]:
        if env == "LST":
            img, size = get_lst(aoi, cfg["start_date"], cfg["end_date"])
        elif env == "Rainfall":
            img, size = get_rainfall(aoi, cfg["start_date"], cfg["end_date"])
        else:
            img, size = get_night_lights(aoi, cfg["start_date"], cfg["end_date"])
        try:
            n = size.getInfo()
        except Exception as exc:
            st.error(f"Earth Engine request failed for {env}: {exc}")
            continue
        if n == 0:
            st.warning(f"No {env} data found for the selected date range. Try widening the dates.")
            continue
        scale = 1000 if env == "LST" else (5566 if env == "Rainfall" else 463)
        results["layers"][env] = {"image": img, "scale": scale, "type": "environmental"}

    for pol in cfg["air_pollutants"]:
        img, size = get_air_pollution(aoi, cfg["start_date"], cfg["end_date"], pol)
        try:
            n = size.getInfo()
        except Exception as exc:
            st.error(f"Earth Engine request failed for {pol}: {exc}")
            continue
        if n == 0:
            st.warning(f"No {pol} observations found for the selected date range. Try widening the dates.")
            continue
        results["layers"][pol] = {"image": img, "scale": 1113, "type": "pollution"}

    results["aoi"] = aoi
    results["geojson"] = geojson
    return results


def compute_statistics(results):
    rows = []
    for name, layer in results["layers"].items():
        if layer["type"] == "truecolor":
            continue
        image = layer["image"]
        try:
            stats = image.reduceRegion(
                reducer=ee.Reducer.mean().combine(ee.Reducer.minMax(), sharedInputs=True).combine(ee.Reducer.stdDev(), sharedInputs=True),
                geometry=results["aoi"], scale=layer["scale"], maxPixels=1e13, bestEffort=True,
            ).getInfo()
        except Exception as exc:
            st.warning(f"Could not compute statistics for {name}: {exc}")
            continue
        prefix = name
        rows.append({
            "Layer": name,
            "Mean": stats.get(f"{prefix}_mean"),
            "Minimum": stats.get(f"{prefix}_min"),
            "Maximum": stats.get(f"{prefix}_max"),
            "Std Dev": stats.get(f"{prefix}_stdDev"),
        })
    return pd.DataFrame(rows)


def compute_timeseries(aoi, index_name, start_date, end_date, cloud_cover, resolution):
    start = dt.datetime.combine(start_date, dt.time.min)
    end = dt.datetime.combine(end_date, dt.time.min)
    n_months = max(1, (end.year - start.year) * 12 + (end.month - start.month) + 1)
    n_months = min(n_months, 24)

    records = []
    cursor = start.replace(day=1)
    for _ in range(n_months):
        period_start = cursor
        period_end = cursor.replace(year=cursor.year + 1, month=1) if cursor.month == 12 else cursor.replace(month=cursor.month + 1)

        if index_name in SPECTRAL_INDICES:
            composite, size = get_sentinel2_composite(aoi, period_start.date(), period_end.date(), cloud_cover)
            if size.getInfo() > 0:
                img = compute_index(composite, index_name)
                mean_val = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=aoi, scale=resolution, maxPixels=1e13, bestEffort=True).getInfo().get(index_name)
                records.append({"Date": period_start.strftime("%Y-%m-%d"), index_name: mean_val})
        elif index_name == "LST":
            img, size = get_lst(aoi, period_start.date(), period_end.date())
            if size.getInfo() > 0:
                mean_val = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=aoi, scale=1000, maxPixels=1e13, bestEffort=True).getInfo().get("LST")
                records.append({"Date": period_start.strftime("%Y-%m-%d"), index_name: mean_val})
        elif index_name == "Rainfall":
            img, size = get_rainfall(aoi, period_start.date(), period_end.date())
            if size.getInfo() > 0:
                sum_val = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=aoi, scale=5566, maxPixels=1e13, bestEffort=True).getInfo().get("Rainfall")
                records.append({"Date": period_start.strftime("%Y-%m-%d"), index_name: sum_val})

        cursor = period_end

    return pd.DataFrame(records)


def render_map_tab(results, cfg):
    fmap = build_map()
    add_aoi_layer(fmap, results["geojson"], name=f"{results['area_label']} boundary")

    for name, layer in results["layers"].items():
        if layer["type"] == "truecolor":
            vis = {"min": 0, "max": 0.3, "bands": ["B4", "B3", "B2"]}
            add_ee_layer(fmap, layer["image"], vis, "Sentinel-2 True Color")
        else:
            vis = VIS_PARAMS.get(name, {"min": 0, "max": 1, "palette": ["blue", "green", "red"]})
            add_ee_layer(fmap, layer["image"], vis, name)
            unit = ENV_INFO.get(name, {}).get("unit", "")
            add_legend(fmap, name, unit)

    LayerControl(collapsed=False).add_to(fmap)
    st_folium(fmap, width=None, height=560, returned_objects=[])
    st.session_state["last_map"] = fmap


def render_statistics_tab(results):
    df = compute_statistics(results)
    if df.empty:
        st.info("No statistics available for the current selection.")
        return
    st.dataframe(df, use_container_width=True)
    st.download_button("Download Statistics (CSV)", df_to_csv_bytes(df), file_name=f"{results['area_label']}_statistics.csv")


def render_timeseries_tab(results, cfg):
    candidates = [n for n in results["layers"] if n in SPECTRAL_INDICES + ["LST", "Rainfall"]]
    if not candidates:
        st.info("Select NDVI/EVI/SAVI/NDWI/MNDWI/NDBI/BSI, LST, or Rainfall to view a time series.")
        return
    index_choice = st.selectbox("Select layer for time series", candidates)
    if st.button("Compute Time Series"):
        with st.spinner("Computing monthly time series through Earth Engine..."):
            df = compute_timeseries(results["aoi"], index_choice, cfg["start_date"], cfg["end_date"], cfg["cloud_cover"], cfg["resolution"])
        if df.empty:
            st.warning("No valid observations found across the selected period.")
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["Date"], y=df[index_choice], mode="lines+markers", name=index_choice))
            fig.update_layout(title=f"{index_choice} Time Series - {results['area_label']}", xaxis_title="Date", yaxis_title=index_choice)
            st.plotly_chart(fig, use_container_width=True)
            st.download_button("Download Time Series (CSV)", df_to_csv_bytes(df), file_name=f"{results['area_label']}_{index_choice}_timeseries.csv")


def render_change_detection_tab(cfg):
    st.markdown("Compare an index between two periods and visualize the difference.")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Period A")
        a_start = st.date_input("Period A - Start", key="a_start", value=cfg["start_date"] - dt.timedelta(days=365))
        a_end = st.date_input("Period A - End", key="a_end", value=cfg["end_date"] - dt.timedelta(days=365))
    with col2:
        st.subheader("Period B")
        b_start = st.date_input("Period B - Start", key="b_start", value=cfg["start_date"])
        b_end = st.date_input("Period B - End", key="b_end", value=cfg["end_date"])

    change_index = st.selectbox("Index for change detection", SPECTRAL_INDICES, index=0)

    if st.button("Run Change Detection"):
        aoi = get_geometry(cfg["area_level"], cfg["province_name"], cfg["district_name"])
        with st.spinner("Retrieving both periods from Earth Engine..."):
            comp_a, size_a = get_sentinel2_composite(aoi, a_start, a_end, cfg["cloud_cover"])
            comp_b, size_b = get_sentinel2_composite(aoi, b_start, b_end, cfg["cloud_cover"])
            n_a, n_b = size_a.getInfo(), size_b.getInfo()

        if n_a == 0 or n_b == 0:
            st.error("No suitable imagery for one or both periods. Try wider date ranges or higher cloud thresholds.")
            return

        img_a = compute_index(comp_a, change_index)
        img_b = compute_index(comp_b, change_index)
        diff = img_b.subtract(img_a).rename(f"{change_index}_change")

        fmap = build_map()
        vis = VIS_PARAMS[change_index]
        diff_vis = {"min": -0.5, "max": 0.5, "palette": ["#b2182b", "#f7f7f7", "#2166ac"]}
        add_ee_layer(fmap, img_a, vis, f"{change_index} - Period A")
        add_ee_layer(fmap, img_b, vis, f"{change_index} - Period B")
        add_ee_layer(fmap, diff, diff_vis, f"{change_index} Change (B - A)")
        LayerControl(collapsed=False).add_to(fmap)
        st_folium(fmap, width=None, height=520, returned_objects=[])

        stats = diff.reduceRegion(
            reducer=ee.Reducer.mean().combine(ee.Reducer.minMax(), sharedInputs=True),
            geometry=aoi, scale=cfg["resolution"], maxPixels=1e13, bestEffort=True,
        ).getInfo()
        st.write("**Change Statistics**")
        st.json(stats)


def render_export_tab(results, cfg):
    if not results["layers"]:
        st.info("Run an analysis first.")
        return

    layer_names = [n for n in results["layers"] if results["layers"][n]["type"] != "truecolor"]
    if not layer_names:
        st.info("No exportable analytical layers in the current selection.")
        return

    choice = st.selectbox("Select layer to export", layer_names)
    layer = results["layers"][choice]
    filename = f"{results['area_label']}_{choice}_{results['start']}_{results['end']}".replace(" ", "_")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Export GeoTIFF to Google Drive"):
            try:
                task = start_geotiff_export(layer["image"], results["aoi"], filename, layer["scale"])
                st.success(f"Export task started: {filename}.tif -> Google Drive folder 'EarthEngineExports'. Task ID: {task.id}")
            except Exception as exc:
                st.error(f"Export failed: {exc}")

    with col2:
        if st.session_state.get("last_map") is not None:
            html_str = map_to_html(st.session_state["last_map"])
            st.download_button("Download Interactive Map (HTML)", data=html_str, file_name=f"{filename}_map.html", mime="text/html")
        else:
            st.caption("Generate the map in the Map tab first to enable HTML export.")

    df = compute_statistics(results)
    if not df.empty:
        st.download_button("Download Statistics (CSV)", df_to_csv_bytes(df), file_name=f"{filename}_statistics.csv")


def render_dataset_info_tab(results):
    for name in results["layers"]:
        if name in INDEX_INFO:
            info = INDEX_INFO[name]
            with st.expander(name):
                st.write("**Dataset:** Sentinel-2 Surface Reflectance Harmonized")
                st.write(f"**Resolution:** {info['resolution']}")
                st.write(f"**Formula:** {info['formula']}")
                st.write(f"**Purpose:** {info['purpose']}")
        elif name in ENV_INFO:
            info = ENV_INFO[name]
            with st.expander(name):
                st.write(f"**Dataset:** {info['dataset']}")
                st.write(f"**Resolution:** {info['resolution']}")
                st.write(f"**Formula:** {info['formula']}")
                st.write(f"**Purpose:** {info['purpose']}")

    if any(n in ("NO2", "SO2", "CO") for n in results["layers"]):
        st.warning(
            "Sentinel-5P atmospheric products represent satellite observations of atmospheric "
            "constituents and should not automatically be interpreted as direct ground-level "
            "pollutant concentrations."
        )
    if "LST" in results["layers"]:
        st.warning("Land Surface Temperature is not the same as near-surface air temperature.")


def main():
    init_session_state()
    header()

    ok, message = initialize_earth_engine()
    if not ok:
        st.error(message)
        st.stop()

    with st.expander("About this tool", expanded=not st.session_state["results"]):
        about_section()

    cfg = sidebar_controls()

    if cfg["generate"]:
        if validate_inputs(cfg):
            with st.spinner("Retrieving and processing satellite data from Google Earth Engine..."):
                results = run_analysis(cfg)
            if results is not None:
                st.session_state["results"] = results
                st.session_state["cfg"] = cfg
                st.success(f"Analysis complete for {results['area_label']} ({results['start']} to {results['end']}).")

    results = st.session_state.get("results")
    cfg_saved = st.session_state.get("cfg") or cfg

    if results:
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
            ["\U0001F5FA\uFE0F Map", "\U0001F4CA Statistics", "\U0001F4C8 Time Series", "\U0001F504 Change Detection", "\U0001F4E5 Export", "\U0001F4D6 Dataset Info"]
        )
        with tab1:
            render_map_tab(results, cfg_saved)
        with tab2:
            render_statistics_tab(results)
        with tab3:
            render_timeseries_tab(results, cfg_saved)
        with tab4:
            render_change_detection_tab(cfg_saved)
        with tab5:
            render_export_tab(results, cfg_saved)
        with tab6:
            render_dataset_info_tab(results)
    else:
        st.info("Configure your area, dates, and analysis in the sidebar, then click 'GENERATE ANALYSIS'.")


if __name__ == "__main__":
    main()
