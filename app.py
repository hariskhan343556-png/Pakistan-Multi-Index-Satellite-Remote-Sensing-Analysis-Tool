import datetime as dt

import ee
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

from modules import boundaries, environmental, exports, gee_auth, indices, sentinel, visualization

st.set_page_config(
    page_title="Pakistan Multi-Index Satellite Remote Sensing Tool",
    page_icon="\U0001F6F0\uFE0F",
    layout="wide",
)

SPECTRAL_INDICES = ["NDVI", "EVI", "SAVI", "NDWI", "MNDWI", "NDBI", "BSI"]
ENV_ANALYSES = ["LST", "Rainfall", "NightLights", "NO2", "SO2", "CO"]
ALL_ANALYSES = SPECTRAL_INDICES + ENV_ANALYSES


def init_session_state():
    defaults = {
        "ee_ready": False,
        "results": {},
        "aoi": None,
        "aoi_geojson": None,
        "area_label": "",
        "s2_composite": None,
        "s2_size": 0,
    }
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
            provinces = boundaries.get_provinces()
        province_name = st.sidebar.selectbox("Select Province", provinces)

    if area_level == "District" and province_name:
        with st.spinner("Loading districts..."):
            districts = boundaries.get_districts(province_name)
        district_name = st.sidebar.selectbox("Select District", districts)

    st.sidebar.header("Date Range")
    default_end = dt.date.today()
    default_start = default_end - dt.timedelta(days=90)
    start_date = st.sidebar.date_input("Start Date", value=default_start, max_value=default_end)
    end_date = st.sidebar.date_input("End Date", value=default_end, max_value=default_end)

    st.sidebar.header("Analysis")
    selected = st.sidebar.multiselect(
        "Select Analysis (multiple allowed)",
        ALL_ANALYSES,
        default=["NDVI"],
    )

    air_pollutants = [a for a in selected if a in ("NO2", "SO2", "CO")]
    spectral_selected = [a for a in selected if a in SPECTRAL_INDICES]
    env_selected = [a for a in selected if a in ("LST", "Rainfall", "NightLights")]

    st.sidebar.header("Settings")
    cloud_cover = st.sidebar.slider("Cloud Cover (%)", 0, 100, 20)
    resolution = st.sidebar.selectbox("Resolution (m)", [10, 20, 30, 60], index=0)
    true_color = st.sidebar.checkbox("Include Sentinel-2 True Color Preview", value=True)

    generate = st.sidebar.button("\U0001F680 GENERATE ANALYSIS", use_container_width=True)

    return {
        "area_level": area_level,
        "province_name": province_name,
        "district_name": district_name,
        "start_date": start_date,
        "end_date": end_date,
        "spectral_selected": spectral_selected,
        "env_selected": env_selected,
        "air_pollutants": air_pollutants,
        "cloud_cover": cloud_cover,
        "resolution": resolution,
        "true_color": true_color,
        "generate": generate,
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
    aoi = boundaries.get_geometry(cfg["area_level"], cfg["province_name"], cfg["district_name"])
    feature = boundaries.get_feature(cfg["area_level"], cfg["province_name"], cfg["district_name"])
    area_label = resolve_area_label(cfg)

    results = {"area_label": area_label, "start": str(cfg["start_date"]), "end": str(cfg["end_date"]), "layers": {}}

    try:
        geojson = feature.getInfo()
    except Exception as exc:
        st.error(f"Unable to retrieve the selected boundary. {exc}")
        return None

    needs_s2 = bool(cfg["spectral_selected"]) or cfg["true_color"]
    if needs_s2:
        composite, size = sentinel.get_sentinel2_composite(
            aoi, cfg["start_date"], cfg["end_date"], cfg["cloud_cover"]
        )
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
            img = indices.compute_index(composite, idx)
            results["layers"][idx] = {"image": img, "scale": cfg["resolution"], "type": "spectral"}

        if cfg["true_color"]:
            results["layers"]["TrueColor"] = {
                "image": sentinel.get_true_color(composite),
                "scale": cfg["resolution"],
                "type": "truecolor",
            }

    for env in cfg["env_selected"]:
        if env == "LST":
            img, size = environmental.get_lst(aoi, cfg["start_date"], cfg["end_date"])
        elif env == "Rainfall":
            img, size = environmental.get_rainfall(aoi, cfg["start_date"], cfg["end_date"])
        else:
            img, size = environmental.get_night_lights(aoi, cfg["start_date"], cfg["end_date"])
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
        img, size = environmental.get_air_pollution(aoi, cfg["start_date"], cfg["end_date"], pol)
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
                reducer=ee.Reducer.mean()
                .combine(ee.Reducer.minMax(), sharedInputs=True)
                .combine(ee.Reducer.stdDev(), sharedInputs=True),
                geometry=results["aoi"],
                scale=layer["scale"],
                maxPixels=1e13,
                bestEffort=True,
            ).getInfo()
        except Exception as exc:
            st.warning(f"Could not compute statistics for {name}: {exc}")
            continue
        band = list(stats.keys())
        prefix = name
        mean_key = f"{prefix}_mean"
        min_key = f"{prefix}_min"
        max_key = f"{prefix}_max"
        std_key = f"{prefix}_stdDev"
        rows.append(
            {
                "Layer": name,
                "Mean": stats.get(mean_key),
                "Minimum": stats.get(min_key),
                "Maximum": stats.get(max_key),
                "Std Dev": stats.get(std_key),
            }
        )
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
        if cursor.month == 12:
            period_end = cursor.replace(year=cursor.year + 1, month=1)
        else:
            period_end = cursor.replace(month=cursor.month + 1)

        if index_name in SPECTRAL_INDICES:
            composite, size = sentinel.get_sentinel2_composite(aoi, period_start.date(), period_end.date(), cloud_cover)
            n = size.getInfo()
            if n > 0:
                img = indices.compute_index(composite, index_name)
                mean_val = img.reduceRegion(
                    reducer=ee.Reducer.mean(), geometry=aoi, scale=resolution, maxPixels=1e13, bestEffort=True
                ).getInfo().get(index_name)
                records.append({"Date": period_start.strftime("%Y-%m-%d"), index_name: mean_val})
        elif index_name == "LST":
            img, size = environmental.get_lst(aoi, period_start.date(), period_end.date())
            n = size.getInfo()
            if n > 0:
                mean_val = img.reduceRegion(
                    reducer=ee.Reducer.mean(), geometry=aoi, scale=1000, maxPixels=1e13, bestEffort=True
                ).getInfo().get("LST")
                records.append({"Date": period_start.strftime("%Y-%m-%d"), index_name: mean_val})
        elif index_name == "Rainfall":
            img, size = environmental.get_rainfall(aoi, period_start.date(), period_end.date())
            n = size.getInfo()
            if n > 0:
                sum_val = img.reduceRegion(
                    reducer=ee.Reducer.mean(), geometry=aoi, scale=5566, maxPixels=1e13, bestEffort=True
                ).getInfo().get("Rainfall")
                records.append({"Date": period_start.strftime("%Y-%m-%d"), index_name: sum_val})

        cursor = period_end

    return pd.DataFrame(records)


def render_map_tab(results, cfg):
    fmap = visualization.build_map()
    visualization.add_aoi_layer(fmap, results["geojson"], name=f"{results['area_label']} boundary")

    for name, layer in results["layers"].items():
        if layer["type"] == "truecolor":
            vis = {"min": 0, "max": 0.3, "bands": ["B4", "B3", "B2"]}
            visualization.add_ee_layer(fmap, layer["image"], vis, "Sentinel-2 True Color")
        else:
            vis = visualization.VIS_PARAMS.get(name, {"min": 0, "max": 1, "palette": ["blue", "green", "red"]})
            visualization.add_ee_layer(fmap, layer["image"], vis, name)
            unit = environmental.ENV_INFO.get(name, {}).get("unit", "")
            visualization.add_legend(fmap, name, unit)

    from folium import LayerControl
    LayerControl(collapsed=False).add_to(fmap)

    st_folium(fmap, width=None, height=560, returned_objects=[])
    st.session_state["last_map"] = fmap


def render_statistics_tab(results):
    df = compute_statistics(results)
    if df.empty:
        st.info("No statistics available for the current selection.")
        return
    st.dataframe(df, use_container_width=True)
    csv_bytes = exports.timeseries_to_csv(df)
    st.download_button("Download Statistics (CSV)", csv_bytes, file_name=f"{results['area_label']}_statistics.csv")


def render_timeseries_tab(results, cfg):
    candidates = [n for n in results["layers"] if n in SPECTRAL_INDICES + ["LST", "Rainfall"]]
    if not candidates:
        st.info("Select NDVI/EVI/SAVI/NDWI/MNDWI/NDBI/BSI, LST, or Rainfall to view a time series.")
        return
    index_choice = st.selectbox("Select layer for time series", candidates)
    if st.button("Compute Time Series"):
        with st.spinner("Computing monthly time series through Earth Engine..."):
            df = compute_timeseries(
                results["aoi"], index_choice, cfg["start_date"], cfg["end_date"], cfg["cloud_cover"], cfg["resolution"]
            )
        if df.empty:
            st.warning("No valid observations found across the selected period.")
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["Date"], y=df[index_choice], mode="lines+markers", name=index_choice))
            fig.update_layout(title=f"{index_choice} Time Series - {results['area_label']}", xaxis_title="Date", yaxis_title=index_choice)
            st.plotly_chart(fig, use_container_width=True)
            csv_bytes = exports.timeseries_to_csv(df)
            st.download_button("Download Time Series (CSV)", csv_bytes, file_name=f"{results['area_label']}_{index_choice}_timeseries.csv")


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
        aoi = boundaries.get_geometry(cfg["area_level"], cfg["province_name"], cfg["district_name"])
        with st.spinner("Retrieving both periods from Earth Engine..."):
            comp_a, size_a = sentinel.get_sentinel2_composite(aoi, a_start, a_end, cfg["cloud_cover"])
            comp_b, size_b = sentinel.get_sentinel2_composite(aoi, b_start, b_end, cfg["cloud_cover"])
            n_a, n_b = size_a.getInfo(), size_b.getInfo()

        if n_a == 0 or n_b == 0:
            st.error("No suitable imagery for one or both periods. Try wider date ranges or higher cloud thresholds.")
            return

        img_a = indices.compute_index(comp_a, change_index)
        img_b = indices.compute_index(comp_b, change_index)
        diff = img_b.subtract(img_a).rename(f"{change_index}_change")

        fmap = visualization.build_map()
        vis = visualization.VIS_PARAMS[change_index]
        diff_vis = {"min": -0.5, "max": 0.5, "palette": ["#b2182b", "#f7f7f7", "#2166ac"]}
        visualization.add_ee_layer(fmap, img_a, vis, f"{change_index} - Period A")
        visualization.add_ee_layer(fmap, img_b, vis, f"{change_index} - Period B")
        visualization.add_ee_layer(fmap, diff, diff_vis, f"{change_index} Change (B - A)")
        from folium import LayerControl
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
                task = exports.start_geotiff_export(layer["image"], results["aoi"], filename, layer["scale"])
                st.success(f"Export task started: {filename}.tif -> Google Drive folder 'EarthEngineExports'. Task ID: {task.id}")
            except Exception as exc:
                st.error(f"Export failed: {exc}")

    with col2:
        if "last_map" in st.session_state:
            html_str = exports.map_to_html(st.session_state["last_map"])
            st.download_button(
                "Download Interactive Map (HTML)",
                data=html_str,
                file_name=f"{filename}_map.html",
                mime="text/html",
            )
        else:
            st.caption("Generate the map in the Map tab first to enable HTML export.")

    df = compute_statistics(results)
    if not df.empty:
        st.download_button("Download Statistics (CSV)", exports.timeseries_to_csv(df), file_name=f"{filename}_statistics.csv")


def render_dataset_info_tab(results):
    for name, layer in results["layers"].items():
        if name in indices.INDEX_INFO:
            info = indices.INDEX_INFO[name]
            with st.expander(name):
                st.write(f"**Dataset:** Sentinel-2 Surface Reflectance Harmonized")
                st.write(f"**Resolution:** {info['resolution']}")
                st.write(f"**Formula:** {info['formula']}")
                st.write(f"**Purpose:** {info['purpose']}")
        elif name in environmental.ENV_INFO:
            info = environmental.ENV_INFO[name]
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

    ok, message = gee_auth.initialize_earth_engine()
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
    cfg_saved = st.session_state.get("cfg", cfg)

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
