# Pakistan Multi-Index Satellite Remote Sensing Tool

An automated, browser-based satellite remote sensing dashboard for Pakistan, built with **Streamlit** and **Google Earth Engine (GEE)**. Select an area (country, province, or district), a date range, and one or more analyses — the app retrieves, cloud-masks, and processes satellite data entirely on Google's servers and returns an interactive map, statistics, time series, and export options. No manual downloading, cropping, or Earth Engine scripting required.

## 1. What It Does

The tool replaces the traditional remote-sensing workflow (search → download → crop → preprocess → script → GIS software) with a single click. It queries Earth Engine directly, applies cloud masking and index formulas server-side, and streams only the rendered map tiles and summary statistics back to your browser.

## 2. Features

- Area selection: full country, any province, or any district, using live Earth Engine administrative boundaries (no hard-coded lists)
- Multi-select analysis: choose several indices/products in one run
- Spectral indices: NDVI, EVI, SAVI, NDWI, MNDWI, NDBI, BSI (Sentinel-2)
- Environmental products: Land Surface Temperature, Rainfall, Night-Time Lights
- Air quality: NO2, SO2, CO (Sentinel-5P atmospheric column observations)
- Automatic cloud/cloud-shadow/cirrus masking using the Sentinel-2 Scene Classification Layer
- Interactive Folium map with layer control, legends, and a Sentinel-2 true-color preview
- Zonal statistics (mean, min, max, standard deviation) via Earth Engine reducers
- Monthly time-series charts (Plotly) computed server-side
- Before/after change detection between two custom periods
- Export: GeoTIFF (to Google Drive), interactive HTML map, and CSV statistics/time series
- Dataset information panel with formulas, resolution, and scientific caveats
- Defensive error handling for empty imagery, invalid dates, and large-AOI processing

## 3. Datasets Used

| Analysis | Dataset | Approx. Resolution |
|---|---|---|
| NDVI / EVI / SAVI / NDWI / MNDWI / NDBI / BSI | `COPERNICUS/S2_SR_HARMONIZED` | 10–20 m |
| Land Surface Temperature | `MODIS/061/MOD11A2` | 1000 m |
| Rainfall | `UCSB-CHG/CHIRPS/DAILY` | ~5.5 km |
| Night-Time Lights | `NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG` | 463 m |
| NO2 / SO2 / CO | `COPERNICUS/S5P/OFFL/L3_NO2` / `L3_SO2` / `L3_CO` | ~1.1 km |
| Administrative boundaries | `FAO/GAUL/2015/level0` / `level1` / `level2` | — |

## 4. Installation

```bash
git clone <your-repo-url>
cd pakistan_remote_sensing
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 5. Google Earth Engine Setup

1. Sign up for Earth Engine access at https://signup.earthengine.google.com (Google account required).
2. Authenticate locally:

```bash
earthengine authenticate
```

This opens a browser window to log in and stores a local credential the Python API will reuse automatically.

## 6. Local Execution

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`. Configure your area, dates, and analysis in the sidebar, then click **GENERATE ANALYSIS**.

## 7. Streamlit Community Cloud Deployment

Local `earthengine authenticate` credentials do not transfer to Streamlit Cloud. Instead, use a **Google Cloud service account**:

1. In Google Cloud Console, create a service account and grant it the "Earth Engine Resource Viewer" role (and register it for Earth Engine access at https://signup.earthengine.google.com/#!/service_accounts).
2. Generate a JSON key for that service account.
3. In your Streamlit Cloud app settings, open **Secrets** and paste the key contents following the structure in `.streamlit/secrets.toml.example` (rename to `secrets.toml` only for local testing — never commit real secrets).
4. Deploy. `modules/gee_auth.py` automatically detects `st.secrets["gee_service_account"]` and authenticates with it; it falls back to local credentials otherwise.

Never hard-code credentials, private keys, tokens, or passwords in the source code.

## 8. Export Functionality

- **GeoTIFF**: Starts an `ee.batch.Export.image.toDrive()` task; the file lands in a `EarthEngineExports` folder in the Google Drive of the authenticated account. Large AOIs (e.g. full-country) may take longer to complete — check the task status in the Earth Engine Code Editor "Tasks" tab or the Earth Engine Python task API.
- **HTML**: Downloads the current interactive Folium map (including layers, legend, and AOI outline) as a standalone HTML file.
- **CSV**: Downloads zonal statistics or time-series data.

## 9. Scientific Limitations

This tool is intended for research, education, visualization, and exploratory geospatial analysis. Satellite-derived indicators are subject to spatial, temporal, atmospheric, and sensor-related limitations. Results should be validated against appropriate ground observations before being used for operational or high-stakes decisions.

- Sentinel-5P atmospheric products (NO2, SO2, CO) represent satellite observations of atmospheric column density and should not be interpreted as direct ground-level pollutant concentrations.
- Land Surface Temperature (MODIS) is not equivalent to near-surface air temperature.
- Cloud masking relies on the Sentinel-2 Scene Classification Layer and may not remove all contamination in persistently cloudy periods.
- Coarser-resolution products (LST, rainfall, night lights, air quality) should not be interpreted at the same spatial precision as 10 m Sentinel-2 indices.

## 10. Future Improvements

- Flood monitoring workflow (SAR-based water extent using Sentinel-1)
- Automated urban growth tracking across multi-year composites
- Crop health and yield-proxy mapping for major agricultural districts
- Batch/report generation (PDF summaries per district)
- User accounts and saved analysis history
- Asynchronous export status tracking inside the app
