# 🇵🇰 Pakistan Multi-Index Satellite Remote Sensing Analysis Tool

A Streamlit web app to select a location in Pakistan, choose a date range and
satellite (Sentinel-2 or Landsat 8/9), and compute/visualize/export seven
spectral indices: **NDVI, EVI, NDWI, MNDWI, NDBI, BSI, SAVI**.

## Features
- Click-to-pick location map + adjustable AOI radius
- Date range picker, satellite selector (Sentinel-2 SR / Landsat 8 SR / Landsat 9 SR)
- All 7 indices computed and shown as colored map overlays
- Per-index stats (mean/min/max)
- Time-series chart across your date range + year-over-year month comparison
- Export everything (GeoTIFF + PNG + CSV) as one ZIP

## Three data modes
| Mode | What it needs | What you get |
|---|---|---|
| **Demo (synthetic data)** | Nothing — works instantly | Procedurally generated, location/date-seeded fake reflectance data, for testing the whole workflow end to end. Clearly labeled as synthetic in the UI. |
| **Upload GeoTIFF bands** | Your own single-band GeoTIFFs (Blue/Green/Red/NIR/SWIR1/SWIR2) | Indices computed on real data you already have (e.g. downloaded from Copernicus Open Access Hub / USGS EarthExplorer). |
| **Google Earth Engine (live)** | A free Earth Engine account + a service-account JSON key | Real, current Sentinel-2 / Landsat 8/9 imagery for any AOI in Pakistan, auto cloud-filtered and composited. |

> Why three modes? Live satellite APIs (Earth Engine, Sentinel Hub, Copernicus)
> require a personal account/login and network access that a hosted sandbox
> cannot hold or reach on your behalf. This app ships with fully working
> integration code for Earth Engine — you just need to run it somewhere with
> internet access and plug in your own free credentials, exactly as you would
> for any GIS project.

## 1. Install & run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Then open the local URL Streamlit prints (usually http://localhost:8501).

## 2. Run it "live" on the internet (streaming/deployed)
Easiest free option — **Streamlit Community Cloud**:
1. Push this folder to a new GitHub repo.
2. Go to https://share.streamlit.io → "New app" → pick the repo/branch → set
   main file to `app.py` → Deploy.
3. You'll get a public URL like `https://yourapp.streamlit.app` that anyone
   can open — this is your "streamed", always-on version of the tool.

Other options: Hugging Face Spaces (Streamlit SDK), Render, Railway, or your
own server (`streamlit run app.py --server.port 8501 --server.address 0.0.0.0`
behind a reverse proxy).

## 3. Getting real satellite data (Google Earth Engine mode)
1. Sign up free at https://earthengine.google.com (approval is usually instant/fast).
2. In Google Cloud Console, create a project + a **service account**, enable
   the Earth Engine API for it, and download a JSON key for that account.
3. In the app sidebar, choose "Google Earth Engine (live)" and upload that
   JSON key file.
4. Pick your AOI, date range, satellite and indices, then click **Compute
   indices**. The app filters by cloud cover, builds a median composite, and
   computes each index server-side on Earth Engine.

Alternative to a service account: run `earthengine authenticate` once on the
machine hosting the app (interactive OAuth), then leave the JSON upload empty.

## 4. Getting real data without Earth Engine (Upload mode)
Download band GeoTIFFs yourself from:
- **Copernicus Browser** (dataspace.copernicus.eu) for Sentinel-2 L2A bands
- **USGS EarthExplorer** (earthexplorer.usgs.gov) for Landsat 8/9 Collection 2 Level-2 SR bands

Then upload the Blue/Green/Red/NIR (and SWIR1/SWIR2 if you have them) bands
in the sidebar and click Compute.

## Index formulas used
| Index | Formula | Use |
|---|---|---|
| NDVI | (NIR−Red)/(NIR+Red) | Vegetation/greenness |
| EVI | 2.5·(NIR−Red)/(NIR+6·Red−7.5·Blue+1) | Enhanced vegetation |
| NDWI | (Green−NIR)/(Green+NIR) | Surface water/moisture |
| MNDWI | (Green−SWIR1)/(Green+SWIR1) | Water extraction (urban-robust) |
| NDBI | (SWIR1−NIR)/(SWIR1+NIR) | Built-up areas |
| BSI | ((SWIR1+Red)−(NIR+Blue))/((SWIR1+Red)+(NIR+Blue)) | Bare soil |
| SAVI | (NIR−Red)(1+L)/(NIR+Red+L), L=0.5 | Vegetation, soil-adjusted |

## Project structure
```
app.py                    Streamlit UI
utils/indices.py          NumPy index formulas (used by Demo & Upload modes)
utils/ee_indices.py       Earth Engine (ee.Image) index formulas (GEE mode)
utils/gee_utils.py        Earth Engine auth + data fetching
utils/raster_utils.py     GeoTIFF/PNG rendering + ZIP export
utils/synthetic_data.py   Demo-mode synthetic band generator
requirements.txt
```

## Notes & limitations
- Demo-mode data is **not real** — it's for exercising the UI/workflow only.
- Earth Engine's `sampleRectangle` (used here for simplicity) is fine for
  small-to-medium AOIs; for very large regions, switch to `ee.batch.Export`
  to Google Drive/Cloud Storage instead of pulling arrays in-app.
- All exported GeoTIFFs are single-band float32, EPSG:4326.
