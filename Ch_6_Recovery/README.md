# Chapter 6 — Post-fire vegetation recovery

Code supporting Chapter 6, *Post-fire vegetation recovery across spatial, ecological and temporal contexts*, of a PhD thesis investigating bushfire severity and post-fire vegetation response. The repository contains Google Earth Engine (GEE) JavaScript for Landsat spectral-index compositing and recovery metrics, alongside Python scripts for statistical analysis of the prepared recovery dataset. Additional GIS processing was undertaken in ArcGIS Pro.

## Research workflow

The analysis combines EO processing, spatial data preparation and statistical modelling:

1. **Google Earth Engine:** Landsat 8/9 Collection 2 Level-2 preprocessing; seasonal NBR, NDVI and EVI composites; a five-year pre-fire baseline (2015–2019); NBR background-offset correction; normalised recovery trajectories and summary metrics; and multiband GeoTIFF exports. The supplied script uses a February–April window. Corresponding seasonal configurations were also used in the research, but are not included here.
2. **ArcGIS Pro:** Further spatial preprocessing and integration of exported EO products with the study-area datasets, including RdNBR-derived initial fire-severity classification into **Low** and **High** (`Sev2Class`) and preparation of spatial analysis variables. These GIS operations were carried out outside the included JavaScript and Python scripts.
3. **Python (Spyder):** Preparation of the long-format analysis table, recovery modelling across temporal, landform and vegetation contexts, canopy-height and post-fire climate analyses, and production of figures and tables.

The Python scripts operate on the prepared analysis dataset. The ArcGIS processing steps are described here as part of the research workflow; ArcGIS models and processing files are not included in this code collection.

## Repository contents

| File | Purpose |
| --- | --- |
| `GEE/GEE_Annual_Spectral_Indices_and_Recovery.js` | Landsat 8/9 seasonal NBR/NDVI/EVI composites, recovery trajectories and metrics, and GeoTIFF exports |
| `00_Config_v2.py` | Shared input/output locations and analysis settings |
| `01_Prepare_Data_v2.py` | Prepare and check the long-format recovery dataset |
| `02_Recovery_Models_v2.py` | Model recovery across severity, time, landform and vegetation contexts |
| `03_CHM_Analysis_v2.py` | Examine canopy-height relationships within EFWL |
| `04_Climate_Scenarios_v2.py` | Analyse post-fire climate scenarios |
| `05_Final_Figures_Tables_v2.py` | Produce principal figures and tables |
| `06_Supplementary_Recovery_v2.py` | Supplementary recovery analyses |
| `07_Supplementary_Climate_v2.py` | Supplementary climate analyses and figures |

## Input data and analysis settings

The Python workflow reads a prepared long-format CSV with repeated observations for recovery stages **T1–T5**. Key fields include sample identifier, time, recovery response, fire severity, landform, vegetation, canopy height and post-fire climate variables. The preparation script accepts `Sev2Class` as the RdNBR-derived severity field, with values `Low` and `High`. See `01_Prepare_Data_v2.py` for supported input column names.

Input and output locations are configured locally through environment variables:

- `RECOVERY_DATA_DIR` — folder containing `02_Long_Format/Recovery_Climate_Long.csv`.
- `RECOVERY_OUTPUT_DIR` — folder for analysis-ready data, model outputs, figures, tables and supplementary results.

The paths are not embedded in the public code. Example settings for a Spyder IPython Console:

```python
import os
os.environ["RECOVERY_DATA_DIR"] = r"YOUR_PRIVATE_DATA_FOLDER"
os.environ["RECOVERY_OUTPUT_DIR"] = r"YOUR_PRIVATE_OUTPUT_FOLDER"
```

Replace these placeholders locally. Re-enter the settings if the Spyder kernel is restarted.

## Running the Python analysis

Install the packages listed in `requirements.txt` in the Python environment used by Spyder. Keep the eight Python scripts in the same folder, and run `01_Prepare_Data_v2.py` through `07_Supplementary_Climate_v2.py` in numerical order. `00_Config_v2.py` is loaded by the analysis scripts and does not need to be run separately.

The scripts write their outputs to the configured `RECOVERY_OUTPUT_DIR`, retaining the analysis subfolder structure defined in `00_Config_v2.py`.

## GEE spectral-index and recovery workflow

`GEE/GEE_Annual_Spectral_Indices_and_Recovery.js` is the supplied Landsat 8/9 processing script. It uses the imported `studyArea` geometry, 100 m export scale and EPSG:3577 projection. Its configured seasonal window is **1 February–30 April** (exclusive end date: 1 May). Five pre-fire annual composites (2015–2019) form a pixel-wise median baseline; a separate 2019 composite supports baseline sensitivity analysis. The post-fire composite is 2020 and recovery stages T1–T5 are 2021–2025.

The script calculates NBR, NDVI and EVI, applying a background/phenological offset **to NBR only**, using an exterior 180 m reference ring with a 1,000 m fallback. It calculates normalised recovery for NBR, NDVI and EVI, T5 magnitude, pixel-wise linear recovery rate, and NBR persistence and sustained time-to-80%-recovery. Three multiband GeoTIFF exports are configured: annual spectral indices, normalised recovery trajectories and recovery trajectory metrics. NDVI and EVI are retained as seasonally matched supporting indicators.

**Seasonal configurations:** The research used corresponding GEE configurations for other seasonal windows. Only the February–April version is supplied here; this repository does not assert that the other seasonal scripts or their exported rasters are included. The configured February–April period should not be interpreted as evidence that this was the selected final analysis season. Refer to the thesis for the seasonal-window evaluation and selected analysis period.

**ArcGIS Pro processing:** Exported EO rasters were used alongside additional ArcGIS Pro processing and spatial data integration. The final Low/High fire-severity classification for the Python analysis was derived from RdNBR in the GIS workflow; it is **not** generated by the GEE script. The script's NBR offset is an explicit GEE processing step; do not interpret it as documentation of every subsequent GIS correction or classification operation.

To run the GEE script, import a study-area asset named `studyArea` (or adjust the variable), check the date windows and export settings, execute it in the GEE Code Editor, and start the three export tasks. The outputs and GIS steps must be validated against the research data and thesis before claiming reproduction of the reported results.

## Research use

The repository documents the analytical code and its relationship to the wider thesis workflow. Research datasets and ArcGIS preprocessing files are maintained separately. Users seeking to apply the scripts to other study areas should review the input schema, temporal windows, classification definitions and analytical settings for their application.
