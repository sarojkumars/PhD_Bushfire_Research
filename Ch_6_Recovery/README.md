# Chapter 6 — Post-fire vegetation recovery

This folder contains research code supporting Chapter 6, Post-fire vegetation recovery across spatial, ecological and temporal contexts, of a PhD thesis investigating bushfire severity and post-fire vegetation response.

The analysis integrates EO data processing, GIS-based spatial analysis and statistical modelling to examine vegetation recovery following the 2019–2020 Australian bushfires.

## Research workflow

The analysis combines EO processing, spatial data preparation and statistical modelling:

1. **Google Earth Engine:**Processing Landsat imagery to derive spectral indices, assess post-fire vegetation recovery and export EO datasets.
2. **ArcGIS Pro:**  Spatial data preparation, integration of EO products, fire-severity classification and preparation of analysis datasets.
3. **Python (Spyder):** Statistical modelling of vegetation recovery across temporal, landform, vegetation, canopy-height and post-fire climate contexts, including visualisation of results.

The Python scripts operate on the prepared analysis dataset. The ArcGIS processing steps are described here as part of the research workflow.

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


## Running the code

The GEE scripts can be run in the Google Earth Engine Code Editor using the required study-area data and appropriate export settings.
The Python scripts use prepared analysis datasets. Install the packages listed in requirements.txt, configure local input and output directories using 00_Config_v2.py, and run the analysis scripts in numerical order.

## Research use

The repository documents the analytical code and its relationship to the wider thesis workflow. Research datasets and ArcGIS preprocessing files are maintained separately. Users seeking to apply the scripts to other study areas should review the input schema, temporal windows, classification definitions and analytical settings for their application.
