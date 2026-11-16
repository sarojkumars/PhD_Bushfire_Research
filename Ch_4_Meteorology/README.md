# Chapter 4 — Meteorological drivers of bushfire severity

This directory contains research scripts associated with the meteorological analysis of the 2009 Black Saturday bushfires, including BARRA-R meteorological data extraction, satellite-derived fire-severity indices and statistical modelling. The scripts document the computational components of the research and complement the methods described in Chapter 4 of the thesis and the associated publication.

## Repository contents

| Directory | Research component |
|---|---|
| `01_BARRA_R_extraction/` | Python scripts for extracting meteorological variables from BARRA-R NetCDF datasets at MODIS active-fire locations, organised by variable group |
| `02_HPC_SLURM/` | SLURM batch scripts for HPC-based meteorological data extraction |
| `03_Modelling/BRT/` | Boosted regression tree analysis in R |
| `03_Modelling/Fuzzy_Forest/` | Fuzzy Forest variable selection and associated analyses in R |
| `03_Modelling/Random_Forest/` | Random Forest modelling in Python |
| `03_Modelling/XGBoost/` | XGBoost modelling in Python |
| `04_GEE_fire_severity/` | Google Earth Engine JavaScript for Landsat 5 NBR, dNBR and RdNBR calculations |
| `05_ArcGIS_workflow/` | Documentation of GIS-based data preparation and analysis |

## Research workflow

1. **Fire observations and spatial preparation:** MODIS active-fire observations and associated spatial datasets were prepared for analysis using ArcGIS Pro.

2. **Meteorological data extraction:** Python scripts were used to extract meteorological variables from BARRA-R NetCDF datasets at MODIS active-fire locations. The scripts cover temperature, humidity, wind, atmospheric pressure, precipitation, soil moisture and surface fluxes. Selected extraction processes were executed on an HPC system using SLURM.

3. **Fire-severity assessment:** Google Earth Engine was used to process pre- and post-fire Landsat 5 imagery and derive NBR, dNBR and RdNBR.

4. **GIS processing:** Spatial data preparation, integration, analysis and phenological correction were undertaken in ArcGIS Pro using GIS tools and manual geoprocessing operations.

5. **Meteorological modelling:** Statistical and ML analyses were conducted using R and Python, including BRT, Fuzzy Forest, RF and XGBoost, to investigate the relationships between meteorological conditions and bushfire severity.

## Model configurations

The scripts retain representative model configurations used during the original analyses.

| Method | Configuration recorded in the script |
|---|---|
| BRT | Tree complexity = 5; learning rate = 0.004; bag fraction = 0.75; maximum trees = 10,000; initial trees = 100; 10-fold cross-validation |
| Fuzzy Forest | WGCNA power = 6; minimum module size = 10; drop fraction = 0.25; number of selected variables = 20; final trees = 5,000 |
| Random Forest | Number of estimators = 100; maximum depth = 5; random seed = 42; 80/20 training–testing split |
| XGBoost | Learning rate = 0.1; number of estimators = 100; maximum depth = 3; 80/20 training–testing split |

### Parameter selection

Model parameters were explored and refined through manual experimentation with different configurations in R and Python, supported by WEKA for additional parameter assessment. The scripts provided represent selected configurations from the original analyses rather than an exhaustive record of all parameter combinations evaluated. The specific settings are retained within the respective scripts.

## Software and computing environment

- **Python:** BARRA-R NetCDF extraction and RF/XGBoost modelling.
- **R:** BRT and Fuzzy Forest analyses.
- **WEKA:** Supporting ML parameter exploration and refinement.
- **Google Earth Engine:** Landsat image processing and spectral fire-severity index calculations.
- **ArcGIS Pro:** Spatial data preparation, phenological correction and GIS analysis.
- **SLURM/HPC:** Batch execution of meteorological data extraction scripts.

Python and R package dependencies are listed in `requirements-python.txt` and `requirements-R.txt`, respectively.

## Research use

These scripts are provided to document the computational approaches and analytical configurations employed during the research. They reflect the software environments and data-processing arrangements used at the time of the study. Some configuration may be required to execute the scripts in different computing environments.

For detailed methodological descriptions, experimental results and scientific interpretation, refer to Chapter 4 of the thesis and the associated publication.