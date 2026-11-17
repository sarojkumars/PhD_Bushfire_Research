# Chapter 5 — Topographic influences on bushfire severity

This repository contains research scripts associated with the investigation of topographic influences on bushfire severity during the 2019–2020 Black Summer bushfires in eastern Victoria, Australia. The scripts document the computational components of the research supporting Chapter 5 of the PhD thesis and the associated publication.

**Associated publication:** https://ieeexplore.ieee.org/document/10059148/

## Repository contents

| Directory | Research component |
|---|---|
| `01_GEE_fire_severity/` | Google Earth Engine scripts for Landsat 8 NBR, dNBR and RdNBR calculations |
| `02_ML_modelling/` | Random Forest (RF), Gradient Boosting (GB) and Support Vector Machine (SVM) classification using Python |
| `03_Results_visualisation/` | Visualisation of fire severity across landform, aspect and profile-curvature classes |
| `04_ArcGIS_processing/` | Summary of ArcGIS-based spatial data preparation, terrain analysis and geoprocessing |

## Research workflow

The research involved the integration of satellite-derived fire-severity indices, topographic variables and ML techniques to investigate the influence of terrain characteristics on the spatial distribution of bushfire severity.

1. **Fire-severity assessment:** Pre- and post-fire Landsat 8 imagery was processed using Google Earth Engine to derive spectral indices, including NBR, dNBR and RdNBR.

2. **Topographic data preparation:** Terrain attributes and associated spatial datasets were prepared and processed in ArcGIS Pro. This included the derivation and classification of topographic variables, landform classification, and preparation of spatial datasets for subsequent analysis.

3. **Spatial processing and correction:** Additional processing, including phenological correction of fire-severity indices, spatial integration, sampling, data extraction and preparation of model input datasets, was undertaken in ArcGIS Pro using GIS tools and manual geoprocessing operations.

4. **ML modelling:** RF, GB and SVM models were developed in Python to examine the relationships between topographic predictors and fire-severity classes. Different model configurations and parameter combinations were explored during model development.

5. **Results visualisation:** Python was used to visualise the variation in fire severity across landform types, aspect classes and profile-curvature categories.

## Model configurations and parameter selection

Model development involved manual experimentation with different hyperparameter combinations to assess model behaviour and identify suitable configurations. This included testing alternative parameter values, some of which remain in the scripts as commented code. Parameter selection was informed by the evaluation of alternative configurations during the research. The scripts represent selected model implementations and experimental settings rather than a complete record of every parameter combination tested. 

The thesis and associated publication provide the final methodological descriptions, model configurations and reported results.

## Software and analytical environment

- **Python:** RF, GB and SVM modelling, model evaluation.
- **Google Earth Engine:** Landsat imagery processing and spectral fire-severity index calculations.
- **ArcGIS:** Visualisation, Terrain analysis, landform classification, phenological correction, spatial data processing, integration and preparation of modelling datasets.


## Research use

The scripts are provided as a record of the computational methods and analytical configurations explored during the research. They complement the methodological descriptions and results presented in Chapter 5 of the thesis and the associated peer-reviewed publication. The repository is intended to support research transparency and provide examples of the analytical workflows used in the study. 
