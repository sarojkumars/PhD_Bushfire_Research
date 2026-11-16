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

Model development involved manual experimentation with different hyperparameter combinations to assess model behaviour and identify suitable configurations. This included testing alternative parameter values, some of which remain in the scripts as commented code.

The configurations shown below represent the active model settings retained in the accompanying scripts.

| Method | Model configuration |
|---|---|
| Random Forest (RF) | 300 trees; Gini criterion; maximum depth = 100; minimum samples split = 2; minimum samples leaf = 1; random state = 42 |
| Gradient Boosting (GB) | 500 trees; maximum depth = 7; learning rate = 0.1; minimum samples split = 3 |
| Support Vector Machine (SVM) | Radial basis function (RBF) kernel; C = 1000 |

Parameter selection was informed by the evaluation of alternative configurations during the research. The scripts represent selected model implementations and experimental settings rather than a complete record of every parameter combination tested. Consequently, individual parameter values in these scripts may differ from those reported for the final models in the publication.

The thesis and associated publication provide the final methodological descriptions, model configurations and reported results.

## Software and analytical environment

- **Python:** RF, GB and SVM modelling, model evaluation and results visualisation.
- **Google Earth Engine:** Landsat imagery processing and spectral fire-severity index calculations.
- **ArcGIS Pro:** Terrain analysis, landform classification, phenological correction, spatial data processing, integration and preparation of modelling datasets.

Python package dependencies are listed in `requirements-python.txt`.

## Running the Python scripts

The modelling scripts use prepared tabular datasets generated through the spatial processing workflow. The model input table contains topographic predictors and one-hot-encoded fire-severity classes.

Set `TOPO_INPUT_CSV` to the prepared modelling dataset. In the provided configuration, the first 18 columns represent predictors and the following four columns represent severity classes.

For visualisation, set `TOPO_VIS_CSV` to the prepared dataset containing the relevant topographic and fire-severity attributes, including `Lform`, `AspClass`, `ProfCurvCls1` and `RdNBR`.

The GEE scripts document the satellite-based spectral index calculations, while additional spatial corrections and processing were performed in ArcGIS Pro.

## Research use

The scripts are provided as a record of the computational methods and analytical configurations explored during the research. They complement the methodological descriptions and results presented in Chapter 5 of the thesis and the associated peer-reviewed publication.

The repository is intended to support research transparency and provide examples of the analytical workflows used in the study. Some configuration may be required when applying the scripts to different datasets or computing environments.