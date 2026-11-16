# Bushfire severity and post-fire vegetation recovery — Research code

This repository contains research code associated with a PhD thesis examining meteorological and topographic influences on bushfire severity, integrated environmental modelling, and post-fire vegetation recovery in Australia. The scripts complement the methods, results and interpretation presented in the thesis and associated publications.

## Research components

| Component | Research focus |
|---|---|
| [Chapter 4 — Meteorology](Chapter4_Meteorology/) | Meteorological analysis of the 2009 Black Saturday fires using BARRA-R, satellite observations, GIS, and statistical/ML modelling |
| [Chapter 5 — Topography](Chapter5_Topography/) | Terrain and landform influences on severity during the 2019–2020 Black Summer fires |
| [Chapter 6 — Recovery](Chapter6_Recovery/) | Post-fire vegetation recovery across temporal, spatial and ecological contexts |
| [Appendix A — Integrated modelling](AppendixA_IntegratedModelling/) | Environmental-domain comparisons, integrated severity modelling, ensemble analyses and model interpretation |

## Analytical workflow

The research combines RS processing, GIS, meteorological reanalysis, topographic and vegetation information, and statistical and ML analyses. Google Earth Engine supports spectral-index calculations and satellite-image analysis, including Landsat 8/9 seasonal recovery products for Chapter 6. ArcGIS Pro was used for terrain and landform classification, phenological/background processing where applicable, RdNBR-derived severity classification, spatial preparation and manual geoprocessing. The Chapter 6 GEE script also explicitly performs an NBR background offset; further GIS processing was undertaken separately. Python and R support data extraction, modelling, evaluation and visualisation; selected extraction tasks used HPC/SLURM.

The chapter and appendix folders contain their respective scripts, dependencies and methodological descriptions. Model settings and analytical examples are retained in the individual scripts. Where manual parameter experimentation was undertaken, the repositories document selected configurations rather than every combination considered. Consult the thesis and associated publications for the final reported analyses.

## Research use

This code is provided as a record of the computational components of the research. Prepared research datasets and GIS processing environments are maintained separately. Local paths and computing settings may need configuration for use in other environments.
