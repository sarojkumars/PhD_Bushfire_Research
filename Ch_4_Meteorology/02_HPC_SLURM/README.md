### HPC batch processing

High-performance computing (HPC) was used to process and extract more than 60+ meteorological and environmental parameters from BARRA-R datasets. SLURM batch scripts were used to manage computationally intensive extraction tasks, enabling efficient processing of large spatiotemporal datasets.

The `batch_hpc.slurm` file provides an example of the SLURM job configuration, including computational resource allocation, software modules and execution of Python extraction scripts. Similar batch-processing workflows were applied across the environmental parameters to support the preparation of predictor datasets for subsequent fire-severity modelling.