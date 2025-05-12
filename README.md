# Multimodal Survival Modeling in the Age of Foundation Models

This is the codebase for the work "Multimodal Survival Modeling in the Age of Foundation Models" by Song et al. 2025. The repo is divided into separate subdirectories each with their own READMEs and instructions. The order of our analysis with links to each subdirectory is outlined below:

1. [Download and Prepare Data](data)
1. [Process and Embed Data](embed)
1. [Model Survival](model)
1. [Inspect and Analyze Results](results)
1. [Manual Comparison Tool](tools)

## Environment
We use conda and pip for environment management. We recommend using [`miniforge`](https://github.com/conda-forge/miniforge) as a portable installation for conda. Regardless of which conda executable you use, environment installation is simply:
```bash
conda env create -f env.yml
conda activate survival
```

## Acknowledgements
This codebase is the culmination and adaptation of several individual components which were used for initial experimentation and hyperparameter tuning, namely:
* https://github.com/mbwangfpdc/tcga-survival
* https://github.com/mbwangfpdc/tcga-summarize
* https://github.com/imadejski/tcga-survival-prediction
* https://github.com/uc-cdis/multimodal-pathology

## Citation
```
TODO: add bibtex citation
```
