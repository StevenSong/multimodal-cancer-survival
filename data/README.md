# Data Preparation

This folder contains tools and instructions for preparing data for our experiments.

### Prerequisites
1. Setup and activate the conda environment
1. Download and install the [GDC Data Transfer Tool](https://gdc.cancer.gov/access-data/gdc-data-transfer-tool)

## Instructions
1. Download [`TCGA-Reports.csv`](https://data.mendeley.com/datasets/hyg5xkznpx/1), pre-extracted by [Kefeli and Tatonetti 2024](https://doi.org/10.1016/j.patter.2024.100933)
1. Compute intersection of cases based on data available for all modalities. Also download clinical data:
    ```bash
    python data-tool.py prepare \
    --reports-path /path/to/TCGA-Reports.csv \
    --clinical-data /path/to/save/clinical.csv \
    --expr-manifest /path/to/save/expr-manifest.txt \
    --hist-manifest /path/tosave/hist-manifest.txt
    ```
1. Download gene expression data using the GDC Data Transfer Tool and the prepared manifest:
    ```bash
    gdc-client download \
    -m /path/to/saved/expr-manifest.txt \
    -d /path/to/save/downloaded-expr
    ```
    * Check to make sure all files are downloaded. If some files fail to download, they can be reattempted by repeating the above command. The transfer tool will treat it as a partial download and will not reattempt files already downloaded.
    * For our experiments, we use pre-computed embeddings from UNI2-h. We therefore do not download the source histology slides using the manifest, however to do so is simple using `gdc-client`.
1. Download pre-computed TCGA histology embeddings from [`MahmoodLab/UNI2-h-features`](MahmoodLab/UNI2-h-features):
    ```bash
    huggingface-cli download \
    MahmoodLab/UNI2-h-features \
    --exclude 'CPTAC/*' 'PANDA/*' \
    --repo-type dataset \
    --local-dir /path/to/save/UNI2-h-features
    cd /path/to/saved/UNI2-h-features/TCGA
    for file in *.tar.gz; do tar xzvf "${file}" && rm "${file}"; done
    ```
1. Organize downloaded data:
    ```bash
    python data-tool.py organize \
    --reports-path /path/to/TCGA-Reports.csv \
    --downloaded-expr /path/to/saved/downloaded-expr \
    --downloaded-hist /path/to/saved/UNI2-h-features/TCGA \
    --organized-expr /path/to/save/organized-expr \
    --organized-hist /path/to/save/organized-hist
    ```
