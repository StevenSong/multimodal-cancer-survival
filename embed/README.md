# Data Embedding

Separate embedding scripts are provided for each modality, described below. Additionally, we provide tooling to first summarize, then embed pathology reports. Our tools expect files to be organized according to the output from our [data preparation/organization stage](../data/README.md).

## Embed RNA-seq
We embed RNA-seq gene expression data using BulkRNABert by Gelard et al. 2025. We have minimally modified the authors' codebase for use in our analysis; our modified version is [forked here](https://github.com/StevenSong/BulkRNABert) and is installed via this project's [`requirements.txt`](../requirements.txt).

**NB:** You still need to download the `checkpoints` folder so that the model weights can be loaded for inference: https://github.com/instadeepai/multiomics-open-research/tree/main/checkpoints

Our version expects a GPU to accelerate inference. To embed the prepared RNA-seq data, run:
```bash
python embed_bulkrnabert.py \
--dataset-folder ../data/expr \
--output-h5 expr.h5 \
--gene-list ../data/bulkrnabert_gene_list.txt \
--rna-seq-column tpm_unstranded \
--model-name bulk_rna_bert_gtex_encode \
--weights-folder /path/to/BulkRNABert/checkpoints \
--aggregation mean
```

## Embed Histology
We use precomputed tile-level embeddings from UNI2 by Chen et al. 2024. Our tool aggregates tile embeddings into a slide-level embedding. To prepare histology embeddings, run:
```bash
python embed_uni2.py \
--dataset-folder ../data/hist \
--output-h5 hist.h5 \
--aggregation mean
```

## Embed Pathology Reports
We embed pathology reports using BioMistral by Labrak et al. 2024. This model was primarily chosen for its biomedical domain adaptation with relatively greater token context length of 2048, as opposed to more specific pathology domain (vision-)language models such as CONCH (length 128), MUSK (length 100), or PRISM (adapts BioGPT length 1024). To prepare pathology report embeddings, run:
```bash
python embed_biomistral.py \
--input-csv ../data/TCGA_Reports.csv \
--output-h5 text.h5
```

### Generate Summaries
While BioMistral allows us to use longer input texts, the information contained within the original pathology reports are often repeptitive and poorly organized in its raw form. We therefore use an LLM to generate summaries of the reports first, after which we can also embed the summarized text using the same utility as above. To generate and embed summaries, run:
```bash
python generate_summaries.py \
--input-csv ../data/TCGA_Reports.csv \
--output-csv ../data/summarized_reports.csv

python embed_biomistral.py \
--input-csv ../data/summarized_reports.csv \
--output-h5 summ.h5
```