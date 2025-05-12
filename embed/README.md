# Data Embedding

Separate embedding scripts are provided for each modality, described below. Additionally, we provide tooling to first summarize, then embed pathology reports. Our tools expect files to be organized according to the output from our [data preparation/organization stage](../data/README.md).

## Embed RNA-seq
We embed RNA-seq gene expression data using BulkRNABert by Gelard et al. 2025[1]. We have minimally modified the authors' codebase for use in our analysis; our modified version is [forked here](https://github.com/StevenSong/BulkRNABert) and is installed via this project's [`requirements.txt`](../requirements.txt).

**NB:** You still need to download the `checkpoints` folder so that the model weights can be loaded for inference: https://github.com/instadeepai/multiomics-open-research/tree/main/checkpoints. While Gelard et al. provide checkpoints for their model trained on TCGA data, we strictly use models which are not trained on TCGA to avoid data leakage.

Our version expects a GPU to accelerate inference. To embed the prepared RNA-seq data, run:
```bash
python embed_expr_bulkrnabert.py \
--dataset-folder ../data/expr \
--output-h5 expr.h5 \
--gene-list ../data/bulkrnabert_gene_list.txt \
--rna-seq-column tpm_unstranded \
--model-name bulk_rna_bert_gtex_encode \
--weights-folder /path/to/BulkRNABert/checkpoints \
--aggregation mean
```

## Embed Histology
We use precomputed tile-level embeddings from UNI2 by Chen et al. 2024[2]. Our tool aggregates tile embeddings into a slide-level embedding. To prepare histology embeddings, run:
```bash
python embed_hist_uni2.py \
--dataset-folder ../data/hist \
--output-h5 hist.h5 \
--aggregation mean
```

## Embed Pathology Reports
We embed pathology reports using BioMistral by Labrak et al. 2024[3]. This model was primarily chosen for its biomedical domain adaptation with relatively greater token context length of 2048, as opposed to more specific pathology domain (vision-)language models such as CONCH (length 128), MUSK (length 100), or PRISM (adapts BioGPT length 1024). To prepare pathology report embeddings, run:
```bash
python embed_text_biomistral.py \
--input-csv ../data/TCGA_Reports.csv \
--output-h5 text.h5
```

## Generate Summaries
While BioMistral allows us to use longer input texts, the information contained within the original pathology reports are often repeptitive and poorly organized in its raw form. We therefore use an LLM to generate summaries of the reports first, after which we can also embed the summarized text using the same utility as above. We generate summaries using Llama-3.1-8B-Instruct by Grattafiori et al. 2024[4]. This model was chosen for its strong general instruction following capabilities. To generate and embed summaries, run:
```bash
python generate_summaries.py \
--input-csv ../data/TCGA_Reports.csv \
--output-csv ../data/summarized_reports.csv

python embed_text_biomistral.py \
--input-csv ../data/summarized_reports.csv \
--output-h5 summ.h5
```

## Alternate Embeddings
We also experiment with other embedding models. Namely, we embed text with Mistral-7B-Instruct-v0.1 by Jiang et al. 2023[5] and expression data with Universal Cell Embedding (UCE) by Rosen et al. 2023[6]. These other models have corresponding scripts in this directory.

### UCE Setup
Embeddings using UCE require some additional setup. We have minimally modified the authors' codebase for use in our analysis; our modified version is [forked here](https://github.com/StevenSong/UCE) and is installed via this project's [`requirements.txt`](../requirements.txt).

You additionally need to download the model weights from FigShare and unzip/untar compressed files to a directory `uce_model_files`:
```bash
wget https://figshare.com/ndownloader/articles/24320806/versions/5 -O temp.zip
unzip temp.zip -d uce_model_files
tar -xvf uce_model_files/protein_embeddings.tar.gz -C uce_model_files
```

UCE embeddings (using their 33-layer model) are then extracted using the following script:
```bash
python embed_expr_bulkrnabert.py \
--dataset-folder ../data/expr \
--output-h5 expr-uce.h5 \
--weights-folder /path/to/uce_model_files \
```

## References
1. [BulkRNABert](https://proceedings.mlr.press/v259/gelard25a.html)
1. [UNI](https://www.nature.com/articles/s41591-024-02857-3)
1. [BioMistral](https://aclanthology.org/2024.findings-acl.348/)
1. [Llama 3.1](https://arxiv.org/abs/2407.21783)
1. [Mistral v0.1](https://arxiv.org/abs/2310.06825)
1. [UCE](https://www.biorxiv.org/content/10.1101/2023.11.28.568918v2)
