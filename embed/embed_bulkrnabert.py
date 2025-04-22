import argparse
import os
from pathlib import Path

import h5py
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from multiomics_open_research.bulk_rna_bert.preprocess import (
    preprocess_rna_seq_for_bulkrnabert,
    preprocess_tcga_rna_seq_dataset,
)
from multiomics_open_research.bulk_rna_bert.pretrained import get_pretrained_model
from tqdm import trange


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-folder", required=True)
    parser.add_argument("--output-h5", required=True)
    parser.add_argument("--gene-list", required=True)
    parser.add_argument("--rna-seq-column", default="tpm_unstranded")
    parser.add_argument("--model-name", default="bulk_rna_bert_gtex_encode")
    parser.add_argument("--weights-folder", required=True)
    args = parser.parse_args()
    return args


def main(args):
    with open(args.gene_list, "r") as f:
        reference_gene_ids = [line.strip() for line in f.readlines()]

    print("Preprocessing dataset")
    df = preprocess_tcga_rna_seq_dataset(
        dataset_path=Path(args.dataset_folder),
        output_file=None,
        reference_gene_ids=reference_gene_ids,
        rna_seq_column=args.rna_seq_column,
    )
    df = df.sort_values(["case_id", "identifier"]).reset_index(drop=True)

    parameters, forward_fn, tokenizer, config = get_pretrained_model(
        model_name=args.model_name,
        embeddings_layers_to_save=(4,),
        checkpoint_directory=args.weights_folder,
    )
    forward_fn = hk.transform(forward_fn)

    case_ids = df["case_id"].to_list()
    file_ids = df["identifier"].to_list()
    df = df.drop(columns=["identifier", "case_id"])

    rna_seq_array = preprocess_rna_seq_for_bulkrnabert(df, config)
    random_key = jax.random.PRNGKey(0)
    outs = []

    print("Generating embeddings")
    if os.path.exists(args.output_h5):
        print(f"Output H5 already exists, will not overwrite existing keys")
    with h5py.File(args.output_h5, mode="a") as h5:
        for i in trange(len(df)):
            case_id = case_ids[i]
            file_id = file_ids[i]
            if case_id in h5 and file_id in h5[case_id]:
                print(f"{case_id}/{file_id} already exists, skipping")
                continue
            elif case_id not in h5:
                h5.create_group(case_id)
            batch_array = rna_seq_array[i:i]  # requires batch dim
            tokens_ids = tokenizer.batch_tokenize(batch_array)
            tokens = jnp.asarray(tokens_ids, dtype=jnp.int32)
            outs = forward_fn.apply(parameters, random_key, tokens)
            embs = np.array(outs["embeddings_4"])
            emb = embs[0]  # unwrap batch dim
            h5[case_id].create_dataset(file_id, data=emb)


if __name__ == "__main__":
    args = parse_args()
    main(args)
