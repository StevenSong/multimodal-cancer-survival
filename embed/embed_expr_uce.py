import argparse
import os
import tempfile

import anndata
import h5py
import pandas as pd
from accelerate import Accelerator
from tqdm import tqdm, trange
from uce.evaluate import AnndataProcessor


def parse_args(tmp_dir):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-h5", required=True)
    parser.add_argument("--weights-folder", required=True)
    parser.add_argument("--preprocessed-cache", default="for_uce.h5ad")
    parser.add_argument("--batch-size", type=int, default=25)
    args = parser.parse_args()

    args.adata_path = args.preprocessed_cache
    args.dir = os.path.join(tmp_dir, "")  # trailing slash
    args.species = "human"
    args.filter = True
    args.skip = True
    args.nlayers = 33
    args.model_loc = os.path.join(args.weights_folder, "33l_8ep_1024t_1280.torch")
    args.spec_chrom_csv_path = os.path.join(args.weights_folder, "species_chrom.csv")
    args.token_file = os.path.join(args.weights_folder, "all_tokens.torch")
    args.protein_embeddings_dir = os.path.join(
        args.weights_folder, "protein_embeddings", ""
    )  # trailing slash
    args.offset_pkl_path = os.path.join(args.weights_folder, "species_offsets.pkl")
    args.model_files_path = args.weights_folder
    args.pad_length = 1536
    args.pad_token_idx = 0
    args.chrom_token_left_idx = 1
    args.chrom_token_right_idx = 2
    args.cls_token_idx = 3
    args.CHROM_TOKEN_OFFSET = 143574
    args.sample_size = 1024
    args.CXG = True
    args.output_dim = 1280
    args.d_hid = 5120
    args.token_dim = 5120
    args.multi_gpu = False
    return args


def prepare_adata_for_uce(dataset_path, preprocessed_cache, debug=None):
    fpaths = []
    for root, _, files in os.walk(dataset_path):
        for f in files:
            if f.endswith(".tsv"):
                case_id = root.replace(dataset_path, "").lstrip("/").split("/")[0]
                fpath = os.path.join(root, f)
                fpaths.append((case_id, fpath))
    if debug is not None:
        fpaths = fpaths[:debug]
    exps = []
    for case_id, fpath in tqdm(fpaths):
        df = pd.read_csv(fpath, sep="\t", skiprows=1)
        exp = df.loc[
            df["gene_type"] == "protein_coding", ["gene_name", "gene_id", "unstranded"]
        ]
        exp = exp.sort_values(["gene_name", "gene_id"]).set_index("gene_name")
        exp.index.name = None
        exp = exp.T.rename(index={"unstranded": case_id})
        exps.append(exp.loc[[case_id]].astype(int))
        for e in exps:
            assert (e.columns == exps[0].columns).all()
    X = pd.concat(exps)

    # sum together duplicate genes
    X = X.T.groupby(level=0).sum().T

    obs = pd.DataFrame(
        {"file_id": [os.path.basename(fpath).split(".")[0] for _, fpath in fpaths]},
        index=[case_id for case_id, _ in fpaths],
    )
    adata = anndata.AnnData(X, obs=obs)

    adata.write_h5ad(preprocessed_cache)
    return adata


def main(args):
    if os.path.exists(args.preprocessed_cache):
        print("Using cached preprocessed dataset")
        adata = anndata.read_h5ad(args.preprocessed_cache)
    else:
        print("Preprocessing dataset")
        adata = prepare_adata_for_uce(
            dataset_path=args.dataset_path,
            preprocessed_cache=args.preprocessed_cache,
            # debug=100,
        )

    print("Generating embeddings")
    accelerator = Accelerator(project_dir=args.dir)
    processor = AnndataProcessor(args, accelerator)
    processor.preprocess_anndata()
    processor.generate_idxs()
    processor.run_evaluation()

    print("Organizing results")
    uce_h5ad = os.path.join(
        tmp_dir,
        os.path.basename(args.preprocessed_cache).replace(".h5ad", "_uce_adata.h5ad"),
    )
    uce_adata = anndata.read_h5ad(uce_h5ad)

    df = uce_adata.obs.reset_index(names="case_id")

    if os.path.exists(args.output_h5):
        print(f"Output H5 already exists, will not overwrite existing keys")
    with h5py.File(args.output_h5, mode="a") as h5:
        for i in trange(len(df)):
            case_id = df["case_id"].iloc[i]
            file_id = df["file_id"].iloc[i]
            if case_id in h5 and file_id in h5[case_id]:
                print(f"{case_id}/{file_id} already exists, skipping")
                continue
            elif case_id not in h5:
                h5.create_group(case_id)
            emb = uce_adata.obsm["X_uce"][i]
            h5[case_id].create_dataset(file_id, data=emb)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp_dir:
        args = parse_args(tmp_dir)
        main(args)
