import argparse
import os
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-folder", required=True)
    parser.add_argument("--output-h5", required=True)
    parser.add_argument("--aggregation", required=True, choices=["mean", "max"])
    args = parser.parse_args()
    return args


def main(args):
    files = []
    for root, _, fs in os.walk(args.dataset_folder):
        for f in fs:
            if f.endswith(".h5"):
                case_id = os.path.basename(root)
                file_id = f[:-3]
                files.append((case_id, file_id, os.path.join(root, f)))

    print("Generating slide-level embeddings")
    if os.path.exists(args.output_h5):
        print(f"Output H5 already exists, will not overwrite existing keys")
    with h5py.File(args.output_h5, mode="a") as h5:
        for case_id, file_id, file_path in tqdm(files):
            if case_id in h5 and file_id in h5[case_id]:
                print(f"{case_id}/{file_id} already exists, skipping")
                continue
            elif case_id not in h5:
                h5.create_group(case_id)
            with h5py.File(file_path, "r") as h5_in:
                tile_embs = h5_in["features"][:]  # 1 x num_patches x 1536
                tile_embs = np.squeeze(tile_embs)
                if args.aggregation == "mean":
                    emb = np.mean(tile_embs, axis=0)
                elif args.aggregation == "max":
                    emb = np.max(tile_embs, axis=0)
                else:
                    raise ValueError(f"Unknown aggregation method: {args.aggregation}")
            h5[case_id].create_dataset(file_id, data=emb)


if __name__ == "__main__":
    args = parse_args()
    main(args)
