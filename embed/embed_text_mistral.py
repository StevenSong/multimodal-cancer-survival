import argparse
import os

import h5py
import numpy as np
import pandas as pd
from tqdm import trange
from vllm import LLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-csv",
        required=True,
        help="Path to report CSV.",
    )
    parser.add_argument(
        "--output-h5",
        required=True,
        help="Path to save extracted report features.",
    )
    args = parser.parse_args()

    return args


def main(args):
    df = pd.read_csv(args.input_csv)

    model = LLM(
        model="mistralai/Mistral-7B-Instruct-v0.1",
        task="embed",
        enforce_eager=True,
    )

    print("Generating report embeddings")
    if os.path.exists(args.output_h5):
        print(f"Output H5 already exists, will not overwrite existing keys")
    with h5py.File(args.output_h5, mode="a") as h5:
        for i in trange(len(df)):
            file_id = df.loc[i, "patient_filename"]
            case_id = file_id.split(".")[0]
            if case_id in h5 and file_id in h5[case_id]:
                print(f"{case_id}/{file_id} already exists, skipping")
                continue
            elif case_id not in h5:
                h5.create_group(case_id)
            report = df.loc[i, "text"]
            output = model.embed([report], use_tqdm=False)
            emb = output[0].outputs.embedding
            emb = np.asarray(emb, dtype=np.float32)
            h5[case_id].create_dataset(file_id, data=emb)


if __name__ == "__main__":
    args = parse_args()
    main(args)
