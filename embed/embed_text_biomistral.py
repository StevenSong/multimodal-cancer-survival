import argparse
import os

import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import trange
from transformers import MistralModel
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
    parser.add_argument("--model-cache", default="model-cache")
    args = parser.parse_args()

    return args


def main(args):
    df = pd.read_csv(args.input_csv)

    if not os.path.exists(args.model_cache):  # need to sanitize state dict
        # BioMistral on HF is configured as MistralForCausalLM
        # and the transformer weights are prefixed with "model.".
        # vLLM requires the model be configured as MistralModel for embeddings
        # so load using huggingface (which takes care of weight prefixes too)
        # and save just the transformer backbone model.
        temp = MistralModel.from_pretrained(
            "BioMistral/BioMistral-7B",
            torch_dtype=torch.bfloat16,
        )
        temp.save_pretrained(args.model_cache, safe_serialization=False)  # TODO errors?
        del temp

    model = LLM(
        model=args.model_cache,
        tokenizer="BioMistral/BioMistral-7B",
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
