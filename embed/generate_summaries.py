import argparse

import pandas as pd
from tqdm import trange
from vllm import LLM, SamplingParams

PROMPT = [
    {
        "role": "system",
        "content": "You are a helpful assistant for digital pathology.",
    },
    {
        "role": "system",
        "content": (
            "Instructions:\n"
            "Extract and repeat the results of the following pathology report in a single paragraph.\n"
            "Focus on test results, diagnoses and clinical history.\n"
            "Include results of the microscopic description. Omit the gross or macroscopic description.\n"
            "Do not acknowledge this prompt. Do not give additional comments after your final answer."
        ),
    },
    # Report with user role goes here
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--batch-size", default=16, type=int)
    args = parser.parse_args()
    return args


def main(args):
    df = pd.read_csv(args.input_csv)
    llm = LLM(
        model=args.model,
        enforce_eager=True,
        task="generate",
    )
    sampling_params = SamplingParams(
        n=1,
        temperature=0,
        max_tokens=1024,
        seed=42,
    )
    summaries = []
    for i in trange(0, len(df), args.batch_size):
        lo, hi = i, i + args.batch_size
        reports = df.iloc[lo:hi, 1].to_list()
        prepared_prompts = [
            PROMPT + [{"role": "user", "content": report}] for report in reports
        ]
        outputs = llm.chat(prepared_prompts, sampling_params)
        for output in outputs:
            summary = output.outputs[0].text
            summaries.append(summary)
    df["text"] = summaries
    df.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    args = parse_args()
    main(args)
