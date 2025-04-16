import argparse
import io
import json
import os
import shutil

import numpy as np
import pandas as pd
import requests

MAX_QUERY_SIZE = 1000000


def df_len_check(df):
    if len(df) == MAX_QUERY_SIZE:
        print(
            f"Retrieved entries equal to MAX_QUERY_SIZE ({MAX_QUERY_SIZE}), "
            "may be missing rows, consider increasing limit"
        )


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        title="mode",
        required=True,
        dest="mode",
        help="See mode-specific help for further options",
    )

    shared_parser = argparse.ArgumentParser(add_help=False)
    shared_parser.add_argument("--reports-path", required=True)

    prepare_parser = subparsers.add_parser("prepare", parents=[shared_parser])
    prepare_parser.add_argument("--clinical-data", required=True)
    prepare_parser.add_argument("--expr-manifest", required=True)
    # prepare_parser.add_argument("--hist-manifest", required=True)

    organize_parser = subparsers.add_parser("organize", parents=[shared_parser])
    organize_parser.add_argument("--downloaded-expr", required=True)
    organize_parser.add_argument("--downloaded-hist", required=True)
    organize_parser.add_argument("--organized-expr", required=True)
    organize_parser.add_argument("--organized-hist", required=True)

    args = parser.parse_args()
    return args


def main(args):
    clins, exprs, hists, texts = get_merged_metadata(args.reports_path)

    if args.mode == "prepare":
        clins.to_csv(args.clinical_data, index=False)

        # create manifest files for gdc data transfer tool
        exprs = exprs.rename(
            columns={"file_name": "filename", "file_size": "size", "md5sum": "md5"}
        )[["id", "filename", "md5", "size", "state"]]
        exprs.to_csv(args.expr_manifest, sep="\t", index=False)
        # No need for hist manifest since using precomputed embeddings
        # hists = hists.rename(
        #     columns={"file_name": "filename", "file_size": "size", "md5sum": "md5"}
        # )[["id", "filename", "md5", "size", "state"]]
        # hists.to_csv(args.hist_manifest, sep="\t", index=False)

        print()
        print(f"Clinical data saved to {args.clinical_data}")
        print()
        print(f"Expr manifest saved to {args.expr_manifest}")
        # print(f"Hist manifest saved to {args.hist_manifest}")
        print()
        print("Use the manifests with the GDC Data Transfer Tool:")
        print("https://gdc.cancer.gov/access-data/gdc-data-transfer-tool")
        print()
        print(
            "After the downloads complete, use the 'organize' mode "
            "of this tool to cleanup the downloaded data"
        )
        print()
    elif args.mode == "organize":
        for name, df, src_dir, dst_dir in [
            ("Expr", exprs, args.downloaded_expr, args.organized_expr),
            ("Hist", hists, args.downloaded_hist, args.organized_hist),
        ]:
            print(f"Organizing {name} data from {src_dir} to {dst_dir}")
            # construct list of expected files and their planned locations
            file_map = dict()
            for _, row in df.iterrows():
                file_name = row["file_name"]
                if name == "Hist":
                    # using precomputed embeddings
                    file_name = file_name.replace(".svs", ".h5")
                case_id = row["case_id"]
                case_path = os.path.join(dst_dir, case_id)
                os.makedirs(case_path, exist_ok=True)
                file_map[file_name] = os.path.join(case_path, file_name)

            # find and organize downloaded files
            for root, _, files in os.walk(src_dir):
                for file_name in files:
                    if file_name in file_map:
                        src_file = os.path.join(root, file_name)
                        dst_file = file_map.pop(file_name)
                        shutil.move(src=src_file, dst=dst_file)
            if len(file_map) > 0:
                not_found_csv = f"{name}-not-found.csv"
                print(f"Some files were not found, saving list to {not_found_csv}")
                not_found = pd.DataFrame(
                    [{"file_name": k, "dst_path": v} for k, v in file_map.items()]
                ).sort_values("file_name")
                not_found.to_csv(not_found_csv, index=False)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


def get_merged_metadata(reports_path):
    exprs, hists = get_expr_hist_metadata()
    clins = get_clin_metadata()

    texts = pd.read_csv(reports_path)
    texts["case_id"] = texts["patient_filename"].str[:12]

    # intersection of cases between all modalities
    clin_cases = set(clins["case_id"])
    expr_cases = set(exprs["case_id"])
    hist_cases = set(hists["case_id"])
    text_cases = set(texts["case_id"])
    cases = clin_cases & expr_cases & hist_cases & text_cases
    print(f"Using {len(cases)} cases across all modalities")

    clins = clins[clins["case_id"].isin(cases)].reset_index(drop=True)
    exprs = exprs[exprs["case_id"].isin(cases)].reset_index(drop=True)
    hists = hists[hists["case_id"].isin(cases)].reset_index(drop=True)
    texts = texts[texts["case_id"].isin(cases)].reset_index(drop=True)
    return clins, exprs, hists, texts


HIST_FILTER = {
    "op": "and",
    "content": [
        {
            "op": "in",
            "content": {
                "field": "files.experimental_strategy",
                "value": ["Diagnostic Slide"],
            },
        },
        {
            "op": "in",
            "content": {"field": "files.data_format", "value": ["SVS"]},
        },
    ],
}

EXPR_FILTER = {
    "op": "and",
    "content": [
        {
            "op": "in",
            "content": {"field": "files.experimental_strategy", "value": ["RNA-Seq"]},
        },
        {
            "op": "in",
            "content": {"field": "files.data_format", "value": ["TSV"]},
        },
        {
            "op": "in",
            "content": {
                "field": "files.data_type",
                "value": ["Gene Expression Quantification"],
            },
        },
    ],
}


def get_expr_hist_metadata():
    response = requests.get(
        "https://api.gdc.cancer.gov/files",
        params={
            "filters": json.dumps(
                {
                    "op": "and",
                    "content": [
                        {
                            "op": "in",
                            "content": {
                                "field": "cases.project.program.name",
                                "value": ["TCGA"],
                            },
                        },
                        {
                            "op": "in",
                            "content": {
                                "field": "cases.samples.tissue_type",
                                "value": ["Tumor"],
                            },
                        },
                        {
                            "op": "or",
                            "content": [HIST_FILTER, EXPR_FILTER],
                        },
                    ],
                }
            ),
            "fields": ",".join(
                [
                    "file_name",
                    "cases.project.project_id",
                    "cases.submitter_id",
                    "experimental_strategy",
                    "file_size",
                    "md5sum",
                    "state",
                ]
            ),
            "format": "TSV",
            "size": str(MAX_QUERY_SIZE),
        },
    )
    df = pd.read_csv(io.StringIO(response.text), sep="\t")
    df = df.rename(
        columns={
            "cases.0.project.project_id": "project",
            "cases.0.submitter_id": "case_id",
        }
    )
    df = df.sort_values(["project", "case_id"])
    df_len_check(df)

    exprs = df[df["experimental_strategy"] == "RNA-Seq"].reset_index(drop=True)
    hists = df[df["experimental_strategy"] == "Diagnostic Slide"].reset_index(drop=True)

    print("Retrieved Expr and Hist metadata")
    return exprs, hists


def get_clin_metadata():
    response = requests.get(
        "https://api.gdc.cancer.gov/cases",
        params={
            "filters": json.dumps(
                {
                    "op": "and",
                    "content": [
                        {
                            "op": "in",
                            "content": {
                                "field": "cases.project.program.name",
                                "value": ["TCGA"],
                            },
                        },
                    ],
                }
            ),
            "fields": ",".join(
                [
                    "project.project_id",
                    "submitter_id",
                    "demographic.days_to_death",
                    "demographic.vital_status",
                    "follow_ups.days_to_follow_up",
                    "demographic.ethnicity",
                    "demographic.gender",
                    "demographic.race",
                ]
            ),
            "format": "JSON",
            "size": str(MAX_QUERY_SIZE),
        },
    )
    data = []
    temp = json.loads(response.content)
    for datum in temp["data"]["hits"]:
        project = datum["project"]["project_id"]
        case_id = datum["submitter_id"]
        sex = np.nan
        race = np.nan
        ethnicity = np.nan
        vital_status = np.nan
        days_to_death = np.nan
        if "demographic" in datum:
            demo = datum["demographic"]
            if "vital_status" in demo:
                vital_status = demo.get("vital_status")
            if "days_to_death" in demo:
                days_to_death = demo["days_to_death"]
            if "gender" in demo:
                sex = demo["gender"]
            if "race" in demo:
                race = demo["race"]
            if "ethnicity" in demo:
                ethnicity = demo["ethnicity"]

        days_to_last_follow_up = np.nan
        if "follow_ups" in datum:
            for x in datum["follow_ups"]:
                fu_dt = x["days_to_follow_up"]
                if isinstance(fu_dt, int):
                    if (
                        np.isnan(days_to_last_follow_up)
                        or fu_dt > days_to_last_follow_up
                    ):
                        days_to_last_follow_up = fu_dt

        data.append(
            {
                "case_id": case_id,
                "project": project,
                "sex": sex,
                "race": race,
                "ethnicity": ethnicity,
                "vital_status": vital_status,
                "days_to_death": days_to_death,
                "days_to_last_follow_up": days_to_last_follow_up,
            }
        )
    clins = pd.DataFrame(data)
    df_len_check(clins)

    # dead = vital_status == "Dead"
    # d2d = days_to_death not nan
    # d2f = days_to_last_follow_up not nan
    # dead d2d d2f include
    #    T   T   T       T
    #    T   T   F       T
    #    T   F   T       F # dead xnor d2d
    #    T   F   F       F # dead xnor d2d
    #    F   T   T       F # dead xnor d2d
    #    F   T   F       F # dead xnor d2d
    #    F   F   T       T
    #    F   F   F       F # d2d or d2f
    clins = clins[
        ~((clins["vital_status"] == "Dead") ^ clins["days_to_death"].notna())
    ].reset_index(drop=True)
    clins = clins[
        clins["days_to_death"].notna() | clins["days_to_last_follow_up"].notna()
    ].reset_index(drop=True)

    for col in ["sex", "race", "ethnicity"]:
        clins[col] = clins[col].fillna("not reported")

    clins = clins.sort_values(["project", "case_id"]).reset_index(drop=True)

    print("Retrieved Clinical data")
    return clins


if __name__ == "__main__":
    args = parse_args()
    main(args)
