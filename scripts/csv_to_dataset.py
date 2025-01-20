# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import os

from mattergen.common.data.dataset import CrystalDataset
from mattergen.common.globals import PROJECT_ROOT

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv-folder",
        type=str,
        required=True,
        help="Path to the folder containing the csv files. All csv files in the folder will be processed (e.g., 'train.csv', 'val.csv', 'test.csv') and the resulting datasets will be placed under {cache_path/dataset_name/filename_without_extension}, e.g, /path/to/project/dataset/mp_20/train.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        help="Name of the dataset (e.g. mp_20. Will be used to create a folder in the cache folder)",
    )
    parser.add_argument(
        "--cache-folder",
        type=str,
        required=True,
        default=f"{PROJECT_ROOT}/datasets",
        help="Path to the cache folder. Defaults to datasets folder in the project root.",
    )
    args = parser.parse_args()
    for file in os.listdir(f"{args.csv_folder}"):
        if file.endswith(".csv"):
            print(f"Processing {args.csv_folder}/{file}")
            CrystalDataset.from_csv(
                csv_path=f"{args.csv_folder}/{file}",
                cache_path=f"{args.cache_folder}/{args.dataset_name}/{file.split('.')[0]}",
            )
