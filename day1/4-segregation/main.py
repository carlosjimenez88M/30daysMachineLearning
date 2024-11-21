#!usr/bin/env python
'''
Segregation System Model
Machine Learning Pipeline Operation
Edge Machine Learning (Raspberry Pi)
2024-11-21
'''


#=====================#
# ---- Libraries ---- #
#=====================#

import os
import logging
import wandb
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
import tempfile

#================================#
# ---- Logger Configuration ---- #
#================================#

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s : %(levelname)s : %(message)s')
logger = logging.getLogger()


#==============================#
# ---- Main Configuration ---- #
#==============================#

def go(args):
    logger.info('Initialize segregation process')
    run = wandb.init(project='end_to_end',
                     job_type='segregation')

    logger.info("Downloading and reading artifact")
    artifact = run.use_artifact(args.input_artifact)
    artifact_path = artifact.file()
    df = pd.read_csv(artifact_path, low_memory=False)

    logger.info("Splitting data into train, val, and test")
    splits = {}

    # Fixing the keys for proper naming
    splits["train"], splits["test"] = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=df[args.stratify] if args.stratify != 'null' else None,
    )
    with tempfile.TemporaryDirectory() as tmpdirname:
        for split, split_df in splits.items():  # Use appropriate variable names
            artifact_name = f"{args.artifact_root}_{split}.csv"
            temp_path = os.path.join(tmpdirname, artifact_name)

            logger.info(f"Uploading the {split} dataset to {artifact_name}")
            split_df.to_csv(temp_path, index=False)

            artifact = wandb.Artifact(
                name=artifact_name,
                type=args.artifact_type,
                description=f"{split} split of dataset {args.input_artifact}",
            )
            artifact.add_file(temp_path)
            logger.info("Logging artifact")
            run.log_artifact(artifact)
            logger.info(f"Artifact {artifact_name} logged successfully.")
            artifact.wait()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split a dataset into train and test",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Fully-qualified name for the input artifact",
        required=True,
    )

    parser.add_argument(
        "--artifact_root",
        type=str,
        help="Root for the names of the produced artifacts. The script will produce 2 artifacts: "
             "{root}_train.csv and {root}_test.csv",
        required=True,
    )

    parser.add_argument(
        "--artifact_type",
        type=str,
        help="Type for the produced artifacts",
        required=True
    )

    parser.add_argument(
        "--test_size",
        help="Fraction of dataset or number of items to include in the test split",
        type=float,
        required=True
    )

    parser.add_argument(
        "--random_state",
        help="An integer number to use to init the random number generator. It ensures repeatibility in the"
             "splitting",
        type=int,
        required=False,
        default=42
    )

    parser.add_argument(
        "--stratify",
        help="If set, it is the name of a column to use for stratified splitting",
        type=str,
        required=False,
        default='null'
    )

    args = parser.parse_args()

    go(args)
