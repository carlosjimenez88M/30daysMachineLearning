#!usr/bin/env python
'''
End to end Machine Learning Project
Step: Download Database
2024-11-20
'''

# Description -------------------------------------
# This system generates process to download dataset
# to initiate the complete pipeline of the system.

#=====================#
# ---- libraries ---- #
#=====================#
import os
import argparse
import logging
import pathlib
import pandas as pd
import tarfile
import urllib.request
import wandb
import tempfile


#================================#
# ---- Logger Configuration ---- #
#================================#

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s : %(levelname)s : %(message)s'
)



logger= logging.getLogger()


#=========================#
# ---- Main Function ---- #
#=========================#

def load_housing_data(args):
    """
    Downloads the housing dataset, extracts it, registers it with W&B,
    and returns the loaded dataset as a pandas DataFrame.
    """
    tarball_path = pathlib.Path("datasets/housing.tgz")
    extracted_csv_path = pathlib.Path("datasets/housing/housing.csv")


    pathlib.Path("datasets").mkdir(parents=True, exist_ok=True)

    if not tarball_path.is_file():
        logger.info(f"Downloading {args.file_url} ...")
        urllib.request.urlretrieve(args.file_url, tarball_path)
        logger.info(f"Downloaded file saved to {tarball_path}")
    if not extracted_csv_path.is_file():
        logger.info(f"Extracting {tarball_path} ...")
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets")
        logger.info(f"Extracted files to datasets directory")

    logger.info(f"Deleting tarball file {tarball_path} ...")
    os.remove(tarball_path)
    logger.info(f"Deleted tarball file {tarball_path}")
    logger.info("Creating W&B artifact")
    with wandb.init(project="end_to_end",
                    job_type="data_preparation") as run:
        artifact = wandb.Artifact(
            name=args.artifact_name,
            type=args.artifact_type,
            description=args.artifact_description,
            metadata={'source_url': args.file_url}
        )

        artifact.add_file(str(extracted_csv_path),
                          name="housing.csv")
        logger.info("Logging artifact to W&B")
        run.log_artifact(artifact)
        artifact.wait()
    logger.info(f"Loading data from {extracted_csv_path}")
    return pd.read_csv(extracted_csv_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download, extract, and process housing data, registering it with W&B",
        fromfile_prefix_chars="@"
    )

    parser.add_argument(
        "--file_url",
        type=str,
        help="URL to the housing dataset tarball",
        required=True
    )

    parser.add_argument(
        "--artifact_name",
        type=str,
        help="Name for the artifact",
        required=True
    )

    parser.add_argument(
        "--artifact_type",
        type=str,
        help="Type for the artifact",
        required=True
    )

    parser.add_argument(
        "--artifact_description",
        type=str,
        help="Description for the artifact",
        required=True,
    )

    args = parser.parse_args()

    df = load_housing_data(args)
    logger.info(f"Dataframe loaded with shape: {df.shape}")
    print(df.head())