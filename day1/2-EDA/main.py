#!usr/bin/env python
'''
Exploratory Data Analysis
Machine learning Pipeline Operation
Edge Machine Learning (Raspberry PI)
2024-11-20
'''

#=====================#
# ---- Libraries ---- #
#=====================#

import logging
import argparse
import wandb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas.plotting import scatter_matrix

#================================#
# ---- Logger Configuration ---- #
#================================#

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s : %(levelname)s : %(message)s')



logger= logging.getLogger()


#================================#
# ---- Main function system ---- #
#================================#


def go(args):
    logger.info('Initialize system data Exploration')
    run = wandb.init(project='end_to_end',
                     job_type='data_exploration')
    logger.info('Downloading artifact')
    artifact = run.use_artifact(args.file_artifact) #'housing_dataset:latest'
    artifact_dir = artifact.download()
    csv_path = f"{artifact_dir}/housing.csv"
    df = pd.read_csv(csv_path)
    logger.info(f'Dataframe head: {df.head()}')

    plt.rc('font', size=14)
    plt.rc('axes', labelsize=14, titlesize=14)
    plt.rc('legend', fontsize=14)
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)

    df.hist(bins=50, figsize=(12, 8))
    wandb.log({"data_histogram": wandb.Image(plt,
                                             caption="Data Distribution Histogram")})



    #==============================================================================#
    #                            HISTOGRAM STRATIFICATION                          #
    #==============================================================================#

    df["income_cat"] = pd.cut(df["median_income"],
                              bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                              labels=[1, 2, 3, 4, 5])

    df["income_cat"].value_counts().sort_index().plot.bar(rot=0, grid=True)
    plt.xlabel("Income category")
    plt.ylabel("Number of districts")
    wandb.log({"Income Distribution": wandb.Image(plt,
                                                  caption="Data Distribution Histogram to stratify")})


    # ==============================================================================#
    #                            DISTRIBUTION POPULATION                            #
    # ==============================================================================#

    df.plot(kind="scatter",
            x="longitude",
            y="latitude",
            grid=True,
            s=df["population"] / 100,
            label="population",
            c="median_house_value",
            cmap="jet",
            colorbar=True,
            legend=True,
            sharex=False,
            figsize=(10, 7))
    wandb.log({"Geographical Data with filters": wandb.Image(plt,
                                                             caption="Geographical Data visualization")})



    # ==============================================================================#
    #                              CORRELATION SCHEMAS                              #
    # ==============================================================================#

    attributes = ["median_house_value",
                  "median_income",
                  "total_rooms",
                  "housing_median_age"]
    scatter_matrix(df[attributes], figsize=(12, 8))
    wandb.log({"Correlation Variables": wandb.Image(plt,
                                                    caption="Correlation Variables visualization")})


    run.finish()





if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Preprocess a dataset",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "--file_artifact",
        type=str,
        help="Fully-qualified name for the input artifact",
        required=True,
    )

    args = parser.parse_args()

    go(args)