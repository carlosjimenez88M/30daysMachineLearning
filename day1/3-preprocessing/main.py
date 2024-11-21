#!/usr/bin/env python
'''
Preprocessing System Model
Machine Learning Pipeline Operation
Edge Machine Learning (Raspberry Pi)
2024-11-20
'''

# ===================== #
# ---- Libraries ---- #
# ===================== #

import logging
import wandb
import argparse
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# ================================ #
# ---- Logger Configuration ---- #
# ================================ #

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s : %(levelname)s : %(message)s')
logger = logging.getLogger()

# ========================= #
# ---- Main Function ---- #
# ========================= #

def go(args):
    """
    Main function to preprocess data, split into train.py and test sets,
    and log the processed data as a W&B artifact.
    Args:
        args: Command-line arguments.
    """
    try:
        logger.info('Initializing preprocessing...')
        run = wandb.init(project='end_to_end',
                         job_type='preprocessing')

        logger.info('Downloading data...')
        artifact = run.use_artifact(args.input_artifact)
        artifact_path = artifact.file()
        df = pd.read_csv(artifact_path)
        logger.info(f'Data shape: {df.shape}')

        # Create income categories for stratification
        df["income_cat"] = pd.cut(df["median_income"],
                                  bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                  labels=[1, 2, 3, 4, 5])

        # Create stratified train.py-test splits
        logger.info('Creating stratified splits...')
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_idx, test_idx in splitter.split(df, df["income_cat"]):
            strat_train_set = df.iloc[train_idx]
            strat_test_set = df.iloc[test_idx]
        logger.info(f'Train shape: {strat_train_set.shape}')
        logger.info(f'Test shape: {strat_test_set.shape}')

        def income_cat_proportions(data):
            return data["income_cat"].value_counts() / len(data)

        # Compare proportions in different splits
        logger.info('Comparing data representation...')
        compare_props = pd.DataFrame({
            "Overall %": income_cat_proportions(df),
            "Stratified %": income_cat_proportions(strat_test_set),
        }).sort_index()
        compare_props.index.name = "Income Category"
        compare_props["Error %"] = (compare_props["Stratified %"] /
                                    compare_props["Overall %"] - 1) * 100
        logger.info("\n" + compare_props.round(2).to_string())

        # Prepare data for preprocessing
        housing = strat_train_set.copy()
        num_cols = housing.select_dtypes(include=["float64", "int64"]).columns.tolist()
        cat_cols = ["ocean_proximity"]

        # Define preprocessing pipelines
        num_pipeline = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("standardize", StandardScaler()),
        ])

        cat_pipeline = Pipeline([
            ("encode", OrdinalEncoder())
        ])

        full_pipeline = ColumnTransformer([
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
        ])

        logger.info('Transforming data...')
        housing_prepared = full_pipeline.fit_transform(housing)
        logger.info(f'Processed data shape: {housing_prepared.shape}')

        # Convert processed data to DataFrame
        processed_df = pd.DataFrame(
            housing_prepared,
            columns=num_cols + cat_cols,
            index=housing.index
        )

        # Reintegrate the income category column
        processed_df["income_cat"] = strat_train_set["income_cat"].values

        # Save processed data locally
        output_path = os.path.abspath(args.output_artifact)
        logger.info(f'Saving processed data to {output_path}...')
        processed_df.to_csv(output_path, index=False)

        logger.info('Logging processed data to W&B...')
        artifact = wandb.Artifact(
            name=args.output_artifact_name,
            type="preprocessed_data",
            description="Preprocessed housing data",
        )
        artifact.add_file(output_path)
        run.log_artifact(artifact)
        artifact.wait()
        logger.info('Preprocessing completed successfully.')

    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess housing data and log artifacts to W&B"
    )

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Name of the input artifact in W&B",
        required=True
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Path to save the processed data",
        required=True
    )

    parser.add_argument(
        "--output_artifact_name",
        type=str,
        help="Name for the output artifact in W&B",
        required=True
    )

    parser.add_argument(
        "--description",
        type=str,
        help="Description of the preprocessing step",
        required=False
    )

    args = parser.parse_args()

    go(args)
