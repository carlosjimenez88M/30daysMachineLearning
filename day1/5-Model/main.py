import os
import wandb
import yaml
import json
import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split
import psutil

def prepare_data(train_artifact, test_artifact):

    train_artifact_path = train_artifact.file()
    test_artifact_path = test_artifact.file()

    train_data = pd.read_csv(train_artifact_path)
    test_data = pd.read_csv(test_artifact_path)

    X = train_data.drop(columns=["median_house_value"])
    y = train_data["median_house_value"]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    X_test = test_data.drop(columns=["median_house_value"])
    y_test = test_data["median_house_value"]

    return X_train, X_val, X_test, y_train, y_val, y_test

def sweep_train(config_defaults, args):
    def train(config=None):
        with wandb.init(config=config) as run:
            config = wandb.config

            train_artifact = run.use_artifact(args.train_artifact)
            test_artifact = run.use_artifact(args.test_artifact)
            X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(train_artifact, test_artifact)

            model = RandomForestRegressor(
                n_estimators=config.n_estimators,
                max_features=config.max_features,
                min_samples_split=config.min_samples_split,
                min_samples_leaf=config.min_samples_leaf,
                bootstrap=config.bootstrap,
                random_state=42,
            )
            cv_scores = cross_val_score(
                model,
                X_train,
                y_train,
                scoring="neg_root_mean_squared_error",
                cv=3
            )
            mean_cv_rmse = -cv_scores.mean()
            wandb.log({"cv_rmse": mean_cv_rmse})

            model.fit(X_train, y_train)

            val_predictions = model.predict(X_val)
            val_rmse = mean_squared_error(y_val, val_predictions, squared=False)
            wandb.log({"val_rmse": val_rmse})

            test_predictions = model.predict(X_test)
            test_rmse = mean_squared_error(y_test, test_predictions, squared=False)
            wandb.log({"test_rmse": test_rmse})

            # Guardar los mejores parámetros si este run tiene el mejor cv_rmse
            best_params_file = args.best_params_output

            # Leer el mejor cv_rmse actual del archivo, si existe
            if os.path.exists(best_params_file):
                with open(best_params_file, 'r') as f:
                    data = json.load(f)
                best_cv_rmse = data.get('best_cv_rmse', float('inf'))
            else:
                best_cv_rmse = float('inf')

            if mean_cv_rmse < best_cv_rmse:
                # Actualizar el archivo con los nuevos mejores parámetros
                best_params = {
                    "n_estimators": config.n_estimators,
                    "max_features": config.max_features,
                    "min_samples_split": config.min_samples_split,
                    "min_samples_leaf": config.min_samples_leaf,
                    "bootstrap": config.bootstrap,
                    "best_cv_rmse": mean_cv_rmse
                }
                with open(best_params_file, 'w') as f:
                    json.dump(best_params, f, indent=4)

    sweep_id = wandb.sweep(config_defaults, project="end_to_end")
    wandb.agent(sweep_id, function=train)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Random Forest with W&B sweeps")
    parser.add_argument("--sweep_config", type=str, required=True, help="Path to the sweep config file")
    parser.add_argument("--train_artifact", type=str, required=True, help="Artifact containing the training dataset")
    parser.add_argument("--test_artifact", type=str, required=True, help="Artifact containing the test dataset")
    parser.add_argument("--output_artifact", type=str, required=True, help="Path to save the trained model")
    parser.add_argument("--best_params_output", type=str, required=True, help="Path to save the best parameters as JSON")
    args = parser.parse_args()

    with open(args.sweep_config, "r") as f:
        sweep_config = yaml.safe_load(f)

    sweep_train(sweep_config, args)


