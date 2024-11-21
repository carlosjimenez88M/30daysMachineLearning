#!usr/bin/env Python
'''
System Model to main.py like a End to end Machine Learning Application
Cap #2 Hands on Machine Learning Applications
'''


#=====================#
# ---- Libraries ---- #
#=====================#

import mlflow
import os
import logging
import hydra
from omegaconf import DictConfig, OmegaConf

#=================================#
# ---- Logging Configuration ---- #
#=================================#

logging.basicConfig(
    level=logging.INFO,
    format= '%(asctime)-15s (message)s%'
)

logger= logging.getLogger()

#=========================#
# ---- Main Function ---- #
#=========================#


@hydra.main(config_path='.', config_name='config')
def go(config: DictConfig):
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]
    root_path = hydra.utils.get_original_cwd()

    _ = mlflow.run(
        os.path.join(root_path,'1-download_dataset'),
        'main',
        parameters={
            'file_url':config['data']['file_url'],
            'artifact_name':config['data']['artifact_name'],
            'artifact_type': config['data']['artifact_type'],
            'artifact_description': config['data']['artifact_description']
        }
    )

    _ = mlflow.run(
        os.path.join(root_path,'2-EDA'),
        'main',
        parameters={'file_artifact': config['eda']['file_artifact']}
    )

    _ = mlflow.run(
        os.path.join(root_path, '3-preprocessing'),
        'main',
        parameters={
            'input_artifact': config['preprocessing']['input_artifact'],
            'output_artifact': config['preprocessing']['output_artifact'],
            'output_artifact_name': config['preprocessing']['output_artifact_name'],
            'description': config['preprocessing']['description'],
        }
    )

    _ = mlflow.run(
        os.path.join(root_path, '4-segregation'),
        'main',
        parameters={
            'input_artifact': config['segregation']['input_artifact'],
            'artifact_root': config['segregation']['artifact_root'],
            'artifact_type': config['segregation']['artifact_type'],
            'test_size': config['segregation']['test_size'],
            'random_state': config['segregation']['random_state'],
            'stratify': config['segregation']['stratify']
        }
    )
    _ = mlflow.run(
        os.path.join(root_path, "5-Model"),
        "sweep_and_train",
        parameters={
            "sweep_config": config["train"]["sweep_config"],
            "train_artifact": f"{config['segregation']['artifact_root']}_train.csv:latest",
            "test_artifact": f"{config['segregation']['artifact_root']}_test.csv:latest",
            "output_artifact": config["train"]["output_artifact"],
            "best_params_output": config["train"]["best_params_output"]
        }
    )



if __name__ == '__main__':
    go()




