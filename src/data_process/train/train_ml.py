import json
import os
import sys
import shutil

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_path)

import hydra
import joblib
import pandas as pd
from omegaconf import OmegaConf, DictConfig
from transformers import set_seed
import tqdm
import pickle
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
import numpy as np
import openpyxl

from src.self_logger import logger, init_logger
from src.utils.train import process_config, print_config
from src.datasets.ml_dataset import ProbioticKFTrainDataset
from src.models.stacked_aggregation_classifier import MLClassModel, MLTrainer

def train_ml(config: DictConfig):
    set_seed(config.train_ml.train.seed)

    data_config = config.train_ml.dataset
    train_dataset, test_dataset, eval_dataset = None, None, None
    
    model = MLClassModel(**config.train_ml.model, split_num=config.train_ml.dataset.split_num)
    trainer = MLTrainer(
        model=model,
        overwrite_output_dir=config.train_ml.train.overwrite_output_dir,
        output_dir=config.train_ml.train.output_dir,
        logging_dir=config.train_ml.train.logging_dir,
        resume_from_checkpoint=config.train_ml.train.resume_from_checkpoint,
    )

    if config.train_ml.train.overwrite_output_dir:
        shutil.rmtree(config.train_ml.train.output_dir, ignore_errors=True)

    if config.train_ml.dataset.train_split is not None:
        train_txt_path = os.path.join(data_config.dest_path, data_config.dataset_name, data_config.train_split+'.txt')
        with open(train_txt_path, 'r') as file:
            seq_paths = np.array([x.rstrip() for x in file.readlines()])
        train_dataset = ProbioticKFTrainDataset(seq_paths=seq_paths, seed=config.train_ml.train.seed, **data_config)
    if train_dataset is not None and config.train_ml.train.do_train:
        logger.info(f"Start training ...")
        train_result, final_train_result = trainer.train(train_dataset)
        model_path = os.path.join(config.train_ml.train.output_dir,"model.pkl")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(trainer.model.model, model_path)
        if trainer.model.final_model is not None:
            final_model_path = os.path.join(config.train_ml.train.output_dir,"final_model.pkl")
            os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
            joblib.dump(trainer.model.final_model, final_model_path)

@hydra.main(config_path="configs", config_name="config.yaml", version_base=None)
def main(config: OmegaConf):
    # check if the config is valid
    config = process_config(config)
    if config.extract_llmr.model._name_ == "NTForClassifier":
        config.train_ml.train.output_dir = os.path.join(config.train_ml.train.output_dir, 'NT_50M')
    elif config.extract_llmr.model._name_ == "EvoForClassifier":
        config.train_ml.train.output_dir = os.path.join(config.train_ml.train.output_dir, 'EVO_7B')
    else:
        raise ValueError(f"Unknown model name: {config.extract_llmr.model._name_}")
    config.train_ml.train.logging_dir = os.path.join(config.train_ml.train.output_dir, "logs")

    os.makedirs(config.train_ml.dataset.dest_path, exist_ok=True)
    os.makedirs(config.train_ml.train.output_dir, exist_ok=True)
    os.makedirs(config.train_ml.train.logging_dir, exist_ok=True)

    init_logger(svr_name="train_ml", log_path=config.enhance_rep.train.logging_dir)
    # logger.info("start train ml...")
    train_ml(config)
    logger.info("train ml completed")

if __name__ == '__main__':
    main()
