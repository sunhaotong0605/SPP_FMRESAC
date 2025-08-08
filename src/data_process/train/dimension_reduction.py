import glob
import os
import pickle
import sys
import multiprocessing


root_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, root_path)

# import comet_ml
import torch, gc
import tqdm
import hydra
import pandas as pd
import numpy as np
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import confusion_matrix
from transformers.trainer_utils import get_last_checkpoint
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding, set_seed

from src import utils
from src.utils import registry
from src.utils.train import print_config, process_config
from src.self_logger import logger, init_logger
from src.datasets.probiotics_dataset import ProbioticSplitEnhanceRepresentationDataset, ProbioticsDataProcess

def dimension_reduction(config: DictConfig):
    set_seed(config.dimension_reduction.train.seed)
    model = utils.config.instantiate(registry.model, config.dimension_reduction.model)
    dataset = utils.config.instantiate(registry.dataset, config.dimension_reduction.dataset, partial=True)
    train_dataset, test_dataset, val_dataset = None, None, None

    # Set up training arguments
    training_args = TrainingArguments(
        label_names=["seqs_labels"],
        remove_unused_columns=False, 
        **config.dimension_reduction.train,
    )

    # Detecting last checkpoint
    last_checkpoint = None
    if os.path.isdir(config.dimension_reduction.train.output_dir) and config.dimension_reduction.train.do_train and not config.dimension_reduction.train.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(config.dimension_reduction.train.output_dir)
        if last_checkpoint is None and len(os.listdir(config.dimension_reduction.train.output_dir)) > 1:  # exclude logs file
            raise ValueError(
                f"Output directory ({config.comet_ml.file}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and config.dimension_reduction.train.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set up Trainer
    # data_collator = DataCollatorWithPadding(tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if config.dimension_reduction.train.do_train else None,
        eval_dataset=val_dataset if config.dimension_reduction.train.do_eval else None,
        compute_metrics=utils.util.compute_metrics,
        # data_collator=data_collator,
    )

    # Train model
    checkpoint = None
    if config.dimension_reduction.train.resume_from_checkpoint is not None:
        checkpoint = config.dimension_reduction.train.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    logger.info(f"load checkpoint from {checkpoint}")
    if checkpoint is not None:
        trainer._load_from_checkpoint(checkpoint)
        logger.info(f'load checkpoint from {checkpoint=} ')


    # Prediction
    with torch.no_grad():
        if config.dimension_reduction.dataset.test_split is not None and config.dimension_reduction.train.do_predict:
            sample_dataset = dataset(
                dest_path=config.dimension_reduction.dataset.llm_rep_path,
                _dest_path=config.dimension_reduction.dataset.ef_path,
                split=config.dimension_reduction.dataset.test_split,
                strategy=config.dimension_reduction.model.strategy
            )
            for sample in tqdm.tqdm(sample_dataset, desc=f"dimension_reduction"):
                gc.collect()
                torch.cuda.empty_cache()

                test_dataset = ProbioticSplitEnhanceRepresentationDataset(
                    seqs_labels=sample['seqs_labels'],
                    manual_feature=sample['manual_feature'],
                    embedding=sample['embedding'],
                    strategy=config.dimension_reduction.model.strategy,
                )

                pickles_dir = os.path.join(config.dimension_reduction.train.output_dir,"pickles")
                os.makedirs(pickles_dir, exist_ok=True)
                output_pkl_path = os.path.join(pickles_dir, f"{sample['sample_name']}.pkl")

                # For probiotic sample predict
                try:
                    trainer.data_collator = test_dataset.data_collator
                except:
                    pass

                ef_feats = None
                prediction_output = trainer.predict(test_dataset)
                if isinstance(prediction_output.predictions, tuple):
                    # output tuple with SequenceClassifierOutput
                    llm_rep, ef_feat = prediction_output.predictions
                    llm_rep = np.squeeze(llm_rep)
                    ef_feat = np.squeeze(ef_feat)
                    embeddings = llm_rep.tolist()
                    ef_feats = ef_feat.tolist()
                else:
                    embeddings = np.squeeze(embeddings)
                    embeddings = embeddings.tolist()

                predict_result = {
                    "sample_name": sample['sample_name'],
                    "seqs_paths": sample['seqs_paths'],
                    "seqs_labels": sample['seqs_labels'],
                    "model_predict": {
                        "embedding": embeddings,
                    }
                }
                with open(output_pkl_path, "wb") as f:
                    pickle.dump(predict_result, f)
                logger.info(f"predict pkl result: {output_pkl_path}")
                if ef_feats is not None:
                    pickles_dir = os.path.join(os.path.dirname(config.dimension_reduction.train.output_dir),'Engineered_features',"pickles")
                    os.makedirs(pickles_dir, exist_ok=True)
                    output_pkl_path = os.path.join(pickles_dir, f"{sample['sample_name']}.pkl")
                    predict_result = {
                        "sample_name": sample['sample_name'],
                        "seqs_paths": sample['seqs_paths'],
                        "seqs_labels": sample['seqs_labels'],
                        "model_predict": {
                            "embedding": ef_feats,
                        }
                    }
                    with open(output_pkl_path, "wb") as f:
                        pickle.dump(predict_result, f)
                    logger.info(f"predict pkl result: {output_pkl_path}")
        else:
            logger.info(f"not valid data split name: {config.dimension_reduction.dataset.test_split}")

@hydra.main(config_path="configs", config_name="config.yaml", version_base=None)
def main(config: OmegaConf):
    # check if the config is valid
    config = process_config(config)
    os.environ["WANDB_MODE"] = "disabled"
    os.makedirs(config.dimension_reduction.train.output_dir, exist_ok=True)

    init_logger(svr_name="dimension_reduction", log_path=config.dimension_reduction.train.logging_dir)
    dimension_reduction(config)

    pbt_dp = ProbioticsDataProcess()
    pbt_dp.probiotics_get_pickles_txt(dir=config.dimension_reduction.train.output_dir)
    pbt_dp.probiotics_get_pickles_txt(dir=os.path.join(os.path.dirname(config.dimension_reduction.train.output_dir),'Engineered_features'))
    pbt_dp.probiotics_split_trainset(dir=config.dimension_reduction.train.output_dir,train1_ratio=0.6)
    pbt_dp.probiotics_split_trainset(dir=os.path.join(os.path.dirname(config.dimension_reduction.train.output_dir),'Engineered_features'),train1_ratio=0.6)

    logger.info("dimension reduction completed​​​")

if __name__ == '__main__':
    main()