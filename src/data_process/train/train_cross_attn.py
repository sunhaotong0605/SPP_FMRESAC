import os
import sys

import torch
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hydra
from omegaconf import DictConfig, OmegaConf
from transformers.trainer_utils import get_last_checkpoint
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding, set_seed

from src import utils
from src.utils import registry
from src.utils.train import print_config, process_config
from src.self_logger import logger, init_logger
 
def train(config: DictConfig):
    set_seed(config.train_cross_attn.train.seed)

    model = utils.config.instantiate(registry.model, config.train_cross_attn.model)
    # Load the tokenizer and dataset
    dataset = utils.config.instantiate(registry.dataset, config.train_cross_attn.dataset, partial=True)
    # tokenizer = model.embedding.get_tokenizer
    train_dataset, test_dataset, eval_dataset = None, None, None
    # data_collator = DataCollatorWithPadding(tokenizer)

    if config.train_cross_attn.dataset.train_split is not None:
        train_dataset = dataset(
            dest_path=config.train_cross_attn.dataset.llm_rep_path,
            _dest_path=config.train_cross_attn.dataset.ef_path,
            split=config.train_cross_attn.dataset.train_split,
            only_features=config.train_cross_attn.dataset.only_features,
        )
        data_collator = train_dataset._data_collator
    if config.train_cross_attn.dataset.test_split is not None:
        test_dataset = dataset(
            dest_path=config.train_cross_attn.dataset.llm_rep_path,
            _dest_path=config.train_cross_attn.dataset.ef_path,
            split=config.train_cross_attn.dataset.test_split,
            only_features=config.train_cross_attn.dataset.only_features,
        )
        data_collator = test_dataset._data_collator
    if config.train_cross_attn.dataset.val_split is not None:
        eval_dataset = dataset(
            dest_path=config.train_cross_attn.dataset.llm_rep_path,
            _dest_path=config.train_cross_attn.dataset.ef_path,
            split=config.train_cross_attn.dataset.val_split,
            only_features=config.train_cross_attn.dataset.only_features,

        )
        data_collator = eval_dataset._data_collator

    # Set up training arguments
    training_args = TrainingArguments(
        label_names=["seqs_labels"],
        # notice: do not wandb
        # report_to=["comet_ml"],
        remove_unused_columns=False,
        save_total_limit=2,
        # load_best_model_at_end=True,
        # load_best_model_at_end=False,
        **config.train_cross_attn.train,
    )

    # Detecting last checkpoint
    last_checkpoint = None
    if os.path.isdir(config.train_cross_attn.train.output_dir) and config.train_cross_attn.train.do_train and not config.train_cross_attn.train.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(config.train_cross_attn.train.output_dir)
        if last_checkpoint is None and len(os.listdir(config.train_cross_attn.train.output_dir)) > 1:  # exclude logs file
            logger.warning(
                f"Output directory ({config.train_cross_attn.train.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and config.train_cross_attn.train.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
 
    # Set up Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if config.train_cross_attn.train.do_train else None,
        eval_dataset=eval_dataset if config.train_cross_attn.train.do_eval else None,
        compute_metrics=None, #utils.util.compute_metrics,
        data_collator=data_collator,
    )

    if config.train_cross_attn.train.overwrite_output_dir:
        # save init model
        logger.info("save init model in {}".format(os.path.join(trainer.args.output_dir, "checkpoint-0")))
        trainer.save_model(output_dir=os.path.join(trainer.args.output_dir, "checkpoint-0"))

    # Fit checkpoint
    checkpoint = None
    if config.train_cross_attn.train.resume_from_checkpoint is not None:
        checkpoint = config.train_cross_attn.train.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint


    if train_dataset is not None and config.train_cross_attn.train.do_train:
        logger.info("start train...")
        logger.info(f"checkpoint from to train: {checkpoint}")
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()

        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
            
    else:
        logger.info(f"stipe train stage and load ckpt from: {checkpoint}")
        if checkpoint is not None:
            trainer._load_from_checkpoint(checkpoint)

    # Evaluation
    if eval_dataset is not None and config.train_cross_attn.train.do_eval:
        logger.info("start evaluation...")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        logger.info("eval result: \n{}".format(metrics))

    # Prediction
    if test_dataset is not None and config.train_cross_attn.train.do_predict:
        logger.info("start predict...")
        prediction_output = trainer.predict(test_dataset)
        if isinstance(prediction_output.predictions, tuple):
            # output tuple with SequenceClassifierOutput
            predictions, embedding = prediction_output.predictions
        else:
            predictions, embedding = prediction_output.predictions, None

        probability = torch.nn.functional.softmax(torch.tensor(predictions), dim=-1).numpy()

        # save to csv
        test_result = {
            "labels": prediction_output.label_ids,
            "predict_0": predictions[:, 0],
            "predict_1": predictions[:, 1],
            "predicts": predictions.argmax(-1),
            "predict_value": predictions.max(-1),
            "prob_0": probability[:, 0],
            "prob_1": probability[:, 1],
            "prob_value": probability.max(-1),
        }
        df_test_result = pd.DataFrame(test_result)
        df_test_result.to_csv(
            os.path.join(trainer.args.output_dir, f"test_result_{config.train_cross_attn.dataset.test_split}.csv"),
            index=False
        )

    logger.info("Finished!")


@hydra.main(config_path="configs", config_name="config.yaml", version_base=None)
def main(config: OmegaConf):
    # check if the config is valid
    config = process_config(config)
    os.environ["WANDB_MODE"] = "disabled"
    if config.train_cross_attn.model.llm_feature == "NTForClassifier":
        config.train_cross_attn.train.output_dir = os.path.join(config.train_cross_attn.train.output_dir, 'NT_50M')
    elif config.train_cross_attn.model.llm_feature == "EvoForClassifier":
        config.train_cross_attn.train.output_dir = os.path.join(config.train_cross_attn.train.output_dir, 'EVO_7B')
    else:
        raise ValueError(f"Unknown model name: {config.train_cross_attn.model.llm_feature}")
    config.train_cross_attn.train.logging_dir = os.path.join(config.train_cross_attn.train.output_dir, "logs")
    os.makedirs(config.train_cross_attn.train.output_dir, exist_ok=True)
    os.makedirs(config.train_cross_attn.train.logging_dir, exist_ok=True)

    init_logger(svr_name="genomicsLLM_Train", log_path=config.train_cross_attn.train.logging_dir)
    logger.info("start training cross attention model...")
    train(config)
    logger.info("train cross attn completed")
    

if __name__ == '__main__':
    main()