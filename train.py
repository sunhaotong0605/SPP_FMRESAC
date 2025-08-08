import os
import sys

import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data_process.cut_seq import cut_seq
from src.data_process.extract_feature import cut_seq_to_fna
from src.data_process.extract_feature import extract_ef
from src.data_process.extract_feature import extract_llmr
from src.data_process.enhance_rep import enhance_rep
from src.data_process.train import train_cross_attn
from src.data_process.train import dimension_reduction
from src.data_process.train import train_ml
from src.data_process.ml_predict import ml_predict

@hydra.main(config_path="configs", config_name="config.yaml", version_base=None)
def main(config: OmegaConf):
    if config.run_train_model:
        cut_seq.main(config)
        cut_seq_to_fna.main(config)
        extract_ef.main(config)
        extract_llmr.main(config)
        dimension_reduction.main(config)
        train_cross_attn.main(config)
        config.enhance_rep.train.resume_from_checkpoint = config.train_cross_attn.train.output_dir
        enhance_rep.main(config)
        train_ml.main(config)
    else:
        raise ValueError("run_train_model must be True")

if __name__ == '__main__':
    main()
    