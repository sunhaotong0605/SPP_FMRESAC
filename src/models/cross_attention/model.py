import os
from typing import Union, Tuple, Optional

from .modeling_esm import MultiHeadAttention
import torch
from torch import nn
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import SequenceClassifierOutput
from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
import numpy as np

from .enhance_rep_config import EnhanceRepresentationConfig, NTConfig, DnaBert2Config, EVOConfig


class EnhanceRepresentation(PreTrainedModel):
    def __init__(
            self, 
            config=EnhanceRepresentationConfig(), 
            llm_feature='nt', 
            device: str = None, 
            num_classes: int = 2,
            strategy: str = 'cross_attention',
            **kwargs
    ):
        super().__init__(config)

        self.config = config
        self.nt_config = config.nt_config
        self.evo_config = config.evo_config
        self.strategy = strategy
 
        # llm representation config
        if llm_feature == "NTForClassifier":
            cross_attention_config = self.nt_config
            cross_attention_config_class = NTConfig
        elif llm_feature == "DnaBert2ForClassifier":
            cross_attention_config = self.dnabert2_config
            cross_attention_config_class = DnaBert2Config
        elif llm_feature == "EvoForClassifier":
            cross_attention_config = self.evo_config
            cross_attention_config_class = EVOConfig

        cross_attention_config.hidden_size = 132
        config.num_heads_omics_cross_attention = 11
        cross_attention_other_omic_size = 132

        self.cross_attention_layer_rna = MultiHeadAttention(
            config=cross_attention_config_class(
                num_attention_heads=config.num_heads_omics_cross_attention,
                attention_head_size=cross_attention_config.hidden_size // config.num_heads_omics_cross_attention,
                hidden_size=cross_attention_config.hidden_size,
                attention_probs_dropout_prob=0,
                max_position_embeddings=0
            ),
            omics_of_interest_size=cross_attention_config.hidden_size,
            other_omic_size=cross_attention_other_omic_size
        )

        class MLPHead(nn.Module):
            def __init__(self, input_dim=132, num_classes=2):
                super().__init__()
                self.classifier = nn.Sequential(
                    nn.Linear(input_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, num_classes)
                )
                self.decoder = nn.Linear(input_dim, 132)
            def forward(self, x):
                logits = self.classifier(x)
                recon = self.decoder(x)
                return logits, recon
        self.mlp_head = MLPHead(input_dim=132, num_classes=num_classes)

    @staticmethod
    def _get_loss_fn(problem_type: str = "cross_entropy"):
        if problem_type == "mse":
            loss_fct = MSELoss(reduction='none')
        elif problem_type == "cross_entropy":
            loss_fct = CrossEntropyLoss()
        elif problem_type == "bce_with_logits":
            loss_fct = BCEWithLogitsLoss()
        else:
            raise ValueError(f"problem_type {problem_type} is not supported.")
        return loss_fct

    def forward(
            self,
            embedding,
            manual_feature,
            manual_feature_mask: Optional[torch.FloatTensor] = None,
            seqs_labels: Optional[torch.LongTensor] = None,
            weight: Optional[torch.FloatTensor] = 0.8,
            **kwargs
    ) -> Union[Tuple, SequenceClassifierOutput]:
        if self.strategy == "finetune": 
            manual_to_embed = self.cross_attention_layer_rna.forward(
                hidden_states=torch.tensor(embedding, dtype=torch.float32),
                encoder_hidden_states=torch.tensor(manual_feature,dtype=torch.float32),
            )['embeddings']
            logits, recon = self.mlp_head(manual_to_embed)
            logits = logits.reshape(-1, logits.shape[-1])
            seqs_labels = seqs_labels.reshape(seqs_labels.shape[0]*seqs_labels.shape[1])

            loss = None
            if seqs_labels is not None:
                seqs_labels = seqs_labels.to(logits.device)
                loss = self._get_loss_fn('cross_entropy')(logits, seqs_labels) * weight + \
                    (self._get_loss_fn('mse')(recon, manual_feature)*manual_feature_mask).sum() / manual_feature_mask.sum() * (1-weight)

            return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=embedding,
                # attentions=outputs.attentions,
            )
        elif self.strategy == "cross_attention":
            # cross attention
            manual_to_embed = self.cross_attention_layer_rna.forward(
                hidden_states=torch.tensor(embedding,dtype=torch.float32),
                encoder_hidden_states=torch.tensor(manual_feature,dtype=torch.float32),
            )["embeddings"]
        elif self.strategy == "dimension_reduction":
            manual_to_embed = (torch.tensor(embedding, dtype=torch.float32), torch.tensor(manual_feature, dtype=torch.float32))
        else:
            raise ValueError(f"strategy {self.strategy} is not supported.")

        return SequenceClassifierOutput(
            # loss=torch.randn(1).to(manual_to_embed.device),
            # logits=torch.randn(manual_to_embed.shape[0],2).to(manual_to_embed.device),
            hidden_states=manual_to_embed,
        )