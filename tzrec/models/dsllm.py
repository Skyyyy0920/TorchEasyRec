# Copyright (c) 2025, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from modelscope import AutoModelForCausalLM
from torch import nn
from torch._tensor import Tensor

from tzrec.datasets.utils import BASE_DATA_GROUP, NEG_DATA_GROUP, Batch
from tzrec.features.feature import BaseFeature
from tzrec.protos.loss_pb2 import LossConfig
from tzrec.models.match_model import MatchModel, MatchTowerWoEG
from tzrec.protos import model_pb2, simi_pb2, tower_pb2
from tzrec.utils.fx_util import fx_arange

torch.fx.wrap('len')
torch.fx.wrap(fx_arange)


class DSLLMTower(MatchTowerWoEG):
    """LLM as Tower for user/item embedding generation.

    Args:
        tower_config (Tower): user/item tower config.
        output_dim (int): user/item output embedding dimension.
        similarity (Similarity): when use COSINE similarity,
            will norm the output embedding.
        feature_groups (list[FeatureGroupConfig]): feature group configs.
        features (list): list of features.
        model_config (ModelConfig): model config containing LLM settings.
    """

    def __init__(
        self,
        tower_config: tower_pb2.LLMTower,
        output_dim: int,
        similarity: simi_pb2.Similarity,
        feature_groups: List[model_pb2.FeatureGroupConfig],
        features: List[BaseFeature],
    ) -> None:
        super().__init__(tower_config, output_dim, similarity, feature_groups, features)
        
        self.model_name = getattr(tower_config, 'model_name', 'Qwen/Qwen3-0.6B')
        self.pooling_method = getattr(tower_config, 'pooling_method', 'mean')  # mean, cls, last
        self.freeze_llm = getattr(tower_config, 'freeze_llm', False)
        
        self.llm_model = AutoModelForCausalLM.from_pretrained(self.model_name)

        if self.freeze_llm:
            for param in self.llm_model.parameters():
                param.requires_grad = False
    
    def _sparse_to_tokens(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """Parse sparse input data from PromptFeature."""
        input_dict = {}
        
        feature_name = self._features[0].name
        if self._tower_config.input == 'user':
            sparse_data = batch.sparse_features[BASE_DATA_GROUP][feature_name]
        elif self._tower_config.input == 'item':
            sparse_data = batch.sparse_features[NEG_DATA_GROUP][feature_name]
        else:
            raise ValueError(f"Invalid tower name: {self._tower_config.input}")
        
        lengths = sparse_data.lengths()
        values = sparse_data.values()

        split_values = torch.split(values, lengths.tolist())
        input_ids = torch.nn.utils.rnn.pad_sequence(split_values, batch_first=True)

        max_length = input_ids.size(1)
        attention_mask = torch.arange(max_length)[None, :].to(lengths.device) < lengths[:, None]
    
        input_dict[f'{feature_name}_input_ids'] = input_ids
        input_dict[f'{feature_name}_attention_mask'] = attention_mask
        
        return input_dict

    def _get_llm_embeddings(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Get embeddings from LLM.""" 
        outputs = self.llm_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        hidden_states = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_dim]

        if self.pooling_method == 'mean':
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            embeddings = sum_embeddings / sum_mask
        elif self.pooling_method == 'cls':
            embeddings = hidden_states[:, 0]
        elif self.pooling_method == 'last':
            batch_size = hidden_states.size(0)
            embeddings = []
            for i in range(batch_size):
                last_token_idx = attention_mask[i].sum() - 1
                embeddings.append(hidden_states[i, last_token_idx])
            embeddings = torch.stack(embeddings)
        else:
            raise ValueError(f"Unsupported pooling method: {self.pooling_method}")

        return embeddings

    def forward(self, batch: Batch) -> torch.Tensor:
        """Forward the tower.

        Args:
            batch (Batch): input batch data.

        Return:
            embedding (Tensor): tower output embedding.
        """
        input_dict = self._sparse_to_tokens(batch)

        input_ids = None
        attention_mask = None
        for key, value in input_dict.items():
            if key.endswith('_input_ids'):
                input_ids = value
            elif key.endswith('_attention_mask'):
                attention_mask = value

        if input_ids is None or attention_mask is None:
            raise ValueError("No valid prompt feature input found")


        embeddings = self._get_llm_embeddings(input_ids, attention_mask)

        if self._similarity == simi_pb2.Similarity.COSINE:
            embeddings = F.normalize(embeddings, p=2.0, dim=1)

        return embeddings


class DSLLM(MatchModel):
    """Deep Structured Large Language Model for recommendation.

    Args:
        model_config (ModelConfig): an instance of ModelConfig.
        features (list): list of features.
        labels (list): list of label names.
        sample_weights (list): sample weight names.
    """

    def __init__(
        self,
        model_config: model_pb2.ModelConfig,
        features: List[BaseFeature],
        labels: List[str],
        sample_weights: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_config, features, labels, sample_weights, **kwargs)

        name_to_feature_group = {x.group_name: x for x in model_config.feature_groups}

        user_group = name_to_feature_group[self._model_config.user_tower.input]
        item_group = name_to_feature_group[self._model_config.item_tower.input]
        user_features = self.get_features_in_feature_groups([user_group])
        item_features = self.get_features_in_feature_groups([item_group])

        self.user_tower = DSLLMTower(
            self._model_config.user_tower,
            self._model_config.output_dim,
            self._model_config.similarity,
            [user_group],
            user_features,
        )

        self.item_tower = DSLLMTower(
            self._model_config.item_tower,
            self._model_config.output_dim,
            self._model_config.similarity,
            [item_group],
            item_features,
        )

        self.use_focal_loss = getattr(model_config, 'use_focal_loss', True)
        self.focal_alpha = getattr(model_config, 'focal_alpha', 0.25)
        self.focal_gamma = getattr(model_config, 'focal_gamma', 2.0)
        self.distributed_negatives = getattr(model_config, 'distributed_negatives', True)

    def sim(self, user_emb: torch.Tensor, item_emb: torch.Tensor) -> torch.Tensor:
        """Calculate user and item embedding similarity with distributed support."""
        if self._in_batch_negative:
            return torch.mm(user_emb, item_emb.T)
        else:
            batch_size = user_emb.size(0)
            pos_item_emb = item_emb[:batch_size]
            neg_item_emb = item_emb[batch_size:]
            
            # user_embs = F.normalize(user_embs, p=2, dim=1)
            # pos_item_embs = F.normalize(pos_item_embs, p=2, dim=1)
            # neg_item_embs = F.normalize(neg_item_embs, p=2, dim=1)

            if self.distributed_negatives and dist.is_initialized():
                group = torch.distributed.group.WORLD
                all_neg_emb = torch.cat(torch.distributed.nn.functional.all_gather(neg_item_emb, group), dim=0)
            else:
                all_neg_emb = neg_item_emb

            total_negs = all_neg_emb.size(0)

            all_item_embs = torch.cat([
                pos_item_emb.unsqueeze(1),  # [B, 1, D]
                all_neg_emb.unsqueeze(0).expand(batch_size, total_negs, -1)  # [B, TN, D]
            ], dim=1)  # [B, 1+TN, D]

            expanded_user_embs = user_emb.unsqueeze(1)  # [B, 1, D]

            logits = torch.sum(expanded_user_embs * all_item_embs, dim=2)  # [B, 1+TN]
            
            return logits

    def _focal_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        probs = torch.sigmoid(logits)
        pt = torch.where(labels.bool(), probs, 1 - probs)

        alpha_t = torch.where(labels.bool(), self.focal_alpha, 1 - self.focal_alpha)
        focal_weight = alpha_t * (1 - pt) ** self.focal_gamma

        return (focal_weight * bce_loss).mean()

    def _loss_impl(
        self,
        predictions: Dict[str, torch.Tensor],
        batch: Batch,
        label: torch.Tensor,
        loss_cfg: LossConfig,
        suffix: str = "",
    ) -> Dict[str, torch.Tensor]:
        """Compute loss with focal loss support."""
        losses = {}
        loss_type = loss_cfg.WhichOneof("loss")
        loss_name = loss_type + suffix
        pred = predictions["similarity" + suffix]

        if self.use_focal_loss:
            labels = torch.zeros_like(pred)
            labels[:, 0] = 1.0
            # TODO: need add focal loss, now I just use softmax_cross_entropy loss but actually calculate focal loss
            # losses["binary_focal_loss" + suffix] = self._focal_loss(pred, labels)
            losses[loss_name] = self._focal_loss(pred, labels)
        else:
            # Use standard cross entropy loss
            label = torch.zeros(pred.size(0), dtype=torch.long, device=pred.device)
            losses[loss_name] = self._loss_modules[loss_name](pred, label)

        return losses

    def predict(self, batch: Batch) -> Dict[str, Tensor]:
        """Forward the model.

        Args:
            batch (Batch): input batch data.

        Return:
            predictions (dict): a dict of predicted result.
        """
        user_tower_emb = self.user_tower(batch)
        item_tower_emb = self.item_tower(batch)

        similarity = self.sim(user_tower_emb, item_tower_emb) / self._model_config.temperature

        return {"similarity": similarity}