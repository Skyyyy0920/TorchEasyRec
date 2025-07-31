# Copyright (c) 2024, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyarrow as pa
from jinja2 import Template
from modelscope import AutoTokenizer

from tzrec.datasets.utils import ParsedData, SparseData
from tzrec.features.feature import (
    BaseFeature,
    FgMode,
)
from tzrec.protos.feature_pb2 import FeatureConfig

SYSTEM_USER_PROMPT = """你是一个用户表征编码器，将以下用户特征转换为适合推荐系统使用的高质量表征向量。"""
SYSTEM_ITEM_PROMPT = """你是一个内容表征编码器，将以下内容特征转化为适合推荐系统使用的高质量表征向量。"""


def tokens_to_sparse(model_inputs: Dict[str, np.ndarray], name: str) -> SparseData:
    """Transfer tokens to SparseData.

    Args:
        model_inputs: tokens，include 'input_ids' and 'attention_mask'
        name: feature name

    Return:
        SparseData, include input_ids and attention_mask.
    """
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]

    bool_mask = attention_mask.bool()
    values = input_ids[bool_mask].cpu().numpy()
    lengths = attention_mask.sum(dim=1).cpu().numpy().astype(np.int32)

    return SparseData(
        name=name,
        values=values,
        lengths=lengths
    )


class PromptFeature(BaseFeature):
    """PromptFeature class for LLM4Rec integration.

    This feature generates prompts for LLM-based recommendation systems,
    supporting both user and item prompt generation.

    Args:
        feature_config (FeatureConfig): a instance of feature config.
        fg_mode (FgMode): input data fg mode.
        fg_encoded_multival_sep (str, optional): multival_sep when fg_mode=FG_NONE
    """

    def __init__(
        self,
        feature_config: FeatureConfig,
        fg_mode: FgMode = FgMode.FG_NONE,
        fg_encoded_multival_sep: Optional[str] = None,
    ) -> None:
        super().__init__(feature_config, fg_mode, fg_encoded_multival_sep)

        self._tokenizer = AutoTokenizer.from_pretrained(f"Qwen/{self.config.tokenizer}")
        self.max_length = self.config.max_length
        
        self.prompt_template_path = self.config.prompt_template_path
        with open(self.prompt_template_path, "r", encoding="utf-8") as f:
            self._prompt_template = Template(f.read())

    @property
    def name(self) -> str:
        """Feature name."""
        return self.config.feature_name

    @property
    def value_dim(self) -> int:
        """Fg value dimension of the feature."""
        if self.config.HasField("value_dim"):
            return self.config.value_dim
        else:
            return 0

    @property
    def output_dim(self) -> int:
        """Output dimension of the feature after embedding."""
        if self.config.HasField("embedding_dim"):
            return self.config.embedding_dim
        else:
            return self.value_dim

    @property
    def is_sparse(self) -> bool:
        """Feature is sparse or dense."""
        return True

    @property
    def prompt_type(self) -> str:
        """Get prompt type (user or item)."""
        return self.config.prompt_type

    def _build_side_inputs(self) -> Optional[List[Tuple[str, str]]]:
        """Input field names with side."""
        if len(self.config.expression) > 0:
            return [tuple(x.split(":")) for x in self.config.expression]
        else:
            return None
        
    def _build_prompt(self, row_data: Dict[str, Any]) -> str:
        """Render the prompt using user-provided Jinja2 template."""
        return self._prompt_template.render(**row_data).strip()

    def _prepare_input(self, prompts: List[str]) -> Dict:
        """Tokenize a list of prompts to token ids in batch."""
        all_messages = []
        system_prompt = SYSTEM_ITEM_PROMPT if self.prompt_type == 'item' else SYSTEM_USER_PROMPT
        for prompt in prompts:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            all_messages.append(messages)
        texts = self._tokenizer.apply_chat_template(
            all_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        model_inputs = self._tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        return model_inputs


    def _parse(self, input_data: Dict[str, pa.Array]) -> ParsedData:
        """Parse input data for the feature impl.

        Args:
            input_data (dict): raw input feature data.

        Return:
            parsed feature data.
        """
        if self.fg_mode == FgMode.FG_NONE:
            # TODO(tianqiong) to be implemented
            pass
        elif self.fg_mode == FgMode.FG_NORMAL:
            # For FG_NORMAL mode, use the fg_op to process inputs
            input_feats = []
            for name in self.inputs:
                x = input_data[name]
                if pa.types.is_list(x.type):
                    x = x.fill_null([])
                input_feats.append(x.tolist())

            batch_size = len(input_feats[0])
            prompts_list = []

            for i in range(batch_size):
                sample_data = {}
                for j, field_name in enumerate(self.inputs):
                    if j < len(input_feats) and i < len(input_feats[j]):
                        sample_data[field_name] = input_feats[j][i]

                # Generate and tokenize prompt
                prompt = self._build_prompt(sample_data)
                if self.prompt_type == 'item':
                    print(prompt)
                prompts_list.append(prompt)

            model_inputs = self._prepare_input(prompts_list)
            parsed_feat = tokens_to_sparse(model_inputs, self.name)
        else:
            raise ValueError(
                f"fg_mode: {self.fg_mode} is not supported for PromptFeature."
            )

        return parsed_feat

    def fg_json(self) -> List[Dict[str, Any]]:
        """Get fg json config."""
        # TODO(tianqiong) to be implemented
        fg_cfg = {
            "feature_type": "prompt_feature",
            "feature_name": self.name,
            "prompt_type": self.prompt_type,
            "expression": list(self.config.expression),
            "value_type": "dense",
            "need_prefix": False,
        }

        if self.config.separator != "\x1d":
            fg_cfg["separator"] = self.config.separator

        return [fg_cfg]
