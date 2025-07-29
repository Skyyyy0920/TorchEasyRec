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
from modelscope import AutoTokenizer

from tzrec.datasets.utils import ParsedData, SparseData
from tzrec.features.feature import (
    BaseFeature,
    FgMode,
)
from tzrec.protos.feature_pb2 import FeatureConfig

CATEGORY_MAPPING = {
    'OTHER': '其他',
    'CHINASTUDIES': '国学',
    'YOGA': '瑜伽',
    'ZHONGYI': '中医',
    'Pilates': '普拉提',
    'SHORT_VIDEO': '短视频',
    'PHONE_PHOTOGRAPHY': '手机摄影',
    'SING': '唱歌',
    'CHUANDA': '穿搭',
    'EAT_THIN': '吃瘦',
    'FIVEFOWLPLAYS': '五禽戏',
    'Cameraphotography': '相机摄影',
    'QIXUE_TIAOLI': '气血调理',
    'LIFE': '生活',
    'GUQIHUOXUE': '古琴活学',
    'DANCE': '舞蹈',
    'ASTROLOGY': '占星',
    'YIJING': '易经',
    'ZMGJ': '正面管教',
    'Videoclip': '视频剪辑',
    'TAROT': '塔罗',
    'shequceshileimu': '社区测试类目',
    np.nan: '未知类别',
    None: '未知类别',
    '': '未知类别'
}
SYSTEM_USER_PROMPT = """你是一个用户表征编码器，将以下用户特征转换为适合推荐系统使用的高质量表征向量。"""
SYSTEM_ITEM_PROMPT = """你是一个内容表征编码器，将以下内容特征转化为适合推荐系统使用的高质量表征向量。"""

def build_user_prompt(row):
    base_template = """用户是来自{city}的{gender}性，账户等级{growth_level}{new_user_tag}。
最近活跃于周{week_day}{day_h}时，历史点击{item_cnt}次内容。
用户关注{follow_cnt}人，拥有{follower_cnt}粉丝，收藏{favorite_cnt}内容。
已购买{buy_camp_cnt}门课程，{category_name_list}。
用户近期点击内容类别：{formatted_category_seq}
近期点击内容标题：{formatted_item_title_seq}
根据以上用户特征生成综合表征向量："""


    def format_seq(seq):
        if len(seq) > 30:
            leng = len(seq)
            seq = seq[leng-30:]
        seq = [f"'{item}'" for item in seq if item.strip() != '']
        return ", ".join([f"{item}" for item in seq])

    def analyze_purchases(cat_list):
        counter = Counter(cat_list)
        total = len(cat_list)

        frequent = [f"{k}{v}门" for k,v in counter.items()]
        return "其中" + "，".join(frequent) if total > 0 else "暂无显著消费倾向"

    def format_category_seq(seq):
        counter = Counter(seq)

        main_cats = {k:v for k,v in counter.items()}
        sorted_cats = sorted(main_cats.items(), key=lambda x: -x[1])

        parts = []
        for cat, cnt in sorted_cats:
            translated = CATEGORY_MAPPING.get(cat, cat)
            parts.append(f"{translated}{cnt}次")

        return "、".join(parts)

    week_day_cn = "一二三四五六七"
    week_day = week_day_cn[row["week_day"]-1]
    new_user = "是新用户，" if row['is_new_user'] == 1 else ""
    if row['gender'] == 0:
        gender = '男'
    elif row['gender'] == 1:
        gender = '女'
    else:
        gender = '未知'

    prompt = base_template.format(
        city=row['city'],
        gender=gender,
        day_h=row['day_h'],
        week_day=week_day,
        is_new_user=new_user,
        growth_level=f"{row['growth_level'] or 0:.0f}",
        new_user_tag="(新用户)" if row['is_new_user'] == 1 else "",
        buy_camp_cnt=f"{row['buy_camp_cnt'] or 0:.0f}",
        item_cnt=f"{row['item_cnt'] or 0:.0f}",
        follow_cnt=f"{row['follow_cnt'] or 0:.0f}",
        follower_cnt=f"{row['follower_cnt'] or 0:.0f}",
        favorite_cnt=f"{row['favorite_cnt'] or 0:.0f}",
        category_name_list=analyze_purchases(row['category_name_list']),
        formatted_category_seq=format_category_seq(row['click_50_seq__category']),
        formatted_item_title_seq=format_seq(row['click_50_seq__item_title']),
    )

    return prompt


def build_item_prompt(row):
    base_template = """内容标题为：{title}，一级标签为{category}。类型为{type}，发布于{pub_time}，{status}被推荐。
作者身份为{author_status}，发布源为{publish}。
内容获得{praise}点赞，{comment}评论，{collect}收藏，{share}分享。
根据以上内容特征生成综合表征向量："""


    if row['pub_time'] is None:
        pub_time = datetime.now().strftime("%Y-%m-%d")
    else:
        pub_time = datetime.fromtimestamp(row['pub_time']).strftime("%Y-%m-%d")
    status = "已" if row['status'] else "未"
    home_mark = "属于" if row['home_mark'] == 'Y' else "不属于"
    club_mark = "属于" if row['club_mark'] == 'Y' else "不属于"

    prompt = base_template.format(
        type=row['item_type'],
        pub_time=pub_time,
        status=status,
        title=row['title'],
        category=CATEGORY_MAPPING.get(row['category'], row['category']),
        author_status=row['author_status'],
        praise=f"{row['praise_count'] or 0:.0f}",
        comment=f"{row['comment_count'] or 0:.0f}",
        collect=f"{row['collect_count'] or 0:.0f}",
        share=f"{row['share_count'] or 0:.0f}",
        publish=row['publish_source'],
        home_mark=home_mark,
        club_mark=club_mark,
    )

    return prompt


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
        """Parse row data to generate prompt."""
        if self.prompt_type == 'user':
            return build_user_prompt(row_data)
        elif self.prompt_type == 'item':
            return build_item_prompt(row_data)
        else:
            raise ValueError(f"Unsupported prompt_type: {self.prompt_type}")

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

    def generate_chat_messages(self, features: List[Dict[str, Any]]) -> List[List[Dict[str, str]]]:
        """Generate chat messages for LLM processing."""
        if self._prompt_type == 'user':
            system_prompt = SYSTEM_USER_PROMPT
            prompt_builder = build_user_prompt
        elif self._prompt_type == 'item':
            system_prompt = SYSTEM_ITEM_PROMPT
            prompt_builder = build_item_prompt
        else:
            raise ValueError(f"Unsupported prompt_type: {self._prompt_type}")

        messages = [
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_builder(feat['sample'])}
            ] for feat in features
        ]

        return messages
