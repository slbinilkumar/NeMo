# Copyright 2018 The Google AI Language Team Authors and
# The HuggingFace Inc. team.
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from transformers import RobertaTokenizer as HF_ROBERTA_TOKENIZER

from nemo.collections.common.tokenizers.huggingface_tokenizer import HuggingFaceTokenizer

__all__ = ['RobertaTokenizer']


class RobertaTokenizer(HF_ROBERTA_TOKENIZER, HuggingFaceTokenizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
