# Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

import string

import pytest

import cudf

from morpheus.utils.cudf_subword_helper import tokenize_text_series


@pytest.mark.parametrize("seq_length", [10, 256, 1024])
@pytest.mark.parametrize("do_lower_case", [False, True])
@pytest.mark.parametrize("add_special_tokens", [False, True])
def test_needs_trunc_error(bert_cased_hash: str, seq_length: int, do_lower_case: bool, add_special_tokens: bool):
    """
    Feeding the subword tokenizer with a string that is too long should raise an error rather than
    a duplicate in the id list
    """

    short_string = string.ascii_lowercase[0:seq_length - 1]

    long_string = list(string.ascii_lowercase)
    while len(long_string) <= seq_length:
        long_string.extend(string.ascii_lowercase)

    long_string = "".join(long_string)

    series = cudf.Series([short_string, long_string])

    # Infer the value of stride the same way that the PreprocessNLPStage does
    stride = (seq_length // 2) + (seq_length // 4)

    with pytest.raises(ValueError):
        tokenize_text_series(vocab_hash_file=bert_cased_hash,
                             do_lower_case=do_lower_case,
                             text_ser=series,
                             seq_len=seq_length,
                             truncation=False,
                             stride=stride,
                             add_special_tokens=add_special_tokens)
