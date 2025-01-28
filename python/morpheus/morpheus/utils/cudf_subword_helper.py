# Copyright (c) 2021-2025, NVIDIA CORPORATION.
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
"""Wrapper around cudf's subword tokenizer"""

import collections
import threading

import cudf
from cudf.core.subword_tokenizer import SubwordTokenizer

_tl = threading.local()

Feature = collections.namedtuple(  # pylint: disable=invalid-name
    "Feature", ["input_ids", "input_mask", "segment_ids"])


def create_vocab_table(vocabpath):
    """
    Create Vocabulary tables from the vocab.txt file

    Parameters
    ----------
    vocabpath : str
        Path of vocablary file

    Returns
    -------
    np.array
        id2vocab: np.array, dtype=<U5

    """
    id2vocab = []
    vocab2id = {}
    import numpy as np
    with open(vocabpath, encoding='UTF-8') as f:
        for index, line in enumerate(f):
            token = line.split()[0]
            id2vocab.append(token)
            vocab2id[token] = index
    return np.array(id2vocab), vocab2id


def create_tokenizer(vocab_hash_file: str, do_lower_case: bool):
    """_summary_

    Parameters
    ----------
    vocab_hash_file : str
        Path to hash file containing vocabulary of words with token-ids. This can be created from the raw vocabulary
        using the `cudf.utils.hash_vocab_utils.hash_vocab` function.
    do_lower_case : bool
        If set to true, original text will be lowercased before encoding.

    Returns
    -------
    cudf.core.subword_tokenizer.SubwordTokenizer
        Subword tokenizer
    """
    tokenizer = SubwordTokenizer(vocab_hash_file, do_lower_case=do_lower_case)

    return tokenizer


def get_cached_tokenizer(vocab_hash_file: str, do_lower_case: bool):
    """
    Get cached subword tokenizer. Creates tokenizer and caches it if it does not already exist.

    Parameters
    ----------
    vocab_hash_file : str
        Path to hash file containing vocabulary of words with token-ids. This can be created from the raw vocabulary
        using the `cudf.utils.hash_vocab_utils.hash_vocab` function.
    do_lower_case : bool
        If set to true, original text will be lowercased before encoding.

    Returns
    -------
    cudf.core.subword_tokenizer.SubwordTokenizer
        Cached subword tokenizer
    """
    hashed_inputs = hash((vocab_hash_file, do_lower_case))

    cached_tokenizers = getattr(_tl, "cached_tokenizers", None)

    # Set the initial dictionary if it's not set
    if (cached_tokenizers is None):
        cached_tokenizers = {}
        _tl.cached_tokenizers = cached_tokenizers

    # Check for cache miss
    if (hashed_inputs not in cached_tokenizers):
        cached_tokenizers[hashed_inputs] = create_tokenizer(vocab_hash_file, do_lower_case)

    return cached_tokenizers[hashed_inputs]


def tokenize_text_series(vocab_hash_file: str,
                         do_lower_case: bool,
                         text_ser: cudf.Series,
                         seq_len: int,
                         stride: int,
                         truncation: bool,
                         add_special_tokens: bool):
    """
    This function tokenizes a text series using the bert subword_tokenizer and vocab-hash

    Parameters
    ----------
    vocab_hash_file : str
        vocab_hash_file to use (Created using `perfect_hash.py` with compact flag)
    do_lower_case : bool
        If set to true, original text will be lowercased before encoding.
    text_ser : cudf.Series
        Text Series to tokenize
    seq_len : int
        Sequence Length to use (We add to special tokens for ner classification job)
    stride : int
        Stride for the tokenizer
    truncation : bool
        If set to true, strings will be truncated and padded to max_length. Each input string will result in exactly one
        output sequence. If set to false, there may be multiple output sequences when the max_length is smaller
        than generated tokens.
    add_special_tokens : bool
        Whether or not to encode the sequences with the special tokens of the BERT classification model.

    Returns
    -------
    collections.namedtuple
        A named tuple with these keys {'input_ids':,'input_mask':,'segment_ids':}

    """
    tokenizer = get_cached_tokenizer(vocab_hash_file, do_lower_case)

    assert tokenizer is not None, "Must create tokenizer first using `create_tokenizer()`"

    if len(text_ser) == 0:
        return Feature(None, None, None)

    max_rows_tensor = len(text_ser) * 2
    max_length = seq_len

    # Preflight check to ensure that the input strings are not too long
    if not truncation:
        max_value_length = text_ser.str.len().max()
        if max_value_length > max_length:
            raise ValueError(
                f"Input strings are too long ({max_value_length}) to be tokenized without truncation seq_len={seq_len}")

    # Call the tokenizer
    tokenizer_output = tokenizer(text_ser,
                                 max_length=max_length,
                                 max_num_rows=max_rows_tensor,
                                 add_special_tokens=add_special_tokens,
                                 padding='max_length',
                                 truncation=truncation,
                                 stride=stride,
                                 return_tensors="cp",
                                 return_token_type_ids=False)

    input_ids = tokenizer_output["input_ids"]
    attention_mask = tokenizer_output["attention_mask"]
    metadata = tokenizer_output["metadata"]

    output_f = Feature(input_ids=input_ids, input_mask=attention_mask, segment_ids=metadata)

    return output_f
