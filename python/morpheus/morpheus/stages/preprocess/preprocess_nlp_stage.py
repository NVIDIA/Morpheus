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

import logging

import mrc

from morpheus.cli.register_stage import register_stage
from morpheus.cli.utils import MorpheusRelativePath
from morpheus.cli.utils import get_package_relative_file
from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.stages.preprocess.preprocess_base_stage import PreprocessBaseStage

logger = logging.getLogger(__name__)


@register_stage(
    "preprocess",
    modes=[PipelineModes.NLP],
    option_args={"vocab_hash_file": {
        "type": MorpheusRelativePath(exists=True, dir_okay=False, resolve_path=True)
    }})
class PreprocessNLPStage(PreprocessBaseStage):
    """
    Prepare NLP input DataFrames for inference.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    vocab_hash_file : str
        Path to hash file containing vocabulary of words with token-ids. This can be created from the raw vocabulary
        using the `cudf.utils.hash_vocab_utils.hash_vocab` function.
    truncation : bool
        If set to true, strings will be truncated and padded to max_length. Each input string will result in exactly one
        output sequence. If set to false, there may be multiple output sequences when the max_length is smaller
        than generated tokens.
    do_lower_case : bool
        If set to true, original text will be lowercased before encoding.
    add_special_tokens : bool
        Whether to encode the sequences with the special tokens of the BERT classification model.
    stride : int
        If `truncation` == False and the tokenized string is larger than max_length, the sequences containing the
        overflowing token-ids can contain duplicated token-ids from the main sequence. If max_length is equal to stride
        there are no duplicated-id tokens. If stride is 80% of max_length, 20% of the first sequence will be repeated on
        the second sequence and so on until the entire sentence is encoded.
    column : str
        Name of the column containing the data that needs to be preprocessed.

    """

    def __init__(self,
                 c: Config,
                 vocab_hash_file: str = "data/bert-base-cased-hash.txt",
                 truncation: bool = False,
                 do_lower_case: bool = False,
                 add_special_tokens: bool = False,
                 stride: int = -1,
                 column: str = "data"):
        super().__init__(c)

        self._column = column
        self._seq_length = c.feature_length
        self._vocab_hash_file = get_package_relative_file(vocab_hash_file)

        if (stride <= 0):
            # Set the stride to 75%. Works well with powers of 2
            self._stride = self._seq_length // 2
            self._stride = self._stride + self._stride // 2
        else:
            # Use the given value
            self._stride = stride

        self._truncation = truncation
        self._do_lower_case = do_lower_case
        self._add_special_tokens = add_special_tokens

    @property
    def name(self) -> str:
        return "preprocess-nlp"

    def supports_cpp_node(self) -> bool:
        return True

    def _get_preprocess_node(self, builder: mrc.Builder):
        import morpheus._lib.stages as _stages
        return _stages.PreprocessNLPStage(builder,
                                          self.unique_name,
                                          self._vocab_hash_file,
                                          self._seq_length,
                                          self._truncation,
                                          self._do_lower_case,
                                          self._add_special_tokens,
                                          self._stride,
                                          self._column)
