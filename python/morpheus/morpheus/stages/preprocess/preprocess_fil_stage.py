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
from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.stages.preprocess.preprocess_base_stage import PreprocessBaseStage

logger = logging.getLogger(__name__)


@register_stage("preprocess", modes=[PipelineModes.FIL])
class PreprocessFILStage(PreprocessBaseStage):
    """
    Prepare FIL input DataFrames for inference.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.

    """

    def __init__(self, c: Config):
        super().__init__(c)

        self._fea_length = c.feature_length
        self.features = c.fil.feature_columns

        assert self._fea_length == len(self.features), \
            f"Number of features in preprocessing {len(self.features)}, does not match configuration {self._fea_length}"

    @property
    def name(self) -> str:
        return "preprocess-fil"

    def supports_cpp_node(self) -> bool:
        return True

    def _get_preprocess_node(self, builder: mrc.Builder):
        import morpheus._lib.stages as _stages
        return _stages.PreprocessFILStage(builder, self.unique_name, self.features)
