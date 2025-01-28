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
import typing

import mrc

from morpheus.cli.register_stage import register_stage
from morpheus.common import TypeId
from morpheus.config import Config
from morpheus.stages.postprocess.add_scores_stage_base import AddScoresStageBase

logger = logging.getLogger(__name__)


@register_stage("add-class", rename_options={"labels": "--label"})
class AddClassificationsStage(AddScoresStageBase):
    """
    Add detected classifications to each message.

    Add classification labels based on probabilities calculated in inference stage. Label indexes will be looked up in
    the Config.class_labels property. Uses default threshold of 0.5 for predictions.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    labels : typing.List[str], default = None, multiple = True, show_default = "[Config.class_labels]"
        Converts probability indexes into classification labels. Each item in the list will determine its index from the
        Config.class_labels property and must be one of the available class labels. Leave as None to add all labels in
        the Config.class_labels property.
    prefix : str, default = ""
        Prefix to add to each label. Allows adding labels different from the `Config.class_labels` property.
    probs_type : TypeId, default = "bool8"
        Datatype of the scores columns.
    threshold : typing.Optional[float], default = 0.5
        Converts all scores to a boolean value using this threshold. If `None`, scores are used, as-is.

    """

    def __init__(self,
                 c: Config,
                 *,
                 labels: typing.List[str] = None,
                 prefix: str = "",
                 probs_type: TypeId = TypeId.BOOL8,
                 threshold: float = 0.5):
        super().__init__(c, labels=labels, prefix=prefix, probs_type=probs_type, threshold=threshold)

    @property
    def name(self) -> str:
        return "add-class"

    def supports_cpp_node(self) -> bool:
        # Enable support by default
        return True

    def _get_cpp_node(self, builder: mrc.Builder):
        import morpheus._lib.stages as _stages

        return _stages.AddClassificationsStage(builder, self.unique_name, self._idx2label, self._threshold)
