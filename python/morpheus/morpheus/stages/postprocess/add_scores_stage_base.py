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

import functools
import logging
import typing
from abc import abstractmethod

import mrc
import mrc.core.operators as ops

from morpheus.common import TypeId
from morpheus.config import Config
from morpheus.messages import ControlMessage
from morpheus.pipeline.execution_mode_mixins import GpuAndCpuMixin
from morpheus.pipeline.pass_thru_type_mixin import PassThruTypeMixin
from morpheus.pipeline.single_port_stage import SinglePortStage

logger = logging.getLogger(__name__)


class AddScoresStageBase(PassThruTypeMixin, GpuAndCpuMixin, SinglePortStage):
    """
    Base class for the `AddScoresStage` and `AddClassificationStage`

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
    probs_type : TypeId
        Datatype of the scores columns.
    threshold : typing.Optional[float]
        Converts all scores to a boolean value using this threshold. If `None`, scores are used, as-is.
    """

    def __init__(self,
                 c: Config,
                 *,
                 labels: typing.List[str] = None,
                 prefix: str = "",
                 probs_type: TypeId,
                 threshold: typing.Optional[float]):
        super().__init__(c)

        self._feature_length = c.feature_length
        self._labels = labels if labels is not None and len(labels) > 0 else c.class_labels
        self._prefix = prefix
        self._threshold = threshold

        self._class_labels = c.class_labels

        # Build the Index to Label map.
        self._idx2label = {}

        for label in self._labels:
            # All labels must be in class_labels in order to get their position
            if (label not in self._class_labels):
                logger.warning("The label '%s' is not in Config.class_labels and will be ignored", label)
                continue

            prefixed_label = self._prefix + label
            self._idx2label[self._class_labels.index(label)] = prefixed_label
            self._needed_columns[prefixed_label] = probs_type

        assert len(self._idx2label) > 0, "No labels were added to the stage"

    def accepted_types(self) -> typing.Tuple:
        """
        Accepted input types for this stage are returned.

        Returns
        -------
        typing.Tuple[`morpheus.messages.ControlMessage`,]
            Accepted input types.

        """
        return (ControlMessage, )

    @abstractmethod
    def _get_cpp_node(self, builder: mrc.Builder):
        pass

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        # Convert the messages to rows of strings
        if self._build_cpp_node():
            node = self._get_cpp_node(builder=builder)
        else:
            node = builder.make_node(
                self.unique_name,
                ops.map(functools.partial(self._add_labels, idx2label=self._idx2label, threshold=self._threshold)))

        builder.make_edge(input_node, node)

        # Return input type unchanged
        return node

    @typing.overload
    @staticmethod
    def _add_labels(msg: ControlMessage, idx2label: dict[int, str],
                    threshold: typing.Optional[float]) -> ControlMessage:
        ...

    @typing.overload
    @staticmethod
    def _add_labels(msg: ControlMessage, idx2label: dict[int, str],
                    threshold: typing.Optional[float]) -> ControlMessage:
        ...

    @staticmethod
    def _add_labels(msg: ControlMessage, idx2label: dict[int, str], threshold: typing.Optional[float]):
        probs = msg.tensors().get_tensor("probs")

        if (probs.shape[1] <= max(idx2label.keys())):
            raise RuntimeError(("Model output did not contain enough columns to fufill the requested labels. "
                                f"Label indexes: {idx2label}, Model output columns: {probs.shape[1]}"))

        if (threshold is not None):
            probs = (probs > threshold).astype(bool)

        # Do these one at a time to prevent failures
        for i, label in idx2label.items():
            msg.payload().set_data(label, probs[:, i])

        # Return the same object
        return msg
