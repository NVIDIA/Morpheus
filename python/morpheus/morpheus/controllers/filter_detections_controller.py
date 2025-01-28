# Copyright (c) 2022-2025, NVIDIA CORPORATION.
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

import numpy as np

from morpheus.common import FilterSource
from morpheus.messages import ControlMessage
from morpheus.utils.type_aliases import NDArrayType

logger = logging.getLogger(__name__)


class FilterDetectionsController:
    """
    Controller class for filtering detections based on a specified threshold and source.

    Parameters
    ----------
    threshold : float
        The threshold value for filtering detections.
    filter_source : `morpheus.common.FilterSource`
        The source used for filtering.
    field_name : str
        The name of the field used for filtering.
    """

    def __init__(self, threshold: float, filter_source: FilterSource, field_name: str) -> None:
        self._threshold = threshold
        self._filter_source = filter_source
        self._field_name = field_name

    @property
    def threshold(self):
        """
        Get the threshold value.
        """
        return self._threshold

    @property
    def filter_source(self):
        """
        Get the filter source.
        """
        return self._filter_source

    @property
    def field_name(self):
        """
        Get the field name.
        """
        return self._field_name

    def _find_detections(self, msg: ControlMessage) -> NDArrayType:
        # Determine the filter source
        if self._filter_source == FilterSource.TENSOR:
            filter_source = msg.tensors().get_tensor(self._field_name)
        else:
            filter_source = msg.payload().get_data(self._field_name).values

        # Get per row detections
        detections = (filter_source > self._threshold)

        if (len(detections.shape) > 1):
            detections = detections.any(axis=1)

        # Surround in False to ensure we get an even number of pairs
        detections = np.concatenate([np.array([False]), detections, np.array([False])])

        return np.where(detections[1:] != detections[:-1])[0].reshape((-1, 2))

    def filter_copy(self, msg: ControlMessage) -> ControlMessage:
        """
        This function uses a threshold value to filter the messages.

        Parameters
        ----------
        msg : `morpheus.messages.ControlMessasge`
            Response message with probabilities calculated from inference results.

        Returns
        -------
        `morpheus.messages.ControlMessage`
            A new message containing a copy of the rows above the threshold.

        """
        if msg is None:
            return None

        true_pairs = self._find_detections(msg)

        # If we didnt have any detections, return None
        if (true_pairs.shape[0] == 0):
            return None

        meta = msg.payload()
        msg.payload(meta.copy_ranges(true_pairs))
        return msg

    def filter_slice(self, msg: ControlMessage) -> list[ControlMessage]:
        """
        This function uses a threshold value to filter the messages.

        Parameters
        ----------
        msg : `morpheus.messages.ControlMessage`
            Response message with probabilities calculated from inference results.

        Returns
        -------
        list[`morpheus.messages.ControlMessage`]
            List of filtered messages.

        """
        # Unfortunately we have to convert this to a list in case there are non-contiguous groups
        output_list = []
        if msg is not None:
            true_pairs = self._find_detections(msg)
            for pair in true_pairs:
                pair = tuple(pair.tolist())
                if ((pair[1] - pair[0]) > 0):
                    sliced_meta = msg.payload().get_slice(*pair)
                    cm = ControlMessage(msg)
                    cm.payload(sliced_meta)
                    output_list.append(cm)

        return output_list

    def update_filter_source(self, message_type: typing.Any):
        """
        This function updates filter source.

        Parameters
        ----------
        message_type : `typing.Any`
            Response message with probabilities calculated from inference results.
        """

        # Unfortunately we have to convert this to a list in case there are non-contiguous groups
        if self._filter_source == FilterSource.Auto:
            self._filter_source = FilterSource.DATAFRAME

            logger.debug(
                "filter_source was set to Auto, inferring a filter source of %s based on an input "
                "message type of %s",
                self._filter_source,
                message_type)
