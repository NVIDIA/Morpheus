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
from functools import reduce

import fsspec
from tqdm import tqdm

from morpheus.common import IndicatorsFontStyle
from morpheus.common import IndicatorsTextColor
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.utils.logger import LogLevels
from morpheus.utils.monitor_utils import MorpheusTqdm
from morpheus.utils.type_aliases import DataFrameType
from morpheus.utils.type_utils import is_dataframe

logger = logging.getLogger(__name__)


class MonitorController:
    """
    Controls and displays throughput numbers at a specific point in the pipeline.

    Parameters
    ----------
    position: int
        Specifies the monitor's position on the console.
    description : str, default = "Progress"
        Name to show for this Monitor Stage in the console window.
    smoothing : float
        Smoothing parameter to determine how much the throughput should be averaged. 0 = Instantaneous, 1 =
        Average.
    unit : str
        Units to show in the rate value.
    delayed_start : bool
        When delayed_start is enabled, the progress bar will not be shown until the first message is received.
        Otherwise, the progress bar is shown on pipeline startup and will begin timing immediately. In large pipelines,
        this option may be desired to give a more accurate timing.
    determine_count_fn : typing.Callable[[typing.Any], int]
        Custom function for determining the count in a message. Gets called for each message. Allows for
        correct counting of batched and sliced messages.
    log_level : `morpheus.utils.logger.LogLevels`, default = 'INFO'
        Enable this stage when the configured log level is at `log_level` or lower.
    tqdm_class: `tqdm`, default = None
        Custom implementation of tqdm if required.
    """

    SupportedTypes = typing.Union[DataFrameType, MessageMeta, ControlMessage, list]
    controller_count: int = 0

    def __init__(
        self,
        position: int,
        description: str,
        smoothing: float,
        unit: str,
        delayed_start: bool,
        determine_count_fn: typing.Callable[[typing.Any], int],
        log_level: LogLevels,
        tqdm_class: tqdm = None,
        text_color: IndicatorsTextColor = IndicatorsTextColor.cyan,
        font_style: IndicatorsFontStyle = IndicatorsFontStyle.bold,
    ):

        self._progress: tqdm = None
        self._position = position
        self._description = description
        self._smoothing = smoothing
        self._unit = unit
        self._delayed_start = delayed_start
        self._determine_count_fn = determine_count_fn
        self._text_color = text_color
        self._font_style = font_style
        self._tqdm_class = tqdm_class if tqdm_class else MorpheusTqdm

        if isinstance(log_level, LogLevels):  # pylint: disable=isinstance-second-argument-not-valid-type
            log_level = log_level.value

        self._log_level = log_level
        self._enabled = None  # defined on first call to _is_enabled

    @property
    def delayed_start(self) -> bool:
        return self._delayed_start

    @property
    def progress(self) -> tqdm:
        return self._progress

    def is_enabled(self) -> bool:
        """
        Returns a boolean indicating whether or not the logger is enabled.
        """

        if self._enabled is None:
            self._enabled = logger.isEnabledFor(self._log_level)

        return self._enabled

    def ensure_progress_bar(self):
        """
        Ensures that the progress bar is initialized and ready for display.
        """

        if (self._progress is None):
            self._progress = self._tqdm_class(desc=self._description,
                                              smoothing=self._smoothing,
                                              dynamic_ncols=True,
                                              unit=(self._unit if self._unit.startswith(" ") else f" {self._unit}"),
                                              mininterval=0.25,
                                              maxinterval=1.0,
                                              miniters=1,
                                              position=self._position)

            self._progress.reset()

    def refresh_progress(self, _):
        """
        Refreshes the progress bar display.
        """
        self._progress.refresh()

    def progress_sink(self, msg: SupportedTypes) -> SupportedTypes:
        """
        Receives a message and determines the count of the message.
        The progress bar is displayed and the progress is updated.

        Parameters
        ----------
        msg: SupportedTypes
            Message that determines the count of the message

        Returns
        -------
        SupportedTypes
        """

        # Make sure the progress bar is shown
        self.ensure_progress_bar()

        if (self._determine_count_fn is None):
            self._determine_count_fn = self.auto_count_fn(msg)

        # Skip incase we have empty objects
        if (self._determine_count_fn is None):
            return msg

        # Do our best to determine the count
        count = self._determine_count_fn(msg)

        self._progress.update(n=count)

        return msg

    def auto_count_fn(self, msg: SupportedTypes) -> typing.Callable[[SupportedTypes], int] | None:
        """
        This is a helper function that is used to determine the count of messages received by the
        monitor.

        Parameters
        ----------
        msg: SupportedTypes
            Message that determines the count of the message

        Returns
        -------
        Message count.

        """

        # pylint: disable=too-many-return-statements

        if (msg is None):
            return None

        # Wait for a list thats not empty
        if (isinstance(msg, list) and len(msg) == 0):
            return None

        if (is_dataframe(msg)):
            return lambda y: len(y.index)

        if (isinstance(msg, MessageMeta)):
            return lambda y: y.count

        if isinstance(msg, ControlMessage):

            def check_df(y):
                df = y.payload().df
                if df is not None:
                    return len(df)

                return 0

            return check_df

        if (isinstance(msg, list)):
            item_count_fn = self.auto_count_fn(msg[0])
            return lambda y: reduce(lambda sum, z, item_count_fn=item_count_fn: sum + item_count_fn(z), y, 0)

        if (isinstance(msg, (str, fsspec.core.OpenFile))):
            return lambda y: 1

        if (hasattr(msg, "__len__")):
            return len  # Return len directly (same as `lambda y: len(y)`)

        raise NotImplementedError(f"Unsupported type: {type(msg)}")

    def sink_on_completed(self):
        """
        Stops the progress bar and prevents the monitors from writing over each other when the last
        stage completes.
        """

        # Ensure that the progress bar exists even if we dont have any values
        self.ensure_progress_bar()

        # Set the name to complete. This refreshes the display
        self.progress.set_description_str(self.progress.desc + "[Complete]")

        self.progress.stop()

        # To prevent the monitors from writing over eachother, stop the monitor when the last stage completes
        MonitorController.controller_count -= 1

        if (MonitorController.controller_count <= 0 and self._tqdm_class.monitor is not None):
            self._tqdm_class.monitor.exit()
            self._tqdm_class.monitor = None
