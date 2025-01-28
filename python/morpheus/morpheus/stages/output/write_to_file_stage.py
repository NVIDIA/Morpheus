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
"""Write to file stage."""

import typing

import mrc
import mrc.core.operators as ops

from morpheus.cli.register_stage import register_stage
from morpheus.common import FileTypes
from morpheus.config import Config
from morpheus.controllers.write_to_file_controller import WriteToFileController
from morpheus.messages import MessageMeta
from morpheus.pipeline.execution_mode_mixins import GpuAndCpuMixin
from morpheus.pipeline.pass_thru_type_mixin import PassThruTypeMixin
from morpheus.pipeline.single_port_stage import SinglePortStage


@register_stage("to-file", rename_options={"include_index_col": "--include-index-col"})
class WriteToFileStage(GpuAndCpuMixin, PassThruTypeMixin, SinglePortStage):
    """
    Write all messages to a file.

    This class writes messages to a file.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    filename : str
        Name of the file to which the messages will be written.
    overwrite : boolean, default = False, is_flag = True
        Overwrite file if exists. Will generate an error otherwise.
    file_type : `morpheus.common.FileTypes`, optional, case_sensitive = False
        Indicates what type of file to write. Specifying 'auto' will determine the file type from the extension.
        Supported extensions: 'csv', 'json' and 'jsonlines'
    include_index_col : bool, default = True
        Write out the index as a column, by default True.
    flush : bool, default = False, is_flag = True
        When `True` flush the output buffer to disk on each message.
    """

    def __init__(self,
                 c: Config,
                 filename: str,
                 overwrite: bool = False,
                 file_type: FileTypes = FileTypes.Auto,
                 include_index_col: bool = True,
                 flush: bool = False):

        super().__init__(c)

        self._controller = WriteToFileController(filename=filename,
                                                 overwrite=overwrite,
                                                 file_type=file_type,
                                                 include_index_col=include_index_col,
                                                 flush=flush)

    @property
    def name(self) -> str:
        """Returns the name of this stage."""
        return "to-file"

    def accepted_types(self) -> typing.Tuple:
        """
        Returns accepted input types for this stage.

        Returns
        -------
        typing.Tuple(`morpheus.pipeline.messages.MessageMeta`, )
            Accepted input types.

        """
        return (MessageMeta, )

    def supports_cpp_node(self):
        """Indicates whether this stage supports a C++ node."""
        return True

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        # Sink to file
        if (self._build_cpp_node()):
            import morpheus._lib.stages as _stages
            to_file_node = _stages.WriteToFileStage(builder,
                                                    self.unique_name,
                                                    self._controller.output_file,
                                                    "w",
                                                    self._controller.file_type,
                                                    self._controller.include_index_col,
                                                    self._controller.flush)
        else:

            to_file_node = builder.make_node(self.unique_name, ops.build(self._controller.node_fn))

        builder.make_edge(input_node, to_file_node)

        return to_file_node
