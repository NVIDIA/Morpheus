# Copyright (c) 2023-2025, NVIDIA CORPORATION.
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
"""To File Sink Module."""

import logging

import mrc

from morpheus.common import FileTypes
from morpheus.controllers.write_to_file_controller import WriteToFileController
from morpheus.utils.module_ids import MORPHEUS_MODULE_NAMESPACE
from morpheus.utils.module_ids import WRITE_TO_FILE
from morpheus.utils.module_utils import register_module

logger = logging.getLogger(__name__)


@register_module(WRITE_TO_FILE, MORPHEUS_MODULE_NAMESPACE)
def write_to_file(builder: mrc.Builder):
    """
    Write all messages to a file.

    Parameters
    ----------
    builder : mrc.Builder
        mrc Builder object.

    Notes
    -----
        Configurable Parameters:
            - filename (str): Path to the output file; Example: `output.csv`; Default: None
            - file_type (FileTypes): Type of file to write; Example: `FileTypes.CSV`; Default: `FileTypes.Auto`
            - flush (bool): If true, flush the file after each write; Example: `false`; Default: false
            - include_index_col (bool): If true, include the index column; Example: `false`; Default: true
            - overwrite (bool): If true, overwrite the file if it exists; Example: `true`; Default: false
    """
    config = builder.get_current_module_config()

    filename = config.get("filename", None)
    overwrite = config.get("overwrite", False)
    flush = config.get("flush", False)
    file_type = config.get("file_type", FileTypes.Auto)
    include_index_col = config.get("include_index_col", True)

    controller = WriteToFileController(filename=filename,
                                       overwrite=overwrite,
                                       file_type=file_type,
                                       include_index_col=include_index_col,
                                       flush=flush)

    node = builder.make_node(WRITE_TO_FILE, mrc.core.operators.build(controller.node_fn))

    # Register input and output port for a module.
    builder.register_module_input("input", node)
    builder.register_module_output("output", node)
