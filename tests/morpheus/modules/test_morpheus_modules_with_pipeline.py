# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import tempfile

import cudf

import morpheus.modules  # noqa: F401 # pylint: disable=unused-import
from morpheus import messages
from morpheus.utils.module_ids import FILTER_CM_FAILED

from .morpheus_module_test_pipeline import MorpheusModuleTestPipeline

PACKET_COUNT = 5
PACKETS_RECEIVED = 0


def test_payload_loader_module():
    df = cudf.DataFrame({
        'col1': [1, 2, 3, 4, 5],
        'col2': [1.1, 2.2, 3.3, 4.4, 5.5],
        'col3': ['a', 'b', 'c', 'd', 'e'],
        'col4': [True, False, True, False, True]
    })

    def gen_data():
        payload = messages.MessageMeta(df)
        for _ in range(PACKET_COUNT):
            config = {"tasks": [{"type": "load", "properties": {"loader_id": "payload", "strategy": "aggregate"}}]}
            msg = messages.ControlMessage(config)
            msg.payload(payload)
            yield msg

    def on_next(control_msg):
        # pylint: disable=global-statement
        global PACKETS_RECEIVED
        PACKETS_RECEIVED += 1
        assert (control_msg.payload().df.equals(df))

    config = {"loaders": [{"id": "payload", "properties": {"file_types": "something", "prop2": "something else"}}]}
    test_pipeline = MorpheusModuleTestPipeline("DataLoader",
                                               "morpheus",
                                               "ModuleDataLoaderTest",
                                               config,
                                               "input",
                                               "output",
                                               gen_data,
                                               on_next,
                                               on_error=None,
                                               on_complete=None)
    test_pipeline.run()

    assert (PACKETS_RECEIVED == PACKET_COUNT)


def test_file_loader_module():
    # pylint: disable=global-statement
    global PACKETS_RECEIVED
    PACKETS_RECEIVED = 0

    df = cudf.DataFrame(
        {
            'col1': [1, 2, 3, 4, 5],
            'col2': [1.1, 2.2, 3.3, 4.4, 5.5],
            'col3': ['a', 'b', 'c', 'd', 'e'],
            'col4': [True, False, True, False, True]
        },
        columns=['col1', 'col2', 'col3', 'col4'])

    files = []
    file_types = ["csv", "parquet", "orc"]
    for ftype in file_types:
        with tempfile.NamedTemporaryFile(suffix=f".{ftype}", delete=False) as _tempfile:
            filename = _tempfile.name

            if ftype == "csv":
                df.to_csv(filename, index=False)
            elif ftype == "parquet":
                df.to_parquet(filename)
            elif ftype == "orc":
                df.to_orc(filename)

            files.append((filename, ftype))

    def gen_data():
        for f in files:
            # Check with the file type
            config = {
                "tasks": [{
                    "type": "load",
                    "properties": {
                        "loader_id": "file", "strategy": "aggregate", "files": [{
                            "path": f[0], "type": f[1]
                        }]
                    }
                }]
            }
            msg = messages.ControlMessage(config)
            yield msg

            # Make sure we can auto-detect the file type
            config = {
                "tasks": [{
                    "type": "load",
                    "properties": {
                        "loader_id": "file", "strategy": "aggregate", "files": [{
                            "path": f[0],
                        }]
                    }
                }]
            }
            msg = messages.ControlMessage(config)
            yield msg

    def on_next(control_msg):
        global PACKETS_RECEIVED
        PACKETS_RECEIVED += 1
        assert (control_msg.payload().df.equals(df))

    config = {"loaders": [{"id": "file", "properties": {"file_types": "something", "prop2": "something else"}}]}
    test_pipeline = MorpheusModuleTestPipeline("DataLoader",
                                               "morpheus",
                                               "ModuleDataLoaderTest",
                                               config,
                                               "input",
                                               "output",
                                               gen_data,
                                               on_next,
                                               on_error=None,
                                               on_complete=None)
    test_pipeline.run()

    assert (PACKETS_RECEIVED == len(files) * 2)

    for f in files:
        os.remove(f[0])


def test_filter_cm_failed():
    # pylint: disable=global-statement
    global PACKETS_RECEIVED
    PACKETS_RECEIVED = 0

    def gen_data():
        msg = messages.ControlMessage()
        msg.set_metadata("cm_failed", "true")
        msg.set_metadata("cm_failed_reason", "cm_failed_reason_1")
        yield msg

        msg = messages.ControlMessage()
        msg.set_metadata("cm_failed", "true")
        yield msg

        msg = messages.ControlMessage()
        msg.set_metadata("cm_failed", "false")
        msg = messages.ControlMessage(config)
        yield msg

    def on_next(control_msg):
        # pylint: disable=global-statement
        global PACKETS_RECEIVED
        if control_msg:
            PACKETS_RECEIVED += 1

    config = {}
    test_pipeline = MorpheusModuleTestPipeline(FILTER_CM_FAILED,
                                               "morpheus",
                                               "filter_cm_failed",
                                               config,
                                               "input",
                                               "output",
                                               gen_data,
                                               on_next,
                                               on_error=None,
                                               on_complete=None)
    test_pipeline.run()

    assert PACKETS_RECEIVED == 1


if (__name__ == "__main__"):
    test_payload_loader_module()
    test_file_loader_module()
    test_filter_cm_failed()
