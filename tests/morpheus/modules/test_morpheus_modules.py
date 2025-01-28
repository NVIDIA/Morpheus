#!/usr/bin/env python
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

import mrc
import pytest

import cudf

import morpheus.modules  # noqa: F401 # pylint: disable=unused-import
from morpheus import messages
from morpheus.utils.module_ids import FILTER_CM_FAILED


# pylint: disable=unused-argument
def on_next(control_msg):
    pass


def on_error():
    pass


def on_complete():
    pass


def test_contains_namespace():
    registry = mrc.ModuleRegistry

    assert registry.contains_namespace("morpheus")


def test_is_version_compatible():
    registry = mrc.ModuleRegistry

    release_version = [int(x) for x in mrc.__version__.split(".")]
    old_release_version = [22, 10, 0]
    no_version_patch = [22, 10]
    no_version_minor_and_patch = [22]

    assert registry.is_version_compatible(release_version)
    assert registry.is_version_compatible(old_release_version) is not True
    assert registry.is_version_compatible(no_version_patch) is not True
    assert registry.is_version_compatible(no_version_minor_and_patch) is not True


def test_get_module():
    registry = mrc.ModuleRegistry

    fn_constructor = registry.get_module_constructor("DataLoader", "morpheus")
    assert fn_constructor is not None

    config = {}
    # pylint: disable=unused-variable,not-callable
    module_instance = fn_constructor("ModuleDataLoaderTest", config)  # noqa: F841 -- we don't need to use it


def test_get_module_with_bad_config_no_loader():
    packet_count = 5

    def init_wrapper(builder: mrc.Builder):

        def gen_data():
            nonlocal packet_count
            for _ in range(packet_count):
                config = {
                    "tasks": [{
                        "type": "load",
                        "properties": {
                            "loader_id": "payload",
                            "strategy": "aggregate",
                        },
                    }]
                }
                msg = messages.ControlMessage(config)
                yield msg

        source = builder.make_source("source", gen_data)

        config = {"loaders": []}
        # This will unpack the config and forward its payload (MessageMeta) to the sink
        data_loader = builder.load_module("DataLoader", "morpheus", "ModuleDataLoaderTest", config)

        sink = builder.make_sink("sink", on_next, on_error, on_complete)

        builder.make_edge(source, data_loader.input_port("input"))
        builder.make_edge(data_loader.output_port("output"), sink)

    pipeline = mrc.Pipeline()
    pipeline.make_segment("main", init_wrapper)

    options = mrc.Options()
    options.topology.user_cpuset = "0-1"

    executor = mrc.Executor(options)
    executor.register_pipeline(pipeline)

    with pytest.raises(Exception):
        executor.start()
        executor.join()


def test_get_module_with_bad_loader_type():
    packet_count = 5

    def init_wrapper(builder: mrc.Builder):

        def gen_data():
            nonlocal packet_count
            for _ in range(packet_count):
                config = {
                    "tasks": [{
                        "type": "load",
                        "properties": {
                            "loader_id": "payload",
                            "strategy": "aggregate",
                        },
                    }]
                }
                msg = messages.ControlMessage(config)
                yield msg

        source = builder.make_source("source", gen_data)

        config = {
            "loaders": [{
                "id": "not_a_loader(tm)",
                "properties": {
                    "file_types": "something",
                    "prop2": "something else",
                },
            }]
        }
        # This will unpack the config and forward its payload (MessageMeta) to the sink
        data_loader = builder.load_module("DataLoader", "morpheus", "ModuleDataLoaderTest", config)

        sink = builder.make_sink("sink", on_next, on_error, on_complete)

        builder.make_edge(source, data_loader.input_port("input"))
        builder.make_edge(data_loader.output_port("output"), sink)

    pipeline = mrc.Pipeline()
    pipeline.make_segment("main", init_wrapper)


def test_get_module_with_bad_control_message():
    packet_count = 5

    def init_wrapper(builder: mrc.Builder):

        def gen_data():
            nonlocal packet_count
            for _ in range(packet_count):
                config = {
                    "tasks": [{
                        "type": "load",
                        "properties": {
                            "loader_id": "not_a_loader(tm)",
                            "strategy": "aggregate",
                        },
                    }]
                }
                msg = messages.ControlMessage(config)
                yield msg

        source = builder.make_source("source", gen_data)

        config = {
            "loaders": [{
                "id": "payload",
                "properties": {
                    "file_types": "something",
                    "prop2": "something else",
                },
            }]
        }
        # This will unpack the config and forward its payload (MessageMeta) to the sink
        data_loader = builder.load_module("DataLoader", "morpheus", "ModuleDataLoaderTest", config)

        sink = builder.make_sink("sink", on_next, on_error, on_complete)

        builder.make_edge(source, data_loader.input_port("input"))
        builder.make_edge(data_loader.output_port("output"), sink)

    with pytest.raises(Exception):
        pipeline = mrc.Pipeline()
        pipeline.make_segment("main", init_wrapper)

        options = mrc.Options()
        options.topology.user_cpuset = "0-1"

        executor = mrc.Executor(options)
        executor.register_pipeline(pipeline)

        executor.start()
        executor.join()


def test_payload_loader_module():
    packet_count = 5
    packets_received = 0
    registry = mrc.ModuleRegistry

    fn_constructor = registry.get_module_constructor("DataLoader", "morpheus")
    assert fn_constructor is not None

    def init_wrapper(builder: mrc.Builder):
        df = cudf.DataFrame({
            "col1": [1, 2, 3, 4, 5],
            "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
            "col3": ["a", "b", "c", "d", "e"],
            "col4": [True, False, True, False, True],
        })

        def gen_data():
            nonlocal packet_count
            config = {
                "tasks": [{
                    "type": "load",
                    "properties": {
                        "loader_id": "payload", "strategy": "aggregate"
                    },
                }]
            }

            payload = messages.MessageMeta(df)
            for _ in range(packet_count):
                msg = messages.ControlMessage(config)
                msg.payload(payload)

                yield msg

        def _on_next(control_msg):
            # pylint: disable=global-statement
            nonlocal packets_received
            packets_received += 1
            assert control_msg.payload().df.equals(df)

        source = builder.make_source("source", gen_data)

        config = {
            "loaders": [{
                "id": "payload",
                "properties": {
                    "file_types": "something",
                    "prop2": "something else",
                },
            }]
        }
        # This will unpack the config and forward its payload (MessageMeta) to the sink
        data_loader = builder.load_module("DataLoader", "morpheus", "ModuleDataLoaderTest", config)

        sink = builder.make_sink("sink", _on_next, on_error, on_complete)

        builder.make_edge(source, data_loader.input_port("input"))
        builder.make_edge(data_loader.output_port("output"), sink)

    pipeline = mrc.Pipeline()
    pipeline.make_segment("main", init_wrapper)

    options = mrc.Options()
    options.topology.user_cpuset = "0-1"

    executor = mrc.Executor(options)
    executor.register_pipeline(pipeline)
    executor.start()
    executor.join()

    assert packets_received == packet_count


# pylint: disable=no-value-for-parameter
def test_file_loader_module(tmp_path):
    packets_received = 0

    df = cudf.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
            "col3": ["a", "b", "c", "d", "e"],
            "col4": [True, False, True, False, True],
        },
        columns=["col1", "col2", "col3", "col4"],
    )

    files = []
    file_types = ["csv", "parquet", "orc"]
    for ftype in file_types:
        filename = (tmp_path / f"file.{ftype}").name

        if ftype == "csv":
            df.to_csv(filename, index=False)
        elif ftype == "parquet":
            df.to_parquet(filename)
        elif ftype == "orc":
            df.to_orc(filename)

        files.append((filename, ftype))

    def init_wrapper(builder: mrc.Builder):

        def gen_data():
            for f in files:
                # Check with the file type
                config = {
                    "tasks": [{
                        "type": "load",
                        "properties": {
                            "loader_id": "file",
                            "strategy": "aggregate",
                            "files": [{
                                "path": f[0], "type": f[1]
                            }],
                        },
                    }]
                }
                msg = messages.ControlMessage(config)
                yield msg

                # Make sure we can auto-detect the file type
                config = {
                    "tasks": [{
                        "type": "load",
                        "properties": {
                            "loader_id": "file",
                            "strategy": "aggregate",
                            "files": [{
                                "path": f[0],
                            }],
                        },
                    }]
                }
                msg = messages.ControlMessage(config)
                yield msg

        def _on_next(control_msg):
            nonlocal packets_received
            packets_received += 1
            assert control_msg.payload().df.equals(df)

        registry = mrc.ModuleRegistry

        fn_constructor = registry.get_module_constructor("DataLoader", "morpheus")
        assert fn_constructor is not None

        source = builder.make_source("source", gen_data)

        config = {
            "loaders": [{
                "id": "file",
                "properties": {
                    "file_types": "something",
                    "prop2": "something else",
                },
            }]
        }
        # This will unpack the config and forward its payload (MessageMeta) to the sink
        data_loader = builder.load_module("DataLoader", "morpheus", "ModuleDataLoaderTest", config)

        sink = builder.make_sink("sink", _on_next, on_error, on_complete)

        builder.make_edge(source, data_loader.input_port("input"))
        builder.make_edge(data_loader.output_port("output"), sink)

    pipeline = mrc.Pipeline()
    pipeline.make_segment("main", init_wrapper)

    options = mrc.Options()
    options.topology.user_cpuset = "0-1"

    executor = mrc.Executor(options)
    executor.register_pipeline(pipeline)
    executor.start()
    executor.join()

    assert packets_received == len(files) * 2

    for f in files:
        os.remove(f[0])


def test_filter_cm_failed():
    packets_received = 0

    def init_wrapper(builder: mrc.Builder):

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

        def _on_next(control_msg):
            nonlocal packets_received
            if control_msg:
                packets_received += 1

        source = builder.make_source("source", gen_data)

        config = {}
        filter_cm_failed_module = builder.load_module(FILTER_CM_FAILED, "morpheus", "filter_cm_failed", config)

        sink = builder.make_sink("sink", _on_next, on_error, on_complete)

        builder.make_edge(source, filter_cm_failed_module.input_port("input"))
        builder.make_edge(filter_cm_failed_module.output_port("output"), sink)

    pipeline = mrc.Pipeline()
    pipeline.make_segment("main", init_wrapper)

    options = mrc.Options()
    options.topology.user_cpuset = "0-1"

    executor = mrc.Executor(options)
    executor.register_pipeline(pipeline)
    executor.start()
    executor.join()

    assert packets_received == 1


# def test_rest_loader_module_get_without_params():
#     packets_received = 0
#     df1 = cudf.DataFrame(
#         {
#             "col1": [1, 2],
#             "col2": ["GET_param1_true", ""],
#         },
#         columns=["col1", "col2"],
#     )
#     df2 = cudf.DataFrame(
#         {
#             "col1": [1, 2],
#             "col2": ["GET_param2_true", ""],
#         },
#         columns=["col1", "col2"],
#     )
#     df = cudf.concat([df1, df2])

#     def init_wrapper(builder: mrc.Builder):
#         def gen_data():
#             config = {
#                 "tasks": [
#                     {
#                         "type": "load",
#                         "properties": {
#                             "loader_id": "rest",
#                             "strategy": "aggregate",
#                             "queries": [
#                                 {
#                                     "method": "GET",
#                                     "endpoint": "0.0.0.0/path?param1=true&param2=false",
#                                     "http_version": "1.0",
#                                     "port": "8081",
#                                 },
#                                 {
#                                     "method": "GET",
#                                     "endpoint": "0.0.0.0/path?param1=false&param2=true",
#                                     "port": "8081",
#                                 },
#                             ],
#                         },
#                     }
#                 ]
#             }
#             msg = messages.ControlMessage(config)
#             yield msg

#         def _on_next(control_msg):
#             nonlocal packets_received
#             packets_received += 1
#             assert control_msg.payload().df.equals(df)

#         registry = mrc.ModuleRegistry

#         fn_constructor = registry.get_module_constructor("DataLoader", "morpheus")
#         assert fn_constructor is not None

#         source = builder.make_source("source", gen_data)

#         config = {
#             "loaders": [
#                 {
#                     "id": "rest",
#                     "properties": {"prop1": "something", "prop2": "something else"},
#                 }
#             ]
#         }
#         data_loader = builder.load_module(
#             "DataLoader", "morpheus", "ModuleDataLoaderTest", config
#         )

#         sink = builder.make_sink("sink", _on_next, on_error, on_complete)

#         builder.make_edge(source, data_loader.input_port("input"))
#         builder.make_edge(data_loader.output_port("output"), sink)

#     pipeline = mrc.Pipeline()
#     pipeline.make_segment("main", init_wrapper)

#     options = mrc.Options()
#     options.topology.user_cpuset = "0-1"

#     executor = mrc.Executor(options)
#     executor.register_pipeline(pipeline)
#     executor.start()
#     executor.join()
#     assert packets_received == 1

# def test_rest_loader_module_get_with_params():
#     packets_received = 0
#     df1 = cudf.DataFrame(
#         {
#             "col1": [1, 2],
#             "col2": ["GET_param1_true", ""],
#         },
#         columns=["col1", "col2"],
#     )
#     df2 = cudf.DataFrame(
#         {
#             "col1": [1, 2],
#             "col2": ["GET_param2_true", ""],
#         },
#         columns=["col1", "col2"],
#     )
#     df = cudf.concat([df1, df2])

#     def init_wrapper(builder: mrc.Builder):
#         def gen_data():
#             config = {
#                 "tasks": [
#                     {
#                         "type": "load",
#                         "properties": {
#                             "loader_id": "rest",
#                             "strategy": "aggregate",
#                             "queries": [
#                                 {
#                                     "method": "GET",
#                                     "endpoint": "0.0.0.0/path?param1=true&param2=false",
#                                     "params": [
#                                         {"param1": "true"},
#                                         {"param2": "true"},
#                                     ],
#                                 },
#                             ],
#                         },
#                     }
#                 ]
#             }
#             msg = messages.ControlMessage(config)
#             yield msg

#         def _on_next(control_msg):
#             nonlocal packets_received
#             packets_received += 1
#             assert control_msg.payload().df.equals(df)

#         registry = mrc.ModuleRegistry

#         fn_constructor = registry.get_module_constructor("DataLoader", "morpheus")
#         assert fn_constructor is not None

#         source = builder.make_source("source", gen_data)

#         config = {
#             "loaders": [
#                 {
#                     "id": "rest",
#                     "properties": {"prop1": "something", "prop2": "something else"},
#                 }
#             ]
#         }
#         data_loader = builder.load_module(
#             "DataLoader", "morpheus", "ModuleDataLoaderTest", config
#         )

#         sink = builder.make_sink("sink", _on_next, on_error, on_complete)

#         builder.make_edge(source, data_loader.input_port("input"))
#         builder.make_edge(data_loader.output_port("output"), sink)

#     pipeline = mrc.Pipeline()
#     pipeline.make_segment("main", init_wrapper)

#     options = mrc.Options()
#     options.topology.user_cpuset = "0-1"

#     executor = mrc.Executor(options)
#     executor.register_pipeline(pipeline)
#     executor.start()
#     executor.join()
#     assert packets_received == 1

# def test_rest_loader_module_post_with_body():
#     packets_received = 0

#     df1 = cudf.DataFrame(
#         {
#             "col1": [1, 2],
#             "col2": ["POST_body_123", ""],
#         },
#         columns=["col1", "col2"],
#     )
#     df2 = cudf.DataFrame(
#         {
#             "col1": [1, 2],
#             "col2": ["POST_body_456", ""],
#         },
#         columns=["col1", "col2"],
#     )
#     df = cudf.concat([df1, df2])

#     def init_wrapper(builder: mrc.Builder):
#         def gen_data():
#             config = {
#                 "tasks": [
#                     {
#                         "type": "load",
#                         "properties": {
#                             "loader_id": "rest",
#                             "strategy": "aggregate",
#                             "queries": [
#                                 {
#                                     "method": "POST",
#                                     "endpoint": "0.0.0.0/path",
#                                     "params": [
#                                         {"something1": "something2"},
#                                     ],
#                                     "content_type": "text/plain",
#                                     "body": "123",
#                                 },
#                                 {
#                                     "method": "POST",
#                                     "endpoint": "0.0.0.0/path",
#                                     "params": [
#                                         {"something1": "something2"},
#                                     ],
#                                     "content_type": "text/plain",
#                                     "body": "456",
#                                 },
#                             ],
#                         },
#                     }
#                 ]
#             }
#             msg = messages.ControlMessage(config)
#             yield msg

#         def _on_next(control_msg):
#             nonlocal packets_received
#             packets_received += 1
#             assert control_msg.payload().df.equals(df)

#         registry = mrc.ModuleRegistry

#         fn_constructor = registry.get_module_constructor("DataLoader", "morpheus")
#         assert fn_constructor is not None

#         source = builder.make_source("source", gen_data)

#         config = {
#             "loaders": [
#                 {
#                     "id": "rest",
#                     "properties": {"prop1": "something", "prop2": "something else"},
#                 }
#             ]
#         }
#         # This will unpack the config and forward its payload (MessageMeta) to the sink
#         data_loader = builder.load_module(
#             "DataLoader", "morpheus", "ModuleDataLoaderTest", config
#         )

#         sink = builder.make_sink("sink", _on_next, on_error, on_complete)

#         builder.make_edge(source, data_loader.input_port("input"))
#         builder.make_edge(data_loader.output_port("output"), sink)

#     pipeline = mrc.Pipeline()
#     pipeline.make_segment("main", init_wrapper)

#     options = mrc.Options()
#     options.topology.user_cpuset = "0-1"

#     executor = mrc.Executor(options)
#     executor.register_pipeline(pipeline)
#     executor.start()
#     executor.join()
#     assert packets_received == 1

# def test_rest_loader_module_x_headers():
#     packets_received = 0

#     df1 = cudf.DataFrame(
#         {
#             "col1": [1, 2],
#             "col2": ["x_header_1", ""],
#         },
#         columns=["col1", "col2"],
#     )
#     df2 = cudf.DataFrame(
#         {
#             "col1": [1, 2],
#             "col2": ["x_header_2", ""],
#         },
#         columns=["col1", "col2"],
#     )
#     df = cudf.concat([df1, df2])

#     def init_wrapper(builder: mrc.Builder):
#         def gen_data():
#             config = {
#                 "tasks": [
#                     {
#                         "type": "load",
#                         "properties": {
#                             "loader_id": "rest",
#                             "strategy": "aggregate",
#                             "queries": [
#                                 {
#                                     "endpoint": "0.0.0.0/path",
#                                     "x-headers": {
#                                         "X-Header1": "x_header1",
#                                     },
#                                 },
#                                 {
#                                     "endpoint": "0.0.0.0/path",
#                                     "x-headers": {
#                                         "X-Header2": "x_header2",
#                                     },
#                                 },
#                             ],
#                         },
#                     }
#                 ]
#             }
#             msg = messages.ControlMessage(config)
#             yield msg

#         def _on_next(control_msg):
#             nonlocal packets_received
#             packets_received += 1
#             assert control_msg.payload().df.equals(df)

#         registry = mrc.ModuleRegistry

#         fn_constructor = registry.get_module_constructor("DataLoader", "morpheus")
#         assert fn_constructor is not None

#         source = builder.make_source("source", gen_data)

#         config = {
#             "loaders": [
#                 {
#                     "id": "rest",
#                     "properties": {"prop1": "something", "prop2": "something else"},
#                 }
#             ]
#         }
#         # This will unpack the config and forward its payload (MessageMeta) to the sink
#         data_loader = builder.load_module(
#             "DataLoader", "morpheus", "ModuleDataLoaderTest", config
#         )

#         sink = builder.make_sink("sink", _on_next, on_error, on_complete)

#         builder.make_edge(source, data_loader.input_port("input"))
#         builder.make_edge(data_loader.output_port("output"), sink)

#     pipeline = mrc.Pipeline()
#     pipeline.make_segment("main", init_wrapper)

#     options = mrc.Options()
#     options.topology.user_cpuset = "0-1"

#     executor = mrc.Executor(options)
#     executor.register_pipeline(pipeline)
#     executor.start()
#     executor.join()
#     assert packets_received == 1

if __name__ == "__main__":
    test_contains_namespace()
    test_is_version_compatible()
    test_get_module()
    test_payload_loader_module()
    # test_file_loader_module()
    test_filter_cm_failed()

    # Flask service needed for RESTDataloader tests
    # test_rest_loader_module_get_without_params()
    # test_rest_loader_module_get_with_params()
    # test_rest_loader_module_post_with_body()
    # test_rest_loader_module_x_headers()
