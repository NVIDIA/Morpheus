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

import mrc
import pytest

# When segment modules are imported, they're added to the module registry.
# To avoid flake8 warnings about unused code, the noqa flag is used during import.
import morpheus.modules  # noqa: F401 # pylint: disable=unused-import
from morpheus.pipeline.pipeline import Pipeline
from morpheus.stages.general.linear_modules_stage import LinearModulesStage
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.utils.module_ids import MORPHEUS_MODULE_NAMESPACE
from morpheus.utils.module_ids import PAYLOAD_BATCHER
from morpheus.utils.module_ids import TO_CONTROL_MESSAGE

TIMESTAMPS = [
    '2023-01-23 06:30:36',
    '2023-01-22 00:30:22',
    '2023-01-23 17:01:23',
    '2023-01-23 06:54:22',
    '2023-01-23 12:18:44',
    '2023-01-22 02:47:47',
    '2023-01-23 16:21:51',
    '2023-01-15 16:08:14',
    '2023-01-15 10:00:22',
    '2023-01-15 20:07:54',
    '2023-01-16 01:29:40',
    '2023-01-23 13:29:14',
    '2023-01-22 11:42:07',
    '2023-01-22 10:49:34',
    '2023-01-16 17:31:41',
    '2023-01-23 22:40:05',
    '2023-01-15 01:49:54',
    '2023-01-16 09:14:35',
    '2023-01-23 03:52:01',
    '2023-01-16 06:59:32'
]


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

    fn_constructor = registry.get_module_constructor("PayloadBatcher", "morpheus")
    assert fn_constructor is not None

    config = {}
    module_instance = fn_constructor("PayloadBatcherTest", config)  # pylint: disable=not-callable
    assert isinstance(module_instance, mrc.core.segment.SegmentModule)


@pytest.mark.gpu_mode
@pytest.mark.parametrize(
    "max_batch_size, raise_on_failure, group_by_columns, disable_max_batch_size, timestamp_column_name, "
    "timestamp_pattern, period, expected_count, expected_exception",
    [
        (10, True, [], False, None, None, None, 2, None),  # uses max_batch_size ignores all other params
        (5, True, [], False, None, None, None, 4, None),  # uses max_batch_size ignores all other params
        (7, True, [], False, None, None, None, 3, None),  # uses max_batch_size ignores all other params
        (10, True, [], True, "timestamp", "%Y-%m-%d %H:%M:%S", "D", 4, None),  # uses timestamp and period, ignores rest
        (10, True, [], True, "timestamp", "%Y-%m-%d %H:%M:%S", "Y", 1, None),  # uses timestamp and period, ignores rest
        (10, True, [], True, "timestamp", "%Y-%m-%d %H:%M:%S", "M", 1, None),  # uses timestamp and period, ignores rest
        (10, True, ["v1"], True, "timestamp", "%Y-%m-%d %H:%M:%S", "M", 9,
         None),  # uses v1,timestamp and period, ignores rest
        (10, True, ["v1", "v2"], True, "timestamp", "%Y-%m-%d %H:%M:%S", "M", 19,
         None),  # uses v1, v2, timestamp and period, ignores rest
        (10, True, ["v1", "v2"], True, None, None, None, 19, None),  # uses group_by_columns v1, v2, ignores rest
        (8, True, [], False, "timestamp", "%Y-%m-%d %H:%M:%S", "M", 3,
         None),  # applies max_batch_size condition on period groups
        (10, True, [], True, "timestamp", None, "D", 4, None),  # uses timestamp and period, ignores rest
        (10,
         True, [],
         True,
         None,
         None,
         None,
         None,
         ValueError("""When disable_max_batch_size is True and group_by_columns must not be empty or None.""")
         ),  # raises error disable_max_batch_size is True and group_by_columns is empty
        (10,
         True,
         None,
         True,
         None,
         None,
         None,
         None,
         ValueError("""When disable_max_batch_size is True and group_by_columns must not be empty or None.""")
         ),  # raises error disable_max_batch_size is True and group_by_columns is empty
    ],
)
def test_custom_params(config,
                       filter_probs_df,
                       max_batch_size,
                       raise_on_failure,
                       group_by_columns,
                       disable_max_batch_size,
                       timestamp_column_name,
                       timestamp_pattern,
                       period,
                       expected_count,
                       expected_exception):

    expected_warning = False
    if timestamp_column_name:
        filter_probs_df["timestamp"] = TIMESTAMPS
        expected_warning = timestamp_pattern is None

    pipe = Pipeline(config)

    # Create the stages
    source_stage = pipe.add_stage(InMemorySourceStage(config, dataframes=[filter_probs_df]))

    to_cm_module_config = {
        "module_id": TO_CONTROL_MESSAGE,
        "module_name": "to_control_message",
        "namespace": MORPHEUS_MODULE_NAMESPACE,
        "meta_data": {
            "data_type": "streaming"
        },
        "tasks": [{
            "type": "inference", "properties": {}
        }]
    }
    payload_batcher_module_config = {
        "module_id": PAYLOAD_BATCHER,
        "module_name": "payload_batcher",
        "namespace": MORPHEUS_MODULE_NAMESPACE,
        "max_batch_size": max_batch_size,
        "raise_on_failure": raise_on_failure,
        "group_by_columns": group_by_columns,
        "disable_max_batch_size": disable_max_batch_size,
        "timestamp_column_name": timestamp_column_name,
        "timestamp_pattern": timestamp_pattern,
        "period": period
    }

    to_control_message_stage = pipe.add_stage(
        LinearModulesStage(config, to_cm_module_config, input_port_name="input", output_port_name="output"))

    payload_batcher_stage = pipe.add_stage(
        LinearModulesStage(config, payload_batcher_module_config, input_port_name="input", output_port_name="output"))

    sink_stage = pipe.add_stage(InMemorySinkStage(config))

    pipe.add_edge(source_stage, to_control_message_stage)
    pipe.add_edge(to_control_message_stage, payload_batcher_stage)
    pipe.add_edge(payload_batcher_stage, sink_stage)

    if expected_exception:
        with pytest.raises(type(expected_exception), match=str(expected_exception)):
            pipe.run()
    elif expected_warning:
        with pytest.warns(UserWarning):
            pipe.run()
            assert len(sink_stage.get_messages()) == expected_count
    else:
        pipe.run()
        assert len(sink_stage.get_messages()) == expected_count


@pytest.mark.gpu_mode
def test_default_params(config, filter_probs_df):

    pipe = Pipeline(config)

    # Create the stages
    source_stage = pipe.add_stage(InMemorySourceStage(config, dataframes=[filter_probs_df]))

    to_cm_module_config = {
        "module_id": TO_CONTROL_MESSAGE,
        "module_name": "to_control_message",
        "namespace": MORPHEUS_MODULE_NAMESPACE,
        "meta_data": {
            "data_type": "streaming"
        },
        "tasks": [{
            "type": "inference", "properties": {}
        }]
    }
    payload_batcher_module_config = {
        "module_id": PAYLOAD_BATCHER, "module_name": "payload_batcher", "namespace": MORPHEUS_MODULE_NAMESPACE
    }

    to_control_message_stage = pipe.add_stage(
        LinearModulesStage(config, to_cm_module_config, input_port_name="input", output_port_name="output"))

    payload_batcher_stage = pipe.add_stage(
        LinearModulesStage(config, payload_batcher_module_config, input_port_name="input", output_port_name="output"))

    sink_stage = pipe.add_stage(InMemorySinkStage(config))

    pipe.add_edge(source_stage, to_control_message_stage)
    pipe.add_edge(to_control_message_stage, payload_batcher_stage)
    pipe.add_edge(payload_batcher_stage, sink_stage)

    pipe.run()

    assert len(sink_stage.get_messages()) == 1
