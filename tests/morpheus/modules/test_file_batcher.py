#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest

import cudf

import morpheus.modules  # noqa: F401 # pylint: disable=unused-import
from morpheus.config import Config
from morpheus.messages import ControlMessage
from morpheus.messages.message_meta import MessageMeta
from morpheus.pipeline import LinearPipeline
from morpheus.pipeline.stage_decorator import source
from morpheus.stages.general.linear_modules_stage import LinearModulesStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.utils.module_ids import FILE_BATCHER
from morpheus.utils.module_ids import MORPHEUS_MODULE_NAMESPACE

# pylint: disable=redundant-keyword-arg


@source
def source_test_stage(filenames: list[str], cm_batching_options: dict) -> ControlMessage:

    df = cudf.DataFrame(filenames, columns=['files'])

    control_message = ControlMessage()

    control_message.set_metadata("batching_options", cm_batching_options)
    control_message.set_metadata("data_type", "payload")

    control_message.payload(MessageMeta(df=df))

    yield control_message


@pytest.fixture(name="default_module_config")
def default_module_config_fixture():
    yield {
        "module_id": FILE_BATCHER,
        "module_name": "file_batcher",
        "namespace": MORPHEUS_MODULE_NAMESPACE,
        "sampling_rate_s": 0,
        "start_time": "2022-08-01T00:00:00",
        "end_time": "2022-08-31T00:00:00",
        "parser_kwargs": None,
        "schema": {
            "schema_str": None, "encoding": None
        }
    }


@pytest.fixture(name="default_file_list")
def default_file_list_fixture():
    yield [
        "DUO_2022-08-01T00_05_06.806Z.json",
        "DUO_2022-08-01T03_02_04.418Z.json",
        "DUO_2022-08-01T06_05_05.064Z.json",
        "DUO_2022-08-02T00_05_06.806Z.json",
        "DUO_2022-08-02T03_02_04.418Z.json",
        "DUO_2022-08-02T06_05_05.064Z.json"
    ]


def test_no_overrides(config: Config, default_module_config, default_file_list):
    pipeline = LinearPipeline(config)

    cm_batching_opts = {
        "sampling_rate_s": 0,
        "start_time": "2022-08-01",
        "end_time": "2022-08-31",
        "parser_kwargs": None,
        "schema": {
            "schema_str": None, "encoding": None
        }
    }

    pipeline.set_source(source_test_stage(config, filenames=default_file_list, cm_batching_options=cm_batching_opts))

    pipeline.add_stage(
        LinearModulesStage(config, default_module_config, input_port_name="input", output_port_name="output"))

    sink_stage = pipeline.add_stage(InMemorySinkStage(config))

    pipeline.run()

    sink_messages = sink_stage.get_messages()
    assert len(sink_messages) == 2
    assert len(sink_messages[0].get_tasks()["load"][0]["files"]) == 3
    assert sink_messages[0].get_tasks()["load"][0]["n_groups"] == 2
    assert len(sink_messages[1].get_tasks()["load"][0]["files"]) == 3
    assert sink_messages[1].get_tasks()["load"][0]["n_groups"] == 2


def test_no_date_matches(config: Config, default_module_config, default_file_list):
    pipeline = LinearPipeline(config)

    cm_batching_opts = {
        "sampling_rate_s": 0,
        "start_time": "2022-09-01",
        "end_time": "2022-09-30",
        "parser_kwargs": None,
        "schema": {
            "schema_str": None, "encoding": None
        }
    }

    pipeline.set_source(source_test_stage(config, filenames=default_file_list, cm_batching_options=cm_batching_opts))

    pipeline.add_stage(
        LinearModulesStage(config, default_module_config, input_port_name="input", output_port_name="output"))

    sink_stage = pipeline.add_stage(InMemorySinkStage(config))

    pipeline.run()

    sink_messages = sink_stage.get_messages()
    assert len(sink_messages) == 0


def test_partial_date_matches(config: Config, default_module_config, default_file_list):
    pipeline = LinearPipeline(config)

    cm_batching_opts = {
        "sampling_rate_s": 0,
        "start_time": "2022-07-30",
        "end_time": "2022-08-02",
        "parser_kwargs": None,
        "schema": {
            "schema_str": None, "encoding": None
        }
    }

    pipeline.set_source(source_test_stage(config, filenames=default_file_list, cm_batching_options=cm_batching_opts))

    pipeline.add_stage(
        LinearModulesStage(config, default_module_config, input_port_name="input", output_port_name="output"))

    sink_stage = pipeline.add_stage(InMemorySinkStage(config))

    pipeline.run()

    sink_messages = sink_stage.get_messages()
    sink_messages = sink_stage.get_messages()
    assert len(sink_messages) == 1
    assert len(sink_messages[0].get_tasks()["load"][0]["files"]) == 3
    assert sink_messages[0].get_tasks()["load"][0]["n_groups"] == 1


def test_override_date_regex(config: Config, default_module_config):
    pipeline = LinearPipeline(config)

    filenames = [
        "DUO_2022-08-01_00_05_06.806Z.json",
        "DUO_2022-08-01_03_02_04.418Z.json",
        "DUO_2022-08-01_06_05_05.064Z.json",
        "DUO_2022-08-02_00_05_06.806Z.json",
        "DUO_2022-08-02_03_02_04.418Z.json",
        "DUO_2022-08-02_06_05_05.064Z.json"
    ]

    cm_date_regex_pattern = (
        r"(?P<year>\d{4})-(?P<month>\d{1,2})-(?P<day>\d{1,2})"
        r"_(?P<hour>\d{1,2})(:|_)(?P<minute>\d{1,2})(:|_)(?P<second>\d{1,2})(?P<microsecond>\.\d{1,6})?Z")

    cm_batching_opts = {
        "sampling_rate_s": 0,
        "start_time": "2022-08-01",
        "end_time": "2022-08-31",
        "iso_date_regex_pattern": cm_date_regex_pattern,
        "parser_kwargs": None,
        "schema": {
            "schema_str": None, "encoding": None
        }
    }

    pipeline.set_source(source_test_stage(config, filenames=filenames, cm_batching_options=cm_batching_opts))

    pipeline.add_stage(
        LinearModulesStage(config, default_module_config, input_port_name="input", output_port_name="output"))

    sink_stage = pipeline.add_stage(InMemorySinkStage(config))

    pipeline.run()

    sink_messages = sink_stage.get_messages()
    assert len(sink_messages) == 2
    assert len(sink_messages[0].get_tasks()["load"][0]["files"]) == 3
    assert sink_messages[0].get_tasks()["load"][0]["n_groups"] == 2
    assert len(sink_messages[1].get_tasks()["load"][0]["files"]) == 3
    assert sink_messages[1].get_tasks()["load"][0]["n_groups"] == 2


def test_sampling_freq(config: Config, default_module_config):
    pipeline = LinearPipeline(config)

    filenames = [
        "DUO_2022-08-01T00_05_06.806Z.json",
        "DUO_2022-08-01T00_05_08.418Z.json",
        "DUO_2022-08-01T00_05_12.064Z.json",
        "DUO_2022-08-02T03_02_06.806Z.json",
        "DUO_2022-08-02T03_02_14.418Z.json",
        "DUO_2022-08-02T03_02_17.064Z.json"
    ]

    cm_batching_opts = {
        "sampling_rate_s": None,
        "sampling": "30S",
        "start_time": "2022-08-01",
        "end_time": "2022-08-31",
        "parser_kwargs": None,
        "schema": {
            "schema_str": None, "encoding": None
        }
    }

    pipeline.set_source(source_test_stage(config, filenames=filenames, cm_batching_options=cm_batching_opts))

    pipeline.add_stage(
        LinearModulesStage(config, default_module_config, input_port_name="input", output_port_name="output"))

    sink_stage = pipeline.add_stage(InMemorySinkStage(config))

    pipeline.run()

    sink_messages = sink_stage.get_messages()
    assert len(sink_messages) == 2
    assert len(sink_messages[0].get_tasks()["load"][0]["files"]) == 1
    assert sink_messages[0].get_tasks()["load"][0]["n_groups"] == 2
    assert len(sink_messages[1].get_tasks()["load"][0]["files"]) == 1
    assert sink_messages[1].get_tasks()["load"][0]["n_groups"] == 2


def test_sampling_pct(config: Config, default_module_config, default_file_list):
    pipeline = LinearPipeline(config)

    cm_batching_opts = {
        "sampling_rate_s": None,
        "sampling": 0.5,
        "start_time": "2022-08-01",
        "end_time": "2022-08-31",
        "parser_kwargs": None,
        "schema": {
            "schema_str": None, "encoding": None
        }
    }

    pipeline.set_source(source_test_stage(config, filenames=default_file_list, cm_batching_options=cm_batching_opts))

    pipeline.add_stage(
        LinearModulesStage(config, default_module_config, input_port_name="input", output_port_name="output"))

    sink_stage = pipeline.add_stage(InMemorySinkStage(config))

    pipeline.run()

    sink_messages = sink_stage.get_messages()
    msg_counts = [len(m.get_tasks()["load"][0]["files"]) for m in sink_messages]
    assert sum(msg_counts) == 3


def test_sampling_fixed(config: Config, default_module_config, default_file_list):
    pipeline = LinearPipeline(config)

    cm_batching_opts = {
        "sampling_rate_s": None,
        "sampling": 5,
        "start_time": "2022-08-01",
        "end_time": "2022-08-31",
        "parser_kwargs": None,
        "schema": {
            "schema_str": None, "encoding": None
        }
    }

    pipeline.set_source(source_test_stage(config, filenames=default_file_list, cm_batching_options=cm_batching_opts))

    pipeline.add_stage(
        LinearModulesStage(config, default_module_config, input_port_name="input", output_port_name="output"))

    sink_stage = pipeline.add_stage(InMemorySinkStage(config))

    pipeline.run()

    sink_messages = sink_stage.get_messages()
    msg_counts = [len(m.get_tasks()["load"][0]["files"]) for m in sink_messages]
    assert sum(msg_counts) == 5
