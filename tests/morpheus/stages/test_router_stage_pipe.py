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

import threading

import pytest

from morpheus.config import ExecutionMode
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.pipeline import Pipeline
from morpheus.pipeline.stage_decorator import stage
from morpheus.stages.general.router_stage import RouterStage
from morpheus.stages.input.in_memory_data_generation_stage import InMemoryDataGenStage
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage


@pytest.mark.parametrize("processing_engines", [0, 4])
def test_router_stage_pipe(config, filter_probs_df, processing_engines: bool):

    keys = ["odd", "even"]

    count = 0

    def determine_route_fn(_: ControlMessage):
        nonlocal count
        count += 1
        return keys[count % len(keys)]

    pipe = Pipeline(config)
    source = pipe.add_stage(InMemorySourceStage(config, dataframes=[filter_probs_df], repeat=5))
    deserialize = pipe.add_stage(DeserializeStage(config))
    router_stage = pipe.add_stage(
        RouterStage(config, keys=keys, key_fn=determine_route_fn, processing_engines=processing_engines))
    sink1 = pipe.add_stage(InMemorySinkStage(config))
    sink2 = pipe.add_stage(InMemorySinkStage(config))

    # Connect the stages
    pipe.add_edge(source, deserialize)
    pipe.add_edge(deserialize, router_stage)
    pipe.add_edge(router_stage.output_ports[0], sink1)
    pipe.add_edge(router_stage.output_ports[1], sink2)

    pipe.run()

    assert len(sink1.get_messages()) == 2, "Expected 2 messages in sink1"
    assert len(sink2.get_messages()) == 3, "Expected 3 messages in sink2"


def test_router_stage_backpressure_pipe(config, filter_probs_df):

    # This test simulates a slow single consumer by blocking the second output port of the router stage The router stage
    # will buffer the messages and block the source stage from sending more data When run as a component, less threads
    # will be used but this system will eventually block. With a runnable, this should be able to run to completion

    # Set the edge buffer size to trigger blocking
    config.edge_buffer_size = 4

    keys = ["odd", "even"]

    count = 0

    release_event = threading.Event()

    def source_fn():

        for i in range(20):
            cm = ControlMessage()
            cm.set_metadata("index", i)
            cm.payload(MessageMeta(filter_probs_df))
            yield cm

        # Release the event to allow the pipeline to continue
        release_event.set()

        # Send more data
        for i in range(20, 30):
            cm = ControlMessage()
            cm.set_metadata("index", i)
            cm.payload(MessageMeta(filter_probs_df))
            yield cm

    def determine_route_fn(_: ControlMessage):
        nonlocal count
        count += 1
        return keys[count % len(keys)]

    pipe = Pipeline(config)

    source = pipe.add_stage(InMemoryDataGenStage(config, data_source=source_fn, output_data_type=ControlMessage))
    router_stage = pipe.add_stage(RouterStage(config, keys=keys, key_fn=determine_route_fn, processing_engines=10))
    sink1 = pipe.add_stage(InMemorySinkStage(config))
    sink2 = pipe.add_stage(InMemorySinkStage(config))

    @stage(execution_modes=[ExecutionMode.CPU, ExecutionMode.GPU])
    def blocking_stage(data: ControlMessage) -> ControlMessage:

        release_event.wait()

        return data

    blocking = pipe.add_stage(blocking_stage(config))

    # Connect the stages
    pipe.add_edge(source, router_stage)
    pipe.add_edge(router_stage.output_ports[0], sink1)
    pipe.add_edge(router_stage.output_ports[1], blocking)
    pipe.add_edge(blocking, sink2)

    pipe.run()

    assert len(sink1.get_messages()) == 15, "Expected 15 messages in sink1"
    assert len(sink2.get_messages()) == 15, "Expected 15 messages in sink2"
