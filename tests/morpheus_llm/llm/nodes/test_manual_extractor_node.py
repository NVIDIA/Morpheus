# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from _utils.llm import execute_node
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus_llm.llm import LLMNodeBase
from morpheus_llm.llm.nodes.extracter_node import ManualExtracterNode


def test_constructor():
    node = ManualExtracterNode(["test"])
    assert isinstance(node, LLMNodeBase)


@pytest.mark.parametrize("input_names", [[], ["a", "b", "a"]], ids=["empty", "duplicate"])
def test_constructor_invalid_input_names(input_names: list[str]):
    with pytest.raises(AssertionError):
        ManualExtracterNode(input_names)


@pytest.mark.parametrize("input_names", [["test"], ["a", "b", "c"], ["a", "b", "c", "d"]])
def test_get_input_names(input_names: list[str]):
    node = ManualExtracterNode(input_names)
    assert len(node.get_input_names()) == len(input_names)


def test_execute():
    insects = ["ant", "bee", "butterfly", "mosquito", "grasshopper"]
    mammals = ["lion", "dolphin", "gorilla", "wolf", "tiger"]
    reptiles = ['lizards', 'snakes', 'turtles', 'frogs', 'toads']
    df = cudf.DataFrame({"insects": insects.copy(), "mammals": mammals.copy(), "reptiles": reptiles.copy()})
    message = ControlMessage()
    message.payload(MessageMeta(df))
    message.set_metadata("llm_message_meta", message.payload())

    task_dict = {"input_keys": ["insects"]}
    node = ManualExtracterNode(["mammals", "reptiles"])
    assert execute_node(node, task_dict=task_dict, input_message=message) == {"mammals": mammals, "reptiles": reptiles}
