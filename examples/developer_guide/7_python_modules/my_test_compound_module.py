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

import mrc
from mrc.core import operators as ops

from morpheus.messages import MessageMeta
from morpheus.utils.module_utils import register_module


# Create and register our two new component modules
# ==========================================
@register_module("my_square_module", "my_module_namespace")
def my_square_module_initialization(builder: mrc.Builder):
    module_config = builder.get_current_module_config()
    field_name = module_config.get("field_name", "data")

    def on_data(msg: MessageMeta) -> MessageMeta:
        with msg.mutable_dataframe() as df:
            df[field_name] = df[field_name]**2

        return msg

    def node_fn(observable: mrc.Observable, subscriber: mrc.Subscriber):
        observable.pipe(ops.map(on_data)).subscribe(subscriber)

    node = builder.make_node("square", mrc.core.operators.build(node_fn))

    builder.register_module_input("input_0", node)
    builder.register_module_output("output_0", node)


@register_module("my_times_three_module", "my_module_namespace")
def my_times_three_module_initialization(builder: mrc.Builder):
    module_config = builder.get_current_module_config()
    field_name = module_config.get("field_name", "data")

    def on_data(msg: MessageMeta) -> MessageMeta:
        with msg.mutable_dataframe() as df:
            df[field_name] = df[field_name] * 3

        return msg

    def node_fn(observable: mrc.Observable, subscriber: mrc.Subscriber):
        observable.pipe(ops.map(on_data)).subscribe(subscriber)

    node = builder.make_node("times_two", mrc.core.operators.build(node_fn))

    builder.register_module_input("input_0", node)
    builder.register_module_output("output_0", node)


# Create and register our new compound operator -- illustrates module chaining
@register_module("my_compound_op_module", "my_module_namespace")
def my_compound_op(builder: mrc.Builder):
    square_module = builder.load_module("my_square_module", "my_module_namespace", "square_module", {})
    times_three_module = builder.load_module("my_times_three_module", "my_module_namespace", "times_three_module", {})

    builder.make_edge(square_module.output_port("output_0"), times_three_module.input_port("input_0"))

    builder.register_module_input("input_0", square_module.input_port("input_0"))
    builder.register_module_output("output_0", times_three_module.output_port("output_0"))


# Create and register our new compound module -- illustrates module nesting
@register_module("my_compound_module", "my_module_namespace")
def my_compound_module(builder: mrc.Builder):
    op_module = builder.load_module("my_compound_op_module", "my_module_namespace", "op_module", {})

    builder.register_module_input("input_0", op_module.input_port("input_0"))
    builder.register_module_output("output_0", op_module.output_port("output_0"))
