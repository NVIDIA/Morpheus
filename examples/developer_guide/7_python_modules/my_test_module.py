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

from morpheus.utils.module_utils import register_module


@register_module("my_test_module", "my_module_namespace")
def my_test_module_initialization(builder: mrc.Builder):
    module_config = builder.get_current_module_config()  # noqa: F841 pylint:disable=unused-variable

    def on_data(data):
        return data

    def node_fn(observable: mrc.Observable, subscriber: mrc.Subscriber):
        observable.pipe(ops.map(on_data)).subscribe(subscriber)

    node = builder.make_node("my_test_module_forwarding_node", mrc.core.operators.build(node_fn))

    builder.register_module_input("input_0", node)
    builder.register_module_output("output_0", node)
