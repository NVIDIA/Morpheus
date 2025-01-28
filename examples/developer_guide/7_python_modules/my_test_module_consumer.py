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

from morpheus.utils.module_utils import register_module


@register_module("my_test_module_consumer", "my_module_namespace")
def my_test_module_consumer_initialization(builder: mrc.Builder):
    consumer_module_config = builder.get_current_module_config()  # noqa: F841 pylint:disable=unused-variable
    module_config = {"some_configuration_parameter": "some_value"}

    my_test_module = builder.load_module("my_test_module", "my_module_namespace", "module_instance_name", module_config)

    builder.register_module_input("input_0", my_test_module.input_port("input_0"))
    builder.register_module_output("output_0", my_test_module.output_port("output_0"))
