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

import types
import typing

import mrc

import morpheus.modules  # noqa: F401 # pylint: disable=unused-import


class MorpheusModuleTestPipeline:

    def __init__(self,
                 module_id: str,
                 registry_namespace: str,
                 module_name: str,
                 module_config: dict,
                 input_port: str,
                 output_port: str,
                 gen_data: types.FunctionType,
                 on_next: typing.Callable[[object], None] = None,
                 on_error: typing.Callable[[BaseException], None] = None,
                 on_complete: typing.Callable[[], None] = None,
                 user_cpuset: str = "0-1"):
        self._module_id = module_id
        self._registry_namespace = registry_namespace
        self._module_name = module_name
        self._module_config = module_config
        self._gen_data = gen_data
        self._input_port = input_port
        self._output_port = output_port
        self._on_next = on_next
        self._on_error = on_error
        self._on_complete = on_complete
        self._user_cpuset = user_cpuset

        def init_wrapper(builder: mrc.Builder):
            source = builder.make_source("source", gen_data)
            module = builder.load_module(
                self._module_id,
                self._registry_namespace,
                self._module_name,
                self._module_config,
            )
            sink = builder.make_sink("sink", self._on_next, self._on_error, self._on_complete)

            builder.make_edge(source, module.input_port(self._input_port))
            builder.make_edge(module.output_port(self._output_port), sink)

        self._pipeline = mrc.Pipeline()
        self._pipeline.make_segment("main", init_wrapper)

    def run(self):
        options = mrc.Options()
        options.topology.user_cpuset = self._user_cpuset

        executor = mrc.Executor(options)
        executor.register_pipeline(self._pipeline)
        executor.start()
        executor.join()
